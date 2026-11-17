#!/usr/bin/env python3
"""Inventory job: READS every video in a bucket (it never changes or deletes anything) and reports
how long, how big and in what resolution / frame rate they are, so the processing cost can be
estimated from facts instead of guesses.

Usage:
    python3 /app/inventory_job.py <base64-encoded JSON>

JSON keys (all optional)
  bucket            default nightowl-intvw-str
  exclude_prefixes  default ["background-replacement-processed/"]
  extensions        default ["mp4", "webm", "mov", "m4v", "mkv"]
  report_url        n8n webhook that receives the anonymous summary (no file names)
  report_key        where the private CSV with every file is saved
                    (default background-replacement-processed/_reports/inventory.csv)
  workers           parallel probes (default 6)
  limit             only the first N videos (for a quick trial)
"""

import base64
import csv
import io
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

import boto3
import httpx
from botocore.config import Config

DO_SPACES_KEY = os.getenv("DO_SPACES_KEY")
DO_SPACES_SECRET = os.getenv("DO_SPACES_SECRET")
DO_SPACES_REGION = os.getenv("DO_SPACES_REGION", "sfo3")
WHOLE_JOB_SECONDS = 4 * 3600

DURATION_BUCKETS = [(0, 120, "under 2 min"), (120, 300, "2-5 min"), (300, 600, "5-10 min"),
                    (600, 1200, "10-20 min"), (1200, 2400, "20-40 min"), (2400, 10 ** 9, "over 40 min")]


def make_s3():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{DO_SPACES_REGION}.digitaloceanspaces.com",
        aws_access_key_id=DO_SPACES_KEY,
        aws_secret_access_key=DO_SPACES_SECRET,
        config=Config(signature_version="s3v4"),
    )


def list_videos(s3, bucket, exclude_prefixes, extensions, limit):
    found = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/") or any(key.startswith(p) for p in exclude_prefixes):
                continue
            ext = key.rsplit(".", 1)[-1].lower() if "." in key else ""
            if ext in extensions:
                found.append((key, int(obj["Size"])))
    return found[:limit] if limit else found


def public_url(bucket, key):
    return f"https://{bucket}.{DO_SPACES_REGION}.digitaloceanspaces.com/{quote(key)}"


def signed_url(s3, bucket, key):
    return s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=3600)


def _fps(stream):
    for field in ("avg_frame_rate", "r_frame_rate"):
        try:
            num, den = stream.get(field, "0/0").split("/")
            value = float(num) / float(den) if float(den) else 0.0
            if 0 < value <= 120:
                return value
        except Exception:
            pass
    return 0.0


def run_ffprobe(url):
    cmd = ["ffprobe", "-v", "error", "-rw_timeout", "30000000", "-print_format", "json",
           "-show_entries", "format=duration:stream=codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate",
           url]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError("ffprobe failed")
    return json.loads(proc.stdout)


def probe_one(key, size, urls):
    row = {"key": key, "size_mb": round(size / 1048576, 1), "duration_s": 0.0, "width": 0, "height": 0,
           "fps": 0.0, "frames": 0, "vcodec": "", "has_audio": False, "error": ""}
    data, last = None, None
    for url in urls():
        try:
            data = run_ffprobe(url)
            break
        except Exception as e:
            last = e
    if data is None:
        row["error"] = "could not read the file" if not isinstance(last, subprocess.TimeoutExpired) else "timeout"
        return row
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    row["has_audio"] = any(s.get("codec_type") == "audio" for s in streams)
    if video is None:
        row["error"] = "no video stream"
        return row
    try:
        row["duration_s"] = round(float(data.get("format", {}).get("duration") or 0), 2)
    except Exception:
        row["duration_s"] = 0.0
    row["width"], row["height"] = int(video.get("width") or 0), int(video.get("height") or 0)
    row["fps"] = round(_fps(video), 2)
    row["vcodec"] = video.get("codec_name", "")
    row["frames"] = int(row["duration_s"] * row["fps"])
    if row["duration_s"] <= 0 or row["fps"] <= 0:
        row["error"] = "length or frame rate unknown"
    return row


def resolution_class(height):
    if height >= 1000:
        return "1080p or larger"
    if height >= 600:
        return "720p"
    return "480p or smaller"


def summarize(rows, total_files):
    ok = [r for r in rows if not r["error"]]
    failed = [r for r in rows if r["error"]]
    out = {
        "files_listed": total_files,
        "files_ok": len(ok),
        "files_failed": len(failed),
        "failure_reasons": {},
        "total_hours": round(sum(r["duration_s"] for r in ok) / 3600, 1),
        "total_frames": sum(r["frames"] for r in ok),
        "total_gb": round(sum(r["size_mb"] for r in rows) / 1024, 1),
        "average_minutes": round(sum(r["duration_s"] for r in ok) / max(1, len(ok)) / 60, 1),
        "by_length": [], "by_resolution": {}, "by_fps": {}, "audio_missing": sum(1 for r in ok if not r["has_audio"]),
        "longest_minutes": sorted((round(r["duration_s"] / 60, 1) for r in ok), reverse=True)[:12],
    }
    for r in failed:
        out["failure_reasons"][r["error"]] = out["failure_reasons"].get(r["error"], 0) + 1
    for lo, hi, name in DURATION_BUCKETS:
        sel = [r for r in ok if lo <= r["duration_s"] < hi]
        out["by_length"].append({"length": name, "videos": len(sel),
                                 "hours": round(sum(r["duration_s"] for r in sel) / 3600, 1),
                                 "frames": sum(r["frames"] for r in sel)})
    for r in ok:
        cls = resolution_class(r["height"])
        slot = out["by_resolution"].setdefault(cls, {"videos": 0, "frames": 0, "megapixel_frames": 0.0})
        slot["videos"] += 1
        slot["frames"] += r["frames"]
        slot["megapixel_frames"] += r["frames"] * r["width"] * r["height"] / 1e6
        fps_name = f"{round(r['fps'])} fps"
        out["by_fps"][fps_name] = out["by_fps"].get(fps_name, 0) + 1
    for slot in out["by_resolution"].values():
        slot["megapixel_frames"] = round(slot["megapixel_frames"])
    out["by_fps"] = dict(sorted(out["by_fps"].items(), key=lambda kv: kv[1], reverse=True)[:10])
    return out


def post_report(url, summary):
    for attempt in range(1, 4):
        try:
            response = httpx.post(url, json=summary, timeout=30)
            print(f"Report sent: HTTP {response.status_code}", flush=True)
            if response.status_code < 400:
                return True
        except Exception as e:
            print(f"Report attempt {attempt} failed: {e}", flush=True)
        time.sleep(5)
    return False


def main(s3=None, urls_for=None) -> int:
    if len(sys.argv) != 2:
        print("Usage: inventory_job.py <base64-encoded JSON>", flush=True)
        return 2
    try:
        cfg = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
    except Exception as e:
        print(f"Bad payload: {e}", flush=True)
        return 2

    bucket = cfg.get("bucket", "nightowl-intvw-str")
    exclude = cfg.get("exclude_prefixes", ["background-replacement-processed/"])
    exts = [e.lower() for e in cfg.get("extensions", ["mp4", "webm", "mov", "m4v", "mkv"])]
    report_url = cfg.get("report_url")
    report_key = cfg.get("report_key", "background-replacement-processed/_reports/inventory.csv")
    workers = int(cfg.get("workers", 6))
    limit = int(cfg.get("limit") or 0)

    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, lambda *_: (print("Stopped: took too long", flush=True), os._exit(1)))
        signal.alarm(WHOLE_JOB_SECONDS)

    s3 = s3 or make_s3()
    videos = list_videos(s3, bucket, exclude, exts, limit)
    print(f"Found {len(videos)} videos to inspect (read-only)", flush=True)

    if urls_for is None:
        def urls_for(key):
            return lambda: [public_url(bucket, key), signed_url(s3, bucket, key)]

    rows, started = [], time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(probe_one, key, size, urls_for(key)) for key, size in videos]
        for i, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if i % 50 == 0 or i == len(futures):
                print(f"  inspected {i}/{len(futures)} ({time.time() - started:.0f}s)", flush=True)
    rows.sort(key=lambda r: r["key"])

    summary = summarize(rows, len(videos))
    print("SUMMARY " + json.dumps(summary), flush=True)

    # private CSV with one line per file (this one DOES contain the file names)
    try:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()) if rows else ["key"])
        writer.writeheader()
        writer.writerows(rows)
        s3.put_object(Bucket=bucket, Key=report_key, Body=buf.getvalue().encode("utf-8"), ContentType="text/csv")
        print(f"CSV saved (private): {report_key}", flush=True)
    except Exception as e:
        print(f"CSV could not be saved: {e}", flush=True)

    if report_url:
        post_report(report_url, summary)
    print("Inventory finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
