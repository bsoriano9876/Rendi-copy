#!/usr/bin/env python3
"""One-off Render job: process ONE video, upload the result, optionally call
back n8n, then exit. Billed per second only while this runs.

Usage:
    python3 /app/run_job_v2.py <base64-encoded JSON>

JSON keys
  video_url      required  public URL of the source video
  background     optional  meeting_light | meeting_dark (default meeting_dark)
  candidate_id   optional  only used for the default output name
  callback_url   optional  if missing/empty, no callback is sent
  ref            optional  echoed back in the callback (e.g. a queue row id)
  max_frames     optional  only process the first N frames (quick tests); default = all
  debug_frames   optional  list of frame numbers: save a diagnosis picture for each next to the video
  output_bucket  optional  default = DO_SPACES_BUCKET
  output_key     optional  default = moodle_test_center/branded_videos/<candidate_id>_<job_id>.mp4
  output_acl     optional  default = public-read (old behaviour); use "private" to keep it private

Payloads from the existing Airtable and Supabase workflows (video_url,
candidate_id, background, callback_url) are accepted exactly as before. This script
uses brand_videos_stream.py (frames are streamed into one ffmpeg encode, so no
multi-GB temp file) and a time limit that scales with the video length.
"""

import base64
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import quote

import boto3
import httpx

# --- time limits -------------------------------------------------------------
# Processing may take TIME_FACTOR seconds per second of video (measured about 4.6x on
# 720p and about 7x on 1080p), never less than MIN and never more than MAX.
MIN_PROCESSING_SECONDS = int(os.getenv("RUN_JOB_MIN_SECONDS", "1200"))
MAX_PROCESSING_SECONDS = int(os.getenv("RUN_JOB_MAX_SECONDS", "21600"))
TIME_FACTOR = 10
UNKNOWN_DURATION_SECONDS = 3600
# Hard stop for the WHOLE script (download, upload and callback included).
WATCHDOG_EXTRA_SECONDS = int(os.getenv("RUN_JOB_WATCHDOG_EXTRA", "900"))

DO_SPACES_KEY = os.getenv("DO_SPACES_KEY")
DO_SPACES_SECRET = os.getenv("DO_SPACES_SECRET")
DO_SPACES_BUCKET = os.getenv("DO_SPACES_BUCKET", "nightowl-bucket")
DO_SPACES_REGION = os.getenv("DO_SPACES_REGION", "sfo3")


def probe_duration(url: str):
    """Video length in seconds, read straight from the URL (None if unknown)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", url],
            capture_output=True, text=True, timeout=90,
        )
        value = float(out.stdout.strip())
        return value if value > 0 else None
    except Exception:
        return None


def processing_limit(duration) -> int:
    if duration is None:
        return UNKNOWN_DURATION_SECONDS
    wanted = duration * TIME_FACTOR + 300
    return int(min(MAX_PROCESSING_SECONDS, max(MIN_PROCESSING_SECONDS, wanted)))


def upload_to_spaces(file_path: Path, bucket: str, key: str, acl: str, content_type: str = "video/mp4") -> str:
    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{DO_SPACES_REGION}.digitaloceanspaces.com",
        aws_access_key_id=DO_SPACES_KEY,
        aws_secret_access_key=DO_SPACES_SECRET,
    )
    extra = {"ContentType": content_type}
    if acl and acl != "private":
        extra["ACL"] = acl
    s3.upload_file(str(file_path), bucket, key, ExtraArgs=extra)
    return f"https://{bucket}.{DO_SPACES_REGION}.digitaloceanspaces.com/{quote(key, safe='/')}"


def send_callback(callback_url: str, body: dict) -> bool:
    """POST the result to n8n. Retries so a brief hiccup doesn't leave a record stuck."""
    for attempt in range(1, 4):
        try:
            response = httpx.post(callback_url, json=body, timeout=30)
            print(f"Callback attempt {attempt}: HTTP {response.status_code}", flush=True)
            if response.status_code < 400:
                return True
        except Exception as e:
            print(f"Callback attempt {attempt} failed: {e}", flush=True)
        time.sleep(5)
    return False


def install_watchdog(seconds: int, callback_url, base_body: dict) -> None:
    def fire(signum, frame):
        print(f"WATCHDOG: job exceeded {seconds} seconds, stopping.", flush=True)
        if callback_url:
            try:
                body = dict(base_body)
                body.update({"status": "failed", "error": f"Watchdog: exceeded {seconds} seconds"})
                httpx.post(callback_url, json=body, timeout=15)
            except Exception:
                pass
        os._exit(1)

    signal.signal(signal.SIGALRM, fire)
    signal.alarm(int(seconds))


def clean_key(key):
    key = str(key).lstrip("/")
    if not key or ".." in key.split("/"):
        raise ValueError("invalid output_key")
    return key


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: run_job.py <base64-encoded JSON>", flush=True)
        return 2

    try:
        payload = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
        video_url = payload["video_url"]
        candidate_id = payload.get("candidate_id") or "video"
        callback_url = payload.get("callback_url") or None
        background = payload.get("background", "meeting_dark")
        output_bucket = payload.get("output_bucket") or DO_SPACES_BUCKET
        output_key = payload.get("output_key")
        output_acl = payload.get("output_acl", "public-read")
        ref = payload.get("ref")
        max_frames = int(payload.get("max_frames") or 0)
        debug_frames = [int(x) for x in (payload.get("debug_frames") or [])]
        if output_key is not None:
            output_key = clean_key(output_key)
    except Exception as e:
        print(f"Bad payload: {e}", flush=True)
        return 2

    job_id = uuid.uuid4().hex[:8]
    if output_key is None:
        output_key = f"moodle_test_center/branded_videos/{candidate_id}_{job_id}.mp4"
    print(f"Job {job_id} started for {candidate_id} ({background})", flush=True)

    base_body = {"candidate_id": candidate_id, "job_id": job_id, "branded_video_url": None}
    if ref is not None:
        base_body["ref"] = ref

    duration = probe_duration(video_url)
    limit = processing_limit(duration)
    print(f"Video length: {duration if duration is not None else 'unknown'} s -> time limit {limit} s", flush=True)
    install_watchdog(limit + WATCHDOG_EXTRA_SECONDS, callback_url, base_body)

    work_dir = Path(tempfile.mkdtemp(prefix=f"job_{job_id}_"))
    urls_file = work_dir / "urls.txt"
    output_dir = work_dir / "output"
    output_dir.mkdir()
    urls_file.write_text(video_url)

    script_dir = Path(__file__).resolve().parent
    bg_filename = (
        "Meeting Background 2.png" if background == "meeting_light" else "Meeting Background 1.png"
    )

    branded_url = None
    error = None
    child_env = dict(os.environ)
    if debug_frames:
        child_env["BG_DEBUG_FRAMES"] = ",".join(str(x) for x in debug_frames)
    try:
        proc = subprocess.run(
            [
                "python3", str(script_dir / "brand_videos_stream.py"),
                "--urls", str(urls_file),
                "--logo", str(script_dir / "test_logo.png"),
                "--output-dir", str(output_dir),
                "--cache-dir", str(work_dir / "cache"),
                "--bg-image", str(script_dir / bg_filename),
                "--segmenter", "mediapipe",
                "--logo-scale", "0.001",
                "--upscale", "0",
            ] + (["--max-frames", str(max_frames)] if max_frames > 0 else []),
            text=True,
            timeout=limit,
            env=child_env,
        )
        if proc.returncode != 0:
            error = f"brand_videos.py exited with code {proc.returncode}"
        else:
            mp4_files = list(output_dir.glob("*.mp4"))
            if not mp4_files:
                error = "No output video was produced"
            else:
                print(f"Output file: {mp4_files[0].stat().st_size / 1e6:.0f} MB, uploading to {output_bucket}", flush=True)
                branded_url = upload_to_spaces(mp4_files[0], output_bucket, output_key, output_acl)
                print("Upload finished", flush=True)
                for pic in sorted(output_dir.glob("debug_frame_*.png")):
                    try:
                        stem = output_key[:-4] if output_key.lower().endswith(".mp4") else output_key
                        pic_url = upload_to_spaces(pic, output_bucket, f"{stem}_debug/{pic.name}", output_acl, "image/png")
                        print(f"Debug picture: {pic_url}", flush=True)
                    except Exception as e:
                        print(f"Debug picture upload failed: {e}", flush=True)
    except subprocess.TimeoutExpired:
        error = f"Timed out after {limit} seconds"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    ok = branded_url is not None
    if error:
        print(f"Job {job_id} failed: {error}", flush=True)

    if callback_url:
        body = dict(base_body)
        body.update({
            "branded_video_url": branded_url,
            "status": "success" if ok else "failed",
            "error": error,
        })
        send_callback(callback_url, body)

    print(f"Job {job_id} finished: {'success' if ok else 'failed'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
