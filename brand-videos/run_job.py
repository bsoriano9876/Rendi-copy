#!/usr/bin/env python3
"""One-off Render job: process ONE candidate video, upload the result,
call back n8n, then exit. Billed per second only while this runs.

Usage:
    python3 /app/run_job.py <base64-encoded JSON>

JSON keys: video_url, candidate_id, background, callback_url
(Passed as base64 so odd characters can never break the shell command.)
"""

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import boto3
import httpx

# Safety stop: a stuck job can never bill for more than this.
# Longest known video (about 3 min) takes about 11 min to process.
MAX_PROCESSING_SECONDS = 1200  # 20 minutes

DO_SPACES_KEY = os.getenv("DO_SPACES_KEY")
DO_SPACES_SECRET = os.getenv("DO_SPACES_SECRET")
DO_SPACES_BUCKET = os.getenv("DO_SPACES_BUCKET", "nightowl-bucket")
DO_SPACES_REGION = os.getenv("DO_SPACES_REGION", "sfo3")


def upload_to_spaces(file_path: Path, candidate_id: str, job_id: str) -> str:
    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{DO_SPACES_REGION}.digitaloceanspaces.com",
        aws_access_key_id=DO_SPACES_KEY,
        aws_secret_access_key=DO_SPACES_SECRET,
    )
    key = f"moodle_test_center/branded_videos/{candidate_id}_{job_id}.mp4"
    s3.upload_file(
        str(file_path),
        DO_SPACES_BUCKET,
        key,
        ExtraArgs={"ACL": "public-read", "ContentType": "video/mp4"},
    )
    return f"https://{DO_SPACES_BUCKET}.{DO_SPACES_REGION}.digitaloceanspaces.com/{key}"


def send_callback(callback_url: str, body: dict) -> bool:
    """POST the result to n8n. Retries so a brief hiccup doesn't leave Airtable stuck on 'queued'."""
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


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: run_job.py <base64-encoded JSON>", flush=True)
        return 2

    try:
        payload = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
        video_url = payload["video_url"]
        candidate_id = payload["candidate_id"]
        callback_url = payload["callback_url"]
        background = payload.get("background", "meeting_dark")
    except Exception as e:
        print(f"Bad payload: {e}", flush=True)
        return 2

    job_id = uuid.uuid4().hex[:8]
    print(f"Job {job_id} started for candidate {candidate_id} ({background})", flush=True)

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
    try:
        proc = subprocess.run(
            [
                "python3", str(script_dir / "brand_videos.py"),
                "--urls", str(urls_file),
                "--logo", str(script_dir / "test_logo.png"),
                "--output-dir", str(output_dir),
                "--bg-image", str(script_dir / bg_filename),
                "--segmenter", "mediapipe",
                "--logo-scale", "0.001",
                "--upscale", "0",
            ],
            text=True,
            timeout=MAX_PROCESSING_SECONDS,
        )
        if proc.returncode != 0:
            error = f"brand_videos.py exited with code {proc.returncode}"
        else:
            mp4_files = list(output_dir.glob("*.mp4"))
            if not mp4_files:
                error = "No output video was produced"
            else:
                branded_url = upload_to_spaces(mp4_files[0], candidate_id, job_id)
    except subprocess.TimeoutExpired:
        error = f"Timed out after {MAX_PROCESSING_SECONDS} seconds"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    ok = branded_url is not None
    if error:
        print(f"Job {job_id} failed: {error}", flush=True)

    send_callback(callback_url, {
        "candidate_id": candidate_id,
        "job_id": job_id,
        "branded_video_url": branded_url,
        "status": "success" if ok else "failed",
        "error": error,
    })

    print(f"Job {job_id} finished: {'success' if ok else 'failed'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
