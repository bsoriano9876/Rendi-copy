#!/usr/bin/env python3
"""One-off test: AI-upscale a short clip via Replicate Real-ESRGAN video.

Cheapest "real detail" approach. Uploads test_10s.mp4, runs FHD upscale,
downloads result, and extracts before/after frames for visual comparison.
Writes a summary to output/AI_UPSCALE_SUMMARY.txt.
"""
import os
import time
import traceback
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
import replicate
import subprocess

HERE = Path(__file__).resolve().parent
SRC = HERE / ".cache" / "test_10s.mp4"
OUT_DIR = HERE / "output"
OUT_VIDEO = OUT_DIR / "test_10s_AI_UPSCALED.mp4"
SUMMARY = OUT_DIR / "AI_UPSCALE_SUMMARY.txt"
MODEL = "lucataco/real-esrgan-video:3e56ce4b57863bd03048b42bc09bdd4db20d427cca5fde9d8ae4dc60e1bb4775"

lines = []
def log(msg):
    print(msg, flush=True)
    lines.append(msg)

def probe(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True)
    return r.stdout.replace("\n", " ").strip()

def main():
    OUT_DIR.mkdir(exist_ok=True)
    t0 = time.time()
    log("=== AI UPSCALE TEST (Replicate Real-ESRGAN video) ===")
    log(f"source: {SRC.name}  ->  {probe(SRC)}")

    log("uploading clip to Replicate...")
    with open(SRC, "rb") as f:
        file_obj = replicate.files.create(f)
    input_url = file_obj.urls["get"]

    log("running real-esrgan-video (resolution=FHD, model=RealESRGAN_x4plus)...")
    output = replicate.run(MODEL, input={
        "video_path": input_url,
        "resolution": "FHD",
        "model": "RealESRGAN_x4plus",
    })
    out_url = output.url if hasattr(output, "url") else str(output)
    log(f"prediction done, downloading result...")
    urllib.request.urlretrieve(out_url, OUT_VIDEO)

    elapsed = time.time() - t0
    log(f"result: {OUT_VIDEO.name}  ->  {probe(OUT_VIDEO)}")
    log(f"size: src={SRC.stat().st_size/1e6:.2f}MB  out={OUT_VIDEO.stat().st_size/1e6:.2f}MB")
    log(f"wall time: {elapsed:.0f}s")

    # before/after frame at ~3s for visual comparison
    for label, src in (("BEFORE_640x480", SRC), ("AFTER_AI", OUT_VIDEO)):
        png = OUT_DIR / f"compare_{label}.png"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "3", "-i", str(src),
                        "-vframes", "1", str(png)], check=False)
        log(f"frame: {png.name}")

    log("=== DONE ===")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"ERROR: {e}")
        log(traceback.format_exc())
    finally:
        SUMMARY.write_text("\n".join(lines))
        print(f"\n[summary written to {SUMMARY}]")
