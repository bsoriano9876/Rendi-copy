#!/usr/bin/env python3
"""AI-enhance a video frame-by-frame via Replicate Real-ESRGAN (+ face_enhance).

Pipeline:
  1. ffmpeg: extract frames at a sane fps (source is mis-tagged 1000fps)
  2. nightmareai/real-esrgan per frame (scale, face_enhance) -- run in parallel
  3. ffmpeg: reassemble frames -> h264, mux original audio

This is the only path that actually works (both Replicate *video* upscale
models are currently broken server-side; the *image* model is rock solid).

Usage:
  python ai_enhance_video.py SRC.mp4 OUT.mp4 [--fps 24] [--scale 2]
                             [--max-seconds N] [--workers 6] [--no-face]
"""
import argparse
import concurrent.futures as cf
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
import replicate

REALESRGAN = "nightmareai/real-esrgan"


def extract_frames(src: Path, frames_dir: Path, fps: int, max_seconds: int) -> int:
    frames_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error"]
    if max_seconds > 0:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-i", str(src), "-vf", f"fps={fps}", str(frames_dir / "f_%05d.png")]
    subprocess.run(cmd, check=True)
    return len(list(frames_dir.glob("f_*.png")))


def upscale_one(version: str, in_png: Path, out_png: Path, scale: int, face: bool):
    # Retry on 429 throttling (free/low-credit accounts are capped to ~6/min,
    # burst 1). Back off and retry rather than dropping the frame.
    for attempt in range(8):
        try:
            with open(in_png, "rb") as f:
                out = replicate.run(
                    f"{REALESRGAN}:{version}",
                    input={"image": f, "scale": scale, "face_enhance": face},
                )
            url = out.url if hasattr(out, "url") else str(out)
            urllib.request.urlretrieve(url, out_png)
            return
        except replicate.exceptions.ReplicateError as e:
            if "throttled" in str(e).lower() or getattr(e, "status", None) == 429:
                time.sleep(11)  # rate limit resets in ~10s
                continue
            raise
    raise RuntimeError(f"giving up on {in_png.name} after repeated throttling")


def reassemble(frames_dir: Path, src: Path, out: Path, fps: int):
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        silent = Path(td) / "video.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-framerate", str(fps),
             "-i", str(frames_dir / "up_%05d.png"),
             "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(silent)],
            check=True)
        # mux original audio
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(silent), "-i", str(src),
             "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac",
             "-shortest", str(out)],
            check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--max-seconds", type=int, default=0, help="0 = whole clip")
    ap.add_argument("--workers", type=int, default=1,
                    help="Concurrent requests. Low-credit Replicate accounts are "
                         "throttled to ~6/min burst 1, so keep this at 1.")
    ap.add_argument("--min-interval", type=float, default=10.5,
                    help="Min seconds between submissions to respect 6/min rate limit")
    ap.add_argument("--no-face", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    version = replicate.models.get(REALESRGAN).latest_version.id
    print(f"real-esrgan version {version}", flush=True)

    with tempfile.TemporaryDirectory() as td:
        fdir = Path(td)
        n = extract_frames(args.src, fdir, args.fps, args.max_seconds)
        print(f"extracted {n} frames @ {args.fps}fps", flush=True)
        if n == 0:
            print("no frames extracted", file=sys.stderr); sys.exit(1)

        frames = sorted(fdir.glob("f_*.png"))

        def work(p: Path):
            out_png = p.with_name("up_" + p.name[2:])  # f_00001.png -> up_00001.png
            upscale_one(version, p, out_png, args.scale, not args.no_face)
            return out_png

        eta_min = n * args.min_interval / 60
        print(f"pacing 1 req every {args.min_interval}s (rate-limited). ETA ~{eta_min:.0f} min", flush=True)
        if args.workers <= 1:
            # Paced serial loop: respect the 6/min rate limit.
            for done, p in enumerate(frames, 1):
                t = time.time()
                work(p)
                if done % 10 == 0 or done == n:
                    print(f"  upscaled {done}/{n}", flush=True)
                if done < n:
                    elapsed = time.time() - t
                    if elapsed < args.min_interval:
                        time.sleep(args.min_interval - elapsed)
        else:
            done = 0
            with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
                for _ in ex.map(work, frames):
                    done += 1
                    if done % 10 == 0 or done == n:
                        print(f"  upscaled {done}/{n}", flush=True)

        print("reassembling + muxing audio...", flush=True)
        reassemble(fdir, args.src, args.out, args.fps)

    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height",
         "-of", "default=noprint_wrappers=1:nokey=1", str(args.out)],
        capture_output=True, text=True)
    print(f"DONE -> {args.out}  ({r.stdout.replace(chr(10),' ').strip()})", flush=True)
    print(f"size {args.out.stat().st_size/1e6:.2f}MB, wall {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
