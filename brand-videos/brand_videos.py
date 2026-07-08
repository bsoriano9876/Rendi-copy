#!/usr/bin/env python3
"""
Batch video background replacement CLI.

Downloads videos from URLs, replaces backgrounds with a branded background
(solid color + centered logo), preserves audio, and outputs to a folder.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# MediaPipe imports (new Tasks API)
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Optional rembg import
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

# Optional replicate import
try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False


@dataclass
class ProcessingConfig:
    """Configuration for video processing."""
    logo_path: Path
    output_dir: Path
    cache_dir: Path
    bg_color: tuple[int, int, int]
    logo_scale: float
    output_ext: str
    model_selection: int
    edge_softness: int
    segmenter: str
    max_frames: int = 0  # 0 = process all frames
    bg_image_path: Optional[Path] = None  # Optional background image
    upscale: int = 0  # Target output height (0 = keep source resolution)


@dataclass
class ProcessingResult:
    """Result of processing a single video."""
    url: str
    success: bool
    output_path: Optional[Path] = None
    error: Optional[str] = None


def check_dependencies() -> list[str]:
    """Check that required external tools are available."""
    missing = []

    if shutil.which("ffmpeg") is None:
        missing.append("ffmpeg")
    if shutil.which("yt-dlp") is None:
        missing.append("yt-dlp")

    return missing


def parse_urls_file(urls_path: Path) -> list[str]:
    """Parse URLs file, skipping comments and blank lines."""
    urls = []
    with open(urls_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return urls


def url_to_filename(url: str) -> str:
    """Generate a filesystem-safe filename from a URL."""
    # Use hash for uniqueness, but also include a readable prefix
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    # Extract any filename-like part from the URL
    url_path = url.split("?")[0].split("/")[-1]
    if url_path and len(url_path) < 50:
        # Sanitize the filename part
        safe_name = "".join(c if c.isalnum() or c in ".-_" else "_" for c in url_path)
        return f"{safe_name}_{url_hash}"
    return url_hash


def download_video(url: str, cache_dir: Path) -> Path:
    """Download video using yt-dlp, returning path to downloaded file."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Get the output filename yt-dlp would use
    filename_base = url_to_filename(url)

    # Check if already cached (any extension)
    for cached in cache_dir.glob(f"{filename_base}.*"):
        if cached.suffix.lower() in (".mp4", ".webm", ".mkv", ".avi", ".mov"):
            # Verify file is not empty (incomplete download)
            if cached.stat().st_size > 0:
                return cached

    # Download with yt-dlp
    output_template = str(cache_dir / f"{filename_base}.%(ext)s")
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-o", output_template,
        url
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr}")

    # Find the downloaded file
    for downloaded in cache_dir.glob(f"{filename_base}.*"):
        if downloaded.suffix.lower() in (".mp4", ".webm", ".mkv", ".avi", ".mov"):
            return downloaded

    raise RuntimeError(f"Downloaded file not found for {url}")


def load_logo(logo_path: Path) -> np.ndarray:
    """Load logo image with alpha channel (BGRA)."""
    img = Image.open(logo_path).convert("RGBA")
    # Convert to BGRA for OpenCV
    arr = np.array(img)
    # RGBA -> BGRA
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)


def load_background_image(bg_image_path: Path, width: int, height: int) -> np.ndarray:
    """Load and resize background image to fit video dimensions."""
    img = Image.open(bg_image_path).convert("RGB")
    img_w, img_h = img.size

    # Calculate scaling to cover the entire frame (crop if needed)
    scale = max(width / img_w, height / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Center crop to exact dimensions
    left = (new_w - width) // 2
    top = (new_h - height) // 2
    img_cropped = img_resized.crop((left, top, left + width, top + height))

    # Convert to BGR for OpenCV
    return cv2.cvtColor(np.array(img_cropped), cv2.COLOR_RGB2BGR)


def create_branded_background(
    width: int,
    height: int,
    bg_color: tuple[int, int, int],
    logo_bgra: np.ndarray,
    logo_scale: float,
    bg_image_path: Optional[Path] = None
) -> np.ndarray:
    """Create branded background with centered logo."""
    # Use background image if provided, otherwise solid color
    if bg_image_path is not None:
        background = load_background_image(bg_image_path, width, height)
    else:
        # Create solid color background (BGR)
        background = np.full((height, width, 3), bg_color[::-1], dtype=np.uint8)

    # Resize logo
    logo_h, logo_w = logo_bgra.shape[:2]
    target_w = int(width * logo_scale)
    scale_factor = target_w / logo_w
    target_h = int(logo_h * scale_factor)

    if target_w > 0 and target_h > 0:
        # Clamp logo size to frame dimensions
        target_w = min(target_w, width)
        target_h = min(target_h, height)

        logo_resized = cv2.resize(logo_bgra, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Calculate center position (ensure non-negative)
        x_offset = max(0, (width - target_w) // 2)
        y_offset = max(0, (height - target_h) // 2)

        # Alpha composite the logo onto background
        logo_bgr = logo_resized[:, :, :3]
        logo_alpha = logo_resized[:, :, 3:4] / 255.0

        roi = background[y_offset:y_offset+target_h, x_offset:x_offset+target_w]
        blended = (logo_alpha * logo_bgr + (1 - logo_alpha) * roi).astype(np.uint8)
        background[y_offset:y_offset+target_h, x_offset:x_offset+target_w] = blended

    return background


def target_dimensions(src_width: int, src_height: int, upscale: int) -> tuple[int, int]:
    """Compute output (width, height) given an optional target height.

    Preserves aspect ratio. Width is rounded to an even number (required by
    most H.264 encoders). Never downscales — upscale=0 or a height <= source
    keeps the source resolution.
    """
    if upscale <= 0 or upscale <= src_height:
        return src_width, src_height
    scale = upscale / src_height
    out_w = int(round(src_width * scale))
    out_w += out_w % 2  # force even
    return out_w, upscale


def get_mediapipe_model_path() -> Path:
    """Download and cache the MediaPipe selfie segmentation model."""
    import urllib.request

    cache_dir = Path.home() / ".cache" / "mediapipe"
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_path = cache_dir / "selfie_segmenter.tflite"
    if not model_path.exists():
        # Download the selfie segmentation model
        model_url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
        print(f"  Downloading MediaPipe model...")
        urllib.request.urlretrieve(model_url, model_path)

    return model_path


def ensure_mp4(video_path: Path, cache_dir: Path) -> Path:
    """Convert video to h264 mp4 with faststart if not already mp4."""
    if video_path.suffix.lower() == ".mp4":
        # Verify moov atom is at the front
        with open(video_path, "rb") as f:
            header = f.read(8)
        if header[4:8] == b"ftyp":
            return video_path

    mp4_path = cache_dir / f"{video_path.stem}_h264.mp4"
    if mp4_path.exists() and mp4_path.stat().st_size > 0:
        return mp4_path

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-c:v", "libx264", "-c:a", "aac",
        "-movflags", "+faststart",
        str(mp4_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")
    return mp4_path


def run_replicate_rvm(video_path: Path, cache_dir: Path) -> Path:
    """Run Robust Video Matting on Replicate, return path to downloaded alpha matte video."""
    import urllib.request

    api_key = os.environ.get("REPLICATE_API_TOKEN")
    if not api_key:
        raise RuntimeError("REPLICATE_API_TOKEN not set in environment / .env")

    matte_cache = cache_dir / f"{video_path.stem}_rvm_matte.mp4"
    if matte_cache.exists() and matte_cache.stat().st_size > 0:
        print(f"  Using cached matte: {matte_cache.name}")
        return matte_cache

    # RVM model requires h264 mp4 with moov atom at front
    mp4_path = ensure_mp4(video_path, cache_dir)

    # Upload to Replicate Files API so the model can fetch it via URL
    print(f"  Uploading to Replicate...", end="", flush=True)
    with open(mp4_path, "rb") as f:
        file_obj = replicate.files.create(f)
    input_url = file_obj.urls["get"]
    print(f" done.")

    version = "73d2128a371922d5d1abf0712a1d974be0e4e2358cc1218e4e34714767232bac"
    rvm_input = {"input_video": input_url, "output_type": "alpha-mask"}

    # Full videos take ~45min of GPU time and Replicate's shared GPUs may
    # preempt the prediction (error code PA). Poll the prediction ourselves and
    # auto-resubmit on preemption until it succeeds.
    max_attempts = 8
    output_url = None
    for attempt in range(1, max_attempts + 1):
        print(f"  RVM attempt {attempt}/{max_attempts}: submitting...", flush=True)
        prediction = replicate.predictions.create(version=version, input=rvm_input)
        print(f"    prediction {prediction.id}", flush=True)

        while prediction.status not in ("succeeded", "failed", "canceled"):
            time.sleep(10)
            prediction.reload()

        if prediction.status == "succeeded":
            out = prediction.output
            output_url = out.url if hasattr(out, "url") else str(out)
            print(f"    succeeded.", flush=True)
            break

        err = (prediction.error or "")
        preempted = "PA" in str(err) or "interrupted" in str(err).lower() \
            or prediction.status == "canceled"
        print(f"    {prediction.status}: {err}", flush=True)
        if not preempted:
            raise RuntimeError(f"RVM prediction failed: {err}")
        # preempted -> retry with a short backoff
        time.sleep(15)
    else:
        raise RuntimeError(f"RVM preempted {max_attempts} times; giving up.")

    print(f"  Downloading matte...", end="", flush=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(output_url, matte_cache)
    print(f" done.")
    return matte_cache


def composite_with_rvm_matte(
    video_path: Path,
    matte_path: Path,
    output_path: Path,
    config: "ProcessingConfig",
) -> None:
    """Composite video frames against branded background using RVM green-screen output."""
    logo_bgra = load_logo(config.logo_path)

    cap_src = cv2.VideoCapture(str(video_path))
    cap_matte = cv2.VideoCapture(str(matte_path))

    if not cap_src.isOpened():
        raise RuntimeError(f"Cannot open source video: {video_path}")
    if not cap_matte.isOpened():
        raise RuntimeError(f"Cannot open matte video: {matte_path}")

    try:
        width = int(cap_src.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_src.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap_src.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps > 300:
            fps = 30.0
        total_frames = int(cap_src.get(cv2.CAP_PROP_FRAME_COUNT))

        # Resolve output resolution (optional upscale, aspect-preserving)
        out_w, out_h = target_dimensions(width, height, config.upscale)

        background = create_branded_background(
            out_w, out_h, config.bg_color, logo_bgra, config.logo_scale,
            bg_image_path=config.bg_image_path
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Lossless intermediate; mux_audio does the single H.264 encode.
            temp_video = Path(temp_dir) / "temp_video.mkv"
            fourcc = cv2.VideoWriter_fourcc(*"FFV1")
            writer = cv2.VideoWriter(str(temp_video), fourcc, fps, (out_w, out_h))
            if not writer.isOpened():
                raise RuntimeError("Cannot create video writer")

            try:
                frame_idx = 0
                while True:
                    ret_src, frame_bgr = cap_src.read()
                    ret_mat, frame_matte = cap_matte.read()
                    if not ret_src or not ret_mat:
                        break

                    # RVM alpha-mask output: white (255) = foreground, black (0) = background.
                    # Resize matte to OUTPUT dims; upscale the source frame to match.
                    frame_matte_resized = cv2.resize(frame_matte, (out_w, out_h))
                    if (out_w, out_h) != (width, height):
                        frame_bgr = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
                    if frame_matte_resized.ndim == 3:
                        gray = cv2.cvtColor(frame_matte_resized, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = frame_matte_resized
                    # mask: 1.0 = keep foreground, 0.0 = replace with background.
                    # Remap to kill RVM's soft halo: faint grays -> 0, solid body -> 1,
                    # keeping a thin feather for a clean edge against the dark background.
                    mask = gray.astype(np.float32) / 255.0
                    mask = np.clip((mask - 0.15) / 0.70, 0.0, 1.0)

                    composited = composite_frame(frame_bgr, background, mask, config.edge_softness)
                    writer.write(composited)

                    frame_idx += 1
                    if total_frames > 0:
                        pct = (frame_idx / total_frames) * 100
                        print(f"\r  Compositing: frame {frame_idx}/{total_frames} ({pct:.1f}%)", end="", flush=True)
            finally:
                writer.release()

            mux_audio(video_path, temp_video, output_path, config.output_ext)
    finally:
        cap_src.release()
        cap_matte.release()


def create_segmenter(model_selection: int, segmenter_type: str):
    """Create the appropriate segmenter based on type."""
    if segmenter_type == "rembg":
        if not REMBG_AVAILABLE:
            raise RuntimeError("rembg not installed. Install with: pip install rembg[cpu]")
        session = new_session("u2net_human_seg")
        return ("rembg", session)
    else:
        # MediaPipe new Tasks API
        model_path = get_mediapipe_model_path()

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            output_category_mask=False,
            output_confidence_masks=True
        )
        segmenter = mp_vision.ImageSegmenter.create_from_options(options)
        return ("mediapipe", segmenter)


def segment_frame(frame_rgb: np.ndarray, segmenter_info: tuple) -> np.ndarray:
    """Segment a frame, returning a float32 mask in [0, 1] as HxW array."""
    segmenter_type, segmenter = segmenter_info

    if segmenter_type == "rembg":
        # rembg returns RGBA image with alpha as mask
        result = remove(frame_rgb, session=segmenter, only_mask=True)
        mask = np.array(result, dtype=np.float32) / 255.0
        # Ensure 2D mask (rembg may return 3D)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return mask
    else:
        # MediaPipe new Tasks API
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = segmenter.segment(mp_image)

        # Prefer the confidence mask: a soft [0,1] probability per pixel that
        # yields far cleaner edges than the hard category mask. The selfie
        # segmenter exposes the person channel at index 0.
        conf = getattr(result, "confidence_masks", None)
        if conf:
            mask = np.asarray(conf[0].numpy_view(), dtype=np.float32)
            # Stretch the mid-confidence band so faint halos drop out while the
            # body stays solid, keeping a thin feather at the silhouette.
            mask = np.clip((mask - 0.30) / 0.40, 0.0, 1.0)
            return mask

        # Fallback: hard category mask (0 = person, 255 = background)
        category_mask = result.category_mask.numpy_view()
        mask = (category_mask == 0).astype(np.float32)
        return mask


def composite_frame(
    frame_bgr: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    edge_softness: int
) -> np.ndarray:
    """Composite foreground over background using mask."""
    # Apply edge softening if requested
    if edge_softness > 0:
        kernel_size = edge_softness if edge_softness % 2 == 1 else edge_softness + 1
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

    # Expand mask dimensions for broadcasting
    mask_3ch = mask[:, :, np.newaxis]

    # Composite: mask * foreground + (1 - mask) * background
    composited = (mask_3ch * frame_bgr + (1 - mask_3ch) * background).astype(np.uint8)
    return composited


def process_video(
    video_path: Path,
    output_path: Path,
    config: ProcessingConfig,
    progress_callback=None
) -> None:
    """Process a single video, replacing background."""
    # Load logo
    logo_bgra = load_logo(config.logo_path)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    segmenter_info = None
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Validate FPS
        if not fps or fps <= 0 or fps > 300:
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Resolve output resolution (optional upscale, aspect-preserving)
        out_w, out_h = target_dimensions(width, height, config.upscale)

        # Create branded background at the OUTPUT resolution so the logo and
        # solid/image background are rendered crisply rather than upscaled.
        background = create_branded_background(
            out_w, out_h, config.bg_color, logo_bgra, config.logo_scale,
            bg_image_path=config.bg_image_path
        )

        # Create segmenter
        segmenter_info = create_segmenter(config.model_selection, config.segmenter)

        # Use context manager for temp directory to ensure cleanup
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write a lossless intermediate (FFV1) so the final H.264 encode in
            # mux_audio is the only lossy step. The old mp4v writer baked in
            # blocking artifacts that -c:v copy then carried straight through.
            temp_video = Path(temp_dir) / "temp_video.mkv"

            fourcc = cv2.VideoWriter_fourcc(*"FFV1")
            writer = cv2.VideoWriter(str(temp_video), fourcc, fps, (out_w, out_h))

            if not writer.isOpened():
                raise RuntimeError("Cannot create video writer")

            try:
                frame_idx = 0
                effective_total = total_frames if config.max_frames == 0 else min(total_frames, config.max_frames)
                skip_factor = 2  # segment every 2nd frame, reuse mask in between
                last_mask = None

                while True:
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break

                    # Check max_frames limit
                    if config.max_frames > 0 and frame_idx >= config.max_frames:
                        break

                    if frame_idx % skip_factor == 0 or last_mask is None:
                        # Convert to RGB for segmentation (at source resolution,
                        # which the segmentation models are tuned for)
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        mask = segment_frame(frame_rgb, segmenter_info)
                        last_mask = mask
                    else:
                        mask = last_mask
                        
                    # Upscale frame and mask to output resolution if needed.
                    # Cubic for the image (sharp), linear for the mask (smooth edge).
                    if (out_w, out_h) != (width, height):
                        frame_bgr = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
                        mask = cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

                    # Composite
                    composited = composite_frame(frame_bgr, background, mask, config.edge_softness)

                    # Write frame
                    writer.write(composited)

                    frame_idx += 1
                    if progress_callback and effective_total > 0:
                        progress_callback(frame_idx, effective_total)

            finally:
                writer.release()

            # Mux audio with FFmpeg (after writer is closed but within temp dir context)
            mux_audio(video_path, temp_video, output_path, config.output_ext)

    finally:
        cap.release()
        # Close segmenter if MediaPipe
        if segmenter_info is not None and segmenter_info[0] == "mediapipe":
            segmenter_info[1].close()


def mux_audio(
    source_video: Path,
    processed_video: Path,
    output_path: Path,
    output_ext: str
) -> None:
    """Mux original audio with processed video."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_ext == ".mp4":
        # Encode the lossless intermediate to high-quality H.264. CRF 18 is
        # visually transparent; yuv420p + faststart keeps it web/player-safe.
        cmd = [
            "ffmpeg", "-y",
            "-i", str(processed_video),
            "-i", str(source_video),
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-c:a", "aac",
            "-shortest",
            str(output_path)
        ]
    else:
        # WebM: re-encode with libvpx-vp9
        cmd = [
            "ffmpeg", "-y",
            "-i", str(processed_video),
            "-i", str(source_video),
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", "libvpx-vp9",
            "-b:v", "2M",
            "-c:a", "libopus",
            "-shortest",
            str(output_path)
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg mux failed: {result.stderr}")


def process_single_url(url: str, index: int, total: int, config: ProcessingConfig) -> ProcessingResult:
    """Process a single URL (for use in process pool)."""
    try:
        print(f"[{index}/{total}] {url}")

        # Download
        print(f"  Downloading...", end="", flush=True)
        video_path = download_video(url, config.cache_dir)
        print(f" done ({video_path.name})")

        # Determine output filename
        output_name = f"{url_to_filename(url)}{config.output_ext}"
        output_path = config.output_dir / output_name

        if config.segmenter == "replicate":
            matte_path = run_replicate_rvm(video_path, config.cache_dir)
            composite_with_rvm_matte(video_path, matte_path, output_path, config)
            print(f"\n  → {output_path}")
        else:
            # Frame-by-frame local segmentation (mediapipe / rembg)
            def progress(current, total_frames):
                pct = (current / total_frames) * 100
                print(f"\r  Processing: frame {current}/{total_frames} ({pct:.1f}%)", end="", flush=True)

            process_video(video_path, output_path, config, progress_callback=progress)
            print(f"\n  → {output_path}")

        return ProcessingResult(url=url, success=True, output_path=output_path)

    except Exception as e:
        print(f"\n  ERROR: {e}")
        return ProcessingResult(url=url, success=False, error=str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Batch video background replacement with branded backgrounds"
    )
    parser.add_argument(
        "--urls", type=Path, required=True,
        help="Text file with one video URL per line"
    )
    parser.add_argument(
        "--logo", type=Path, required=True,
        help="Logo image (PNG with alpha preferred)"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for branded videos"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".cache"),
        help="Cache directory for downloads (default: .cache)"
    )
    parser.add_argument(
        "--bg-color", type=str, default="245,245,245",
        help="Background color as R,G,B (default: 245,245,245)"
    )
    parser.add_argument(
        "--bg-image", type=Path, default=None,
        help="Background image (overrides --bg-color if provided)"
    )
    parser.add_argument(
        "--logo-scale", type=float, default=0.25,
        help="Logo width as fraction of frame width (default: 0.25)"
    )
    parser.add_argument(
        "--output-ext", type=str, choices=[".mp4", ".webm"], default=".mp4",
        help="Output format (default: .mp4)"
    )
    parser.add_argument(
        "--model", type=int, choices=[0, 1], default=1,
        help="MediaPipe model selection (default: 1 for landscape)"
    )
    parser.add_argument(
        "--edge-softness", type=int, default=5,
        help="Gaussian blur kernel for mask edges (default: 5, 0 to disable)"
    )
    parser.add_argument(
        "--parallel", type=int, default=1,
        help="Number of videos to process concurrently (default: 1)"
    )
    parser.add_argument(
        "--segmenter", type=str, choices=["mediapipe", "rembg", "replicate"], default="mediapipe",
        help="Segmentation backend: mediapipe, rembg, or replicate (RVM, best quality)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Maximum frames to process per video (0 = all, for testing)"
    )
    parser.add_argument(
        "--upscale", type=int, default=0,
        help="Target output height, e.g. 720 or 1080 (0 = keep source; never downscales)"
    )

    args = parser.parse_args()

    # Validate inputs
    errors = []

    if not args.urls.exists():
        errors.append(f"URLs file not found: {args.urls}")

    if not args.logo.exists():
        errors.append(f"Logo file not found: {args.logo}")
    else:
        try:
            Image.open(args.logo)
        except Exception as e:
            errors.append(f"Cannot read logo file: {e}")

    missing_deps = check_dependencies()
    if missing_deps:
        errors.append(f"Missing dependencies: {', '.join(missing_deps)}")

    if args.segmenter == "rembg" and not REMBG_AVAILABLE:
        errors.append("rembg not installed. Install with: pip install rembg[cpu]")

    if args.segmenter == "replicate":
        if not REPLICATE_AVAILABLE:
            errors.append("replicate not installed. Run: pip install replicate")
        elif not os.environ.get("REPLICATE_API_TOKEN"):
            errors.append("REPLICATE_API_TOKEN not set. Add it to .env or export it.")

    # Validate background image if provided
    if args.bg_image is not None:
        if not args.bg_image.exists():
            errors.append(f"Background image not found: {args.bg_image}")
        else:
            try:
                Image.open(args.bg_image)
            except Exception as e:
                errors.append(f"Cannot read background image: {e}")

    # Parse background color
    try:
        bg_color = tuple(int(x) for x in args.bg_color.split(","))
        if len(bg_color) != 3 or not all(0 <= c <= 255 for c in bg_color):
            raise ValueError()
    except (ValueError, TypeError):
        errors.append(f"Invalid background color: {args.bg_color} (expected R,G,B)")
        bg_color = (245, 245, 245)

    # Validate logo_scale
    if args.logo_scale <= 0 or args.logo_scale > 2.0:
        errors.append(f"Logo scale must be between 0.0 and 2.0, got {args.logo_scale}")

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(1)

    # Parse URLs
    urls = parse_urls_file(args.urls)
    if not urls:
        print("ERROR: No URLs found in file", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(urls)} URLs to process")
    print(f"Output directory: {args.output_dir}")
    print(f"Segmenter: {args.segmenter}")
    print(f"Parallel workers: {args.parallel}")
    print()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create config
    config = ProcessingConfig(
        logo_path=args.logo,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        bg_color=bg_color,
        logo_scale=args.logo_scale,
        output_ext=args.output_ext,
        model_selection=args.model,
        edge_softness=args.edge_softness,
        segmenter=args.segmenter,
        max_frames=args.max_frames,
        bg_image_path=args.bg_image,
        upscale=args.upscale
    )

    # Process videos
    results: list[ProcessingResult] = []

    if args.parallel == 1:
        # Sequential processing
        for i, url in enumerate(urls, 1):
            result = process_single_url(url, i, len(urls), config)
            results.append(result)
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(process_single_url, url, i, len(urls), config): url
                for i, url in enumerate(urls, 1)
            }

            for future in as_completed(futures):
                try:
                    result = future.result(timeout=3600)  # 1 hour timeout per video
                except Exception as e:
                    url = futures[future]
                    result = ProcessingResult(url=url, success=False, error=str(e))
                results.append(result)

    # Summary
    print("\n" + "=" * 60)
    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]

    print(f"Completed: {len(successes)}/{len(results)} videos")

    if failures:
        print(f"\nFailed ({len(failures)}):")
        for r in failures:
            print(f"  - {r.url}")
            print(f"    Error: {r.error}")
        sys.exit(1)
    else:
        print("\nAll videos processed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
