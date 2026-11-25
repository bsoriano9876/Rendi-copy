#!/usr/bin/env python3
"""
STREAMING VARIANT of brand_videos.py (used by run_job_v2.py).

Differences from brand_videos.py:
  * finished frames are piped straight into ONE ffmpeg process that encodes H.264 and adds
    the source audio, so there is no multi-GB lossless intermediate file (Render jobs only
    have 2 GB of /tmp);
  * the constant-frame-rate re-encode is skipped when ffprobe shows the source is already CFR;
  * progress is logged every 100 frames instead of on every frame.

Batch video background replacement CLI.

Downloads videos from URLs, replaces backgrounds with a branded background
(solid color + centered logo), preserves audio, and outputs to a folder.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
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


ENCODE_CRF = os.getenv("BG_ENCODE_CRF", "23")
ENCODE_PRESET = os.getenv("BG_ENCODE_PRESET", "veryfast")
# Caps keep the output small: never above this bitrate, and never above ~1 GB in total.
ENCODE_MAXRATE_KBPS = int(os.getenv("BG_ENCODE_MAXRATE_KBPS", "5000"))
ENCODE_MAX_BYTES = int(os.getenv("BG_ENCODE_MAX_BYTES", "1000000000"))


def disk_report(label: str) -> None:
    """Print how much of /tmp is used and by what (Render jobs only have 2 GB of /tmp)."""
    try:
        root = Path(tempfile.gettempdir())
        sizes = {}
        total = 0
        for entry in root.iterdir():
            size = 0
            if entry.is_dir():
                for r, _, files in os.walk(entry):
                    for f in files:
                        try:
                            size += os.path.getsize(os.path.join(r, f))
                        except OSError:
                            pass
            else:
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = 0
            sizes[entry.name] = size
            total += size
        top = sorted(sizes.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top_text = ", ".join(f"{name} {size / 1e6:.0f} MB" for name, size in top)
        print(f"  [disk] {label}: /tmp total {total / 1e6:.0f} MB; largest: {top_text}", flush=True)
    except Exception as e:
        print(f"  [disk] {label}: could not measure ({e})", flush=True)


def needs_cfr(path: Path) -> bool:
    """True unless ffprobe clearly shows a constant frame rate (browser webm is always re-timed)."""
    if path.suffix.lower() in (".webm", ".mkv"):
        return True
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate,avg_frame_rate",
             "-of", "default=noprint_wrappers=1", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        values = {}
        for line in out.stdout.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                values[k.strip()] = v.strip()

        def frac(s):
            a, b = s.split("/")
            return float(a) / float(b) if float(b) else 0.0

        r, avg = frac(values["r_frame_rate"]), frac(values["avg_frame_rate"])
        if r <= 0 or avg <= 0:
            return True
        return abs(r - avg) / r > 0.01
    except Exception:
        return True


MASK_CLEANUP = os.getenv("BG_MASK_CLEANUP", "1") == "1"


SEGMENTER_CHOICE = os.getenv("BG_SEGMENTER", "rvm")          # "rvm" = Robust Video Matting (default), "mediapipe" = old selfie model
RVM_STRICT = os.getenv("BG_RVM_STRICT", "1") == "1"          # 1 = fail if RVM cannot start (better than a poor result); 0 = fall back to MediaPipe
RVM_WIDTH = int(os.getenv("BG_RVM_WIDTH", "384"))            # the model looks at a copy of the frame this wide
ORT_THREADS = max(1, int(os.getenv("BG_ORT_THREADS", "1")))  # a Render job has 1 CPU; more threads only fight for it
CV_THREADS = max(1, int(os.getenv("BG_CV_THREADS", "1")))
cv2.setNumThreads(CV_THREADS)
RVM_RATIO = float(os.getenv("BG_RVM_RATIO", "0.5"))          # internal downsample ratio of the model
RVM_EVERY = max(1, int(os.getenv("BG_RVM_EVERY", "2")))      # run the model on every Nth frame, reuse the matte in between
MAX_LONG = int(os.getenv("BG_MAX_LONG", "1920"))             # very large videos (4K and up) are scaled down to at most this long side ...
MAX_SHORT = int(os.getenv("BG_MAX_SHORT", "1080"))           # ... and this short side. Saves memory; 1080p and smaller are untouched. 0 = never scale.
RVM_FILENAME = "rvm_mobilenetv3_fp32.onnx"
RVM_URL = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/" + RVM_FILENAME


class RvmMatter:
    """Robust Video Matting (ONNX, CPU): a model made for people in video. Unlike the selfie model it does not
    treat furniture next to the person, photos of other people or call thumbnails as 'person'."""

    @staticmethod
    def _model_path() -> Path:
        beside = Path(__file__).resolve().parent / RVM_FILENAME
        if beside.exists() and beside.stat().st_size > 10_000_000:
            return beside
        cache = Path(tempfile.gettempdir()) / RVM_FILENAME
        if cache.exists() and cache.stat().st_size > 10_000_000:
            return cache
        last = None
        for attempt in range(1, 4):
            try:
                part = cache.with_suffix(".part")
                with urllib.request.urlopen(RVM_URL, timeout=90) as resp, open(part, "wb") as out:
                    shutil.copyfileobj(resp, out)
                if part.stat().st_size < 10_000_000:
                    raise RuntimeError("downloaded file is too small")
                part.replace(cache)
                return cache
            except Exception as e:  # network hiccup: try again
                last = e
                time.sleep(3 * attempt)
        raise RuntimeError(f"could not download the RVM model: {last}")

    def __init__(self):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = ORT_THREADS
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.sess = ort.InferenceSession(str(self._model_path()), sess_options=opts, providers=["CPUExecutionProvider"])
        self.states = [np.zeros((1, 1, 1, 1), np.float32) for _ in range(4)]
        self.ratio = np.array([RVM_RATIO], np.float32)

    @classmethod
    def create(cls):
        try:
            matter = cls()
            print(f"  Matting: Robust Video Matting (width {RVM_WIDTH}, ratio {RVM_RATIO}, every {RVM_EVERY} frame(s))", flush=True)
            return matter
        except Exception as e:
            if RVM_STRICT:
                raise RuntimeError(f"RVM matting is not available: {e}")
            print(f"  WARNING: RVM matting is not available ({e}); using the MediaPipe model instead", flush=True)
            return None

    def matte(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        lw = RVM_WIDTH
        lh = max(8, int(round(h * lw / w)))
        small = cv2.resize(frame_bgr, (lw, lh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        src = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
        fgr, pha, *self.states = self.sess.run(None, {
            "src": src, "r1i": self.states[0], "r2i": self.states[1],
            "r3i": self.states[2], "r4i": self.states[3], "downsample_ratio": self.ratio,
        })
        alpha = cv2.resize(pha[0, 0], (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip((alpha - 0.02) / 0.96, 0.0, 1.0).astype(np.float32)


class MaskRefiner:
    """Cleans the person mask frame after frame:
      * near-binary mask (no half-transparent 'fog');
      * keeps ONLY the main person (plus pieces right next to them, such as a raised hand);
      * locks onto where the person was in the previous frame, so thumbnails, other
        participants or dark areas that the model flags now and then never pop in;
      * smooths the mask over time so edges do not flicker.
    Works at low resolution, so it costs only a few ms per segmented frame."""

    LOW_W = 480
    GATE = 0.35        # the subject may move this much (relative to its size) between segmentations
    NEAR = 0.10        # other pieces count as part of the person if their centre is this close to it
    BLEND_OLD = 0.35   # weight of the previous mask when smoothing over time

    def __init__(self):
        self.prev_box = None
        self.prev_soft = None
        self._open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    @staticmethod
    def _grow(box, frac, w, h):
        x0, y0, x1, y1 = box
        mx, my = (x1 - x0) * frac, (y1 - y0) * frac
        return (max(0, int(x0 - mx)), max(0, int(y0 - my)), min(w, int(x1 + mx)), min(h, int(y1 + my)))

    def refine(self, mask: np.ndarray) -> np.ndarray:
        H, W = mask.shape[:2]
        lw = self.LOW_W
        lh = max(8, int(round(H * lw / W)))
        small = cv2.resize(mask.astype(np.float32), (lw, lh), interpolation=cv2.INTER_LINEAR)
        binary = (np.clip((small - 0.5) / 0.12, 0.0, 1.0) > 0.5).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self._open)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._close)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)

        keep = None
        if n > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            main = int(np.argmax(areas)) + 1
            if self.prev_box is not None:
                gx0, gy0, gx1, gy1 = self._grow(self.prev_box, self.GATE, lw, lh)
                best_overlap = 0
                for i in range(1, n):
                    overlap = int((labels[gy0:gy1, gx0:gx1] == i).sum())
                    if overlap > best_overlap:
                        best_overlap, main = overlap, i
            mx, my, mw, mh = stats[main, 0], stats[main, 1], stats[main, 2], stats[main, 3]
            nx0, ny0, nx1, ny1 = self._grow((mx, my, mx + mw, my + mh), self.NEAR, lw, lh)
            keep = np.zeros_like(binary)
            for i in range(1, n):
                cx, cy = cents[i]
                if i == main or (nx0 <= cx <= nx1 and ny0 <= cy <= ny1):
                    keep[labels == i] = 1
            if int(keep.sum()) < 0.005 * lw * lh:
                keep = None

        if keep is None:
            # no person found in this frame: hold the previous mask (or fall back to the soft mask on the very first frames)
            if self.prev_soft is not None:
                return cv2.resize(self.prev_soft, (W, H), interpolation=cv2.INTER_LINEAR)
            return np.clip((mask.astype(np.float32) - 0.30) / 0.40, 0.0, 1.0)

        ys, xs = np.where(keep > 0)
        self.prev_box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        keep = cv2.erode(keep, np.ones((3, 3), np.uint8))
        soft = cv2.GaussianBlur(keep.astype(np.float32), (5, 5), 0)
        if self.prev_soft is not None and self.prev_soft.shape == soft.shape:
            soft = (1.0 - self.BLEND_OLD) * soft + self.BLEND_OLD * self.prev_soft
        self.prev_soft = soft
        return cv2.resize(soft, (W, H), interpolation=cv2.INTER_LINEAR)


def probe_streams(path: Path) -> dict:
    """Per-stream info (duration, frame count) read with ffprobe; {} if it fails."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration,nb_frames,r_frame_rate",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        info = {}
        for st in json.loads(out.stdout).get("streams", []):
            info.setdefault(st.get("codec_type"), st)
        return info
    except Exception:
        return {}


def _dur(stream) -> float:
    try:
        return float(stream.get("duration"))
    except Exception:
        return -1.0


DEBUG_FRAMES = set(
    int(x) for x in os.getenv("BG_DEBUG_FRAMES", "").replace(" ", "").split(",") if x.strip().isdigit()
)


def save_debug_sheet(path: Path, frame_bgr, raw_mask, kept_mask, result_bgr) -> None:
    """2x2 picture for diagnosis: original | result / what the model says (heat map) | what we keep."""
    try:
        def fit(img, w=640):
            h = max(1, int(round(img.shape[0] * w / img.shape[1])))
            return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

        def label(img, text):
            out = img.copy()
            cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
            cv2.putText(out, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            return out

        size = (frame_bgr.shape[1], frame_bgr.shape[0])
        raw = cv2.resize(np.clip(raw_mask, 0, 1).astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
        kept = cv2.resize(np.clip(kept_mask, 0, 1).astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
        heat = cv2.applyColorMap((raw * 255).astype(np.uint8), cv2.COLORMAP_JET)
        kept_img = cv2.cvtColor((kept * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        a = label(fit(frame_bgr), "original frame")
        b = label(fit(result_bgr), "result")
        c = label(fit(heat), "model: red = sure person, blue = background")
        d = label(fit(kept_img), "kept after clean-up (white = kept)")
        sheet = np.vstack([np.hstack([a, b]), np.hstack([c, d])])
        cv2.imwrite(str(path), sheet)
        print(f"  [debug] saved {path.name}: model says person for {100 * float((raw > 0.5).mean()):.1f}% of the frame, kept {100 * float((kept > 0.5).mean()):.1f}%", flush=True)
    except Exception as e:
        print(f"  [debug] could not save the debug picture: {e}", flush=True)


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


def cap_dimensions(src_w: int, src_h: int) -> tuple[int, int]:
    """Very large videos (4K and up) are processed at no more than MAX_LONG x MAX_SHORT; smaller ones are untouched."""
    if MAX_LONG <= 0 or MAX_SHORT <= 0 or src_w <= 0 or src_h <= 0:
        return src_w, src_h
    long_side, short_side = max(src_w, src_h), min(src_w, src_h)
    scale = min(1.0, MAX_LONG / long_side, MAX_SHORT / short_side)
    if scale >= 1.0:
        return src_w, src_h
    w, h = int(round(src_w * scale)), int(round(src_h * scale))
    return w + (w % 2), h + (h % 2)


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

    # Native blend (about 6x faster than the numpy formula, same result within 1/255)
    mask = np.ascontiguousarray(mask, dtype=np.float32)
    return cv2.blendLinear(frame_bgr, background, mask, 1.0 - mask)


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
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # cv2.CAP_PROP_FPS is unreliable for browser-recorded WebM (often
        # variable frame rate), which drifts the silent output video's
        # duration away from the real audio duration ("fast forward" mouth).
        # Derive fps from ffprobe's actual duration instead, so
        # total_frames / fps always matches real playback time.
        cv2_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = cv2_fps
        duration = None
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
                capture_output=True, text=True, check=True
            )
            duration = float(probe.stdout.strip())
            if duration > 0 and total_frames > 0:
                fps = total_frames / duration
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
            pass  # fall back to cv2's fps below if ffprobe fails

        if not fps or fps <= 0 or fps > 300:
            fps = 30.0

        print(f"[fps-debug] cv2_fps={cv2_fps} ffprobe_duration={duration} total_frames={total_frames} computed_fps={fps}", flush=True)
        
        # Resolve output resolution (optional upscale, aspect-preserving)
        out_w, out_h = target_dimensions(width, height, config.upscale)
        cap_w, cap_h = cap_dimensions(width, height)
        pre_resize = config.upscale <= 0 and (cap_w, cap_h) != (width, height)
        if pre_resize:
            out_w, out_h = cap_w, cap_h
        print(f"  [resolution] source {width}x{height}" + (f", scaled down to {out_w}x{out_h} before processing (memory)" if pre_resize else ", kept as is"), flush=True)

        # Create branded background at the OUTPUT resolution so the logo and
        # solid/image background are rendered crisply rather than upscaled.
        background = create_branded_background(
            out_w, out_h, config.bg_color, logo_bgra, config.logo_scale,
            bg_image_path=config.bg_image_path
        )

        # Create the matting model (RVM by default); the MediaPipe segmenter is only needed as a fallback
        matter = RvmMatter.create() if SEGMENTER_CHOICE == "rvm" else None
        if matter is None:
            segmenter_info = create_segmenter(config.model_selection, config.segmenter)

        # Pipe finished frames into ONE ffmpeg process that encodes H.264 and adds the
        # source audio. No lossless intermediate file, so disk use stays small.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        clip_frames = total_frames if config.max_frames == 0 else min(total_frames, config.max_frames)
        clip_seconds = max(1.0, clip_frames / fps)
        maxrate_kbps = int(max(500, min(ENCODE_MAXRATE_KBPS, ENCODE_MAX_BYTES * 8 / 1000 / clip_seconds)))
        print(f"  Encoding: crf {ENCODE_CRF}, preset {ENCODE_PRESET}, max {maxrate_kbps} kbps, mask clean-up {'on' if MASK_CLEANUP else 'off'}", flush=True)
        src_info = probe_streams(video_path)
        print(f"  [streams] source: video {_dur(src_info.get('video', {})):.1f}s, audio {_dur(src_info.get('audio', {})):.1f}s", flush=True)
        print("  [version] stream v8: RVM matting (width %d, %d thread(s)) + timing log + 4K cap" % (RVM_WIDTH, ORT_THREADS), flush=True)
        refiner = MaskRefiner() if (MASK_CLEANUP and matter is None) else None
        raw_mask = None
        disk_report("before encoding")
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "ffmpeg.log"
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{out_w}x{out_h}", "-r", str(fps),
                "-i", "pipe:0",
                "-i", str(video_path),
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-threads", os.getenv("BG_FFMPEG_THREADS", "2"),
                "-c:v", "libx264", "-crf", ENCODE_CRF, "-preset", ENCODE_PRESET,
                # accurate RGB->YUV conversion: matches the old FFV1 path (plain bgr24 input shifted colours ~3 levels)
                "-vf", "scale=flags=accurate_rnd+full_chroma_int+bicubic",
                "-maxrate", f"{maxrate_kbps}k", "-bufsize", f"{2 * maxrate_kbps}k",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                "-c:a", "aac",
                "-shortest",
                str(output_path),
            ]
            with open(log_path, "wb") as log_file:
                encoder = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log_file)
                try:
                    frame_idx = 0
                    effective_total = total_frames if config.max_frames == 0 else min(total_frames, config.max_frames)
                    skip_factor = RVM_EVERY if matter is not None else 2  # run the model every Nth frame, reuse the mask in between
                    last_mask = None
                    tm = {"read": 0.0, "matte": 0.0, "composite": 0.0, "write": 0.0}
                    tm_frames, tm_calls, tm_start = 0, 0, time.perf_counter()

                    while True:
                        _t = time.perf_counter()
                        ret, frame_bgr = cap.read()
                        tm["read"] += time.perf_counter() - _t
                        if not ret:
                            break

                        # Check max_frames limit
                        if config.max_frames > 0 and frame_idx >= config.max_frames:
                            break

                        if pre_resize:
                            frame_bgr = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)

                        if frame_idx % skip_factor == 0 or last_mask is None:
                            _t = time.perf_counter()
                            tm_calls += 1
                            if matter is not None:
                                mask = matter.matte(frame_bgr)
                                raw_mask = mask
                            else:
                                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                                mask = segment_frame(frame_rgb, segmenter_info)
                                raw_mask = mask
                                if refiner is not None:
                                    mask = refiner.refine(mask)
                            tm["matte"] += time.perf_counter() - _t
                            last_mask = mask
                        else:
                            mask = last_mask

                        if not pre_resize and (out_w, out_h) != (width, height):
                            frame_bgr = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
                            mask = cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

                        _t = time.perf_counter()
                        composited = composite_frame(frame_bgr, background, mask, config.edge_softness)
                        tm["composite"] += time.perf_counter() - _t
                        if frame_idx in DEBUG_FRAMES and raw_mask is not None:
                            save_debug_sheet(output_path.parent / f"debug_frame_{frame_idx}.png", frame_bgr, raw_mask, mask, composited)

                        _t = time.perf_counter()
                        try:
                            encoder.stdin.write(np.ascontiguousarray(composited).tobytes())
                        except (BrokenPipeError, OSError):
                            break  # ffmpeg stopped; its log explains why (checked below)
                        tm["write"] += time.perf_counter() - _t

                        frame_idx += 1
                        tm_frames += 1
                        if frame_idx % 100 == 0:
                            wall = time.perf_counter() - tm_start
                            per = lambda k, n: 1000.0 * tm[k] / max(1, n)
                            print(f"  [timing] last {tm_frames} frames: {tm_frames / wall:.1f} fps | per frame: read {per('read', tm_frames):.0f} ms, matte {per('matte', tm_frames):.0f} ms ({tm_calls} calls, {per('matte', tm_calls):.0f} ms each), composite {per('composite', tm_frames):.0f} ms, write {per('write', tm_frames):.0f} ms", flush=True)
                            tm = {k: 0.0 for k in tm}
                            tm_frames, tm_calls, tm_start = 0, 0, time.perf_counter()
                        if frame_idx % 500 == 0:
                            try:
                                so_far = output_path.stat().st_size if output_path.exists() else 0
                            except OSError:
                                so_far = 0
                            print(f"  [disk] output file so far: {so_far / 1e6:.0f} MB at frame {frame_idx}", flush=True)
                            disk_report(f"frame {frame_idx}")
                        if progress_callback and effective_total > 0:
                            progress_callback(frame_idx, effective_total)
                finally:
                    try:
                        encoder.stdin.close()
                    except Exception:
                        pass
                    if encoder.poll() is None:
                        try:
                            encoder.wait(timeout=600)
                        except subprocess.TimeoutExpired:
                            encoder.kill()
                    rc = encoder.wait()
            if rc != 0:
                detail = log_path.read_text(errors="replace")[-1500:] if log_path.exists() else ""
                raise RuntimeError(f"FFmpeg encode failed (code {rc}): {detail}")
            try:
                print(f"  Encode finished: output file {output_path.stat().st_size / 1e6:.0f} MB", flush=True)
            except OSError:
                pass
            disk_report("after encoding")
            out_info = probe_streams(output_path)
            ov, oa = out_info.get("video", {}), out_info.get("audio", {})
            print(f"  [streams] output: video {_dur(ov):.1f}s ({ov.get('nb_frames')} frames, {ov.get('r_frame_rate')}), audio {_dur(oa):.1f}s", flush=True)
            expected = clip_frames / fps
            if _dur(ov) > 0 and abs(_dur(ov) - expected) > 0.03 * expected:
                print(f"  WARNING: output video is {_dur(ov):.1f}s but {clip_frames} frames at {fps:.3f} fps should be {expected:.1f}s", flush=True)
            if oa and _dur(oa) > 0 and _dur(oa) < 0.9 * _dur(ov):
                print(f"  WARNING: output audio ({_dur(oa):.1f}s) is shorter than the video ({_dur(ov):.1f}s)", flush=True)

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
            "-preset", "ultrafast",
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

        if needs_cfr(video_path):
            normalized_path = video_path.with_name(video_path.stem + "_cfr.mp4")
            normalize_cmd = [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vsync", "cfr", "-r", "30",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
                "-c:a", "aac",
                str(normalized_path)
            ]
            result = subprocess.run(normalize_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                video_path = normalized_path
            else:
                print(f"  WARNING: CFR normalization failed, using original file: {result.stderr[:300]}")
        else:
            print("  Source is already constant frame rate: skipping re-encode.", flush=True)
            
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
                if current % 100 == 0 or current == total_frames:
                    pct = (current / total_frames) * 100
                    print(f"  Processing: frame {current}/{total_frames} ({pct:.1f}%)", flush=True)

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
