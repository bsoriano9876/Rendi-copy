# CLAUDE.md — brand-videos

Guidance for Claude Code when working in the `brand-videos/` subproject.

## What This Does

Two capabilities for call-center interview clips:
1. **AI enhance** — upscale + face-restore low-res footage (`ai_enhance_video.py`)
2. **Brand background** — replace the background with a branded one, preserving audio (`brand_videos.py`)

The intended full pipeline is a **two-stage chain**:

```
source.mp4 → [ai_enhance_video.py] → enhanced.mp4 → [brand_videos.py] → final.mp4
              upscale + face-restore                  segment + brand bg
```

These two stages are NOT yet wired into a single command. A `--ai-enhance` flag on
`brand_videos.py` is the proposed next step (not built yet).

## Critical Context

**Source footage is 640×480, mis-tagged at 1000fps.** Real content is ~24–30fps.
ALWAYS normalize fps on frame extraction (`-vf fps=24`) or frame counts explode to
10,000 for a 10s clip. No compositing/encoding fix improves quality past the source
resolution — AI super-resolution is the only real lever. `face_enhance=True` matters
most for talking-head footage.

**Replicate account has <$5 credit → throttled to 6 requests/min, burst 1.** This is
the dominant constraint. Keep `--workers 1` and ~10.5s `--min-interval`. Adding ≥$5
credit lifts the throttle and allows parallel workers.

**Both Replicate *video* upscale models are broken for this account (do not retry as-is):**
- `lucataco/real-esrgan-video` → "Cog: Got error trying to upload output files" (server-side)
- `topazlabs/video-upscale` → "source.container is required" for every input format tried
  (file URL, file handle, data URI, predictions API + clean .mp4 URL)

**The working AI-enhance path is the *image* model applied per-frame:**
`nightmareai/real-esrgan` with `scale=2, face_enhance=True`. Rock solid (90M+ runs).
Confirmed working 640×480 → 1280×960.

## Commands

### Stage 1 — AI enhance (per-frame Real-ESRGAN via Replicate)

```bash
# Short proof (1s ≈ 24 frames, ~5 min at the rate limit)
python ai_enhance_video.py SRC.mp4 OUT.mp4 --fps 24 --scale 2 --max-seconds 1

# Full clip (whole video)
python ai_enhance_video.py SRC.mp4 OUT.mp4 --fps 24 --scale 2 --max-seconds 0

# Flags: --workers (keep 1 while throttled), --min-interval (default 10.5s),
#        --no-face (disable face_enhance), --scale (2 or 4)
```

Pipeline: ffmpeg extract frames @ fps → Real-ESRGAN per frame (parallel/paced,
retries on 429) → ffmpeg reassemble to H.264 CRF18 → mux original audio.

### Stage 2 — brand background

`brand_videos.py` only accepts URLs (always calls `download_video`), so it CANNOT
take a local file from the CLI. To composite a local enhanced clip, call the
functions directly (see the working snippet below).

```bash
# Normal URL-based usage (segmenters: mediapipe=free local, rembg=local, replicate=RVM paid)
python brand_videos.py --urls urls.txt --logo logo.png --output-dir ./output \
  --bg-image "ChatGPT Image May 10, 2026, 09_28_29 AM.png" --segmenter mediapipe --upscale 0
```

Local-file compositing (used to brand `test_10s_AI_ENHANCED.mp4`):

```python
from pathlib import Path
import brand_videos as bv
cfg = bv.ProcessingConfig(
    logo_path=Path('output/_noop_logo.png'),  # 1x1 transparent = no logo
    output_dir=Path('output'), cache_dir=Path('.cache'),
    bg_color=(20,20,40), logo_scale=0.001, output_ext='.mp4',
    model_selection=1, edge_softness=7, segmenter='mediapipe',
    max_frames=0,
    bg_image_path=Path('ChatGPT Image May 10, 2026, 09_28_29 AM.png'),
    upscale=0,
)
bv.process_video(Path('output/test_10s_AI_ENHANCED.mp4'),
                 Path('output/test_10s_BRANDED_nightowl.mp4'), cfg)
```

## Brand Assets

- **Background image:** `ChatGPT Image May 10, 2026, 09_28_29 AM.png` (1536×1024) —
  NIGHTOWL night scene (owl + moon + city skyline). This is the brand background.
- `test_logo.png` (200×100) is a **placeholder blue rectangle**, NOT a real logo.
  A real transparent NIGHTOWL logo PNG is needed if a logo overlay is wanted.
- Company: nightowl.consulting. Suggested solid brand color: dark navy ~`20,20,40`.

## Cost & Time (measured, per 1-minute video @ 24fps = 1440 frames)

Real-ESRGAN billed ~3.7s/frame on Nvidia T4 @ $0.000225/sec (observed via API metrics).

| Stage | Cost (1-min video) |
|---|---|
| AI enhance (1440 × 3.7s × $0.000225) | **~$1.20** |
| Brand bg — MediaPipe/rembg (local) | **$0.00** |
| Brand bg — RVM on Replicate L40S | ~$0.12–0.30 |
| **Total (enhance + free local bg)** | **≈ $1.20** |
| **Total (enhance + RVM bg)** | **≈ $1.40** |

**Time is the real bottleneck, not cost:**
- At <$5 credit (6 req/min): ~4.3 HOURS just for the enhance stage of a 1-min clip.
- With ≥$5 credit (parallel ~20×): ~4–9 minutes.

GPU pricing: T4 $0.000225/s, L40S $0.000975/s (replicate.com/pricing).

## brand_videos.py — encoding improvements already made

- Switched off the old `mp4v` (MPEG-4 Part 2, blocky) writer → **lossless FFV1
  intermediate** + single **libx264 CRF18** final encode in `mux_audio`.
- Added `--upscale HEIGHT` (aspect-preserving, even-width, never downscales).
- MediaPipe now uses the **soft confidence mask** (with mid-band stretch) instead of
  the hard `category_mask == 0` threshold → cleaner edges. Hard mask is the fallback.

## Known Issues / Quality Notes

- **MediaPipe edge halo:** source was shot on a light background, so a faint light
  halo can linger around hair/shoulders. RVM (`--segmenter replicate`) gives
  noticeably cleaner hair edges (paid). Or tune `edge_softness` / the mask mid-band.
- **Per-frame face restoration can shimmer** in motion (GFPGAN/CodeFormer ignore
  temporal continuity). On the test clip, sampled stills were consistent — but verify
  at full playback speed before shipping a batch.
- `ai_enhance_video.py` pipes long runs through `tail`, which buffers until exit — no
  live progress on backgrounded runs. Don't pipe through tail if you need live output.

## Reference Outputs (in output/)

- `test_10s_AI_ENHANCED.mp4` — 10s clip, AI-enhanced to 1280×960 (no bg change). ✅ good
- `test_10s_BRANDED_nightowl.mp4` — above + NIGHTOWL background composited. ✅ good
- `compare_BEFORE_640x480.png` / `compare_AFTER_realesrgan_img.png` — single-frame
  upscale before/after proof.

## Scratch files (safe to delete)

`test_ai_upscale.py` (first failed-model experiment), various `output/*.png` check
frames. The keeper scripts are `brand_videos.py` and `ai_enhance_video.py`.
