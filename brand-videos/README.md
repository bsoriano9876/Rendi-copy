# Brand Videos - Background Replacement CLI

Batch video background replacement tool. Downloads videos from URLs, replaces backgrounds with a branded background (solid color + centered logo), preserves audio, and outputs to a folder.

## Features

- Downloads videos from URLs using yt-dlp (supports most video hosting platforms)
- Replaces background using MediaPipe selfie segmentation or rembg
- Applies branded background with centered logo
- Preserves original audio track
- Parallel processing support
- Configurable edge softening for cleaner composites

## Installation

### 1. Python Dependencies

```bash
cd brand-videos
pip install -r requirements.txt
```

### 2. FFmpeg

**macOS (Homebrew):**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html and add to PATH.

### 3. yt-dlp

Should be installed via requirements.txt, but if needed:
```bash
pip install yt-dlp
```

### 4. Optional: rembg (higher quality segmentation)

```bash
pip install rembg[cpu]
```

## Quickstart

```bash
# Basic usage
python brand_videos.py \
  --urls urls.txt \
  --logo company_logo.png \
  --output-dir ./output

# With custom background color and parallel processing
python brand_videos.py \
  --urls urls.txt \
  --logo logo.png \
  --output-dir ./output \
  --bg-color 30,30,30 \
  --parallel 4
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--urls PATH` | Required | Text file with one video URL per line |
| `--logo PATH` | Required | Logo image (PNG with alpha preferred) |
| `--output-dir PATH` | Required | Output directory for branded videos |
| `--cache-dir PATH` | `.cache` | Cache directory for downloads |
| `--bg-color R,G,B` | `245,245,245` | Background color (light gray) |
| `--logo-scale FLOAT` | `0.25` | Logo width as fraction of frame width |
| `--output-ext` | `.mp4` | Output format: `.mp4` or `.webm` |
| `--model` | `1` | MediaPipe model: 0 (general) or 1 (landscape) |
| `--edge-softness INT` | `5` | Gaussian blur kernel for mask edges (0 to disable) |
| `--parallel N` | `1` | Number of videos to process concurrently |
| `--segmenter` | `mediapipe` | Segmentation backend: `mediapipe` or `rembg` |

## URLs File Format

```text
# Comments start with #
# Blank lines are ignored

https://example.com/video1.mp4
https://example.com/video2.webm
https://vimeo.com/123456789
```

## Output Formats

### `.mp4` (default, recommended)

- Fastest processing (copies video codec, no re-encoding)
- Audio transcoded to AAC
- Compatible with most players

### `.webm`

- Re-encodes video with libvpx-vp9
- Uses libopus for audio
- Smaller file size
- Significantly slower (full video re-encode)

## Segmenter Options

### MediaPipe (default)

- Fast, lightweight
- Good for talking-head videos
- May have some edge artifacts

```bash
python brand_videos.py --segmenter mediapipe ...
```

### rembg

- Higher quality segmentation using U2Net
- Cleaner edges, better hair handling
- Slower processing
- Requires additional installation: `pip install rembg[cpu]`

```bash
python brand_videos.py --segmenter rembg ...
```

Use rembg when:
- Edge quality is critical
- Processing time is not a constraint
- Videos have complex backgrounds or fine details (hair, accessories)

## Performance

### CPU Performance (tested on Apple M1)

| Segmenter | Resolution | FPS | Notes |
|-----------|------------|-----|-------|
| MediaPipe | 720p | ~15 fps | Good for batch processing |
| MediaPipe | 1080p | ~8 fps | Acceptable |
| rembg | 720p | ~3 fps | Use for quality-critical work |
| rembg | 1080p | ~1.5 fps | Slow but highest quality |

### Parallel Scaling

`--parallel N` scales nearly linearly on multi-core machines:

| Workers | Speedup (approx) |
|---------|------------------|
| 1 | 1x |
| 2 | 1.9x |
| 4 | 3.5x |
| 8 | 6x |

Note: Memory usage increases with parallel workers. For 1080p videos, expect ~500MB per worker.

## Future Optimizations

For faster processing on machines with NVIDIA GPUs:

1. Install ONNX Runtime with CUDA support
2. Use rembg with GPU acceleration

```bash
pip install onnxruntime-gpu
```

This is not implemented in the current version but would provide 5-10x speedup for the segmentation step.

## Troubleshooting

### "ffmpeg not found"

Ensure FFmpeg is installed and in your PATH:
```bash
ffmpeg -version
```

### "yt-dlp failed"

Some URLs may require authentication or be geoblocked. Try:
```bash
yt-dlp --verbose "YOUR_URL"
```

### "Cannot open video"

The video format may not be supported by OpenCV. Try converting first:
```bash
ffmpeg -i input.webm -c:v libx264 output.mp4
```

### Memory issues with parallel processing

Reduce the number of workers:
```bash
python brand_videos.py --parallel 2 ...
```

### Poor segmentation quality

1. Try the rembg segmenter: `--segmenter rembg`
2. Increase edge softness: `--edge-softness 9`
3. Ensure good lighting in source videos

## Running Tests

```bash
pytest tests/
```

The smoke test downloads a short public video and verifies the pipeline produces valid output.
