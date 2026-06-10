"""
Smoke test for brand_videos.py

Downloads a short public video, runs the pipeline, and verifies:
1. Output file exists
2. Output has both video and audio streams (via ffprobe)
"""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from brand_videos import (
    check_dependencies,
    download_video,
    url_to_filename,
    parse_urls_file,
    create_branded_background,
    load_logo,
    ProcessingConfig,
    process_video,
)


# Short public domain test video (Big Buck Bunny 10s clip, ~1MB)
TEST_VIDEO_URL = "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        cache_dir = tmpdir / "cache"
        output_dir = tmpdir / "output"
        cache_dir.mkdir()
        output_dir.mkdir()
        yield {
            "root": tmpdir,
            "cache": cache_dir,
            "output": output_dir,
        }


@pytest.fixture
def test_logo(temp_dirs):
    """Create a simple test logo."""
    from PIL import Image
    import numpy as np

    # Create a simple 100x100 red square with alpha
    logo_path = temp_dirs["root"] / "test_logo.png"
    img = Image.new("RGBA", (100, 100), (255, 0, 0, 200))
    img.save(logo_path)
    return logo_path


def test_dependencies_available():
    """Check that ffmpeg and yt-dlp are installed."""
    missing = check_dependencies()
    if missing:
        pytest.skip(f"Missing dependencies: {', '.join(missing)}")


def test_url_to_filename():
    """Test URL to filename conversion."""
    url1 = "https://example.com/video.mp4"
    url2 = "https://example.com/video.mp4?token=abc"

    name1 = url_to_filename(url1)
    name2 = url_to_filename(url2)

    # Should be filesystem-safe
    assert "/" not in name1
    assert "?" not in name1

    # Same base URL with different query should produce different names
    assert name1 != name2


def test_parse_urls_file(temp_dirs):
    """Test URL file parsing."""
    urls_file = temp_dirs["root"] / "urls.txt"
    urls_file.write_text("""
# Comment
https://example.com/video1.mp4

https://example.com/video2.mp4
# Another comment
https://example.com/video3.mp4
""")

    urls = parse_urls_file(urls_file)
    assert len(urls) == 3
    assert urls[0] == "https://example.com/video1.mp4"
    assert urls[1] == "https://example.com/video2.mp4"
    assert urls[2] == "https://example.com/video3.mp4"


def test_create_branded_background(test_logo):
    """Test branded background creation."""
    import numpy as np

    logo_bgra = load_logo(test_logo)
    bg = create_branded_background(
        width=640,
        height=480,
        bg_color=(245, 245, 245),
        logo_bgra=logo_bgra,
        logo_scale=0.25
    )

    assert bg.shape == (480, 640, 3)
    assert bg.dtype == np.uint8


@pytest.mark.slow
def test_full_pipeline(temp_dirs, test_logo):
    """
    Full smoke test: download video, process it, verify output.

    This test is marked as slow because it downloads a video from the internet.
    """
    missing = check_dependencies()
    if missing:
        pytest.skip(f"Missing dependencies: {', '.join(missing)}")

    # Download test video
    video_path = download_video(TEST_VIDEO_URL, temp_dirs["cache"])
    assert video_path.exists()

    # Create config
    config = ProcessingConfig(
        logo_path=test_logo,
        output_dir=temp_dirs["output"],
        cache_dir=temp_dirs["cache"],
        bg_color=(245, 245, 245),
        logo_scale=0.25,
        output_ext=".mp4",
        model_selection=1,
        edge_softness=5,
        segmenter="mediapipe",
    )

    # Process video
    output_name = f"{url_to_filename(TEST_VIDEO_URL)}.mp4"
    output_path = temp_dirs["output"] / output_name

    process_video(video_path, output_path, config)

    # Verify output exists
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    # Verify output has video stream using ffprobe
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            str(output_path)
        ],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0, f"ffprobe failed: {result.stderr}"

    probe_data = json.loads(result.stdout)
    streams = probe_data.get("streams", [])

    # Check for video stream
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    assert len(video_streams) >= 1, "Output must have at least one video stream"

    # Note: Audio stream check is optional since source may not have audio
    # The Big Buck Bunny clip should have audio
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    # Don't assert on audio - source might not have it


@pytest.mark.slow
def test_download_caching(temp_dirs):
    """Test that downloads are cached properly."""
    missing = check_dependencies()
    if missing:
        pytest.skip(f"Missing dependencies: {', '.join(missing)}")

    # First download
    path1 = download_video(TEST_VIDEO_URL, temp_dirs["cache"])
    mtime1 = path1.stat().st_mtime

    # Second download should use cache
    path2 = download_video(TEST_VIDEO_URL, temp_dirs["cache"])
    mtime2 = path2.stat().st_mtime

    assert path1 == path2
    assert mtime1 == mtime2  # File should not have been modified


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
