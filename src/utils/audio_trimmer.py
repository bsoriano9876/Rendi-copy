"""
Audio Trimming Utilities.

Trims audio files to a specified duration, extracting from the middle.
Uses Python's wave module for WAV files (no external dependencies).
"""

import wave
import os
import logging
from typing import Optional, Tuple


logger = logging.getLogger(__name__)


def get_wav_duration(audio_path: str) -> float:
    """
    Get the duration of a WAV file in seconds.

    Args:
        audio_path: Path to WAV file

    Returns:
        Duration in seconds
    """
    with wave.open(audio_path, 'rb') as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
        return frames / float(rate)


def trim_wav_middle(
    input_path: str,
    output_path: str,
    max_duration: float = 45.0,
) -> Tuple[bool, dict]:
    """
    Trim a WAV file to max_duration seconds, extracting from the middle.

    If the audio is shorter than max_duration, it is copied as-is.
    If longer, extracts the middle segment.

    Args:
        input_path: Path to input WAV file
        output_path: Path for output WAV file
        max_duration: Maximum duration in seconds (default 45)

    Returns:
        Tuple of (success, details_dict)
    """
    try:
        with wave.open(input_path, 'rb') as wav_in:
            # Get audio parameters
            n_channels = wav_in.getnchannels()
            sample_width = wav_in.getsampwidth()
            framerate = wav_in.getframerate()
            n_frames = wav_in.getnframes()

            total_duration = n_frames / float(framerate)

            if total_duration <= max_duration:
                # Audio is short enough, just copy it
                wav_in.rewind()
                frames = wav_in.readframes(n_frames)

                with wave.open(output_path, 'wb') as wav_out:
                    wav_out.setnchannels(n_channels)
                    wav_out.setsampwidth(sample_width)
                    wav_out.setframerate(framerate)
                    wav_out.writeframes(frames)

                return True, {
                    "trimmed": False,
                    "original_duration": total_duration,
                    "output_duration": total_duration,
                    "output_path": output_path,
                }

            # Calculate middle segment
            target_frames = int(max_duration * framerate)
            start_frame = (n_frames - target_frames) // 2

            # Seek to start position and read frames
            wav_in.setpos(start_frame)
            frames = wav_in.readframes(target_frames)

            # Write output
            with wave.open(output_path, 'wb') as wav_out:
                wav_out.setnchannels(n_channels)
                wav_out.setsampwidth(sample_width)
                wav_out.setframerate(framerate)
                wav_out.writeframes(frames)

            output_duration = len(frames) / (n_channels * sample_width * framerate)

            logger.info(
                "Trimmed %s: %.1fs -> %.1fs (middle segment)",
                os.path.basename(input_path),
                total_duration,
                max_duration,
            )

            return True, {
                "trimmed": True,
                "original_duration": total_duration,
                "output_duration": max_duration,
                "start_time": start_frame / framerate,
                "end_time": (start_frame + target_frames) / framerate,
                "output_path": output_path,
            }

    except wave.Error as e:
        logger.error("Wave error processing %s: %s", input_path, e)
        return False, {"error": f"Wave error: {e}"}
    except FileNotFoundError:
        logger.error("File not found: %s", input_path)
        return False, {"error": f"File not found: {input_path}"}
    except Exception as e:
        logger.exception("Error trimming %s", input_path)
        return False, {"error": str(e)}


def estimate_text_for_duration(
    full_text: str,
    original_duration: float,
    target_duration: float,
    start_time: float,
) -> str:
    """
    Estimate the portion of text that corresponds to the trimmed audio segment.

    This is a rough approximation assuming uniform speaking rate.

    Args:
        full_text: The complete transcription
        original_duration: Original audio duration in seconds
        target_duration: Target (trimmed) duration in seconds
        start_time: Start time of trimmed segment in seconds

    Returns:
        Estimated text for the trimmed segment
    """
    if not full_text or original_duration <= 0:
        return full_text

    # Calculate character positions based on time ratios
    chars_per_second = len(full_text) / original_duration

    start_char = int(start_time * chars_per_second)
    end_char = int((start_time + target_duration) * chars_per_second)

    # Adjust to word boundaries
    # Find start of word
    while start_char > 0 and full_text[start_char - 1] not in ' \n\t':
        start_char -= 1

    # Find end of word
    while end_char < len(full_text) and full_text[end_char - 1] not in ' \n\t.!?':
        end_char += 1

    trimmed_text = full_text[start_char:end_char].strip()

    return trimmed_text
