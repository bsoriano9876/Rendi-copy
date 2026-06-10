"""
SpeechAce Pronunciation Assessment Module.

Uses SpeechAce's Score Speech API for open-ended/spontaneous speech assessment.
Evaluates pronunciation, fluency, grammar, vocabulary, and coherence.

API Documentation: https://api-docs.speechace.com/
"""

import logging
import time
from typing import Optional

import requests

from ..config import SPEECHACE_API_KEY, SPEECHACE_API_URL, SPEECHACE_DIALECT
from ..utils.logging_utils import build_error_result


logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAYS = [1, 2, 4]  # Exponential backoff in seconds

# Score Text API endpoint (for scripted speech with reference text)
SPEECHACE_TEXT_API_URL = "https://api.speechace.co/api/scoring/text/v9/json"


def assess_pronunciation_speechace_text(
    audio_file: str,
    reference_text: str,
    language: str = None,
    api_key: str = None,
    include_fluency: bool = True,
) -> dict:
    """
    Perform pronunciation assessment using SpeechAce Score Text API.

    This API is for scripted speech where you provide the expected text.
    It compares the audio against the reference text and scores pronunciation.

    Args:
        audio_file: Path to the audio file (WAV, MP3, etc.)
        reference_text: The expected text that should be spoken
        language: Dialect code (e.g., "en-us", "en-gb"). Defaults to config.
        api_key: SpeechAce API key (optional, uses env var if not provided)
        include_fluency: Include fluency metrics in response

    Returns:
        Dictionary containing assessment results or error
    """
    key = api_key or SPEECHACE_API_KEY
    dialect = language or SPEECHACE_DIALECT

    if not key:
        return build_error_result(
            "SpeechAce API key not configured. Set SPEECHACE_API_KEY in .env",
            stage="configuration",
            audio_file=audio_file,
            language=dialect,
        )

    # Build request URL with API key as query parameter
    params = {
        "key": key,
        "dialect": dialect,
    }

    # Prepare form data
    form_data = {
        "text": reference_text,
    }
    if include_fluency:
        form_data["include_fluency"] = "1"

    last_error = None
    response = None

    for attempt in range(MAX_RETRIES):
        try:
            with open(audio_file, "rb") as f:
                ext = audio_file.lower().rsplit(".", 1)[-1]
                mime_types = {
                    "wav": "audio/wav",
                    "mp3": "audio/mpeg",
                    "m4a": "audio/mp4",
                    "webm": "audio/webm",
                    "ogg": "audio/ogg",
                }
                mime_type = mime_types.get(ext, "audio/wav")

                files = {
                    "user_audio_file": (audio_file.split("/")[-1], f, mime_type)
                }

                response = requests.post(
                    SPEECHACE_TEXT_API_URL,
                    params=params,
                    data=form_data,
                    files=files,
                    timeout=120,
                )
                response.raise_for_status()
                break

        except requests.exceptions.Timeout as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                logger.warning("SpeechAce API timeout (attempt %d/%d), retrying in %ds",
                               attempt + 1, MAX_RETRIES, delay)
                time.sleep(delay)
            else:
                return build_error_result(
                    f"SpeechAce API timeout after {MAX_RETRIES} retries",
                    error=e,
                    stage="speechace_api_call",
                    audio_file=audio_file,
                )

        except requests.exceptions.ConnectionError as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                logger.warning("SpeechAce connection error (attempt %d/%d), retrying in %ds",
                               attempt + 1, MAX_RETRIES, delay, e)
                time.sleep(delay)
            else:
                return build_error_result(
                    f"SpeechAce connection error after {MAX_RETRIES} retries: {e}",
                    error=e,
                    stage="speechace_api_call",
                    audio_file=audio_file,
                )

        except requests.exceptions.HTTPError as e:
            return build_error_result(
                f"SpeechAce API HTTP error: {e}",
                error=e,
                stage="speechace_api_call",
                audio_file=audio_file,
                status_code=response.status_code if response else None,
            )

        except FileNotFoundError as e:
            return build_error_result(
                f"Audio file not found: {audio_file}",
                error=e,
                stage="file_read",
            )

        except Exception as e:
            return build_error_result(
                f"Unexpected error: {e}",
                error=e,
                stage="speechace_api_call",
                audio_file=audio_file,
            )

    # Parse response
    try:
        result_data = response.json()
    except (ValueError, AttributeError) as e:
        return build_error_result(
            f"Failed to parse SpeechAce response as JSON: {e}",
            error=e,
            stage="parse_response",
            audio_file=audio_file,
        )

    if result_data.get("status") != "success":
        error_msg = result_data.get("detail_message") or result_data.get("short_message") or "Unknown API error"
        return build_error_result(
            f"SpeechAce API error: {error_msg}",
            stage="speechace_api_response",
            audio_file=audio_file,
            api_response=result_data,
        )

    # Extract scores from response
    text_score = result_data.get("text_score", {})

    # Get word-level scores
    word_score_list = text_score.get("word_score_list", [])

    # Calculate average quality score from words
    word_scores = [w.get("quality_score", 0) for w in word_score_list if w.get("quality_score") is not None]
    avg_word_score = sum(word_scores) / len(word_scores) if word_scores else None

    # Get fluency scores if available
    fluency_score = text_score.get("fluency_score", {})

    # Build normalized result
    result = {
        "transcription": reference_text,  # For Score Text, we know the expected text
        "scores": {
            "pronunciation": avg_word_score,
            "fluency": fluency_score.get("overall_score") if fluency_score else None,
        },
        "word_score_list": word_score_list,
        "fluency_score": fluency_score,
        "final_score": avg_word_score,
        "quota_remaining": result_data.get("quota_remaining"),
    }

    # Get IELTS/CEFR scores if present
    speechace_scores = text_score.get("speechace_score", {})
    ielts_scores = text_score.get("ielts_score", {})
    if speechace_scores:
        result["speechace_score"] = speechace_scores
        if speechace_scores.get("pronunciation"):
            result["final_score"] = speechace_scores.get("pronunciation")
    if ielts_scores:
        result["ielts_score"] = ielts_scores

    logger.info(
        "SpeechAce Text assessment complete for %s: score=%.1f, quota_remaining=%s",
        audio_file,
        result["final_score"] or 0,
        result["quota_remaining"],
    )

    return result


def assess_pronunciation_speechace(
    audio_file: str,
    language: str = None,
    api_key: str = None,
    include_ielts_feedback: bool = True,
) -> dict:
    """
    Perform pronunciation assessment using SpeechAce Score Speech API.

    This API is designed for spontaneous/open-ended speech where no reference
    text is provided. It transcribes the audio and scores pronunciation,
    fluency, grammar, vocabulary, and coherence.

    Args:
        audio_file: Path to the audio file (WAV, MP3, M4A, WEBM, OGG, AIFF)
        language: Dialect code (e.g., "en-us", "en-gb"). Defaults to config.
        api_key: SpeechAce API key (optional, uses env var if not provided)
        include_ielts_feedback: Include detailed IELTS feedback metrics

    Returns:
        Dictionary containing:
        - transcription: Recognized speech text
        - scores: Dict with pronunciation, fluency, grammar, vocabulary, coherence
        - speechace_score: Raw SpeechAce 0-100 scaled scores
        - ielts_score: IELTS-scaled scores (0-9) if requested
        - final_score: Overall score from speechace_score.overall
        - word_score_list: Per-word pronunciation scores
        - quota_remaining: API quota remaining
        - error: Present if assessment failed
    """
    key = api_key or SPEECHACE_API_KEY
    dialect = language or SPEECHACE_DIALECT

    if not key:
        return build_error_result(
            "SpeechAce API key not configured. Set SPEECHACE_API_KEY in .env",
            stage="configuration",
            audio_file=audio_file,
            language=dialect,
        )

    # Build request URL with API key as query parameter
    params = {
        "key": key,
        "dialect": dialect,
    }

    # Prepare form data
    form_data = {}
    if include_ielts_feedback:
        form_data["include_ielts_feedback"] = "1"

    last_error = None
    response = None

    for attempt in range(MAX_RETRIES):
        try:
            # Open audio file for upload (reopen on each retry)
            with open(audio_file, "rb") as f:
                # Determine MIME type from extension
                ext = audio_file.lower().rsplit(".", 1)[-1]
                mime_types = {
                    "wav": "audio/wav",
                    "mp3": "audio/mpeg",
                    "m4a": "audio/mp4",
                    "webm": "audio/webm",
                    "ogg": "audio/ogg",
                    "aiff": "audio/aiff",
                }
                mime_type = mime_types.get(ext, "audio/wav")

                files = {
                    "user_audio_file": (audio_file.split("/")[-1], f, mime_type)
                }

                response = requests.post(
                    SPEECHACE_API_URL,
                    params=params,
                    data=form_data,
                    files=files,
                    timeout=120,  # Generous timeout for audio processing
                )
                response.raise_for_status()
                break  # Success, exit retry loop

        except requests.exceptions.Timeout as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                logger.warning(
                    "SpeechAce API timeout (attempt %d/%d), retrying in %ds",
                    attempt + 1, MAX_RETRIES, delay
                )
                time.sleep(delay)
            else:
                logger.error("SpeechAce API timeout after %d attempts", MAX_RETRIES)
                return build_error_result(
                    f"SpeechAce API timeout after {MAX_RETRIES} retries",
                    error=e,
                    stage="speechace_api_call",
                    audio_file=audio_file,
                    language=dialect,
                )

        except requests.exceptions.ConnectionError as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                logger.warning(
                    "SpeechAce API connection error (attempt %d/%d), retrying in %ds: %s",
                    attempt + 1, MAX_RETRIES, delay, e
                )
                time.sleep(delay)
            else:
                logger.error("SpeechAce API connection error after %d attempts: %s",
                             MAX_RETRIES, e)
                return build_error_result(
                    f"SpeechAce API connection error after {MAX_RETRIES} retries: {e}",
                    error=e,
                    stage="speechace_api_call",
                    audio_file=audio_file,
                    language=dialect,
                )

        except requests.exceptions.HTTPError as e:
            # Don't retry HTTP errors (4xx, 5xx) - they're likely permanent
            logger.error("SpeechAce API HTTP error for %s: %s", audio_file, e)
            return build_error_result(
                f"SpeechAce API HTTP error: {e}",
                error=e,
                stage="speechace_api_call",
                audio_file=audio_file,
                language=dialect,
                status_code=response.status_code if response else None,
                response_text=response.text[:500] if response else None,
            )

        except FileNotFoundError as e:
            return build_error_result(
                f"Audio file not found: {audio_file}",
                error=e,
                stage="file_read",
                audio_file=audio_file,
            )

        except Exception as e:
            logger.exception("Unexpected error calling SpeechAce API for %s", audio_file)
            return build_error_result(
                f"Unexpected error: {e}",
                error=e,
                stage="speechace_api_call",
                audio_file=audio_file,
                language=dialect,
            )

    # Parse response
    try:
        result_data = response.json()
    except (ValueError, AttributeError) as e:
        return build_error_result(
            f"Failed to parse SpeechAce response as JSON: {e}",
            error=e,
            stage="parse_response",
            audio_file=audio_file,
            response_text=response.text[:500] if response else None,
        )

    # Check for API-level errors
    if result_data.get("status") != "success":
        error_msg = result_data.get("detail_message") or result_data.get("short_message") or "Unknown API error"
        return build_error_result(
            f"SpeechAce API error: {error_msg}",
            stage="speechace_api_response",
            audio_file=audio_file,
            language=dialect,
            api_status=result_data.get("status"),
            api_response=result_data,
        )

    # Extract scores from response
    # The Score Speech API returns data under "speech_score"
    speech_score = result_data.get("speech_score", {})

    # Get SpeechAce 0-100 scores (primary)
    speechace_scores = speech_score.get("speechace_score", {})

    # Get IELTS-scaled scores (optional)
    ielts_scores = speech_score.get("ielts_score", {})

    # Build normalized result matching OpenAI assessment format
    result = {
        "transcription": speech_score.get("transcript", ""),
        "scores": {
            "pronunciation": speechace_scores.get("pronunciation"),
            "fluency": speechace_scores.get("fluency"),
            "grammar": speechace_scores.get("grammar"),
            "coherence": speechace_scores.get("coherence"),
            "vocabulary": speechace_scores.get("vocab"),
        },
        "speechace_score": speechace_scores,
        "ielts_score": ielts_scores,
        "final_score": speechace_scores.get("overall"),
        "word_score_list": speech_score.get("word_score_list", []),
        "quota_remaining": result_data.get("quota_remaining"),
    }

    # Check for score issues (warnings about relevance, length, etc.)
    score_issues = speech_score.get("score_issue_list", [])
    if score_issues:
        result["score_issues"] = score_issues
        for issue in score_issues:
            logger.warning(
                "SpeechAce score issue [%s]: %s - %s",
                issue.get("status"),
                issue.get("short_message"),
                issue.get("detail_message"),
            )

    logger.info(
        "SpeechAce assessment complete for %s: overall=%.1f, quota_remaining=%s",
        audio_file,
        result["final_score"] or 0,
        result["quota_remaining"],
    )

    return result


def print_assessment(assessment: dict):
    """Pretty print the SpeechAce assessment results."""
    if "error" in assessment:
        print(f"\nError: {assessment['error']}")
        if assessment.get("error_context"):
            print(f"Context: {assessment['error_context']}")
        return

    print("\n" + "=" * 60)
    print("PRONUNCIATION ASSESSMENT RESULTS (SpeechAce)")
    print("=" * 60)

    print(f"\nTranscription:\n{assessment.get('transcription', 'N/A')}")

    print("\n--- SpeechAce Scores (0-100) ---")
    scores = assessment.get("scores", {})
    for key in ["pronunciation", "fluency", "grammar", "vocabulary", "coherence"]:
        value = scores.get(key)
        if isinstance(value, (int, float)):
            print(f"  {key.capitalize():14}: {value:.1f}")
        else:
            print(f"  {key.capitalize():14}: N/A")

    # Show IELTS scores if available
    ielts = assessment.get("ielts_score", {})
    if ielts:
        print("\n--- IELTS Scores (0-9) ---")
        for key in ["pronunciation", "fluency", "grammar", "vocab", "coherence", "overall"]:
            value = ielts.get(key)
            if isinstance(value, (int, float)):
                print(f"  {key.capitalize():14}: {value:.1f}")

    # Show score issues if any
    issues = assessment.get("score_issues", [])
    if issues:
        print("\n--- Score Issues (Warnings) ---")
        for issue in issues:
            status = issue.get("status", "unknown")
            short_msg = issue.get("short_message", "")
            detail_msg = issue.get("detail_message", "")
            print(f"  [{status}] {short_msg}: {detail_msg}")

    # Show problematic words if available
    word_scores = assessment.get("word_score_list", [])
    low_score_words = [w for w in word_scores if w.get("quality_score", 100) < 60]
    if low_score_words:
        print(f"\n--- Words Needing Improvement ({len(low_score_words)} identified) ---")
        for word in low_score_words[:10]:  # Show top 10
            print(f"  '{word.get('word', '?')}': {word.get('quality_score', 'N/A')}")

    final_score = assessment.get("final_score")
    if final_score is not None:
        print("\n" + "=" * 60)
        print(f"  FINAL SCORE: {final_score:.1f}/100")
        print("=" * 60)

    quota = assessment.get("quota_remaining")
    if quota is not None:
        print(f"\n(API Quota remaining: {quota})")
