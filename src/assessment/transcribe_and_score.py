"""
Two-Stage Pronunciation Assessment: Transcribe then Score.

Stage 1: Use Whisper to transcribe audio (gets exact words spoken)
Stage 2: Use Gemini Pro to score pronunciation quality based on transcription + audio analysis
"""

import os
import json
import logging
import re
import base64
from typing import Optional, Tuple

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Try to import google-genai
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


def transcribe_with_whisper(audio_file: str) -> Tuple[str, dict]:
    """
    Transcribe audio using OpenAI Whisper.

    Returns:
        Tuple of (transcription_text, metadata_dict)
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)

    with open(audio_file, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            language="en"
        )

    return response.text, {
        "duration": getattr(response, 'duration', None),
        "language": getattr(response, 'language', 'en'),
        "segments": getattr(response, 'segments', [])
    }


def score_with_gemini(
    transcription: str,
    audio_file: str,
    model: str = "gemini-2.5-flash"
) -> dict:
    """
    Score pronunciation using Gemini Pro with both transcription and audio.

    The transcription tells Gemini WHAT was said.
    The audio tells Gemini HOW it was said.
    """
    if not GENAI_AVAILABLE:
        raise ImportError("google-genai not installed")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=api_key)

    # Read audio
    with open(audio_file, 'rb') as f:
        audio_bytes = f.read()

    ext = audio_file.lower().rsplit(".", 1)[-1]
    mime_type = {"wav": "audio/wav", "mp3": "audio/mp3"}.get(ext, "audio/wav")

    prompt = f"""You are a strict pronunciation assessor for English learners applying to US call centers.

TRANSCRIPTION (what the speaker said):
"{transcription}"

Listen to the audio and evaluate HOW WELL they pronounced these words.

SCORING CRITERIA - Be STRICT and use the FULL 0-100 range:

1. INTELLIGIBILITY (25 points max)
   - Could a US customer understand this without asking to repeat?
   - 0-5: Mostly unintelligible
   - 6-12: Understandable with significant effort
   - 13-18: Understandable with some effort
   - 19-22: Easily understood
   - 23-25: Crystal clear

2. PHONEME ACCURACY (25 points max)
   - Are consonants and vowels pronounced correctly?
   - Count errors: th->d, v->b, r->l, vowel substitutions, dropped sounds
   - 0-5: Many errors (10+)
   - 6-12: Several errors (6-9)
   - 13-18: Some errors (3-5)
   - 19-22: Few errors (1-2)
   - 23-25: No errors

3. FLUENCY & RHYTHM (25 points max)
   - Is speech smooth or choppy/halting?
   - 0-5: Very choppy, word-by-word
   - 6-12: Frequent pauses, many fillers
   - 13-18: Some hesitations
   - 19-22: Mostly smooth
   - 23-25: Natural flow

4. ACCENT IMPACT (25 points max)
   - How much does accent affect communication?
   - 0-5: Very heavy accent, distracting
   - 6-12: Heavy accent, noticeable effort needed
   - 13-18: Moderate accent, clearly non-native
   - 19-22: Slight accent, professional
   - 23-25: Native-like or no accent

FINAL SCORE = Sum of 4 categories (0-100)

CALIBRATION - Match these benchmarks:
- Score 10-25: Speaker is HARD TO UNDERSTAND, would fail call center screening
- Score 30-45: Heavy accent, UNDERSTANDABLE WITH EFFORT, needs training
- Score 50-65: Moderate accent, CLEARLY NON-NATIVE but functional
- Score 70-80: Good English, SLIGHT ACCENT, professional quality
- Score 85-100: NATIVE-LIKE, excellent for any role

IMPORTANT:
- Do NOT default everyone to 50-70
- A poor speaker should get 20-40
- An excellent speaker should get 80-95
- Listen carefully and differentiate

Respond in JSON only:
{{
    "scores": {{
        "intelligibility": <0-25>,
        "phoneme_accuracy": <0-25>,
        "fluency_rhythm": <0-25>,
        "accent_impact": <0-25>
    }},
    "specific_errors": ["list pronunciation errors heard"],
    "final_score": <0-100, sum of categories>,
    "category": "poor|below_average|average|good|excellent",
    "would_hire": true|false,
    "summary": "one sentence assessment"
}}"""

    response = client.models.generate_content(
        model=model,
        contents=[
            prompt,
            types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
        ]
    )

    # Parse response
    text = response.text
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        result = json.loads(json_match.group())
    else:
        raise ValueError(f"No JSON in response: {text[:200]}")

    # Round to nearest 5
    if "final_score" in result:
        result["final_score"] = round(result["final_score"] / 5) * 5
        result["final_score"] = max(10, min(100, result["final_score"]))

    return result


def assess_pronunciation(
    audio_file: str,
    gemini_model: str = "gemini-2.5-flash"
) -> dict:
    """
    Full two-stage pronunciation assessment.

    Stage 1: Whisper transcription
    Stage 2: Gemini scoring
    """
    result = {
        "audio_file": audio_file,
        "transcription": None,
        "final_score": 0,
        "error": None
    }

    try:
        # Stage 1: Transcribe
        logger.info(f"Transcribing: {audio_file}")
        transcription, metadata = transcribe_with_whisper(audio_file)
        result["transcription"] = transcription
        result["whisper_metadata"] = metadata

        if not transcription.strip():
            result["error"] = "Empty transcription"
            return result

        # Stage 2: Score
        logger.info(f"Scoring with Gemini: {gemini_model}")
        score_result = score_with_gemini(transcription, audio_file, gemini_model)
        result.update(score_result)

    except Exception as e:
        logger.error(f"Assessment failed: {e}")
        result["error"] = str(e)

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python transcribe_and_score.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]
    result = assess_pronunciation(audio_file)
    print(json.dumps(result, indent=2))
