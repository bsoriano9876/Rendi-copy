"""
Gemini-based Pronunciation Assessment Module.

Uses Google's Gemini model with native audio understanding for pronunciation scoring.
"""

import os
import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import google-genai
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("google-genai not installed. Run: pip install google-genai")


def get_gemini_client():
    """Get configured Gemini client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment")
    return genai.Client(api_key=api_key)


PRONUNCIATION_PROMPT = """You are a STRICT pronunciation assessor for call center hiring.

Listen to this audio and determine: Would you hire this person for a US call center?

FIRST, answer these questions:
1. Can you understand EVERY word clearly without replaying? (YES/NO)
2. Does the speaker sound like they could be from the US? (YES/NO)
3. Do you hear the speaker struggle with any sounds? (YES/NO)
4. Would a US customer complain about this accent? (YES/NO)

SCORING BASED ON ANSWERS:
- If Q1=NO (can't understand everything): Score 10-30
- If Q1=YES but Q4=YES (customers would complain): Score 30-50
- If Q1=YES, Q4=NO, but Q2=NO (not native-sounding): Score 50-70
- If Q1=YES, Q4=NO, Q2=mostly YES: Score 70-90
- If all positive (sounds American/native): Score 90-100

COUNT SPECIFIC ERRORS:
- Each consonant error (th->d, v->b, etc.): note it
- Each vowel error: note it
- Each stress error: note it
- More than 5 errors = score below 50
- More than 10 errors = score below 30

CRITICAL: You MUST use the FULL range of scores:
- 10-25 = Difficult to understand, heavy accent, many errors
- 30-45 = Understandable with effort, heavy accent
- 50-65 = Clear enough but obviously non-native
- 70-85 = Good English, slight accent
- 90-100 = Native or indistinguishable from native

DO NOT DEFAULT TO 60-70 FOR EVERYONE. Differentiate quality levels.

Respond in JSON:
{
    "transcription": "what speaker said",
    "binary_answers": {
        "q1_understand_all": "YES|NO",
        "q2_sounds_american": "YES|NO",
        "q3_struggles_with_sounds": "YES|NO",
        "q4_customers_complain": "YES|NO"
    },
    "error_count": <number of specific errors>,
    "specific_errors": ["list each error"],
    "final_score": <10-100, use FULL range>,
    "category": "poor|below_average|average|good|excellent",
    "summary": "one sentence"
}"""


def assess_pronunciation_gemini(
    audio_file: str,
    language: str = "en-US",
    model: str = "gemini-2.0-flash"
) -> dict:
    """
    Assess pronunciation using Google Gemini with native audio understanding.

    Args:
        audio_file: Path to audio file (WAV, MP3, etc.)
        language: Target language (default en-US)
        model: Gemini model to use

    Returns:
        Dictionary with assessment results
    """
    if not GENAI_AVAILABLE:
        return {
            "error": "google-genai not installed",
            "final_score": 0,
            "summary": "Failed: google-genai package not available"
        }

    try:
        client = get_gemini_client()

        # Read audio file
        with open(audio_file, 'rb') as f:
            audio_bytes = f.read()

        # Determine MIME type
        ext = audio_file.lower().rsplit(".", 1)[-1]
        mime_types = {
            "wav": "audio/wav",
            "mp3": "audio/mp3",
            "m4a": "audio/m4a",
            "ogg": "audio/ogg",
            "webm": "audio/webm"
        }
        mime_type = mime_types.get(ext, "audio/wav")

        logger.info(f"Sending audio to Gemini ({model}): {audio_file}")

        # Make request with inline audio
        response = client.models.generate_content(
            model=model,
            contents=[
                PRONUNCIATION_PROMPT,
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
            ]
        )

        response_text = response.text
        logger.debug(f"Gemini response: {response_text[:500]}")

        # Parse JSON response
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            result = json.loads(json_match.group())
        else:
            raise ValueError(f"No JSON found in response: {response_text[:200]}")

        # Ensure final_score is rounded to nearest 5
        if "final_score" in result:
            raw_score = result["final_score"]
            result["final_score"] = round(raw_score / 5) * 5
            result["final_score"] = max(10, min(100, result["final_score"]))

        return result

    except Exception as e:
        logger.error(f"Gemini assessment failed: {e}")
        return {
            "error": str(e),
            "final_score": 0,
            "summary": f"Failed: {str(e)}"
        }


def test_gemini_connection() -> bool:
    """Test if Gemini API is accessible."""
    if not GENAI_AVAILABLE:
        print("google-genai not installed")
        return False

    try:
        client = get_gemini_client()
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=["Say 'hello' in one word."]
        )
        print(f"Gemini test response: {response.text}")
        return True
    except Exception as e:
        print(f"Gemini test failed: {e}")
        return False


if __name__ == "__main__":
    # Test connection
    print("Testing Gemini connection...")
    test_gemini_connection()
