# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Automated pronunciation assessment for call center interview recordings. The system fetches video records from Airtable, converts video to audio, scores pronunciation using AI, and writes scores back to Airtable. Runs daily via GitHub Actions cron job.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # then fill in API keys
```

Required `.env` keys:
```
OPENAI_API_KEY=          # Whisper transcription + GPT-4o
GEMINI_API_KEY=          # Gemini 2.5 Flash scoring
AIRTABLE_API_KEY=        # Airtable PAT
AIRTABLE_BASE_ID=        # app...
AIRTABLE_TEST_RESULTS_TABLE_ID=   # tbl...
RENDI_API_KEY=           # video-to-audio conversion
```

## Common Commands

```bash
# Single audio file assessment
python assess.py audio.wav
python assess.py audio.wav --provider azure

# Airtable batch cron job
python cron.py --dry-run         # preview without writing
python cron.py --batch-size 50
python cron.py --reprocess       # re-score already-scored records

# Validation scripts
python validate_two_stage.py     # compare two-stage vs existing scores
python test_two_stage.py         # test extreme cases (best/worst speakers)
python compare_services.py       # compare OpenAI vs SpeechAce vs Azure
```

No formal test suite — validation is done via the compare/validate scripts above.

## Architecture

**Data flow:**
1. `cron.py` fetches Airtable records with video URLs (created after Feb 1, 2026)
2. `src/utils/audio_converter.py` calls Rendi API to convert video → WAV
3. Assessment module scores the WAV file
4. `src/airtable/records.py` writes score back to "Video 1 score" field

**Assessment strategy — current best: Two-Stage (Whisper + Gemini)**

`src/assessment/transcribe_and_score.py` implements the production approach:
1. OpenAI Whisper transcribes audio (determines WHAT was said)
2. Gemini 2.5 Flash scores pronunciation given transcription (evaluates HOW it was said)

This two-stage separation achieves 0.58 correlation vs 0.16 for single-stage GPT-4o. Single-stage models compress all speakers into a narrow range; giving Gemini the transcript as context breaks this.

**Other assessment modules (for comparison/fallback):**
- `src/assessment/openai_assessment.py` — GPT-4o multimodal, prompt versions V2–V7
- `src/assessment/gemini_assessment.py` — Gemini-only (single-stage baseline)
- `src/assessment/speechace_assessment.py` — SpeechAce third-party API (evaluated, not production)
- `src/assessment/azure_assessment.py` — Azure Speech SDK (legacy)

**Scoring dimensions (two-stage, 25 pts each → 100 total):**
1. Intelligibility — can a US customer understand without repeating?
2. Phoneme Accuracy — correct consonants/vowels?
3. Fluency/Rhythm — smooth vs choppy/halting?
4. Accent Impact — how much does accent impede communication?

**Prompt versions in `src/assessment/prompt_builder.py`:**
V2–V7 were iterative attempts to solve score compression in single-stage OpenAI. All failed; the two-stage approach supersedes them. They remain for reference.

## Key Design Decisions

**Why two-stage beats single-stage:** GPT-4o-audio-preview and Gemini both exhibit score compression when scoring audio alone — they cluster all speakers in 50–65 regardless of actual quality. Providing the Whisper transcript as explicit context lets Gemini differentiate quality instead of defaulting to "sounds okay."

**SpeechAce was evaluated and rejected for production:** +40.9 point upward bias vs existing scores, 0.30 correlation, 45-second audio limit, and score clustering in 70–94 range. Files remain in `src/assessment/speechace_assessment.py` and `compare_speechace.py`.

**Airtable field ordering:** `cron.py` writes "Video 1 score" before "Pronunciation Assessment Score" — this order matters for the downstream Airtable automations.

## Known Issues

- **Rendi API 403**: Video-to-audio conversion credentials may be expired. Use pre-downloaded audio from `data/batch_assessment/audio/` for testing.
- **SpeechAce 45-sec limit**: Use `src/utils/audio_trimmer.py` → `trim_wav_middle()` as workaround.
- **High-score compression in two-stage**: Speakers with old scores 90–100 tend to score 45–60 in two-stage (may reflect genuine accent detection rather than a bug).

## Deployment

GitHub Actions cron (`.github/workflows/cron.yml`) runs `cron.py` daily. Designed to run on Render as a scheduled task. Batch size defaults to 200 records per run.
