"""Test two-stage (Whisper + Gemini Pro) assessment on extreme cases."""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

from src.assessment.transcribe_and_score import assess_pronunciation

EXTREME_CASES = [
    {"name": "WENLYN LAGANZON", "existing_score": 10, "audio_index": "001"},
    {"name": "Lerry Kim Minorca", "existing_score": 10, "audio_index": "014"},
    {"name": "Daniel Arvin Sedo", "existing_score": 90, "audio_index": "022"},
    {"name": "Carla Villalon", "existing_score": 100, "audio_index": "095"},
]

AUDIO_DIR = "data/batch_assessment/audio"

def main():
    print("=" * 70)
    print("TWO-STAGE ASSESSMENT (Whisper + Gemini Pro)")
    print("=" * 70)

    results = []

    for case in EXTREME_CASES:
        import glob
        pattern = os.path.join(AUDIO_DIR, f"{case['audio_index']}*.wav")
        matches = glob.glob(pattern)
        if not matches:
            print(f"\nSkipping {case['name']}: Audio not found")
            continue

        audio_file = matches[0]
        print(f"\n--- {case['name']} (Old score: {case['existing_score']}) ---")

        try:
            result = assess_pronunciation(audio_file)

            score = result.get("final_score", 0)
            category = result.get("category", "N/A")
            scores = result.get("scores", {})
            summary = result.get("summary", "")[:60]
            transcription = result.get("transcription", "")[:80]
            errors = result.get("specific_errors", [])[:3]

            diff = score - case['existing_score']

            print(f"Transcription: {transcription}...")
            print(f"Score: {score} | Category: {category} | Diff: {diff:+d}")
            if scores:
                print(f"  Intelligibility: {scores.get('intelligibility', 0)}/25")
                print(f"  Phoneme: {scores.get('phoneme_accuracy', 0)}/25")
                print(f"  Fluency: {scores.get('fluency_rhythm', 0)}/25")
                print(f"  Accent: {scores.get('accent_impact', 0)}/25")
            if errors:
                print(f"Errors: {', '.join(errors[:3])}")
            print(f"Summary: {summary}")

            results.append({
                "name": case['name'],
                "existing_score": case['existing_score'],
                "new_score": score,
                "category": category,
                "diff": diff
            })

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "name": case['name'],
                "existing_score": case['existing_score'],
                "new_score": 0,
                "category": "error",
                "diff": -case['existing_score']
            })

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Name':<30} {'Old':>6} {'New':>6} {'Category':>15} {'Diff':>6}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<30} {r['existing_score']:>6} {r['new_score']:>6} {r['category']:>15} {r['diff']:>+6d}")

if __name__ == "__main__":
    main()
