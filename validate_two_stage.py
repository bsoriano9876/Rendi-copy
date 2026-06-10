"""Validate two-stage (Whisper + Gemini) assessment against existing scores."""
import os
import sys
import json
import glob
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

from src.assessment.transcribe_and_score import assess_pronunciation

AUDIO_DIR = "data/batch_assessment/audio"
ENTRIES_FILE = "data/batch_assessment/selected_entries.json"


def main():
    # Load existing scores
    with open(ENTRIES_FILE) as f:
        entries = json.load(f)

    # Create lookup by name
    score_lookup = {e["name"]: e["existing_score"] for e in entries}
    url_lookup = {e["name"]: e["url"] for e in entries}

    # Get audio files
    audio_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))[:20]

    print("=" * 70)
    print("TWO-STAGE VALIDATION (Whisper + Gemini Flash)")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Samples: {len(audio_files)}")
    print("=" * 70)

    results = []

    for i, audio_file in enumerate(audio_files):
        # Extract name from filename
        basename = os.path.basename(audio_file)
        # Format: 001_WENLYN_LAGANZON.wav -> WENLYN LAGANZON
        parts = basename.replace(".wav", "").split("_", 1)
        if len(parts) > 1:
            name = parts[1].replace("_", " ")
        else:
            name = basename

        existing_score = score_lookup.get(name, None)
        if existing_score is None:
            # Try case-insensitive match
            for k, v in score_lookup.items():
                if k.lower().replace(".", "") == name.lower().replace(".", ""):
                    existing_score = v
                    name = k
                    break

        print(f"\n--- [{i+1}/{len(audio_files)}] {name} (Old: {existing_score}) ---")

        try:
            result = assess_pronunciation(audio_file)
            new_score = result.get("final_score", 0)
            category = result.get("category", "N/A")
            errors = result.get("specific_errors", [])[:3]

            diff = new_score - (existing_score or 0)
            print(f"New Score: {new_score} | Category: {category} | Diff: {diff:+.0f}")
            if errors:
                print(f"Errors: {errors[0][:60]}...")

            results.append({
                "name": name,
                "video_url": url_lookup.get(name, ""),
                "existing_score": existing_score,
                "new_score": new_score,
                "category": category,
                "difference": diff,
                "transcription": result.get("transcription", "")[:200],
                "errors": errors,
                "scores": result.get("scores", {})
            })

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "name": name,
                "existing_score": existing_score,
                "new_score": 0,
                "error": str(e)
            })

    # Calculate statistics
    valid_results = [r for r in results if r.get("new_score", 0) > 0 and r.get("existing_score")]

    if valid_results:
        import numpy as np
        existing = [r["existing_score"] for r in valid_results]
        new = [r["new_score"] for r in valid_results]

        correlation = np.corrcoef(existing, new)[0, 1] if len(valid_results) > 1 else 0
        mean_diff = np.mean([r["difference"] for r in valid_results])

        existing_mean = np.mean(existing)
        existing_std = np.std(existing)
        new_mean = np.mean(new)
        new_std = np.std(new)

    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    if valid_results:
        print(f"\nStatistics:")
        print(f"  Correlation (Pearson r): {correlation:.3f}")
        print(f"  Mean difference: {mean_diff:+.1f}")
        print(f"  Existing scores - Mean: {existing_mean:.1f}, Std: {existing_std:.1f}")
        print(f"  New scores - Mean: {new_mean:.1f}, Std: {new_std:.1f}")

        # Agreement rate
        within_10 = sum(1 for r in valid_results if abs(r["difference"]) <= 10)
        within_20 = sum(1 for r in valid_results if abs(r["difference"]) <= 20)
        print(f"  Agreement within ±10: {within_10}/{len(valid_results)} ({100*within_10/len(valid_results):.1f}%)")
        print(f"  Agreement within ±20: {within_20}/{len(valid_results)} ({100*within_20/len(valid_results):.1f}%)")

    print(f"\n{'Name':<30} {'Old':>6} {'New':>6} {'Diff':>6} {'Category':>15}")
    print("-" * 70)
    for r in sorted(valid_results, key=lambda x: x["existing_score"]):
        print(f"{r['name']:<30} {r['existing_score']:>6.0f} {r['new_score']:>6} {r['difference']:>+6.0f} {r.get('category', 'N/A'):>15}")

    # Save results
    report = {
        "timestamp": datetime.now().isoformat(),
        "samples": len(results),
        "valid": len(valid_results),
        "correlation": correlation if valid_results else 0,
        "mean_difference": mean_diff if valid_results else 0,
        "results": results
    }

    report_file = f"data/reports/two_stage_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {report_file}")


if __name__ == "__main__":
    main()
