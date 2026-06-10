#!/usr/bin/env python3
"""
Compare SpeechAce API vs Existing "Video 1 Score" in Airtable.

Uses pre-existing audio files from batch_assessment directory, trims them
to 45 seconds (from the middle), and compares SpeechAce scores against
existing scores.

Usage:
    python compare_speechace.py [--samples N] [--dialect en-us]
"""

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from statistics import mean, stdev
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from src.assessment.speechace_assessment import assess_pronunciation_speechace_text
from src.utils.audio_trimmer import trim_wav_middle, estimate_text_for_duration, get_wav_duration


# Paths to pre-existing data
AUDIO_DIR = Path("data/batch_assessment/audio")
RESULTS_DIR = Path("data/batch_assessment/results")


def pearson_correlation(x: list, y: list) -> float:
    """Calculate Pearson correlation coefficient between two lists."""
    n = len(x)
    if n != len(y) or n < 2:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))

    sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
    sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)

    denominator = (sum_sq_x * sum_sq_y) ** 0.5

    if denominator == 0:
        return 0.0

    return numerator / denominator


def get_preexisting_samples(max_samples: int = 50) -> list:
    """
    Get samples from pre-existing batch assessment results.

    Returns records that have:
    - An audio file in data/batch_assessment/audio/
    - A result file with transcription in data/batch_assessment/results/
    - An existing_score (Video 1 score) and openai_final_score

    Args:
        max_samples: Maximum number of samples to return

    Returns:
        List of sample dicts with audio_path, transcription, existing_score, openai_score
    """
    samples = []

    # Get all result files
    result_files = sorted(RESULTS_DIR.glob("*.json"))

    for result_file in result_files:
        if len(samples) >= max_samples:
            break

        # Check for corresponding audio file
        audio_name = result_file.stem + ".wav"
        audio_path = AUDIO_DIR / audio_name

        if not audio_path.exists():
            continue

        # Load result data
        try:
            with open(result_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        # Check for required fields
        assessment = data.get("assessment", {})
        transcription = assessment.get("transcription")
        existing_score = data.get("existing_score")
        openai_score = data.get("openai_final_score") or assessment.get("final_score")

        # Skip if missing transcription or scores
        if not transcription or existing_score is None:
            continue

        # Skip if transcription is too short (likely an error)
        if len(transcription) < 50:
            continue

        samples.append({
            "name": data.get("name", result_file.stem),
            "audio_path": str(audio_path),
            "transcription": transcription,
            "existing_score": float(existing_score),
            "openai_score": float(openai_score) if openai_score else None,
            "result_file": str(result_file),
        })

    return samples


def compare_speechace(samples: int = 50, dialect: str = "en-us", max_audio_duration: float = 45.0):
    """Compare SpeechAce against existing scores using pre-existing audio files."""

    print("=" * 70)
    print("SERVICE COMPARISON: SpeechAce vs Existing Scores")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Target samples: {samples}")
    print(f"Dialect: {dialect}")
    print(f"Max audio duration: {max_audio_duration}s")
    print("=" * 70)

    # Get pre-existing samples
    records = get_preexisting_samples(max_samples=samples)

    if not records:
        print("No valid samples found in data/batch_assessment/!")
        print("Make sure audio files and result JSON files exist.")
        return

    print(f"\nFound {len(records)} valid samples with audio and transcriptions")

    results = []
    failures = []

    for i, record in enumerate(records):
        print(f"\n--- Record {i+1}/{len(records)}: {record['name'][:40]} ---")
        print(f"  Existing (Video 1) score: {record['existing_score']}")
        if record['openai_score']:
            print(f"  OpenAI score: {record['openai_score']}")

        # Create temp file for trimmed audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            trimmed_path = tmp.name

        try:
            # Get original duration
            original_duration = get_wav_duration(record['audio_path'])
            print(f"  Original audio: {original_duration:.1f}s")

            # Trim audio to max_audio_duration from the middle
            success, trim_info = trim_wav_middle(
                record['audio_path'],
                trimmed_path,
                max_duration=max_audio_duration,
            )

            if not success:
                print(f"  SKIP: Trim failed - {trim_info.get('error', 'unknown')}")
                failures.append({
                    "name": record["name"],
                    "reason": f"Trim failed: {trim_info.get('error', 'unknown')}"
                })
                continue

            if trim_info.get("trimmed"):
                print(f"  Trimmed: {trim_info['start_time']:.1f}s - {trim_info['end_time']:.1f}s")

            # Estimate text for trimmed segment
            if trim_info.get("trimmed"):
                reference_text = estimate_text_for_duration(
                    record['transcription'],
                    original_duration,
                    max_audio_duration,
                    trim_info['start_time'],
                )
                print(f"  Reference text (estimated): {reference_text[:60]}...")
            else:
                reference_text = record['transcription']
                print(f"  Reference text: {reference_text[:60]}...")

            # Run SpeechAce assessment
            print(f"  Running SpeechAce assessment...")
            speechace_result = assess_pronunciation_speechace_text(
                trimmed_path,
                reference_text,
                language=dialect,
            )

            if "error" in speechace_result:
                error_msg = speechace_result.get('error', 'unknown')
                print(f"  ERROR: {error_msg}")
                failures.append({
                    "name": record["name"],
                    "reason": f"SpeechAce failed: {error_msg}"
                })
                continue

            speechace_score = speechace_result.get("final_score")

            if speechace_score is not None:
                result = {
                    "name": record["name"],
                    "existing_score": record["existing_score"],
                    "openai_score": record["openai_score"],
                    "speechace_score": speechace_score,
                    "diff_vs_existing": speechace_score - record["existing_score"],
                    "diff_vs_openai": (speechace_score - record["openai_score"]) if record["openai_score"] else None,
                    "speechace_scores": speechace_result.get("scores", {}),
                    "word_count": len(speechace_result.get("word_score_list", [])),
                    "trimmed": trim_info.get("trimmed", False),
                    "audio_duration": trim_info.get("output_duration", original_duration),
                }
                results.append(result)

                diff_str = f"+{result['diff_vs_existing']:.1f}" if result['diff_vs_existing'] >= 0 else f"{result['diff_vs_existing']:.1f}"
                print(f"  SpeechAce Score: {speechace_score:.1f} | Diff vs Existing: {diff_str}")

                if record["openai_score"]:
                    diff_openai = f"+{result['diff_vs_openai']:.1f}" if result['diff_vs_openai'] >= 0 else f"{result['diff_vs_openai']:.1f}"
                    print(f"  Diff vs OpenAI: {diff_openai}")

                # Show quota remaining periodically
                quota = speechace_result.get("quota_remaining")
                if quota is not None and quota >= 0:
                    print(f"  (Quota remaining: {quota})")
            else:
                print(f"  ERROR: No score returned")
                failures.append({
                    "name": record["name"],
                    "reason": "No score returned from SpeechAce"
                })

        finally:
            if os.path.exists(trimmed_path):
                os.remove(trimmed_path)

    # Generate analysis and report
    if results:
        generate_report(results, failures, samples, dialect)
    else:
        print("\nNo successful comparisons to report!")


def generate_report(results: list, failures: list, target_samples: int, dialect: str):
    """Generate comprehensive comparison report."""

    existing_scores = [r["existing_score"] for r in results]
    speechace_scores = [r["speechace_score"] for r in results]
    differences = [r["diff_vs_existing"] for r in results]

    # OpenAI comparison (if available)
    openai_scores = [r["openai_score"] for r in results if r["openai_score"] is not None]
    sa_for_openai = [r["speechace_score"] for r in results if r["openai_score"] is not None]

    # Calculate statistics
    existing_mean = mean(existing_scores)
    existing_std = stdev(existing_scores) if len(existing_scores) > 1 else 0
    existing_min = min(existing_scores)
    existing_max = max(existing_scores)

    sa_mean = mean(speechace_scores)
    sa_std = stdev(speechace_scores) if len(speechace_scores) > 1 else 0
    sa_min = min(speechace_scores)
    sa_max = max(speechace_scores)

    correlation_existing = pearson_correlation(existing_scores, speechace_scores)
    correlation_openai = pearson_correlation(openai_scores, sa_for_openai) if openai_scores else None

    # Agreement rate (within +/- 10 points)
    agreement_count = sum(1 for d in differences if abs(d) <= 10)
    agreement_rate = agreement_count / len(results) * 100

    # Score distribution buckets
    buckets = [(0, 30), (30, 50), (50, 70), (70, 85), (85, 101)]
    bucket_labels = ["0-29", "30-49", "50-69", "70-84", "85-100"]

    existing_dist = []
    sa_dist = []
    for low, high in buckets:
        existing_dist.append(sum(1 for s in existing_scores if low <= s < high))
        sa_dist.append(sum(1 for s in speechace_scores if low <= s < high))

    # Notable discrepancies (>20 point difference)
    discrepancies = [r for r in results if abs(r["diff_vs_existing"]) > 20]

    # Build report
    report_lines = [
        "=" * 80,
        "SERVICE COMPARISON REPORT",
        "SpeechAce vs Existing Scores (Using Pre-existing Audio)",
        "=" * 80,
        f"Date: {datetime.now().strftime('%B %d, %Y')}",
        f"Dialect: {dialect}",
        f"Target Samples: {target_samples}",
        f"Successful Comparisons: {len(results)}",
        f"Failures: {len(failures)}",
        "",
        "=" * 80,
        "EXECUTIVE SUMMARY",
        "=" * 80,
        "",
        f"Correlation vs Existing (Video 1): {correlation_existing:.3f}",
    ]

    if correlation_openai is not None:
        report_lines.append(f"Correlation vs OpenAI: {correlation_openai:.3f}")

    report_lines.extend([
        f"Agreement rate (within +/-10 points): {agreement_rate:.1f}%",
        f"Mean difference (SpeechAce - Existing): {mean(differences):+.1f}",
        "",
    ])

    # Interpretation
    if correlation_existing > 0.7:
        interp = "Strong positive correlation - SpeechAce scores track well with existing service."
    elif correlation_existing > 0.4:
        interp = "Moderate correlation - SpeechAce and existing service show some agreement."
    elif correlation_existing > 0:
        interp = "Weak correlation - SpeechAce and existing service scores differ significantly."
    else:
        interp = "No correlation or negative - Services measure speech quality differently."

    report_lines.extend([
        f"Interpretation: {interp}",
        "",
        "=" * 80,
        "SCORE STATISTICS",
        "=" * 80,
        "",
        f"{'Metric':<20} {'Existing':>15} {'SpeechAce':>15} {'Difference':>15}",
        "-" * 65,
        f"{'Mean':<20} {existing_mean:>15.1f} {sa_mean:>15.1f} {sa_mean - existing_mean:>+15.1f}",
        f"{'Std Dev':<20} {existing_std:>15.1f} {sa_std:>15.1f} {sa_std - existing_std:>+15.1f}",
        f"{'Min':<20} {existing_min:>15.1f} {sa_min:>15.1f} {sa_min - existing_min:>+15.1f}",
        f"{'Max':<20} {existing_max:>15.1f} {sa_max:>15.1f} {sa_max - existing_max:>+15.1f}",
        f"{'Range':<20} {existing_max - existing_min:>15.1f} {sa_max - sa_min:>15.1f} {(sa_max - sa_min) - (existing_max - existing_min):>+15.1f}",
        "",
    ])

    # OpenAI comparison if available
    if openai_scores:
        openai_mean = mean(openai_scores)
        openai_std = stdev(openai_scores) if len(openai_scores) > 1 else 0
        report_lines.extend([
            "--- Comparison with OpenAI Scores ---",
            f"OpenAI Mean: {openai_mean:.1f} | SpeechAce Mean: {sa_mean:.1f} | Diff: {sa_mean - openai_mean:+.1f}",
            f"Correlation (SpeechAce vs OpenAI): {correlation_openai:.3f}",
            "",
        ])

    report_lines.extend([
        "=" * 80,
        "SCORE DISTRIBUTION",
        "=" * 80,
        "",
        f"{'Range':<12} {'Existing':>15} {'SpeechAce':>15}",
        "-" * 45,
    ])

    for label, old_count, sa_count in zip(bucket_labels, existing_dist, sa_dist):
        report_lines.append(f"{label:<12} {old_count:>15} {sa_count:>15}")

    report_lines.extend([
        "",
        "=" * 80,
        "INDIVIDUAL RESULTS",
        "=" * 80,
        "",
        f"{'Name':<30} {'Exist':>7} {'OpenAI':>7} {'SA':>7} {'Diff':>7}",
        "-" * 65,
    ])

    for r in sorted(results, key=lambda x: x["diff_vs_existing"]):
        openai_str = f"{r['openai_score']:.0f}" if r['openai_score'] else "N/A"
        diff_str = f"+{r['diff_vs_existing']:.0f}" if r['diff_vs_existing'] >= 0 else f"{r['diff_vs_existing']:.0f}"
        report_lines.append(
            f"{r['name'][:30]:<30} {r['existing_score']:>7.0f} {openai_str:>7} "
            f"{r['speechace_score']:>7.0f} {diff_str:>7}"
        )

    report_lines.extend([
        "",
        "=" * 80,
        f"NOTABLE DISCREPANCIES (>20 point difference): {len(discrepancies)} cases",
        "=" * 80,
        "",
    ])

    if discrepancies:
        for r in sorted(discrepancies, key=lambda x: abs(x["diff_vs_existing"]), reverse=True):
            direction = "HIGHER" if r["diff_vs_existing"] > 0 else "LOWER"
            report_lines.extend([
                f"Name: {r['name']}",
                f"  Existing Score: {r['existing_score']:.0f}",
                f"  OpenAI Score: {r['openai_score']:.0f}" if r['openai_score'] else "",
                f"  SpeechAce Score: {r['speechace_score']:.0f} ({direction} by {abs(r['diff_vs_existing']):.0f} points)",
                "",
            ])
    else:
        report_lines.append("No discrepancies >20 points found.")

    if failures:
        report_lines.extend([
            "",
            "=" * 80,
            f"FAILURES ({len(failures)} records)",
            "=" * 80,
            "",
        ])
        for i, f in enumerate(failures, 1):
            report_lines.append(f"{i}. {f['name']}: {f['reason']}")

    # Bias analysis
    higher_count = sum(1 for d in differences if d > 0)
    lower_count = sum(1 for d in differences if d < 0)
    report_lines.extend([
        "",
        "=" * 80,
        "CONCLUSION",
        "=" * 80,
        "",
        f"SpeechAce shows {correlation_existing:.2f} correlation with existing scores.",
        "",
        f"Bias analysis:",
        f"  - SpeechAce scored HIGHER: {higher_count} times ({higher_count/len(results)*100:.1f}%)",
        f"  - SpeechAce scored LOWER: {lower_count} times ({lower_count/len(results)*100:.1f}%)",
        f"  - Mean bias: {mean(differences):+.1f} points",
        "",
    ])

    if correlation_openai is not None:
        report_lines.append(f"SpeechAce shows {correlation_openai:.2f} correlation with OpenAI scores.")

    report_lines.extend([
        "",
        "=" * 80,
        f"Report generated: {datetime.now().isoformat()}",
        "=" * 80,
    ])

    # Write report to file
    report_text = "\n".join(report_lines)

    output_file = "data/reports/speechace_comparison_report.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(report_text)

    # Also save JSON for further analysis
    json_file = f"data/reports/speechace_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "dialect": dialect,
            "target_samples": target_samples,
            "successful": len(results),
            "failures": len(failures),
            "correlation_vs_existing": correlation_existing,
            "correlation_vs_openai": correlation_openai,
            "agreement_rate": agreement_rate,
            "mean_difference": mean(differences),
            "existing_stats": {
                "mean": existing_mean,
                "std": existing_std,
                "min": existing_min,
                "max": existing_max,
            },
            "speechace_stats": {
                "mean": sa_mean,
                "std": sa_std,
                "min": sa_min,
                "max": sa_max,
            },
            "results": results,
            "failures": failures,
        }, f, indent=2)

    # Print report
    print("\n" + report_text)
    print(f"\nReport saved to: {output_file}")
    print(f"JSON data saved to: {json_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare SpeechAce vs existing scores using pre-existing audio'
    )
    parser.add_argument('--samples', type=int, default=50,
                        help='Number of samples to compare (default: 50)')
    parser.add_argument('--dialect', type=str, default='en-us',
                        choices=['en-us', 'en-gb', 'en-au'],
                        help='SpeechAce dialect (default: en-us)')
    parser.add_argument('--max-duration', type=float, default=45.0,
                        help='Max audio duration in seconds (default: 45)')
    args = parser.parse_args()

    compare_speechace(args.samples, args.dialect, args.max_duration)


if __name__ == "__main__":
    main()
