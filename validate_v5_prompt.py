#!/usr/bin/env python3
"""
Validate V5 Deduction-Based Prompt Against Existing Scores.

This script tests the V5 deduction prompt using pre-converted audio files
from data/batch_assessment/audio/ and compares against existing scores.

Usage:
    python validate_v5_prompt.py [--samples N]
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, os.path.dirname(__file__))

from src.assessment.openai_assessment import assess_pronunciation_openai


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


def load_test_data():
    """Load audio files and their existing scores."""
    audio_dir = Path("data/batch_assessment/audio")
    entries_file = Path("data/batch_assessment/selected_entries.json")

    if not entries_file.exists():
        print(f"Error: {entries_file} not found")
        return []

    with open(entries_file) as f:
        entries = json.load(f)

    # Map names to existing scores
    name_to_score = {e["name"]: e["existing_score"] for e in entries}

    # Find audio files and match with scores
    test_data = []
    audio_files = sorted(audio_dir.glob("*.wav"))

    for audio_file in audio_files:
        # Extract name from filename (format: 000_FirstName_LastName.wav)
        parts = audio_file.stem.split("_", 1)
        if len(parts) < 2:
            continue

        name_from_file = parts[1].replace("_", " ")

        # Try to match with entries
        matched_score = None
        matched_name = None

        for entry_name, score in name_to_score.items():
            # Case-insensitive partial match
            if entry_name.lower().replace(" ", "") == name_from_file.lower().replace(" ", ""):
                matched_score = score
                matched_name = entry_name
                break
            elif name_from_file.lower() in entry_name.lower() or entry_name.lower() in name_from_file.lower():
                matched_score = score
                matched_name = entry_name
                break

        if matched_score is not None:
            test_data.append({
                "audio_file": str(audio_file),
                "name": matched_name,
                "existing_score": matched_score,
            })

    return test_data


def run_validation(samples: int = 50):
    """Run V5 prompt validation against existing scores."""

    print("=" * 70)
    print("V5 DEDUCTION PROMPT VALIDATION")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Target samples: {samples}")
    print("=" * 70)

    # Load test data
    test_data = load_test_data()

    if not test_data:
        print("No test data found!")
        return

    print(f"\nFound {len(test_data)} audio files with existing scores")
    test_data = test_data[:samples]

    results = []
    failures = []

    for i, data in enumerate(test_data):
        print(f"\n--- Sample {i+1}/{len(test_data)}: {data['name'][:40]} ---")
        print(f"  Existing score: {data['existing_score']}")
        print(f"  Audio: {Path(data['audio_file']).name}")

        try:
            # Run V5 assessment
            result = assess_pronunciation_openai(
                data["audio_file"],
                prompt_version="v5"
            )

            v5_score = result.get("final_score")
            total_deductions = result.get("total_deductions", 0)
            deductions = result.get("deductions", {})

            if v5_score is not None:
                diff = v5_score - data["existing_score"]
                results.append({
                    "name": data["name"],
                    "existing_score": data["existing_score"],
                    "v5_score": v5_score,
                    "difference": diff,
                    "total_deductions": total_deductions,
                    "deductions": deductions,
                    "summary": result.get("summary", ""),
                    "transcription": result.get("transcription", "")[:200],
                })

                diff_str = f"+{diff:.0f}" if diff >= 0 else f"{diff:.0f}"
                print(f"  V5 Score: {v5_score:.0f} | Diff: {diff_str} | Deductions: {total_deductions}")
            else:
                error = result.get("error", "Unknown error")
                print(f"  ERROR: {error}")
                failures.append({
                    "name": data["name"],
                    "reason": error
                })

        except Exception as e:
            print(f"  EXCEPTION: {e}")
            failures.append({
                "name": data["name"],
                "reason": str(e)
            })

    # Generate report
    if results:
        generate_report(results, failures, samples)
    else:
        print("\nNo successful assessments to report!")


def generate_report(results: list, failures: list, target_samples: int):
    """Generate comprehensive validation report."""

    existing_scores = [r["existing_score"] for r in results]
    v5_scores = [r["v5_score"] for r in results]
    differences = [r["difference"] for r in results]

    # Calculate statistics
    existing_mean = mean(existing_scores)
    existing_std = stdev(existing_scores) if len(existing_scores) > 1 else 0
    existing_min = min(existing_scores)
    existing_max = max(existing_scores)

    v5_mean = mean(v5_scores)
    v5_std = stdev(v5_scores) if len(v5_scores) > 1 else 0
    v5_min = min(v5_scores)
    v5_max = max(v5_scores)

    correlation = pearson_correlation(existing_scores, v5_scores)

    # Agreement rate (within ±10 points)
    agreement_count = sum(1 for d in differences if abs(d) <= 10)
    agreement_rate = agreement_count / len(results) * 100

    # Score distribution buckets
    buckets = [(0, 30), (30, 50), (50, 70), (70, 85), (85, 101)]
    bucket_labels = ["0-29", "30-49", "50-69", "70-84", "85-100"]

    existing_dist = []
    v5_dist = []
    for low, high in buckets:
        existing_dist.append(sum(1 for s in existing_scores if low <= s < high))
        v5_dist.append(sum(1 for s in v5_scores if low <= s < high))

    # Notable discrepancies (>20 point difference)
    discrepancies = [r for r in results if abs(r["difference"]) > 20]

    # Build report
    report_lines = [
        "=" * 80,
        "V5 DEDUCTION PROMPT VALIDATION REPORT",
        "=" * 80,
        f"Date: {datetime.now().strftime('%B %d, %Y %H:%M')}",
        f"Target Samples: {target_samples}",
        f"Successful Assessments: {len(results)}",
        f"Failures: {len(failures)}",
        "",
        "=" * 80,
        "EXECUTIVE SUMMARY",
        "=" * 80,
        "",
        f"Correlation coefficient (Pearson r): {correlation:.3f}",
        f"Agreement rate (within ±10 points): {agreement_rate:.1f}%",
        f"Mean difference (V5 - Existing): {mean(differences):+.1f}",
        "",
    ]

    # Interpretation
    if correlation > 0.7:
        interp = "STRONG correlation - V5 tracks well with existing scores!"
    elif correlation > 0.5:
        interp = "MODERATE correlation - V5 shows reasonable agreement."
    elif correlation > 0.3:
        interp = "WEAK correlation - V5 differs from existing scores."
    else:
        interp = "VERY WEAK correlation - V5 scores differently than existing."

    report_lines.extend([
        f"Interpretation: {interp}",
        "",
        "=" * 80,
        "SCORE STATISTICS",
        "=" * 80,
        "",
        f"{'Metric':<20} {'Existing':>15} {'V5 Prompt':>15} {'Target':>15}",
        "-" * 65,
        f"{'Mean':<20} {existing_mean:>15.1f} {v5_mean:>15.1f} {'40-55':>15}",
        f"{'Std Dev':<20} {existing_std:>15.1f} {v5_std:>15.1f} {'>10':>15}",
        f"{'Min':<20} {existing_min:>15.1f} {v5_min:>15.1f} {'':>15}",
        f"{'Max':<20} {existing_max:>15.1f} {v5_max:>15.1f} {'':>15}",
        f"{'Range':<20} {existing_max - existing_min:>15.1f} {v5_max - v5_min:>15.1f} {'>50':>15}",
        "",
        "=" * 80,
        "SCORE DISTRIBUTION",
        "=" * 80,
        "",
        f"{'Range':<12} {'Existing':>15} {'V5 Prompt':>15}",
        "-" * 45,
    ])

    for label, old_count, v5_count in zip(bucket_labels, existing_dist, v5_dist):
        report_lines.append(f"{label:<12} {old_count:>15} {v5_count:>15}")

    report_lines.extend([
        "",
        "=" * 80,
        "INDIVIDUAL RESULTS (sorted by difference)",
        "=" * 80,
        "",
        f"{'Name':<35} {'Exist':>8} {'V5':>8} {'Diff':>8} {'Deductions':>12}",
        "-" * 75,
    ])

    for r in sorted(results, key=lambda x: x["difference"]):
        diff_str = f"+{r['difference']:.0f}" if r['difference'] >= 0 else f"{r['difference']:.0f}"
        ded_str = f"{r['total_deductions']}" if r['total_deductions'] else "N/A"
        report_lines.append(f"{r['name'][:35]:<35} {r['existing_score']:>8.0f} {r['v5_score']:>8.0f} {diff_str:>8} {ded_str:>12}")

    # Success criteria evaluation
    report_lines.extend([
        "",
        "=" * 80,
        "SUCCESS CRITERIA EVALUATION",
        "=" * 80,
        "",
        f"1. Std Dev > 10: {'PASS' if v5_std > 10 else 'FAIL'} (V5 std dev: {v5_std:.1f})",
        f"2. Range > 50 pts: {'PASS' if (v5_max - v5_min) > 50 else 'FAIL'} (V5 range: {v5_max - v5_min:.0f})",
        f"3. Correlation > 0.45: {'PASS' if correlation > 0.45 else 'FAIL'} (r: {correlation:.3f})",
        f"4. Mean ~50: {'PASS' if 40 <= v5_mean <= 60 else 'FAIL'} (V5 mean: {v5_mean:.1f})",
        "",
    ])

    if discrepancies:
        report_lines.extend([
            "=" * 80,
            f"NOTABLE DISCREPANCIES (>20 point difference): {len(discrepancies)} cases",
            "=" * 80,
            "",
        ])

        for r in sorted(discrepancies, key=lambda x: abs(x["difference"]), reverse=True)[:10]:
            direction = "HIGHER" if r["difference"] > 0 else "LOWER"
            report_lines.extend([
                f"Name: {r['name']}",
                f"  Existing: {r['existing_score']:.0f} | V5: {r['v5_score']:.0f} ({direction} by {abs(r['difference']):.0f})",
                f"  Total deductions: {r['total_deductions']}",
                f"  Summary: {r.get('summary', 'N/A')[:100]}",
                "",
            ])

    report_lines.extend([
        "=" * 80,
        f"Report generated: {datetime.now().isoformat()}",
        "=" * 80,
    ])

    # Write report
    report_text = "\n".join(report_lines)

    os.makedirs("data/reports", exist_ok=True)
    output_file = "data/reports/v5_validation_report.txt"
    with open(output_file, 'w') as f:
        f.write(report_text)

    # Save JSON
    json_file = f"data/reports/v5_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "target_samples": target_samples,
            "successful": len(results),
            "failures": len(failures),
            "correlation": correlation,
            "agreement_rate": agreement_rate,
            "existing_stats": {
                "mean": existing_mean,
                "std": existing_std,
                "min": existing_min,
                "max": existing_max,
            },
            "v5_stats": {
                "mean": v5_mean,
                "std": v5_std,
                "min": v5_min,
                "max": v5_max,
            },
            "results": results,
            "failures": failures,
        }, f, indent=2)

    # Print report
    print("\n" + report_text)
    print(f"\nReport saved to: {output_file}")
    print(f"JSON data saved to: {json_file}")


def main():
    parser = argparse.ArgumentParser(description='Validate V5 deduction prompt')
    parser.add_argument('--samples', type=int, default=50, help='Number of samples to validate')
    args = parser.parse_args()

    run_validation(args.samples)


if __name__ == "__main__":
    main()
