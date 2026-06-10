"""Test Gemini on extreme score cases (10, 90, 100)."""
import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.assessment.gemini_assessment import assess_pronunciation_gemini

# Extreme cases from selected_entries.json
EXTREME_CASES = [
    {"name": "WENLYN LAGANZON", "existing_score": 10, "audio_index": "001"},
    {"name": "Lerry Kim Minorca", "existing_score": 10, "audio_index": "014"},
    {"name": "Daniel Arvin Sedo", "existing_score": 90, "audio_index": "022"},
    {"name": "Carla Villalon", "existing_score": 100, "audio_index": "095"},
]

AUDIO_DIR = "data/batch_assessment/audio"

def main():
    print("=" * 70)
    print("EXTREME CASES V5 TEST")
    print("=" * 70)
    
    results = []
    
    for case in EXTREME_CASES:
        audio_file = os.path.join(AUDIO_DIR, f"{case['audio_index']}_{case['name'].replace(' ', '_').replace('.', '_')}.wav")
        
        # Check if file exists
        if not os.path.exists(audio_file):
            # Try alternate naming
            import glob
            pattern = os.path.join(AUDIO_DIR, f"{case['audio_index']}*.wav")
            matches = glob.glob(pattern)
            if matches:
                audio_file = matches[0]
            else:
                print(f"\nSkipping {case['name']}: Audio file not found")
                continue
        
        print(f"\n--- {case['name']} (Old score: {case['existing_score']}) ---")
        print(f"Audio: {os.path.basename(audio_file)}")
        
        try:
            result = assess_pronunciation_gemini(
                audio_file=audio_file,
                language="en-US"
            )

            gemini_score = result.get("final_score", 0)
            category = result.get("category", "N/A")
            summary = result.get("summary", "")[:70]
            dims = result.get("dimension_scores", {})

            diff = gemini_score - case['existing_score']
            print(f"Gemini Score: {gemini_score} | Category: {category} | Diff: {diff:+d}")
            if dims:
                print(f"Dimensions: intel={dims.get('intelligibility',0)} phoneme={dims.get('phoneme_accuracy',0)} fluency={dims.get('fluency',0)}")
            print(f"Summary: {summary}")

            results.append({
                "name": case['name'],
                "existing_score": case['existing_score'],
                "gemini_score": gemini_score,
                "category": category,
                "diff": diff
            })
            
        except Exception as e:
            print(f"ERROR: {e}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Name':<30} {'Old':>6} {'Gemini':>7} {'Cat':>12} {'Diff':>6}")
    print("-" * 65)
    for r in results:
        print(f"{r['name']:<30} {r['existing_score']:>6} {r['gemini_score']:>7} {r['category']:>12} {r['diff']:>+6d}")

if __name__ == "__main__":
    main()
