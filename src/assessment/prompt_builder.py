"""
Pronunciation Assessment Prompt Builder Module.

ANTI-COMPRESSION DESIGN DECISIONS:
==================================
This module implements several techniques to prevent GPT-4o from compressing
scores toward the mean (the "regression to middle" problem):

1. NO NARROW RANGE INSTRUCTIONS: We explicitly avoid phrases like "most speakers
   score 50-70" which cause the model to anchor all outputs to that range.

2. EVENLY SPACED CALIBRATION EXAMPLES: Instead of clustering examples near the
   mean (old: 51, 52, 64, 64, 84), we use evenly spaced anchors across the full
   range (24, 41, 58, 74, 91) to teach the model the entire distribution.

3. ASCENDING ORDER (WORST TO BEST): Examples are ordered from lowest to highest
   score. This leverages recency bias - the model pays more attention to later
   examples, so we end with high-quality examples to counteract the tendency
   to compress upward scores.

4. CHAIN-OF-THOUGHT BEFORE SCORING: The model must evaluate categorical dimensions
   (poor/fair/good/excellent) BEFORE generating a numeric score. This forces
   deliberate reasoning rather than pattern-matching to a mean.

5. EXPLICIT ANTI-COMPRESSION INSTRUCTION: A direct instruction at the top of
   the prompt tells the model NOT to cluster scores in the middle range.

6. DIMENSION-TO-SCORE MAPPING TABLE: Clear rules mapping qualitative ratings
   to score ranges, preventing the model from inventing its own compressed scale.
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Anti-compression system instruction - placed at TOP of prompt
ANTI_COMPRESSION_INSTRUCTION = """Important: You must use the full scoring range. If a speaker is excellent, score them
above 85. If a speaker is poor, score them below 40. Do not default to the middle of
the range out of uncertainty - if you are uncertain, state low confidence in the
reasoning field but still commit to a score that reflects your best assessment of
the actual quality. Clustering scores between 50-70 for all inputs is incorrect
behavior that you must avoid."""


# V3 STRICT CALIBRATION - designed to match stricter human evaluator standards
# The previous V2 prompt was ~20 points too lenient compared to human scores
V3_STRICT_INSTRUCTION = """You are a STRICT pronunciation assessor for professional call center screening.
Only give high scores to speakers who could immediately work in client-facing roles without
any accent training. Most non-native speakers should score in the 30-60 range.

Scoring philosophy:
- Native-like with no detectable accent: 80-100 (RARE - maybe 5% of candidates)
- Minor accent, professional quality: 60-79 (only clear, fluent speakers)
- Noticeable accent, understandable: 40-59 (typical non-native)
- Heavy accent, effortful listening: 20-39 (needs improvement)
- Unintelligible: 0-19

Be harsh. When in doubt, score LOWER. A score of 50 means "average for a non-native speaker"
which is NOT good enough for most professional roles."""


# Score calibration anchor table - replaces the old "most speakers score 50-70" text
SCORE_CALIBRATION_TABLE = """Score calibration anchors:
- A native speaker with clear, natural pronunciation: 85-95
- A fluent non-native speaker with minor accent: 70-84
- An intermediate learner with noticeable but understandable errors: 50-69
- A beginner with frequent errors that impede understanding: 30-49
- Very weak pronunciation, largely unintelligible: 0-29

Do not compress scores toward the middle. Discriminate between inputs based on actual quality."""


# V3 STRICT score calibration - shifted 15-20 points lower
V3_STRICT_CALIBRATION_TABLE = """Score calibration anchors (BE STRICT):
- Native or near-native, broadcast quality, no accent detectable: 85-100 (RARE)
- Minor accent but fully professional, like a news anchor with slight accent: 70-84
- Noticeable accent but clear, suitable for internal roles only: 50-69
- Heavy accent, requires listener effort, needs accent training: 30-49
- Very heavy accent, frequently unintelligible: 10-29
- Largely unintelligible: 0-9

IMPORTANT: Most call center candidates score 30-60. A score of 60 means "decent but not
excellent." Only give 70+ to speakers who could immediately handle VIP client calls.
When uncertain between two scores, always choose the LOWER one."""


# Chain-of-thought dimension evaluation - forces categorical reasoning before numeric output
DIMENSION_EVALUATION_INSTRUCTIONS = """Before outputting a score, you MUST evaluate these four dimensions:

1. Phoneme accuracy - Are individual sounds produced correctly?
   Rate: poor / fair / good / excellent

2. Rhythm and stress - Are word stress and sentence rhythm natural?
   Rate: poor / fair / good / excellent

3. Fluency - Is speech smooth and connected, or halting and segmented?
   Rate: poor / fair / good / excellent

4. Intelligibility - Could a native listener understand without effort?
   Rate: poor / fair / good / excellent

Then map your dimension ratings to a score range:
- All excellent                    -> 85-100
- Mostly good/excellent            -> 70-84
- Mixed fair and good              -> 50-69
- Mostly fair or one poor          -> 30-49
- Any dimension rated poor         -> below 30 (use judgment on severity)

Output your dimension ratings FIRST, then derive the final score from them."""


# V3 STRICT dimension mapping - stricter thresholds
V3_STRICT_DIMENSION_INSTRUCTIONS = """Before outputting a score, evaluate these four dimensions STRICTLY:

1. Phoneme accuracy - Are individual sounds produced correctly?
   - excellent: Native-like, no detectable errors
   - good: Minor errors, 1-2 per sentence max
   - fair: Several errors per sentence but understandable
   - poor: Frequent errors, some sounds unrecognizable

2. Rhythm and stress - Are word stress and sentence rhythm natural?
   - excellent: Sounds like a native speaker
   - good: Natural flow with minor stress errors
   - fair: Noticeable non-native patterns but comprehensible
   - poor: Very choppy, robotic, or wrong stress patterns

3. Fluency - Is speech smooth and connected, or halting and segmented?
   - excellent: Smooth, no hesitations, natural pace
   - good: Minor hesitations, good recovery
   - fair: Noticeable pauses, some filler words
   - poor: Frequent pauses, halting delivery

4. Intelligibility - Could a native listener understand without effort?
   - excellent: Effortless understanding, clear as native
   - good: Easy to understand despite accent
   - fair: Understandable with some listener effort
   - poor: Requires significant concentration or repetition

STRICT SCORING (dimension -> score mapping):
- All excellent                    -> 85-100 (RARE)
- Mostly excellent, one good       -> 75-84
- All good                         -> 65-74
- Mostly good, one fair            -> 55-64
- Mixed fair and good              -> 45-54
- Mostly fair                      -> 35-44
- Any poor dimension               -> 20-34
- Multiple poor dimensions         -> 0-19

BE HARSH. Most non-native speakers should score 30-55."""


# V3 STRICT calibration examples - shifted ~20 points lower to match human evaluator standards
# Same speakers as V2, but with stricter scores
V3_STRICT_CALIBRATION_EXAMPLES = [
    {
        "score": 15,
        "dimension_ratings": {
            "phoneme_accuracy": "poor",
            "rhythm_and_stress": "poor",
            "fluency": "poor",
            "intelligibility": "poor"
        },
        "description": "Very weak speaker. Severe phoneme errors throughout - vowels and consonants frequently substituted or omitted. Speech is largely unintelligible without significant effort. Heavy mother-tongue interference makes most words unrecognizable. Completely flat intonation with no natural English rhythm.",
        "transcription_snippet": "I... wan to... tell you about... my... job... experien... I work... in... [unintelligible]... three year...",
        "reasoning": "All four dimensions poor. Nearly unintelligible. Score in 10-20 range."
    },
    {
        "score": 28,
        "dimension_ratings": {
            "phoneme_accuracy": "poor",
            "rhythm_and_stress": "fair",
            "fluency": "fair",
            "intelligibility": "fair"
        },
        "description": "Weak speaker with heavy errors. Multiple phoneme substitutions per sentence (th->d, v->b, final consonants dropped). Heavy accent makes listening effortful. Some phrases are clear but others require multiple listens. Rhythm is choppy with frequent pauses mid-phrase.",
        "transcription_snippet": "Hello, my name is Maria. I hab work in customer serbice for tree years. I am... uh... berry good at helping de customers wit deir problems.",
        "reasoning": "One poor dimension (phoneme accuracy), others fair. Heavy accent requires effort to understand. Score in 20-35 range."
    },
    {
        "score": 42,
        "dimension_ratings": {
            "phoneme_accuracy": "fair",
            "rhythm_and_stress": "fair",
            "fluency": "fair",
            "intelligibility": "good"
        },
        "description": "Below-average speaker. Consistent systematic errors (word-final consonant clusters simplified, vowel quality issues). Noticeable accent with non-native rhythm. Can be understood but listener must pay attention. Not suitable for high-profile client calls without training.",
        "transcription_snippet": "Good morning. My name is Jun and I have been working in technical support for about five years now. I enjoy helping people solve their computer problems.",
        "reasoning": "Mostly fair ratings with good intelligibility. Understandable but clearly non-native. Typical candidate - score in 35-50 range."
    },
    {
        "score": 55,
        "dimension_ratings": {
            "phoneme_accuracy": "fair",
            "rhythm_and_stress": "good",
            "fluency": "good",
            "intelligibility": "good"
        },
        "description": "Average speaker. Minor but consistent phoneme errors. Generally clear and understandable with good flow. Accent is noticeable but not distracting. Could handle routine internal calls but would benefit from accent coaching for client-facing roles.",
        "transcription_snippet": "Thank you for calling. I understand you're having issues with your account. Let me pull up your information and see how I can help you today.",
        "reasoning": "Mixed fair and good. Decent but not professional quality. Would pass basic screening but not top tier. Score in 50-60 range."
    },
    {
        "score": 72,
        "dimension_ratings": {
            "phoneme_accuracy": "good",
            "rhythm_and_stress": "good",
            "fluency": "good",
            "intelligibility": "excellent"
        },
        "description": "Good speaker with mostly accurate pronunciation. Only occasional minor errors. Clear and easily understood. Natural-sounding rhythm with appropriate stress. Accent is present but mild and professional. Suitable for client-facing roles.",
        "transcription_snippet": "Thank you for the opportunity to interview today. I've spent the last seven years developing my skills in project management, and I'm particularly proud of leading our team through a successful product launch.",
        "reasoning": "Mostly good with excellent intelligibility. Professional quality, would pass client screening. Score in 65-75 range."
    },
    {
        "score": 88,
        "dimension_ratings": {
            "phoneme_accuracy": "excellent",
            "rhythm_and_stress": "excellent",
            "fluency": "excellent",
            "intelligibility": "excellent"
        },
        "description": "Excellent speaker with near-native accuracy. All phonemes produced correctly. Natural connected speech with proper linking and reduction. No detectable accent or only very subtle traces. Broadcast quality - could work in any VIP client-facing role.",
        "transcription_snippet": "I'm excited to discuss how my background in data analytics could contribute to your team's goals. In my current role, I've implemented machine learning models that reduced customer churn by fifteen percent.",
        "reasoning": "All dimensions excellent. Near-native quality, extremely rare. Score in 85-95 range."
    }
]


# Calibration examples - evenly spaced across 20-95, ordered ASCENDING (worst to best)
# This ordering leverages recency bias toward the extremes
CALIBRATION_EXAMPLES = [
    {
        "score": 24,
        "dimension_ratings": {
            "phoneme_accuracy": "poor",
            "rhythm_and_stress": "poor",
            "fluency": "poor",
            "intelligibility": "poor"
        },
        "description": "Very weak speaker. Severe phoneme errors throughout - vowels and consonants frequently substituted or omitted. Speech is largely unintelligible without significant effort. Heavy mother-tongue interference makes most words unrecognizable. Completely flat intonation with no natural English rhythm. A native listener would struggle to understand even simple phrases.",
        "transcription_snippet": "I... wan to... tell you about... my... job... experien... I work... in... [unintelligible]... three year...",
        "reasoning": "All four dimensions rated poor. Severe phoneme errors, no natural rhythm, halting delivery, and very low intelligibility. Score in bottom range."
    },
    {
        "score": 41,
        "dimension_ratings": {
            "phoneme_accuracy": "poor",
            "rhythm_and_stress": "fair",
            "fluency": "fair",
            "intelligibility": "fair"
        },
        "description": "Poor speaker with heavy errors. Multiple phoneme substitutions per sentence (th->d, v->b, final consonants dropped). Heavy accent makes listening effortful. Some phrases are clear but others require multiple listens. Rhythm is choppy with frequent pauses mid-phrase. A patient listener can understand the main message but details are lost.",
        "transcription_snippet": "Hello, my name is Maria. I hab work in customer serbice for tree years. I am... uh... berry good at helping de customers wit deir problems.",
        "reasoning": "Phoneme accuracy is poor (multiple substitutions), but speaker maintains some fluency and is partially intelligible. One poor dimension with others fair places this in 30-49 range."
    },
    {
        "score": 58,
        "dimension_ratings": {
            "phoneme_accuracy": "fair",
            "rhythm_and_stress": "fair",
            "fluency": "good",
            "intelligibility": "good"
        },
        "description": "Average learner with noticeable but understandable errors. Consistent systematic errors (word-final consonant clusters simplified, vowel quality issues) but these don't prevent comprehension. Noticeable accent but speech flows reasonably well. Native listeners can understand without major effort but would clearly identify speaker as non-native.",
        "transcription_snippet": "Good morning. My name is Jun and I have been working in technical support for about five years now. I enjoy helping people solve their computer problems and I think I would be a good fit for this position.",
        "reasoning": "Mixed fair and good ratings across dimensions. Understandable with systematic errors. This is a typical intermediate learner - score in 50-69 range."
    },
    {
        "score": 74,
        "dimension_ratings": {
            "phoneme_accuracy": "good",
            "rhythm_and_stress": "good",
            "fluency": "good",
            "intelligibility": "excellent"
        },
        "description": "Good speaker with mostly accurate pronunciation. Only occasional minor errors (slight vowel coloring, minor stress shifts on low-frequency words). Clear and easily understood. Natural-sounding rhythm with appropriate sentence stress. Accent is present but mild and does not impede communication at all. Suitable for professional communication roles.",
        "transcription_snippet": "Thank you for the opportunity to interview today. I've spent the last seven years developing my skills in project management, and I'm particularly proud of leading our team through a successful product launch last quarter.",
        "reasoning": "Mostly good with excellent intelligibility. Minor accent present but communication is effortless. This is a competent professional speaker - score in 70-84 range."
    },
    {
        "score": 91,
        "dimension_ratings": {
            "phoneme_accuracy": "excellent",
            "rhythm_and_stress": "excellent",
            "fluency": "excellent",
            "intelligibility": "excellent"
        },
        "description": "Excellent speaker with near-native accuracy. All phonemes produced correctly with appropriate allophones. Natural connected speech with proper linking and reduction. Intonation patterns match native speaker expectations. No detectable accent or only very subtle traces that a linguist might notice. Broadcast quality - could work in any client-facing role without accent being noticed.",
        "transcription_snippet": "I'm excited to discuss how my background in data analytics could contribute to your team's goals. In my current role, I've implemented machine learning models that reduced customer churn by fifteen percent, and I'm eager to bring that same analytical approach to the challenges you're facing.",
        "reasoning": "All four dimensions rated excellent. Near-native pronunciation with natural prosody. This represents top-tier performance - score in 85-100 range."
    }
]


# JSON output schema with chain-of-thought fields
OUTPUT_SCHEMA = """{
  "transcription": "full transcription of what was said",
  "dimension_ratings": {
    "phoneme_accuracy": "<poor|fair|good|excellent>",
    "rhythm_and_stress": "<poor|fair|good|excellent>",
    "fluency": "<poor|fair|good|excellent>",
    "intelligibility": "<poor|fair|good|excellent>"
  },
  "scores": {
    "accuracy": <number 0-100>,
    "fluency": <number 0-100>,
    "pronunciation": <number 0-100>,
    "prosody": <number 0-100>
  },
  "score": <number 0-100, the final overall score>,
  "confidence": "<low|medium|high>",
  "reasoning": "<one sentence explaining why this score was assigned>",
  "words": [
    {"word": "problematic_word", "error_type": "description of the error"}
  ]
}"""


def format_example(example: dict, index: int) -> str:
    """Format a single calibration example for the prompt."""
    lines = [
        f"Example {index} - Score: {example['score']}",
        f"Situation: {example['description']}",
        f"Sample speech: \"{example['transcription_snippet']}\"",
        "Dimension ratings:"
    ]

    for dim, rating in example['dimension_ratings'].items():
        dim_display = dim.replace('_', ' ').title()
        lines.append(f"  - {dim_display}: {rating}")

    lines.append(f"Reasoning: {example['reasoning']}")

    return "\n".join(lines)


def build_assessment_prompt(language: str = "en-US") -> str:
    """
    Build the complete pronunciation assessment prompt with anti-compression design.

    Args:
        language: Target language code (e.g., "en-US", "en-GB")

    Returns:
        Complete prompt string ready for use with GPT-4o
    """
    # Sort examples ascending by score (worst to best) - critical for recency bias
    sorted_examples = sorted(CALIBRATION_EXAMPLES, key=lambda x: x["score"])

    prompt_parts = [
        # 1. Anti-compression instruction at the TOP
        ANTI_COMPRESSION_INSTRUCTION,
        "",

        # 2. Role and task description
        f"You are a pronunciation assessor evaluating non-native English speakers.",
        f"The speaker's target language is: {language}",
        "",

        # 3. Score calibration table (replaces narrow range instructions)
        SCORE_CALIBRATION_TABLE,
        "",

        # 4. Dimension evaluation instructions (chain-of-thought)
        DIMENSION_EVALUATION_INSTRUCTIONS,
        "",

        # 5. Calibration examples (ascending order - worst to best)
        "--- CALIBRATION EXAMPLES (study the full range of scores) ---",
        ""
    ]

    for i, example in enumerate(sorted_examples, 1):
        prompt_parts.append(format_example(example, i))
        prompt_parts.append("")

    prompt_parts.extend([
        "--- END EXAMPLES ---",
        "",

        # 6. Output format specification
        "You MUST respond with valid JSON in this exact format:",
        OUTPUT_SCHEMA,
        "",

        # 7. Final reminder
        "CRITICAL: Use the dimension ratings to derive your score. Do not skip the ratings. "
        "Do not compress all scores to 50-70. Match your score to the calibration examples above."
    ])

    return "\n".join(prompt_parts)


def get_system_message() -> str:
    """Get the system message for the OpenAI API call."""
    return (
        "You are a strict pronunciation assessor. "
        "Always respond with valid JSON only, no markdown formatting. "
        "You must evaluate dimension ratings before assigning a score. "
        "Use the full 0-100 range - do not compress scores to the middle."
    )


# For backward compatibility - can be imported and used directly
def get_assessment_prompt_v2(language: str = "en-US") -> str:
    """
    Alias for build_assessment_prompt for backward compatibility.
    """
    return build_assessment_prompt(language)


def build_assessment_prompt_v3(language: str = "en-US") -> str:
    """
    Build V3 STRICT pronunciation assessment prompt.

    This version is recalibrated to match stricter human evaluator standards.
    Mean scores should be around 35-45 instead of 55-65.

    Args:
        language: Target language code (e.g., "en-US", "en-GB")

    Returns:
        Complete prompt string ready for use with GPT-4o
    """
    # Sort examples ascending by score (worst to best)
    sorted_examples = sorted(V3_STRICT_CALIBRATION_EXAMPLES, key=lambda x: x["score"])

    prompt_parts = [
        # 1. Strict instruction at the TOP
        V3_STRICT_INSTRUCTION,
        "",

        # 2. Role and task description
        f"You are evaluating a candidate for a call center position.",
        f"Target language: {language}",
        "",

        # 3. Strict score calibration table
        V3_STRICT_CALIBRATION_TABLE,
        "",

        # 4. Strict dimension evaluation instructions
        V3_STRICT_DIMENSION_INSTRUCTIONS,
        "",

        # 5. Calibration examples (ascending order)
        "--- CALIBRATION EXAMPLES (study the score range carefully) ---",
        ""
    ]

    for i, example in enumerate(sorted_examples, 1):
        prompt_parts.append(format_example(example, i))
        prompt_parts.append("")

    prompt_parts.extend([
        "--- END EXAMPLES ---",
        "",

        # 6. Output format specification
        "You MUST respond with valid JSON in this exact format:",
        OUTPUT_SCHEMA,
        "",

        # 7. Final strict reminder
        "CRITICAL: Be STRICT. Most candidates should score 30-55. "
        "Only exceptional speakers with near-native pronunciation score above 70. "
        "When uncertain, choose the LOWER score."
    ])

    return "\n".join(prompt_parts)


def get_assessment_prompt_v3(language: str = "en-US") -> str:
    """
    Alias for build_assessment_prompt_v3.
    """
    return build_assessment_prompt_v3(language)


# =============================================================================
# V4 PROMPT: 10-POINT DIMENSION SCALES FOR FINER GRANULARITY
# =============================================================================
# The V2/V3 prompts use 4 categories (poor/fair/good/excellent) which causes
# score compression because "fair" covers a 20-point range. V4 uses 1-10 scales
# with explicit anchors at each level to force finer differentiation.

V4_INSTRUCTION = """You are a pronunciation assessor evaluating non-native English speakers for call center positions.

Your task is to rate EACH dimension on a 1-10 scale with specific anchors. This granular rating
prevents score compression and ensures different speakers receive appropriately different scores.

IMPORTANT: The dimensions are rated 1-10, NOT 1-100. After rating all dimensions, you will
calculate a final score by averaging and scaling to 0-100."""


V4_DIMENSION_SCALES = """Rate each dimension on a 1-10 scale using these anchors:

## PHONEME ACCURACY (1-10)
How correctly are individual sounds produced?

1 - Unintelligible: Most phonemes unrecognizable, severe L1 interference
2 - Very Poor: Frequent severe errors, many words unrecognizable
3 - Poor: Multiple errors per sentence, requires significant effort to decode
4 - Weak: Consistent errors (th->d, v->b, etc.), heavy accent
5 - Below Average: Several noticeable errors, clear non-native patterns
6 - Average: Some errors but mostly correct, moderate accent
7 - Above Average: Occasional minor errors, mild accent
8 - Good: Rare errors, only on difficult words, slight accent
9 - Very Good: Near-native accuracy, minimal detectable accent
10 - Excellent: Native-like, no detectable errors

## RHYTHM AND STRESS (1-10)
Are word stress and sentence rhythm natural?

1 - No rhythm: Completely flat, robotic, no stress patterns
2 - Very Poor: Wrong stress on most words, very choppy
3 - Poor: Frequent wrong stress, unnatural pauses
4 - Weak: Often misplaced stress, non-native cadence obvious
5 - Below Average: Some wrong stress, noticeably non-native rhythm
6 - Average: Mostly correct stress, some non-native patterns
7 - Above Average: Generally natural rhythm, occasional issues
8 - Good: Natural flow, minor stress issues on complex words
9 - Very Good: Near-native rhythm and stress patterns
10 - Excellent: Indistinguishable from native speaker

## FLUENCY (1-10)
Is speech smooth and connected or halting?

1 - Extremely Halting: Long pauses, can't form sentences
2 - Very Poor: Frequent long pauses, word-by-word delivery
3 - Poor: Many pauses, frequent filler words (um, uh, like)
4 - Weak: Noticeable hesitations, some false starts
5 - Below Average: Occasional pauses, some filler words
6 - Average: Generally smooth with minor hesitations
7 - Above Average: Smooth delivery, rare hesitations
8 - Good: Fluid speech, natural pauses only
9 - Very Good: Effortless flow, connected speech
10 - Excellent: Perfect fluency, broadcast quality

## INTELLIGIBILITY (1-10)
How easily can a native listener understand?

1 - Unintelligible: Cannot understand even with maximum effort
2 - Very Poor: Only isolated words understood
3 - Poor: Main idea unclear, requires multiple replays
4 - Weak: Understandable with significant effort and concentration
5 - Below Average: Understandable with effort, some parts unclear
6 - Average: Generally clear, occasional listener effort needed
7 - Above Average: Clear with minimal effort despite accent
8 - Good: Easily understood, accent not distracting
9 - Very Good: Effortless understanding, very clear
10 - Excellent: Crystal clear, native-like clarity"""


V4_SCORING_FORMULA = """SCORING CALCULATION:

1. Rate each dimension 1-10 using the anchors above
2. Calculate: final_score = (phoneme + rhythm + fluency + intelligibility) / 4 * 10

Example calculations:
- All 5s (below average): (5+5+5+5)/4*10 = 50
- All 4s (weak): (4+4+4+4)/4*10 = 40
- Mix of 3-5 (poor to below average): (3+4+5+4)/4*10 = 40
- All 7s (above average): (7+7+7+7)/4*10 = 70
- Mix of 6-8 (average to good): (6+7+8+7)/4*10 = 70

DISTRIBUTION EXPECTATIONS:
- Most non-native speakers score 35-55 (dimension ratings of 3-6)
- Good speakers score 55-75 (dimension ratings of 5-8)
- Excellent speakers score 75-90 (dimension ratings of 8-10)
- Perfect 100 is extremely rare (all 10s)"""


V4_CALIBRATION_EXAMPLES = [
    {
        "dimension_scores": {"phoneme": 2, "rhythm": 2, "fluency": 2, "intelligibility": 2},
        "final_score": 20,
        "description": "Very weak speaker. Most words unrecognizable, severe mother-tongue interference. Speech is word-by-word with long pauses. Native listener cannot follow the message.",
        "reasoning": "All dimensions at level 2 (very poor). Calculation: (2+2+2+2)/4*10 = 20"
    },
    {
        "dimension_scores": {"phoneme": 3, "rhythm": 4, "fluency": 3, "intelligibility": 4},
        "final_score": 35,
        "description": "Poor speaker. Multiple phoneme errors per sentence (th->d, dropped consonants). Choppy rhythm with frequent pauses. Understandable with significant effort.",
        "reasoning": "Mix of 3s and 4s (poor to weak). Calculation: (3+4+3+4)/4*10 = 35"
    },
    {
        "dimension_scores": {"phoneme": 5, "rhythm": 4, "fluency": 5, "intelligibility": 5},
        "final_score": 48,
        "description": "Below-average speaker. Noticeable errors and accent but message is clear. Some hesitations and non-native rhythm. Typical call center candidate.",
        "reasoning": "Mostly 5s with one 4 (below average). Calculation: (5+4+5+5)/4*10 = 47.5 ≈ 48"
    },
    {
        "dimension_scores": {"phoneme": 6, "rhythm": 5, "fluency": 6, "intelligibility": 6},
        "final_score": 58,
        "description": "Average speaker. Some errors but mostly correct pronunciation. Generally smooth with minor issues. Clear enough for routine calls.",
        "reasoning": "Mix of 5s and 6s (below average to average). Calculation: (6+5+6+6)/4*10 = 57.5 ≈ 58"
    },
    {
        "dimension_scores": {"phoneme": 7, "rhythm": 7, "fluency": 7, "intelligibility": 8},
        "final_score": 73,
        "description": "Good speaker. Occasional minor errors, mild accent. Natural rhythm and smooth delivery. Easily understood - suitable for client calls.",
        "reasoning": "Mostly 7s with one 8 (above average to good). Calculation: (7+7+7+8)/4*10 = 72.5 ≈ 73"
    },
    {
        "dimension_scores": {"phoneme": 9, "rhythm": 9, "fluency": 9, "intelligibility": 10},
        "final_score": 93,
        "description": "Excellent speaker. Near-native accuracy and rhythm. Effortless fluency. Crystal clear intelligibility. Broadcast quality.",
        "reasoning": "Mostly 9s with one 10 (very good to excellent). Calculation: (9+9+9+10)/4*10 = 92.5 ≈ 93"
    }
]


V4_OUTPUT_SCHEMA = """{
  "transcription": "full transcription of what was said",
  "dimension_scores": {
    "phoneme_accuracy": <1-10>,
    "rhythm_and_stress": <1-10>,
    "fluency": <1-10>,
    "intelligibility": <1-10>
  },
  "score": <0-100, calculated as average of dimensions * 10>,
  "confidence": "<low|medium|high>",
  "reasoning": "<brief explanation of ratings and calculation>",
  "words": [
    {"word": "problematic_word", "error_type": "description"}
  ]
}"""


def format_v4_example(example: dict, index: int) -> str:
    """Format a single V4 calibration example."""
    lines = [
        f"Example {index} - Final Score: {example['final_score']}",
        f"Situation: {example['description']}",
        "Dimension scores (1-10):"
    ]

    scores = example['dimension_scores']
    lines.append(f"  - Phoneme Accuracy: {scores['phoneme']}")
    lines.append(f"  - Rhythm and Stress: {scores['rhythm']}")
    lines.append(f"  - Fluency: {scores['fluency']}")
    lines.append(f"  - Intelligibility: {scores['intelligibility']}")
    lines.append(f"Reasoning: {example['reasoning']}")

    return "\n".join(lines)


def build_assessment_prompt_v4(language: str = "en-US") -> str:
    """
    Build V4 pronunciation assessment prompt with 10-point dimension scales.

    Key innovation: Forces 1-10 ratings per dimension instead of 4 categories,
    providing 10x more granularity and reducing score compression.

    Args:
        language: Target language code (e.g., "en-US", "en-GB")

    Returns:
        Complete prompt string ready for use with GPT-4o
    """
    sorted_examples = sorted(V4_CALIBRATION_EXAMPLES, key=lambda x: x["final_score"])

    prompt_parts = [
        # 1. Instruction
        V4_INSTRUCTION,
        "",
        f"Target language: {language}",
        "",

        # 2. Dimension scales with explicit 1-10 anchors
        V4_DIMENSION_SCALES,
        "",

        # 3. Scoring formula
        V4_SCORING_FORMULA,
        "",

        # 4. Calibration examples
        "--- CALIBRATION EXAMPLES ---",
        ""
    ]

    for i, example in enumerate(sorted_examples, 1):
        prompt_parts.append(format_v4_example(example, i))
        prompt_parts.append("")

    prompt_parts.extend([
        "--- END EXAMPLES ---",
        "",

        # 5. Output format
        "You MUST respond with valid JSON in this exact format:",
        V4_OUTPUT_SCHEMA,
        "",

        # 6. Final reminder
        "CRITICAL: Rate each dimension 1-10 using the specific anchors above. "
        "Do NOT default to middle values (5-6) for all dimensions. "
        "A poor speaker should get 2-4s, an average speaker 5-6s, a good speaker 7-8s. "
        "Calculate final score as: (sum of 4 dimensions) / 4 * 10"
    ])

    return "\n".join(prompt_parts)


def get_assessment_prompt_v4(language: str = "en-US") -> str:
    """Alias for build_assessment_prompt_v4."""
    return build_assessment_prompt_v4(language)


# =============================================================================
# V5 PROMPT: DEDUCTION-BASED SCORING FOR BETTER DIFFERENTIATION
# =============================================================================
# The V2-V4 prompts ask "how good is this?" which leads to clustering toward
# safe middle scores. V5 inverts this: "start at 100, what's wrong?"
# This forces explicit error identification and produces wider score distribution.

V5_INSTRUCTION = """You are a strict pronunciation assessor using DEDUCTION-BASED scoring.

CRITICAL INSTRUCTIONS:
1. Start at 100 points
2. Listen carefully and identify ALL pronunciation issues
3. Apply mandatory deductions for each error found
4. Final score = 100 + total_deductions
5. Be thorough and use the FULL SCORING RANGE

MANDATORY SCORE DIFFERENTIATION:
- A native speaker with perfect pronunciation = score 95-100 (rare)
- A fluent non-native with minimal errors = score 80-90
- A speaker with noticeable but understandable errors = score 50-65
- A speaker with heavy accent requiring effort = score 30-45
- A speaker who is difficult to understand = score 10-25

DO NOT CLUSTER ALL SPEAKERS IN 40-60 RANGE. Listen carefully to distinguish quality levels.

You MUST check EVERY category below and list specific errors found."""


V5_DEDUCTION_RUBRIC = """DEDUCTION CATEGORIES (check ALL categories):

## 1. PHONEME ERRORS (max deduction: -25)
Listen for consonant and vowel pronunciation errors.

Deductions:
- Consonant substitution (th->d, v->b, r->l, f->p, etc.): -2 each
- Consonant omission (dropped final consonants): -2 each
- Vowel quality error (wrong vowel sound): -2 each
- Consonant cluster simplification: -2 each

Count errors. Poor speakers have 8-12+ errors. Good speakers have 0-3.

## 2. WORD STRESS ERRORS (max deduction: -10)
Check if syllables are stressed correctly.

Deductions:
- Wrong syllable stressed: -3 each

Poor speakers stress words incorrectly throughout. Good speakers rarely err.

## 3. FLUENCY ISSUES (max deduction: -15)
Listen for hesitations, fillers, and pauses.

Deductions:
- Filler words (um, uh, like, you know): -1 each (max -5)
- Unnatural pause mid-phrase: -2 each
- Word-by-word delivery (no connected speech): -8
- Frequent self-corrections: -3

Poor speakers are halting and choppy. Excellent speakers flow naturally.

## 4. PROSODY/RHYTHM ISSUES (max deduction: -15)
Check intonation and rhythm patterns.

Deductions:
- Monotone delivery (no pitch variation): -10
- Choppy/robotic delivery: -8
- Unnatural rhythm/timing: -5
- Wrong sentence intonation: -3

Poor speakers sound robotic. Excellent speakers have natural melody.

## 5. INTELLIGIBILITY IMPACT (max deduction: -20)
How much effort does a native listener need?

Deductions (choose ONE):
- Crystal clear, effortless understanding: -0
- Easily understood with minimal accent: -3
- Clear but noticeable accent: -6
- Understandable but requires some attention: -10
- Understandable with effort, some unclear parts: -14
- Frequently unclear, requires concentration: -17
- Very difficult to understand: -20

## 6. OVERALL ACCENT SEVERITY (max deduction: -15)
How strong is the non-native accent?

Deductions (choose ONE):
- Native or near-native: -0
- Very slight accent: -3
- Noticeable but mild accent: -6
- Moderate accent, clearly non-native: -9
- Heavy accent affecting clarity: -12
- Very heavy accent, significant L1 interference: -15

SCORING RULES:
- Apply deductions from EACH category
- Sum all deductions
- Raw score = 100 + total_deductions
- ROUND to nearest 5: Final score must be one of: 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100
- FLOOR: 10 (minimum possible score)
- CEILING: 100

FINAL SANITY CHECK (MANDATORY):
After calculating the score, verify it matches the speaker's OVERALL quality:
- If the speaker is HARD TO UNDERSTAND, score MUST be 10-30 (add more deductions if needed)
- If the speaker SOUNDS NATIVE, score MUST be 85+ (reduce deductions if needed)
- If all speakers are getting 40-60, you are not differentiating enough

CALIBRATION GUIDE:
- Very heavy accent + many errors + hard to understand = score 10-25
- Heavy accent + several errors + requires effort = score 30-45
- Moderate accent + some errors + understandable = score 50-60
- Light accent + minor errors + clear = score 65-80
- Near-native + rare errors + excellent = score 85-100

EXPECTED DEDUCTION TOTALS:
- Very poor (10-25): -75 to -90+ deductions
- Poor (30-45): -55 to -75 deductions
- Average (50-60): -40 to -55 deductions
- Good (65-80): -20 to -40 deductions
- Excellent (85-100): 0 to -15 deductions"""


V5_CALIBRATION_EXAMPLES = [
    {
        "final_score": 10,
        "deductions": {
            "phoneme_errors": {"items": ["th->d (x6)", "v->b (x3)", "dropped consonants (x4)"], "total": -25},
            "stress_errors": {"items": ["wrong stress (x3)"], "total": -9},
            "fluency": {"items": ["word-by-word delivery", "many fillers"], "total": -13},
            "prosody": {"items": ["completely monotone", "robotic"], "total": -15},
            "intelligibility": {"level": "very difficult to understand", "total": -20},
            "accent": {"level": "very heavy L1 interference", "total": -15}
        },
        "total_deductions": -97,
        "raw_score": 3,
        "description": "Very poor speaker. Heavy L1 interference. Word-by-word delivery. Native listener cannot follow most of the speech."
    },
    {
        "final_score": 20,
        "deductions": {
            "phoneme_errors": {"items": ["th->d (x5)", "v->b (x2)", "dropped consonants (x3)"], "total": -20},
            "stress_errors": {"items": ["wrong stress (x2)"], "total": -6},
            "fluency": {"items": ["very choppy", "fillers (x5)"], "total": -12},
            "prosody": {"items": ["monotone delivery"], "total": -10},
            "intelligibility": {"level": "frequently unclear", "total": -17},
            "accent": {"level": "very heavy accent", "total": -15}
        },
        "total_deductions": -80,
        "raw_score": 20,
        "description": "Poor speaker. Many phoneme errors. Heavy accent makes understanding difficult. Choppy delivery."
    },
    {
        "final_score": 35,
        "deductions": {
            "phoneme_errors": {"items": ["th->d (x4)", "vowel errors (x3)"], "total": -14},
            "stress_errors": {"items": ["wrong stress (x2)"], "total": -6},
            "fluency": {"items": ["fillers (x4)", "pauses (x2)"], "total": -8},
            "prosody": {"items": ["choppy rhythm"], "total": -8},
            "intelligibility": {"level": "understandable with effort", "total": -14},
            "accent": {"level": "heavy accent", "total": -12}
        },
        "total_deductions": -62,
        "raw_score": 38,
        "description": "Below-average speaker. Multiple errors. Heavy accent. Understandable but requires effort."
    },
    {
        "final_score": 50,
        "deductions": {
            "phoneme_errors": {"items": ["th->d (x3)", "vowel errors (x2)"], "total": -10},
            "stress_errors": {"items": ["wrong stress (x1)"], "total": -3},
            "fluency": {"items": ["fillers (x3)", "pause (x1)"], "total": -5},
            "prosody": {"items": ["slightly unnatural"], "total": -5},
            "intelligibility": {"level": "understandable with some attention", "total": -10},
            "accent": {"level": "moderate accent", "total": -9}
        },
        "total_deductions": -42,
        "raw_score": 58,
        "description": "Average speaker. Noticeable errors and accent. Understandable but clearly non-native."
    },
    {
        "final_score": 65,
        "deductions": {
            "phoneme_errors": {"items": ["th->d (x1)", "vowel errors (x1)"], "total": -4},
            "stress_errors": {"items": ["none"], "total": 0},
            "fluency": {"items": ["fillers (x2)"], "total": -2},
            "prosody": {"items": ["minor rhythm variation"], "total": -3},
            "intelligibility": {"level": "clear but noticeable accent", "total": -6},
            "accent": {"level": "noticeable but mild", "total": -6}
        },
        "total_deductions": -21,
        "raw_score": 79,
        "description": "Above average speaker. Few errors. Mild accent. Clear communication."
    },
    {
        "final_score": 80,
        "deductions": {
            "phoneme_errors": {"items": ["minor vowel coloring (x1)"], "total": -2},
            "stress_errors": {"items": ["none"], "total": 0},
            "fluency": {"items": ["occasional filler (x1)"], "total": -1},
            "prosody": {"items": ["natural"], "total": 0},
            "intelligibility": {"level": "easily understood with minimal accent", "total": -3},
            "accent": {"level": "very slight accent", "total": -3}
        },
        "total_deductions": -9,
        "raw_score": 91,
        "description": "Good speaker. Rare minor errors. Very slight accent. Professional communication."
    },
    {
        "final_score": 100,
        "deductions": {
            "phoneme_errors": {"items": ["none detected"], "total": 0},
            "stress_errors": {"items": ["none"], "total": 0},
            "fluency": {"items": ["perfect flow"], "total": 0},
            "prosody": {"items": ["natural native-like intonation"], "total": 0},
            "intelligibility": {"level": "crystal clear", "total": 0},
            "accent": {"level": "native or near-native", "total": 0}
        },
        "total_deductions": 0,
        "raw_score": 100,
        "description": "Excellent speaker. Native or near-native. No detectable issues. Broadcast quality."
    }
]


V5_OUTPUT_SCHEMA = """{
    "transcription": "full transcription of what was said",
    "deductions": {
        "phoneme_errors": {
            "items": ["specific error descriptions"],
            "total": <negative number or 0>
        },
        "stress_errors": {
            "items": ["specific errors"],
            "total": <negative number or 0>
        },
        "fluency": {
            "items": ["specific issues"],
            "total": <negative number or 0>
        },
        "prosody": {
            "items": ["specific issues"],
            "total": <negative number or 0>
        },
        "intelligibility": {
            "level": "description of effort needed",
            "total": <negative number or 0>
        },
        "accent": {
            "level": "severity description",
            "total": <negative number or 0>
        }
    },
    "total_deductions": <sum of all category totals>,
    "raw_score": <100 + total_deductions>,
    "final_score": <MUST be one of: 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95>,
    "confidence": "low|medium|high",
    "summary": "one sentence explaining the main issues"
}"""


def format_v5_example(example: dict, index: int) -> str:
    """Format a single V5 calibration example."""
    lines = [
        f"Example {index} - Final Score: {example['final_score']}",
        f"Situation: {example['description']}",
        "Deductions applied:"
    ]

    for category, data in example['deductions'].items():
        cat_name = category.replace('_', ' ').title()
        if isinstance(data, dict):
            items = data.get('items', [])
            total = data.get('total', 0)
            level = data.get('level', '')
            if level:
                lines.append(f"  - {cat_name}: {level} ({total})")
            elif items:
                lines.append(f"  - {cat_name}: {', '.join(items[:3])}... ({total})")
            else:
                lines.append(f"  - {cat_name}: none ({total})")

    lines.append(f"Total deductions: {example['total_deductions']}")
    lines.append(f"Raw score: 100 + ({example['total_deductions']}) = {example['raw_score']}")
    lines.append(f"Final score (after floor/ceiling): {example['final_score']}")

    return "\n".join(lines)


def build_assessment_prompt_v5(language: str = "en-US") -> str:
    """
    Build V5 deduction-based pronunciation assessment prompt.

    Key innovation: Inverts the question from "how good is this?" to
    "start at 100, what's wrong?" This forces explicit error identification
    and naturally produces wider score distribution.

    Args:
        language: Target language code (e.g., "en-US", "en-GB")

    Returns:
        Complete prompt string ready for use with GPT-4o
    """
    sorted_examples = sorted(V5_CALIBRATION_EXAMPLES, key=lambda x: x["final_score"])

    prompt_parts = [
        # 1. Core instruction
        V5_INSTRUCTION,
        "",
        f"Target language: {language}",
        "",

        # 2. Deduction rubric
        V5_DEDUCTION_RUBRIC,
        "",

        # 3. Calibration examples
        "--- CALIBRATION EXAMPLES (study the deduction patterns) ---",
        ""
    ]

    for i, example in enumerate(sorted_examples, 1):
        prompt_parts.append(format_v5_example(example, i))
        prompt_parts.append("")

    prompt_parts.extend([
        "--- END EXAMPLES ---",
        "",

        # 4. Output format
        "You MUST respond with valid JSON in this exact format:",
        V5_OUTPUT_SCHEMA,
        "",

        # 5. Final reminder
        "CRITICAL REMINDERS:",
        "1. Check EVERY deduction category - do not skip any",
        "2. List SPECIFIC errors found (not generic descriptions)",
        "3. Calculate raw score: 100 + total_deductions",
        "4. ROUND final_score to nearest 5: Must be 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, or 95",
        "5. Score ranges: Poor 15-35, Below-average 40-50, Average 55-65, Good 70-80, Excellent 85-95"
    ])

    return "\n".join(prompt_parts)


def get_assessment_prompt_v5(language: str = "en-US") -> str:
    """Alias for build_assessment_prompt_v5."""
    return build_assessment_prompt_v5(language)


def get_v5_system_message() -> str:
    """Get the system message for V5 deduction-based assessment."""
    return (
        "You are a strict pronunciation assessor using deduction-based scoring. "
        "Start at 100 and apply mandatory deductions for each error category. "
        "Always respond with valid JSON only, no markdown formatting. "
        "Be thorough in finding errors - missing errors leads to score inflation."
    )


# =============================================================================
# V6 PROMPT: TWO-STAGE CATEGORIZATION-THEN-SCORING
# =============================================================================
# V5 still clusters scores because the model applies similar deductions to all.
# V6 forces explicit categorization FIRST, then scores within the category.

V6_INSTRUCTION = """You are a pronunciation assessor. Your task is to evaluate the speaker in TWO steps:

STEP 1: CATEGORIZE THE SPEAKER
Listen to the audio and categorize the speaker into ONE of these 5 categories:

Category A - EXCELLENT (score 85-100):
- Sounds native or near-native
- No accent or barely detectable accent
- Perfect or near-perfect pronunciation
- Natural fluency with no hesitations
- Would be mistaken for a native speaker

Category B - GOOD (score 70-84):
- Very clear pronunciation with rare minor errors
- Slight accent that doesn't impede communication
- Natural speech flow
- Suitable for professional client-facing roles

Category C - AVERAGE (score 50-69):
- Understandable but clearly non-native
- Noticeable accent and some pronunciation errors
- Generally smooth but may have occasional hesitations
- Typical educated non-native speaker

Category D - BELOW AVERAGE (score 30-49):
- Requires listener effort to understand
- Heavy accent with frequent errors
- Choppy or halting speech
- Would need training for professional roles

Category E - POOR (score 10-29):
- Very difficult to understand
- Very heavy accent
- Many pronunciation and fluency errors
- Significant communication barrier

STEP 2: SCORE WITHIN THE CATEGORY
After categorizing, assign a specific score WITHIN that category's range.

IMPORTANT: You must FIRST decide the category, THEN the score. Do NOT skip to scoring.
Most non-native speakers fall in categories C, D, or E. Categories A and B are rare."""


V6_OUTPUT_SCHEMA = """{
    "transcription": "full transcription of what was said",
    "category": "A|B|C|D|E",
    "category_justification": "why this category was chosen (1-2 sentences)",
    "score_within_category": "explanation of where within the range (low/mid/high)",
    "final_score": <number from the category's range, rounded to nearest 5>,
    "summary": "one sentence overall assessment"
}"""


def build_assessment_prompt_v6(language: str = "en-US") -> str:
    """
    Build V6 categorization-then-scoring prompt.

    Key innovation: Forces model to categorize speaker quality FIRST,
    then score within that category. This prevents all scores clustering
    in the middle.
    """
    return f"""{V6_INSTRUCTION}

Target language: {language}

SCORING GUIDE BY CATEGORY:
- Category A (EXCELLENT): 85, 90, 95, or 100
- Category B (GOOD): 70, 75, 80, or 85
- Category C (AVERAGE): 50, 55, 60, or 65
- Category D (BELOW AVERAGE): 30, 35, 40, or 45
- Category E (POOR): 10, 15, 20, or 25

EXAMPLES:

Example 1 - Category E (POOR), Score: 15
Audio characteristics: Word-by-word delivery, many th->d substitutions, dropped consonants,
monotone, very heavy accent. Native listener struggles to follow.
Category justification: Very difficult to understand, significant communication barrier.

Example 2 - Category D (BELOW AVERAGE), Score: 35
Audio characteristics: Heavy accent, multiple phoneme errors, choppy delivery with pauses,
understandable with concentration.
Category justification: Requires effort to understand, would need training.

Example 3 - Category C (AVERAGE), Score: 55
Audio characteristics: Moderate accent, some phoneme and stress errors, generally smooth
delivery, understandable.
Category justification: Clearly non-native but understandable.

Example 4 - Category B (GOOD), Score: 75
Audio characteristics: Slight accent, rare minor errors, natural flow, easily understood.
Category justification: Clear speech suitable for professional roles.

Example 5 - Category A (EXCELLENT), Score: 90
Audio characteristics: Virtually no accent, perfect pronunciation, natural native-like flow.
Category justification: Would be mistaken for a native speaker.

You MUST respond with valid JSON in this exact format:
{V6_OUTPUT_SCHEMA}

CRITICAL: First determine the category based on overall quality, then score within that range.
Do not default to Category C for everyone. Listen carefully to differentiate."""


def get_v6_system_message() -> str:
    """Get the system message for V6 categorization-based assessment."""
    return (
        "You are a pronunciation assessor. First categorize the speaker into A/B/C/D/E, "
        "then score within that category. Most speakers are C, D, or E. "
        "Categories A and B are rare. Respond with valid JSON only."
    )


# =============================================================================
# V7 PROMPT: BINARY QUESTIONS APPROACH
# =============================================================================
# V5 and V6 both fail because the model defaults to middle ranges.
# V7 uses a series of binary YES/NO questions to force explicit decisions.

V7_INSTRUCTION = """You are evaluating English pronunciation. Answer each question below with YES or NO, then use your answers to determine the score.

MANDATORY QUESTIONS (answer ALL):

Q1. INTELLIGIBILITY TEST
Can you understand at least 90% of what the speaker says without re-listening?
- YES = intelligible | NO = hard to understand

Q2. NATIVE LISTENER EFFORT
Would a native speaker understand this WITHOUT any concentration or effort?
- YES = effortless | NO = requires effort

Q3. ACCENT SEVERITY
Does the speaker have a STRONG accent that immediately stands out?
- YES = strong accent | NO = mild or no accent

Q4. PHONEME ERRORS
Do you hear FREQUENT consonant/vowel errors (th->d, v->b, wrong vowels)?
- YES = many errors | NO = few/no errors

Q5. FLUENCY
Is the speech smooth and connected, or choppy/halting?
- SMOOTH = good fluency | CHOPPY = poor fluency

Q6. OVERALL: Could this person work in a US call center without accent training?
- YES = professional ready | NO = needs training

SCORING BASED ON ANSWERS:

If Q1=NO (unintelligible): Score 10-25
If Q1=YES but Q2=NO and Q3=YES and Q4=YES: Score 30-45
If Q1=YES and Q2=mixed and Q3=mixed: Score 50-65
If Q1=YES and Q2=YES and Q3=NO and Q6=YES: Score 70-85
If all answers positive (near-native): Score 85-100

BE STRICT: Most non-native speakers score 30-60. Only give 70+ to clearly professional speakers.
Only give 85+ to speakers who sound native or near-native.
Give 10-25 to speakers you struggle to understand."""


V7_OUTPUT_SCHEMA = """{
    "transcription": "what the speaker said",
    "answers": {
        "q1_intelligible": "YES|NO",
        "q2_effortless": "YES|NO",
        "q3_strong_accent": "YES|NO",
        "q4_many_errors": "YES|NO",
        "q5_fluency": "SMOOTH|CHOPPY",
        "q6_professional_ready": "YES|NO"
    },
    "score_reasoning": "brief explanation based on answers",
    "final_score": <score rounded to nearest 5>
}"""


def build_assessment_prompt_v7(language: str = "en-US") -> str:
    """Build V7 binary questions prompt."""
    return f"""{V7_INSTRUCTION}

Target language: {language}

SCORING EXAMPLES:

Example 1 - Score: 15
Answers: Q1=NO, Q2=NO, Q3=YES, Q4=YES, Q5=CHOPPY, Q6=NO
Reasoning: Cannot understand most of what was said. Heavy accent and errors.

Example 2 - Score: 35
Answers: Q1=YES, Q2=NO, Q3=YES, Q4=YES, Q5=CHOPPY, Q6=NO
Reasoning: Understandable but requires effort. Heavy accent and frequent errors.

Example 3 - Score: 55
Answers: Q1=YES, Q2=NO, Q3=YES, Q4=NO, Q5=SMOOTH, Q6=NO
Reasoning: Understandable, moderate accent, few errors, smooth delivery.

Example 4 - Score: 75
Answers: Q1=YES, Q2=YES, Q3=NO, Q4=NO, Q5=SMOOTH, Q6=YES
Reasoning: Easily understood, mild accent, rare errors, professional quality.

Example 5 - Score: 95
Answers: Q1=YES, Q2=YES, Q3=NO, Q4=NO, Q5=SMOOTH, Q6=YES
Reasoning: Native-like pronunciation, no detectable accent.

Respond with valid JSON only:
{V7_OUTPUT_SCHEMA}"""


def get_v7_system_message() -> str:
    """Get the system message for V7 binary questions assessment."""
    return (
        "You are evaluating English pronunciation. Answer YES/NO questions honestly, "
        "then score based on your answers. Be strict. Respond with valid JSON only."
    )


if __name__ == "__main__":
    # Print the prompt for inspection
    prompt = build_assessment_prompt("en-US")
    print("=" * 80)
    print("GENERATED PROMPT")
    print("=" * 80)
    print(prompt)
    print("=" * 80)
    print(f"\nPrompt length: {len(prompt)} characters")

    # Verify calibration examples
    scores = [ex["score"] for ex in CALIBRATION_EXAMPLES]
    print(f"Example scores: {sorted(scores)}")
    print(f"Score range: {min(scores)} to {max(scores)}")
    print(f"Has score >= 85: {any(s >= 85 for s in scores)}")
    print(f"Has score <= 30: {any(s <= 30 for s in scores)}")
