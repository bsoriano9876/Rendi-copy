# LinkedIn Post: AI for Pronunciation Assessment

---

**What building an AI pronunciation scorer taught me about prompt engineering**

I spent weeks trying to get GPT-4o to score candidate pronunciation consistently. Here's what actually worked—and what completely failed.

**The problem:**

We needed to screen call center candidates at scale. Score their English pronunciation from 0-100. Simple, right?

First attempt: basic prompt. "Score this audio for pronunciation accuracy."

Result? Every candidate scored 75-85. A PhD and a struggling speaker got nearly identical scores. Useless.

**What DIDN'T work:**

❌ **Telling the model to "be strict"** — It just shifted all scores down by 10 points. Same compression, different range.

❌ **Detailed rubrics** — "Deduct 5 points for mispronunciation" sounds logical. The model ignored it and kept defaulting to safe middle scores.

❌ **Asking for justification first** — I thought reasoning would help. Instead, the model would write generous explanations, then feel committed to high scores.

❌ **Single-shot examples** — One "this is a 60" example wasn't enough. The model would anchor everything relative to that one sample.

**What WORKED:**

✅ **Evenly-spaced calibration examples** — Not just one example. Five examples spanning the FULL range: scores of 24, 41, 58, 74, and 91. This broke the compression.

✅ **Chain-of-thought with categories BEFORE numbers** — Force the model to rate each dimension as "poor/fair/good/excellent" BEFORE giving numeric scores. This creates cognitive commitment.

✅ **Explicit anti-compression instructions** — Literally telling the model: "Do NOT cluster scores in 70-85. Different speakers have different abilities. Your scores MUST reflect this."

✅ **Dimension-to-score mapping tables** — Clear rules like "Poor = 20-40, Fair = 40-60" gave the model permission to use low scores.

✅ **Ascending order examples** — Worst-to-best ordering leverages recency bias. The model sees the high-quality example last and calibrates better.

**The result:**

Correlation with human scores jumped from 0.27 to 0.53. Score range expanded from 15 points to 50+ points. Finally, differentiation.

**The meta-lesson:**

Prompt engineering isn't about cleverness. It's about understanding model biases and designing around them.

LLMs want to be safe. They hedge. They compress. They avoid extremes.

Your job is to give them permission—and scaffolding—to be precise.

---

What prompt engineering patterns have you discovered? I'd love to hear what's worked in your domain.

#AI #PromptEngineering #LLM #GPT4 #MachineLearning #BuildingInPublic #HRTech

---
