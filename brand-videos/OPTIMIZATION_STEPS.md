# Optimization Steps for `test.py`

We're going to optimize this script **iteratively**. One change at a time, measured against a baseline.

The mindset: **prove each change is safe and beneficial before moving on.** If a step shows no improvement, or breaks output, we revert and rethink.

---

## Step 0 — Lock down a baseline

Before changing anything, we need to know what "correct and current speed" looks like.

### 0.1 — Pick a fixed test slice

Open `test.py`. Find this block around **line 371**:

```python
    # Build list of remaining work items
    pending = []
    for idx, row in df.iterrows():
        if idx < start_idx:
            continue
        pending.append((idx, row))
```

**Add one line** right after `pending.append(...)`'s loop ends:

```python
    pending = pending[:20]   # TEMP: baseline slice — remove before full run
```

Why 20: large enough to be representative, small enough to iterate fast.

### 0.2 — Add wall-clock timing

Find the batch loop around **line 449**:

```python
    # Process in batches of `concurrency` to preserve order + allow checkpoints
    for batch_start in range(0, len(pending), concurrency):
```

**Just above that line**, add:

```python
    t_start = time.time()
```

Then find the end of `run()` around **line 498**, just before `print(f"\nResults written to ...")`:

```python
    _remove_cache(output)  # clean up cache after successful write
    print(f"\nResults written to {output + output_file}")
```

**Add two lines** just before the final `print`:

```python
    elapsed = time.time() - t_start
    print(f"\n=== {len(rows)} rows in {elapsed:.1f}s "
          f"({len(rows)/elapsed:.2f} rows/sec) ===")
    print(f"  latency p50={out_df['latency_s'].median():.2f}s "
          f"p95={out_df['latency_s'].quantile(0.95):.2f}s "
          f"max={out_df['latency_s'].max():.2f}s")
```

### 0.3 — Run the baseline

```bash
python test.py
```

Then rename the output file so future runs don't overwrite it:

```bash
# Windows:
move "C:\...\Sean_data_cert7.xlsx" "C:\...\baseline.xlsx"
```

### 0.4 — Expected output (right direction)

You should see something like:

```
=== 20 rows in 62.4s (0.32 rows/sec) ===
  latency p50=3.10s p95=4.20s max=5.50s
```

Numbers will vary, but the **shape** should be:
- `rows/sec` will be low (under 1.0) — that's expected for serial calls.
- `p50` ≈ `elapsed / 20` roughly — confirms serial execution.

**Write these numbers down. This is your reference point.**

If you see errors in the output Excel's `error` column, fix those first — optimization can't proceed on a broken baseline.

---

## Step 1 — Add `requests.Session` (connection reuse)

This is the smallest possible change. It reuses the underlying TCP/TLS connection across requests instead of opening a new one each time. **No concurrency change yet** — we isolate one variable at a time.

### 1.1 — Make the change

Near the top of `test.py`, after the imports (around **line 46**):

```python
import pandas as pd
import requests
```

**Add this line below**:

```python
SESSION = requests.Session()
```

Then find `call_intent_service` around **line 148**:

```python
    t0 = time.time()
    try:
        resp = requests.post(url, json=body, stream=True, timeout=timeout)
```

**Change `requests.post` to `SESSION.post`**:

```python
    t0 = time.time()
    try:
        resp = SESSION.post(url, json=body, stream=True, timeout=timeout)
```

That's it. One added line, one changed word.

### 1.2 — Run it

```bash
python test.py
```

Rename the output:

```bash
move "C:\...\Sean_data_cert7.xlsx" "C:\...\step1_session.xlsx"
```

### 1.3 — Diff against baseline

Create a small diff script — save as `brand-videos/diff_outputs.py`:

```python
import sys
import pandas as pd

a = pd.read_excel(sys.argv[1]).sort_values("query").reset_index(drop=True)
b = pd.read_excel(sys.argv[2]).sort_values("query").reset_index(drop=True)

stable_cols = ["query", "actual_intent", "granular_intent",
               "citation_count", "jurisdictions", "resolve_jurisdictions"]
stable_cols = [c for c in stable_cols if c in a.columns and c in b.columns]

diff = a[stable_cols].compare(b[stable_cols])
print(f"Rows compared: {len(a)}")
print(f"Differing rows on stable columns: {len(diff)}")
if len(diff):
    print(diff)
else:
    print("OUTPUTS MATCH ✓")
```

Run it:

```bash
python brand-videos/diff_outputs.py baseline.xlsx step1_session.xlsx
```

### 1.4 — Expected output (right direction)

**Diff output:**
```
Rows compared: 20
Differing rows on stable columns: 0
OUTPUTS MATCH ✓
```

If `actual_intent` differs anywhere → **stop.** The Session shouldn't change classification results. Something else is wrong (maybe the service itself is nondeterministic — investigate before continuing).

**Speed output (in the script's own print):**
```
=== 20 rows in 56.8s (0.35 rows/sec) ===
  latency p50=2.85s p95=4.00s max=5.30s
```

What you're looking for:
- `elapsed` is slightly lower than baseline (~5-15% improvement typical)
- `p50` is slightly lower (~50-200ms per request saved on TCP/TLS reuse)
- `rows/sec` is slightly higher

**Don't expect a huge win here.** The Session change alone is modest. Its real value is **setting up the foundation** for Step 2 (concurrency) — without a Session, parallel requests would each open their own connection and overwhelm the connection pool.

### 1.5 — Decision point

| Result | Action |
|---|---|
| Diff clean + faster | ✅ Keep change, move to Step 2 |
| Diff clean + same speed | ✅ Keep change (zero downside), move to Step 2 |
| Diff clean + slower | ⚠️ Unusual — re-run 2x to rule out noise. If consistent, investigate. |
| Diff has differences | 🛑 Revert and investigate before proceeding |

---

## What's next (preview, don't do yet)

- **Step 2:** Bump `concurrency` from 1 to 5. Measure. This is where the real speedup lives.
- **Step 3:** Bump to 10, then 20. Find the knee of the curve.
- **Step 4:** Fix the `_remove_cache` / `_load_cache` ordering bug at line 363-364.
- **Step 5:** Remove the `print(body)` noise at line 397.
- **Step 6:** Remove the `pending[:20]` slice and run on full data.

Each step gets the same treatment: change → run → diff → measure → decide.

---

## Rules for the whole process

1. **One change per step.** If Step 2 breaks output, you know exactly what broke it.
2. **Diff every time.** Speed without correctness is worthless.
3. **Keep the baseline file.** Every step diffs against the original baseline, not the previous step — that way drift can't accumulate silently.
4. **Write down the numbers.** A table of `(step, elapsed, p50, errors)` makes the wins (and regressions) obvious.

When you've finished Step 1 and the diff is clean, tell me and I'll write Step 2.
