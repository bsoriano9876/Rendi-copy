"""
End-to-end evaluation of the deployed intent service.

Reads test cases from an Excel file, calls the deployed intent service
/predict endpoint for each row, and writes the full intent response
(intent classification, jurisdictions, citation validation, post-processing
overrides) to an output Excel file.

This exercises the FULL intent chain:
    LLM intent detection
    + query normalization
    + jurisdiction extraction & resolution
    + citation validation
    + all post-processing override rules

Excel input columns (required):
    query              - The user query to classify

Excel input columns (optional):
    expected_intent    - Expected intent for accuracy comparison
    chat_history       - JSON string of chat history array (default: [])
    initial_intent     - Pre-specified intent override (default: "")
    intent_input_type  - Business type hint (default: "")
    data_source        - JSON string of data source array (default: [])
    metadata           - JSON string of metadata dict (default: {})

Usage:
    python eval_intent_e2e.py \
        --input eval_cases.xlsx \
        --output eval_e2e_results.xlsx \
        --host https://<intent-service-host> \
        --asset-id 4160
"""

import argparse
import ast
import json
import pickle
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests


# ----------------------------------------------------------------------
# SSE parsing
# ----------------------------------------------------------------------

def parse_sse_stream(response: requests.Response) -> list[dict]:
    """Parse a Server-Sent Events stream into a list of JSON payloads."""
    events = []
    buffer = ""
    try:
        for line in response.iter_lines(decode_unicode=True):
            if line is None:
                continue
            if line.startswith("data:"):
                data_str = line[len("data:"):].strip()
                if data_str:
                    buffer = data_str
            elif line == "" and buffer:
                try:
                    events.append(json.loads(buffer))
                except json.JSONDecodeError as e:
                    print(f"    [WARN] SSE JSON decode error: {e} | buffer: {buffer[:200]}")
                buffer = ""
        if buffer:
            try:
                events.append(json.loads(buffer))
            except json.JSONDecodeError as e:
                print(f"    [WARN] SSE trailing JSON decode error: {e} | buffer: {buffer[:200]}")
    except Exception as e:
        print(f"    [ERROR] SSE stream read failed: {type(e).__name__}: {e}")
    return events


def extract_fields(events: list[dict]) -> dict:
    """Walk SSE events and pull out every field from the IntentResp."""
    result = {
        "intent_result": "",
        "granular_intent": "",
        "contextual_query": "",
        "normalized_query": "",
        "answer": "",
        "citation_count": 0,
        "citation_validation": "",
        "jurisdictions": "",
        "resolve_jurisdictions": "",
        "error": "",
    }

    for event in events:
        finished = event.get("finished", False)
        content = event.get("content", {})

        if event.get("failed"):
            result["error"] = str(content)[:500]
            continue

        if not isinstance(content, dict):
            continue

        msg_type = str(content.get("msg_type", ""))
        data = content.get("data", {})

        if "INTENT_RESULT" in msg_type and isinstance(data, dict):
            result["intent_result"] = data.get("intent_result", result["intent_result"])
            result["granular_intent"] = data.get("granular_intent", result["granular_intent"])

        if finished:
            result["intent_result"] = content.get("intent_result", result["intent_result"])
            result["granular_intent"] = content.get("granular_intent", result["granular_intent"])
            result["contextual_query"] = content.get("contextual_query", "")
            result["normalized_query"] = content.get("normalized_query", "")
            result["answer"] = content.get("answer", "")

            citations = content.get("citations", [])
            result["citation_count"] = len(citations) if isinstance(citations, list) else 0

            cv = content.get("citation_validation", {})
            result["citation_validation"] = cv.get("citation_validation", "") if isinstance(cv, dict) else ""

            qj = content.get("query_jurisdiction") or {}
            result["jurisdictions"] = ", ".join(qj.get("jurisdictions", [])) if isinstance(qj, dict) else ""

            rj = content.get("resolve_jurisdiction") or {}
            result["resolve_jurisdictions"] = ", ".join(rj.get("jurisdictions", [])) if isinstance(rj, dict) else ""

    return result


# ----------------------------------------------------------------------
# Service caller
# ----------------------------------------------------------------------

def call_intent_service(host: str, body: dict, timeout: int = 120) -> dict:
    """POST to the intent service /predict and return parsed results."""
    url = f"{host.rstrip('/')}/predict"
    query_short = str(body.get("query", ""))[:60]
    print(f":)))) Calling intent service for query: {query_short} | URL: {url}")

    t0 = time.time()
    try:
        resp = requests.post(url, json=body, stream=True, timeout=timeout)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            events = parse_sse_stream(resp)
            if not events:
                print(f"    [WARN] No SSE events received for query: {query_short}")
            result = extract_fields(events)
        else:
            try:
                payload = resp.json()
            except json.JSONDecodeError as e:
                raw = resp.text[:300]
                print(f"    [ERROR] JSON decode failed for query: {query_short}")
                print(f"        Status: {resp.status_code} | Content-Type: {content_type}")
                print(f"        Body: {raw}")
                raise ValueError(f"Non-JSON response (status {resp.status_code}): {raw}") from e

            if "content" in payload and isinstance(payload["content"], dict):
                payload = payload["content"]
            result = {
                "intent_result": payload.get("intent_result", ""),
                "granular_intent": payload.get("granular_intent", ""),
                "contextual_query": payload.get("contextual_query", ""),
                "normalized_query": payload.get("normalized_query", ""),
                "answer": payload.get("answer", ""),
                "citation_count": len(payload.get("citations", [])),
                "citation_validation": (payload.get("citation_validation") or {}).get("citation_validation", ""),
                "jurisdictions": ", ".join((payload.get("query_jurisdiction") or {}).get("jurisdictions", [])),
                "resolve_jurisdictions": ", ".join((payload.get("resolve_jurisdiction") or {}).get("jurisdictions", [])),
                "error": "",
            }

        result["latency_s"] = round(time.time() - t0, 2)
        result["status_code"] = resp.status_code

    except requests.ConnectionError as e:
        elapsed = round(time.time() - t0, 2)
        print(f"    [ERROR] Connection failed for query: {query_short} ({elapsed}s)")
        print(f"        URL: {url}")
        print(f"        {type(e).__name__}: {e}")
        result = _empty_result(str(e)[:500], elapsed, None)

    except requests.Timeout as e:
        elapsed = round(time.time() - t0, 2)
        print(f"    [ERROR] Request timed out after {elapsed}s for query: {query_short}")
        print(f"        Timeout setting: {timeout}s")
        result = _empty_result(f"Timeout after {elapsed}s: {e}", elapsed, None)

    except requests.HTTPError as e:
        elapsed = round(time.time() - t0, 2)
        status = getattr(e.response, "status_code", None)
        resp_body = ""
        try:
            resp_body = e.response.text[:300] if e.response else ""
        except Exception:
            pass
        print(f"    [ERROR] HTTP {status} for query: {query_short} ({elapsed}s)")
        print(f"        Response: {resp_body}")
        result = _empty_result(f"HTTP {status}: {e}", elapsed, status)

    except requests.RequestException as e:
        elapsed = round(time.time() - t0, 2)
        print(f"    [ERROR] Request failed for query: {query_short} ({elapsed}s)")
        print(f"        {type(e).__name__}: {e}")
        result = _empty_result(
            str(e)[:500],
            elapsed,
            getattr(getattr(e, "response", None), "status_code", None),
        )

    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        print(f"    [ERROR] Unexpected error for query: {query_short} ({elapsed}s)")
        traceback.print_exc()
        result = _empty_result(f"{type(e).__name__}: {e}", elapsed, None)

    return result


def _empty_result(error: str, latency_s: float, status_code) -> dict:
    """Return an empty result dict with the given error information."""
    return {
        "intent_result": "", "granular_intent": "", "contextual_query": "",
        "normalized_query": "", "answer": "", "citation_count": 0,
        "citation_validation": "", "jurisdictions": "", "resolve_jurisdictions": "",
        "error": str(error)[:500],
        "latency_s": latency_s,
        "status_code": status_code,
    }


# ----------------------------------------------------------------------
# Caching helpers
# ----------------------------------------------------------------------

def _cache_path(output: str) -> Path:
    """Derive a .pkl cache path from the output file path."""
    return Path(output).with_suffix(".cache.pkl")


def _load_cache(output: str) -> list:
    """Load cached rows from a previous interrupted run, or return []."""
    p = _cache_path(output)
    if p.exists():
        try:
            with open(p, "rb") as f:
                rows = pickle.load(f)
            print(f"Resuming from cache ({len(rows)} rows already done): {p}")
            return rows
        except Exception as e:
            print(f"WARNING: failed to load cache {p}: {e} \u2014 starting fresh")
    return []


def _save_cache(rows: list, output: str):
    """Persist rows to the cache file."""
    p = _cache_path(output)
    with open(p, "wb") as f:
        pickle.dump(rows, f)


def _remove_cache(output: str):
    """Delete the cache file after successful completion."""
    p = _cache_path(output)
    if p.exists():
        p.unlink()
        print(f"Cache file removed: {p}")


_CACHE_INTERVAL = 5  # save cache every N queries


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _json_cell(val, default):
    """Parse a JSON string from an Excel cell, or return default."""
    if pd.isna(val) or val == "":
        return default
    if isinstance(val, (list, dict)):
        return val
    s = str(val)
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        pass
    # Fall back to ast.literal_eval for Python-style literals (single quotes, etc.)
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, dict)):
            return parsed
    except (ValueError, SyntaxError):
        pass
    print(f"    [WARN] Failed to parse cell: {s[:200]} | returning default")
    return default


def _print_summary(out_df: pd.DataFrame):
    print("\n" + "=" * 60)
    if "match" in out_df.columns:
        has_expected = out_df[out_df["expected_intent"] != ""]
        matched = has_expected[has_expected["match"] == True]  # noqa: E712
        n, total = len(matched), len(has_expected)
        print(
            f"Accuracy: {n}/{total} ({n / total * 100:.1f}%)"
            if total else "Accuracy: N/A (no expected_intent)"
        )

        if total > 0:
            print("\nPer-intent breakdown:")
            for val in sorted(has_expected["expected_intent"].unique()):
                sub = has_expected[has_expected["expected_intent"] == val]
                ok = sub[sub["match"] == True]  # noqa: E712
                print(f"  {val:25s} {len(ok)}/{len(sub)} ({len(ok) / len(sub) * 100:.0f}%)")

    print("\nActual intent distribution:")
    for val, cnt in out_df["actual_intent"].value_counts().items():
        print(f"  {val:25s} {cnt}")

    errs = out_df[out_df["error"] != ""]
    if len(errs):
        print(f"\nErrors: {len(errs)}/{len(out_df)}")

    print(f"\nAvg latency: {out_df['latency_s'].mean():.2f}s")
    print("=" * 60)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def run():
    # host = "http://intent-svc.rag-us.use1.dev-searchplatform.nl.lexis.com/"
    host = "http://localhost:8080/"

    # df = pd.read_excel(r"C:\Users\wangj38\Desktop\git_copy_of_code\gen-ai-platform\rag\plans\lexis_plus_ai_us\test_data\injection_data\...xlsx")
    df = pd.read_excel(
        r"C:\Users\wangj38\Desktop\git_copy_of_code\gen-ai-platform\rag\plans\lexis_plus_ai_us\test_data\Sean_data.xlsx"
    )

    # df = df[df['data_source'].notna()]   # filter out rows with empty data_source
    # if "Query" not in df.columns:
    #     raise ValueError("Excel file must have a 'Query' column")

    # --- Resume from cache if available ---
    # output = r"C:\Users\wangj38\Desktop\git_copy_of_code\gen-ai-platform\rag\plans\lexis_plus_ai_us\test_data\injection_data"
    # output_file = r"\eval_intent_e2e_results_upload2.xlsx"
    # output_file = r"\samples_output.xlsx"

    output = r"C:\Users\wangj38\Desktop\git_copy_of_code\gen-ai-platform\rag\plans\lexis_plus_ai_us\test_data"
    output_file = r"\Sean_data_cert7.xlsx"

    _remove_cache(output)
    rows = _load_cache(output)
    start_idx = len(rows)

    print(f"Total test cases: {len(df)}  |  Already cached: {start_idx}")
    # print(f"Host: {args.host}\n")

    # Build list of remaining work items
    pending = []
    for idx, row in df.iterrows():
        if idx < start_idx:
            continue
        pending.append((idx, row))

    concurrency = 1

    def _process_row(idx, row):
        query = str(row["query"])
        try:
            body = {
                "query": query,
                "chat_history": _json_cell(row.get("chat_history"), []),
                "initial_intent": "",
                "intent_input_type": "",
                "metadata": _json_cell(row.get("metadata"), {}),
                "headers": {},
                "tracing_info": {"asset_id": "3363"},
                "streaming": False,
                "highlighted_section": "",
                "data_source": _json_cell(row.get("data_source"), []),
                "contextaiinput": None,
                "feature_flags": {"intentClassificationRefinement": True},
            }

            print(body)
            result = call_intent_service(host, body, timeout=100)

            out = {
                "query": query,
                "actual_intent": result["intent_result"],
                **{
                    k: result[k]
                    for k in (
                        "granular_intent", "contextual_query", "normalized_query",
                        "citation_count", "citation_validation",
                        "jurisdictions", "resolve_jurisdictions",
                    )
                },
                "answer": (result["answer"] or "")[:200],
                "latency_s": result["latency_s"],
                "error": result["error"],
            }

            if "expected_intent" in df.columns:
                exp = str(row.get("expected_intent", "") or "")
                out["expected_intent"] = exp
                out["match"] = (
                    exp.strip().lower() == result["intent_result"].strip().lower()
                    if exp else ""
                )

            # Carry over ALL extra columns from the input row that aren't
            # already in the output dict (preserves every column from the
            # source spreadsheet, not just a hard-coded subset).
            for col in df.columns:
                if col not in out:
                    val = row.get(col, "")
                    out[col] = val if not pd.isna(val) else ""

            return idx, out, result

        except Exception as e:
            print(f"    [ERROR] _process_row failed for row {idx} (query: {query[:60]})")
            traceback.print_exc()
            error_result = _empty_result(f"{type(e).__name__}: {e}", 0.0, None)
            out = {
                "query": query, "actual_intent": "", "granular_intent": "",
                "contextual_query": "", "normalized_query": "",
                "citation_count": 0, "citation_validation": "",
                "jurisdictions": "", "resolve_jurisdictions": "",
                "answer": "", "latency_s": 0.0,
                "error": f"{type(e).__name__}: {str(e)[:400]}",
            }
            return idx, out, error_result

    # Process in batches of `concurrency` to preserve order + allow checkpoints
    for batch_start in range(0, len(pending), concurrency):
        batch = pending[batch_start: batch_start + concurrency]
        batch_results = {}

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(_process_row, idx, row): idx
                for idx, row in batch
            }
            for future in as_completed(futures):
                try:
                    idx, out, result = future.result()
                    batch_results[idx] = (out, result)
                except Exception as e:
                    failed_idx = futures[future]
                    print(f"    [ERROR] Future failed for row {failed_idx}: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    error_result = _empty_result(f"{type(e).__name__}: {e}", 0.0, None)
                    out = {
                        "query": "(unknown)", "actual_intent": "", "granular_intent": "",
                        "contextual_query": "", "normalized_query": "",
                        "citation_count": 0, "citation_validation": "",
                        "jurisdictions": "", "resolve_jurisdictions": "",
                        "answer": "", "latency_s": 0.0,
                        "error": f"{type(e).__name__}: {str(e)[:400]}",
                    }
                    batch_results[failed_idx] = (out, error_result)

        # Append in original row order
        for idx, row in batch:
            out, result = batch_results[idx]
            rows.append(out)
            intent = result.get("intent_result", "") if isinstance(result, dict) else ""
            latency = result.get("latency_s", 0) if isinstance(result, dict) else 0
            query_short = out["query"][:80]
            print(
                f"  [{idx + 1}/{len(df)}] {query_short} ... "
                f"-> {intent or 'ERROR'} ({latency}s)"
            )

        # Checkpoint after each batch
        if len(rows) % _CACHE_INTERVAL < concurrency:
            _save_cache(rows, output)
            print(f"    [checkpoint saved \u2014 {len(rows)} rows]")

    out_df = pd.DataFrame(rows)
    _print_summary(out_df)
    out_df.to_excel(output + output_file, index=False)
    _remove_cache(output)  # clean up cache after successful write
    print(f"\nResults written to {output + output_file}")


if __name__ == "__main__":
    run()