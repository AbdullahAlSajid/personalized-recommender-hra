"""
anchor_selector.py

Uses Groq llama-3.3-70b-versatile to:
  Stage 1 — Evaluate all 15 anchor candidates (3 per difficulty band)
  Stage 2 — Select the single best anchor per band

Inputs:
    anchor_candidates_shortlist.csv              — from preliminary scoring script
    all_texts_preliminary_difficulty_scores.csv  — for full text bodies

Outputs:
    anchor_candidates_evaluated.csv   — all 15 candidates with LLM ratings
    final_anchors.csv                 — one row per band (5 rows total)

Usage:
    pip install groq pandas python-dotenv
    python anchor_selector.py

    # Skip evaluation if already done, re-run selection only:
    python anchor_selector.py --select-only
"""

import argparse
import json
import os
import re
import time
import sys
from pathlib import Path

import pandas as pd
from groq import Groq
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────

SHORTLIST_CSV   = "anchor_candidates_shortlist.csv"
FULL_SCORES_CSV = "all_texts_preliminary_difficulty_scores.csv"
EVALUATED_CSV   = "anchor_candidates_evaluated.csv"
FINAL_CSV       = "final_anchors.csv"

TEXT_ID_COL     = "sanity_text_id"

MODEL_NAME      = "llama-3.3-70b-versatile"
TEMPERATURE     = 0.2
MAX_RETRIES     = 3
RETRY_DELAY     = 5
SLEEP_BETWEEN   = 1.0

BAND_ORDER = ["Band_1", "Band_2", "Band_3", "Band_4", "Band_5"]
BAND_LABELS = {
    "Band_1": "Very Easy",
    "Band_2": "Easy",
    "Band_3": "Medium",
    "Band_4": "Hard",
    "Band_5": "Very Hard",
}

BAND_TARGET_DIFFICULTY = {
    "Band_1": 1,
    "Band_2": 2,
    "Band_3": 3,
    "Band_4": 4,
    "Band_5": 5,
}

# ── Prompts ───────────────────────────────────────────────────────────────────

EVALUATION_SYSTEM = """
You are helping with readability-based anchor selection for Norwegian school texts.

Your task:
1. Read a Norwegian text passage.
2. Summarize it briefly in plain English.
3. Evaluate reading difficulty for children aged 9-11.
4. Score the text on six dimensions from 1 to 5.
5. Give an overall difficulty score and an anchor suitability score.

Scoring scale:
  1 = very easy / very low demand
  2 = easy / low demand
  3 = moderate
  4 = hard / high demand
  5 = very hard / very high demand

Dimension definitions:
  vocab_difficulty         — complexity and rarity of vocabulary; domain-specific or uncommon words
  sentence_complexity      — sentence length, subordinate clauses, syntactic depth
  abstractness             — how abstract vs concrete the ideas are
  background_knowledge     — prior knowledge a child needs to understand the text
  cohesion_structure       — how clearly the text guides the reader; logical flow and transitions
  inferential_demand       — how much a reader must infer beyond what is stated

Anchor suitability (overall_anchor_suitability_1to5):
  A good anchor is unambiguously representative of its band, well-formed,
  complete, and typical of Norwegian children's educational texts.
  Penalize: very short fragments, texts that are mostly lists or images,
  borderline texts that could belong in an adjacent band.

Rules:
  - Judge for a 9-11-year-old reader, not an adult.
  - Do not confuse interesting topic with high difficulty.
  - Do not over-penalize short passages just because they are short.

Return ONLY valid JSON. No markdown. No explanation outside the JSON.
"""

EVALUATION_SCHEMA = {
    "english_summary": "string",
    "manual_vocab_difficulty_1to5": 0,
    "manual_sentence_complexity_1to5": 0,
    "manual_abstractness_1to5": 0,
    "manual_background_knowledge_1to5": 0,
    "manual_cohesion_structure_1to5": 0,
    "manual_inferential_demand_1to5": 0,
    "overall_difficulty_1to5": 0,
    "overall_anchor_suitability_1to5": 0,
    "manual_notes": "string",
}


def build_evaluation_prompt(band_label: str, title: str, text: str, stats: dict) -> str:
    stats_str = (
        f"Word count: {stats.get('word_count', '?')} | "
        f"Avg sentence length: {stats.get('avg_sentence_length', '?')} words | "
        f"Avg word length: {stats.get('avg_word_length', '?')} chars | "
        f"Long word ratio: {stats.get('long_word_ratio', '?')} | "
        f"Lexical diversity: {stats.get('lexical_diversity', '?')}"
    )
    words = text.split()
    body  = " ".join(words[:1500])
    if len(words) > 1500:
        body += " [...]"

    schema_str = json.dumps(EVALUATION_SCHEMA, indent=2)

    return f"""Evaluate this Norwegian text candidate.

Candidate metadata:
  Difficulty band: {band_label}
  Title: {title}
  Surface statistics: {stats_str}

Norwegian text:
{body}

Return JSON with exactly these keys:
{schema_str}"""


SELECTION_SYSTEM = """
You are selecting the single best anchor text for a Norwegian reading difficulty scale.

You will receive evaluation results for 3 candidate texts from the same difficulty band.
Select the ONE text that best serves as an anchor for that band.

A good anchor is:
  - Unambiguously representative of its difficulty level (not a borderline case)
  - Complete and well-formed (not a fragment or a list-only text)
  - Typical of Norwegian children's educational content
  - Free from unusual features (e.g. purely numerical, image-heavy)

Return ONLY valid JSON. No markdown. No explanation outside the JSON.
"""


def build_selection_prompt(band: str, band_label: str, candidates: list) -> str:
    lines = [f"Band: {band} ({band_label})\n\nCandidates:\n"]
    for c in candidates:
        lines.append(
            f"--- Candidate {c.get('candidate_rank_within_band', '?')} "
            f"(text_id: {c[TEXT_ID_COL]}) ---\n"
            f"Title: {c.get('title', '')}\n"
            f"Overall difficulty:   {c.get('overall_difficulty_1to5', '?')}/5\n"
            f"Anchor suitability:   {c.get('overall_anchor_suitability_1to5', '?')}/5\n"
            f"Vocab:                {c.get('manual_vocab_difficulty_1to5', '?')}/5\n"
            f"Sentence complexity:  {c.get('manual_sentence_complexity_1to5', '?')}/5\n"
            f"Abstractness:         {c.get('manual_abstractness_1to5', '?')}/5\n"
            f"Background knowledge: {c.get('manual_background_knowledge_1to5', '?')}/5\n"
            f"Cohesion:             {c.get('manual_cohesion_structure_1to5', '?')}/5\n"
            f"Inferential demand:   {c.get('manual_inferential_demand_1to5', '?')}/5\n"
            f"Notes: {c.get('manual_notes', '')}\n"
        )
    lines.append(
        '\nReturn JSON:\n'
        '{\n'
        f'  "selected_text_id": "<one of the text_ids above>",\n'
        f'  "selection_reason": "<two or three sentences explaining why>"\n'
        '}'
    )
    return "\n".join(lines)


# ── API helper ────────────────────────────────────────────────────────────────

def call_groq(client: Groq, system: str, user: str) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                max_tokens=600,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"    [!] API error (attempt {attempt}/{MAX_RETRIES}): {e}")
            if "rate_limit" in str(e).lower() or "429" in str(e):
                print("    Rate limited — waiting 60s...")
                time.sleep(60)
            elif attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                raise


def parse_json(raw: str) -> dict:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    raw = re.sub(r'^```[a-z]*\n?', '', raw)
    raw = re.sub(r'\n?```$', '', raw)
    match = re.search(r'\{.*\}', raw, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Could not parse JSON from: {raw[:200]}")


def clamp(value, lo=1, hi=5):
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return ""


def normalize_evaluation(result: dict) -> dict:
    score_fields = [
        "manual_vocab_difficulty_1to5",
        "manual_sentence_complexity_1to5",
        "manual_abstractness_1to5",
        "manual_background_knowledge_1to5",
        "manual_cohesion_structure_1to5",
        "manual_inferential_demand_1to5",
        "overall_difficulty_1to5",
        "overall_anchor_suitability_1to5",
    ]
    out = {k: result.get(k, "") for k in EVALUATION_SCHEMA}
    for f in score_fields:
        out[f] = str(clamp(out[f]))
    out["english_summary"]   = str(out.get("english_summary", "")).strip()
    out["estimated_age_fit"] = str(out.get("estimated_age_fit", "")).strip()
    out["manual_notes"]      = str(out.get("manual_notes", "")).strip()
    return out


# ── Stage 1: Evaluate ─────────────────────────────────────────────────────────

def run_evaluation(client: Groq, shortlist: pd.DataFrame, full_scores: pd.DataFrame) -> pd.DataFrame:
    full_map = full_scores.set_index(TEXT_ID_COL).to_dict("index")

    for col in list(EVALUATION_SCHEMA.keys()) + ["llm_status", "llm_error"]:
        if col not in shortlist.columns:
            shortlist[col] = ""

    total = len(shortlist)
    print(f"\n{'─'*62}")
    print(f"  Stage 1 — Evaluating {total} candidates ({total} API calls)")
    print(f"{'─'*62}")

    for i, (idx, row) in enumerate(shortlist.iterrows(), 1):
        text_id    = str(row[TEXT_ID_COL])
        title      = str(row.get("title", ""))
        band       = str(row.get("difficulty_band", ""))
        band_label = BAND_LABELS.get(band, band)

        full_row = full_map.get(text_id, {})
        text     = str(full_row.get("full_text", row.get("text_preview", "")))
        stats    = {
            "word_count":          full_row.get("word_count",          row.get("word_count", "?")),
            "avg_sentence_length": full_row.get("avg_sentence_length", row.get("avg_sentence_length", "?")),
            "avg_word_length":     full_row.get("avg_word_length",     row.get("avg_word_length", "?")),
            "long_word_ratio":     full_row.get("long_word_ratio",     row.get("long_word_ratio", "?")),
            "lexical_diversity":   full_row.get("lexical_diversity",   row.get("lexical_diversity", "?")),
        }

        print(f"  [{i}/{total}] {band_label:<12} — {title[:48]}")

        prompt = build_evaluation_prompt(band_label, title, text, stats)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw        = call_groq(client, EVALUATION_SYSTEM, prompt)
                parsed     = parse_json(raw)
                normalized = normalize_evaluation(parsed)

                for k, v in normalized.items():
                    shortlist.at[idx, k] = v
                shortlist.at[idx, "llm_status"] = "ok"
                shortlist.at[idx, "llm_error"]  = ""

                print(
                    f"         overall={normalized.get('overall_difficulty_1to5','?')}/5  "
                    f"suitability={normalized.get('overall_anchor_suitability_1to5','?')}/5  "
                    f"age={normalized.get('estimated_age_fit','?')}"
                )
                break

            except Exception as e:
                msg = f"Attempt {attempt}: {type(e).__name__}: {e}"
                print(f"    [!] {msg}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)
                else:
                    shortlist.at[idx, "llm_status"] = "failed"
                    shortlist.at[idx, "llm_error"]  = msg

        # Save after every row — progress is never lost on crash
        shortlist.to_csv(EVALUATED_CSV, index=False, encoding="utf-8-sig")

        if i < total:
            time.sleep(SLEEP_BETWEEN)

    return shortlist


# ── Stage 2: Select ───────────────────────────────────────────────────────────

def run_selection(client: Groq, evaluated: pd.DataFrame) -> pd.DataFrame:
    anchor_rows = []

    print(f"\n{'─'*62}")
    print(f"  Stage 2 — Selecting one anchor per band (5 API calls)")
    print(f"{'─'*62}")

    for band in BAND_ORDER:
        label   = BAND_LABELS[band]
        band_df = evaluated[evaluated["difficulty_band"] == band].copy()
        target  = BAND_TARGET_DIFFICULTY[band]

        if band_df.empty:
            print(f"\n  [!] No candidates found for {band} — skipping")
            continue

        candidates = band_df.to_dict("records")

        print(f"\n  {band} ({label}) — candidates:")
        for c in candidates:
            print(
                f"    [{c.get('candidate_rank_within_band','?')}] "
                f"{str(c.get('title',''))[:45]:<47} "
                f"overall={c.get('overall_difficulty_1to5','?')}/5  "
                f"suit={c.get('overall_anchor_suitability_1to5','?')}/5"
            )

        selected_row     = None
        selection_reason = ""

        # LLM selection
        try:
            prompt   = build_selection_prompt(band, label, candidates)
            raw      = call_groq(client, SELECTION_SYSTEM, prompt)
            decision = parse_json(raw)

            selected_id      = str(decision.get("selected_text_id", "")).strip()
            selection_reason = decision.get("selection_reason", "")

            match = band_df[band_df[TEXT_ID_COL].astype(str) == selected_id]
            if not match.empty:
                selected_row = match.iloc[0].to_dict()
            else:
                print(f"    [!] LLM returned unknown id '{selected_id}' — using fallback")

        except Exception as e:
            print(f"    [!] LLM selection failed: {e} — using fallback")

        # Fallback: highest suitability, tiebreak by closest difficulty to band target
        if selected_row is None:
            selection_reason = "Automatic fallback: highest suitability score"
            band_df["_suit"] = pd.to_numeric(
                band_df["overall_anchor_suitability_1to5"], errors="coerce"
            ).fillna(0)
            band_df["_dist"] = (
                pd.to_numeric(band_df["overall_difficulty_1to5"], errors="coerce")
                .fillna(3)
                .sub(target)
                .abs()
            )
            band_df      = band_df.sort_values(["_suit", "_dist"], ascending=[False, True])
            selected_row = band_df.iloc[0].to_dict()

        selected_row["final_anchor_pick_yes_no"] = "yes"
        selected_row["selection_reason"]         = selection_reason
        anchor_rows.append(selected_row)

        print(f"  → Selected: {str(selected_row.get('title',''))[:52]}")
        if selection_reason:
            print(f"    Reason:   {selection_reason[:100]}")

        time.sleep(SLEEP_BETWEEN)

    return pd.DataFrame(anchor_rows)


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(anchors: pd.DataFrame):
    print(f"\n{'═'*62}")
    print(f"  FINAL ANCHORS")
    print(f"{'═'*62}")
    print(f"  {'Band':<10} {'Label':<12} {'Diff':>5} {'Suit':>5}  Title")
    print(f"  {'─'*60}")
    for _, row in anchors.iterrows():
        print(
            f"  {str(row.get('difficulty_band','')):<10} "
            f"{str(row.get('difficulty_band_label','')):<12} "
            f"{str(row.get('overall_difficulty_1to5','?')):>4}/5 "
            f"{str(row.get('overall_anchor_suitability_1to5','?')):>4}/5  "
            f"{str(row.get('title',''))[:38]}"
        )
    print(f"\n  Saved → {FINAL_CSV}")
    print(f"  Saved → {EVALUATED_CSV}")
    print(f"\n  IMPORTANT: Open final_anchors.csv and read each selected text.")
    print(f"  Confirm the choices make sense before running the difficulty")
    print(f"  scoring pipeline. If a selection looks wrong:")
    print(f"    1. Edit final_anchor_pick_yes_no in anchor_candidates_evaluated.csv")
    print(f"    2. Re-run: python anchor_selector.py --select-only\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM-assisted anchor selection.")
    parser.add_argument(
        "--select-only", action="store_true",
        help="Skip evaluation, load existing evaluated CSV, re-run selection only.",
    )
    args = parser.parse_args()

    load_dotenv(Path(__file__).parent.parent.parent / ".env")
    api_key = "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"
    if not api_key:
        print("Error: GROQ_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    client = Groq(api_key=api_key)
    print(f"  Model: {MODEL_NAME}")

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    if args.select_only:
        if not Path(EVALUATED_CSV).exists():
            print(f"Error: {EVALUATED_CSV} not found. Run without --select-only first.")
            sys.exit(1)
        evaluated = pd.read_csv(EVALUATED_CSV, dtype=str)
        print(f"  Loaded existing evaluations from {EVALUATED_CSV} ({len(evaluated)} rows)")
    else:
        for path in [SHORTLIST_CSV, FULL_SCORES_CSV]:
            if not Path(path).exists():
                print(f"Error: {path} not found.")
                sys.exit(1)

        shortlist   = pd.read_csv(SHORTLIST_CSV, dtype=str)
        full_scores = pd.read_csv(FULL_SCORES_CSV, dtype=str)
        print(f"  Loaded {len(shortlist)} candidates from shortlist")
        print(f"  Loaded {len(full_scores)} full text records")

        evaluated = run_evaluation(client, shortlist, full_scores)
        print(f"\n  Evaluations complete → {EVALUATED_CSV}")

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    anchors = run_selection(client, evaluated)
    anchors.to_csv(FINAL_CSV, index=False, encoding="utf-8-sig")

    print_summary(anchors)


if __name__ == "__main__":
    main()