"""
difficulty_pipeline.py

Assigns difficulty scores (1-5) to all 157 Norwegian children's texts
using anchor-based LLM comparison, with optional pairwise validation.

Stages:
    1  Anchor comparison — compare each text against 5 anchors (785 API calls)
    2  Pairwise validation — compare 40-text sample pairwise (~780 API calls)
       then compute Kendall's tau between anchor-rank and pairwise-rank

Usage:
    # Run Stage 1 only (anchor comparison):
    python difficulty_pipeline.py

    # Run Stage 2 only (pairwise validation, after Stage 1 is complete):
    python difficulty_pipeline.py --validate-only

    # Re-run Stage 1 from scratch (ignores checkpoint):
    python difficulty_pipeline.py --fresh

Inputs:
    final_anchors_v2.csv          — 5 selected anchor texts
    all_texts_lix_scores_v2.csv   — all 157 texts with full text bodies

Outputs:
    difficulty_scores.csv         — all 157 texts with difficulty 1-5
    difficulty_checkpoint.json    — Stage 1 progress (auto-saved after each text)
    validation_sample.csv         — 40-text sample used for pairwise validation
    pairwise_results.csv          — all pairwise comparison results
    validation_metrics.json       — Kendall's tau and agreement statistics
"""

import argparse
import json
import os
import re
import sys
import time
import random
from pathlib import Path
from itertools import combinations

import pandas as pd
from scipy import stats as scipy_stats
from groq import Groq
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────

ANCHORS_CSV      = "final_anchors_v2.csv"
ALL_TEXTS_CSV    = "all_texts_lix_scores_v2.csv"
SCORES_CSV       = "difficulty_scores.csv"
CHECKPOINT_JSON  = "difficulty_checkpoint.json"
VALIDATION_CSV   = "validation_sample.csv"
PAIRWISE_CSV     = "pairwise_results.csv"
METRICS_JSON     = "validation_metrics.json"

TEXT_ID_COL      = "sanity_text_id"

MODEL_NAME       = "llama-3.3-70b-versatile"
TEMPERATURE      = 0.1      # low temperature for more consistent scoring
MAX_RETRIES      = 3
RETRY_DELAY      = 5
SLEEP_BETWEEN    = 1.0

VALIDATION_SAMPLE_SIZE = 40
RANDOM_SEED            = 42

BAND_TO_DIFFICULTY = {
    "Band_1": 1,
    "Band_2": 2,
    "Band_3": 3,
    "Band_4": 4,
    "Band_5": 5,
}

BAND_LABELS = {
    "Band_1": "Very Easy",
    "Band_2": "Easy",
    "Band_3": "Medium",
    "Band_4": "Hard",
    "Band_5": "Very Hard",
}

# ── Prompts ───────────────────────────────────────────────────────────────────

ANCHOR_COMPARISON_SYSTEM = """
You are an expert in Norwegian language education and reading difficulty assessment
for children aged 9-11 (Norwegian primary school grades 4-6).

You will receive:
1. A target text to score
2. Five anchor texts, each representing a difficulty level from 1 to 5

Your task:
- Read the target text carefully
- Compare it to the five anchor texts
- Assign a difficulty score from 1 to 5

Difficulty scale:
  1 = Very Easy  — similar to or easier than the Level 1 anchor
  2 = Easy       — similar to or easier than the Level 2 anchor
  3 = Medium     — similar to or easier than the Level 3 anchor
  4 = Hard       — similar to or easier than the Level 4 anchor
  5 = Very Hard  — similar to or harder than the Level 5 anchor

When comparing, consider these dimensions in order of importance:
  1. vocab_difficulty     — domain-specific, technical, or uncommon vocabulary
  2. background_knowledge — prior knowledge a 9-11-year-old needs
  3. abstractness         — how abstract vs concrete the concepts are
  4. sentence_complexity  — sentence length and syntactic depth
  5. inferential_demand   — how much the reader must infer

Rules:
  - Judge for a 9-11-year-old Norwegian reader, not an adult
  - A text about a familiar everyday topic with simple vocabulary is easier
    than a text about a technical topic even if sentence lengths are similar
  - Do not confuse interesting or engaging topic with high difficulty
  - If the text falls between two anchors, choose the closer one

Return ONLY valid JSON. No markdown. No explanation outside the JSON.
{
  "difficulty": <integer 1-5>,
  "closest_anchor": <integer 1-5>,
  "easier_or_harder": "easier" | "same" | "harder",
  "reasoning": "<one sentence explaining the score>"
}
"""


def build_anchor_comparison_prompt(target: dict, anchors: list[dict]) -> str:
    target_words = target["full_text"].split()
    target_body  = " ".join(target_words[:1200])
    if len(target_words) > 1200:
        target_body += " [...]"

    anchor_section = ""
    for a in anchors:
        level  = BAND_TO_DIFFICULTY.get(a.get("difficulty_band", ""), "?")
        label  = BAND_LABELS.get(a.get("difficulty_band", ""), "")
        words  = str(a.get("full_text", "")).split()
        body   = " ".join(words[:400])
        if len(words) > 400:
            body += " [...]"
        anchor_section += (
            f"\n--- ANCHOR LEVEL {level} ({label}) ---\n"
            f"Title: {a.get('title', '')}\n"
            f"{body}\n"
        )

    return (
        f"TARGET TEXT TO SCORE:\n"
        f"Title: {target.get('title', '')}\n"
        f"{target_body}\n\n"
        f"{'─'*50}\n"
        f"ANCHOR TEXTS (reference scale):\n"
        f"{anchor_section}"
    )


PAIRWISE_SYSTEM = """
You are an expert in Norwegian language education and reading difficulty assessment
for children aged 9-11 (Norwegian primary school grades 4-6).

You will receive two Norwegian texts. Your task is to decide which one is
harder to read and understand for a 9-11-year-old Norwegian child.

Consider:
  1. Vocabulary difficulty — domain-specific or uncommon words
  2. Background knowledge required — assumed familiarity with concepts
  3. Abstractness — concrete vs abstract ideas
  4. Sentence complexity — length and syntactic depth

Rules:
  - Judge for a 9-11-year-old Norwegian reader, not an adult
  - Do not confuse interesting topic with high difficulty
  - If genuinely equal, choose the one with more technical vocabulary

Return ONLY valid JSON. No markdown. No explanation outside the JSON.
{
  "harder_text": "A" | "B",
  "confidence": "high" | "medium" | "low",
  "reasoning": "<one sentence explaining the choice>"
}
"""


def build_pairwise_prompt(text_a: dict, text_b: dict) -> str:
    def truncate(t):
        words = str(t.get("full_text", "")).split()
        body  = " ".join(words[:600])
        if len(words) > 600:
            body += " [...]"
        return body

    return (
        f"TEXT A:\n"
        f"Title: {text_a.get('title', '')}\n"
        f"{truncate(text_a)}\n\n"
        f"{'─'*50}\n\n"
        f"TEXT B:\n"
        f"Title: {text_b.get('title', '')}\n"
        f"{truncate(text_b)}\n\n"
        f"Which text is harder for a 9-11-year-old Norwegian child to read?"
    )


# ── API helper ────────────────────────────────────────────────────────────────

def call_groq(client: Groq, system: str, user: str, max_tokens: int = 300) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                max_tokens=max_tokens,
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


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    if Path(CHECKPOINT_JSON).exists():
        with open(CHECKPOINT_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(results: dict):
    with open(CHECKPOINT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# ── Stage 1: Anchor comparison ────────────────────────────────────────────────

def run_anchor_comparison(client: Groq, texts: list[dict], anchors: list[dict]) -> list[dict]:
    checkpoint = load_checkpoint()
    results    = dict(checkpoint)
    remaining  = [t for t in texts if t[TEXT_ID_COL] not in results]
    total      = len(texts)
    done       = len(results)

    if done > 0:
        print(f"  Resuming: {done}/{total} done, {len(remaining)} remaining.")

    print(f"\n{'─'*62}")
    print(f"  Stage 1 — Anchor comparison ({len(remaining)} remaining / {total} total)")
    print(f"{'─'*62}")

    for i, text in enumerate(remaining, done + 1):
        text_id = text[TEXT_ID_COL]
        title   = text.get("title", "")

        print(f"  [{i}/{total}] {title[:55]}")

        prompt = build_anchor_comparison_prompt(text, anchors)

        success = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw    = call_groq(client, ANCHOR_COMPARISON_SYSTEM, prompt)
                parsed = parse_json(raw)

                difficulty = int(parsed.get("difficulty", 0))
                if difficulty not in (1, 2, 3, 4, 5):
                    raise ValueError(f"Invalid difficulty: {difficulty}")

                results[text_id] = {
                    TEXT_ID_COL:        text_id,
                    "title":            title,
                    "difficulty":       difficulty,
                    "closest_anchor":   parsed.get("closest_anchor", ""),
                    "easier_or_harder": parsed.get("easier_or_harder", ""),
                    "reasoning":        parsed.get("reasoning", ""),
                    "lix":              text.get("lix", ""),
                    "hybrid_score":     text.get("hybrid_score", ""),
                    "word_count":       text.get("word_count", ""),
                    "error":            None,
                }

                print(f"         difficulty={difficulty}/5  "
                      f"anchor={parsed.get('closest_anchor','?')}  "
                      f"{parsed.get('easier_or_harder','?')}")
                success = True
                break

            except Exception as e:
                msg = f"Attempt {attempt}: {type(e).__name__}: {e}"
                print(f"    [!] {msg}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)
                else:
                    results[text_id] = {
                        TEXT_ID_COL:        text_id,
                        "title":            title,
                        "difficulty":       None,
                        "closest_anchor":   None,
                        "easier_or_harder": None,
                        "reasoning":        None,
                        "lix":              text.get("lix", ""),
                        "hybrid_score":     text.get("hybrid_score", ""),
                        "word_count":       text.get("word_count", ""),
                        "error":            msg,
                    }

        # Save after every text — never lose progress on crash
        save_checkpoint(results)

        if i < total:
            time.sleep(SLEEP_BETWEEN)

    return list(results.values())


# ── Stage 1 export ────────────────────────────────────────────────────────────

def export_scores(results: list[dict]):
    df = pd.DataFrame(results)

    # Difficulty distribution
    print(f"\n{'─'*50}")
    print(f"  DIFFICULTY DISTRIBUTION")
    print(f"{'─'*50}")
    dist = df["difficulty"].value_counts().sort_index()
    for level, count in dist.items():
        bar = "█" * int(count * 30 / dist.max())
        print(f"  Level {level}  {count:>4} texts  {bar}")

    errors = df[df["error"].notna()]
    if len(errors) > 0:
        print(f"\n  [!] {len(errors)} texts failed:")
        for _, row in errors.iterrows():
            print(f"      - {row['title'][:50]}")

    df.to_csv(SCORES_CSV, index=False, encoding="utf-8-sig")
    print(f"\n  Saved → {SCORES_CSV}")


# ── Stage 2: Pairwise validation ──────────────────────────────────────────────

def run_pairwise_validation(client: Groq, scores_df: pd.DataFrame, all_texts: list[dict]):
    text_map = {t[TEXT_ID_COL]: t for t in all_texts}

    # Sample 40 texts stratified by difficulty level
    random.seed(RANDOM_SEED)
    sample_ids = []

    scored = scores_df[scores_df["difficulty"].notna()].copy()
    scored["difficulty"] = scored["difficulty"].astype(int)

    per_level = VALIDATION_SAMPLE_SIZE // 5
    for level in range(1, 6):
        level_texts = scored[scored["difficulty"] == level][TEXT_ID_COL].tolist()
        n           = min(per_level, len(level_texts))
        sample_ids.extend(random.sample(level_texts, n))

    # Top up to 40 if any level had fewer texts
    remaining_ids = [
        tid for tid in scored[TEXT_ID_COL].tolist()
        if tid not in sample_ids
    ]
    while len(sample_ids) < VALIDATION_SAMPLE_SIZE and remaining_ids:
        pick = random.choice(remaining_ids)
        sample_ids.append(pick)
        remaining_ids.remove(pick)

    sample_df = scored[scored[TEXT_ID_COL].isin(sample_ids)].copy()
    sample_df.to_csv(VALIDATION_CSV, index=False, encoding="utf-8-sig")

    print(f"\n{'─'*62}")
    print(f"  Stage 2 — Pairwise validation ({len(sample_ids)} texts)")
    print(f"  Total comparisons: {len(sample_ids) * (len(sample_ids)-1) // 2}")
    print(f"{'─'*62}")

    pairs    = list(combinations(sample_ids, 2))
    total    = len(pairs)
    pairwise = []

    # Load existing pairwise results if resuming
    done_pairs = set()
    if Path(PAIRWISE_CSV).exists():
        existing = pd.read_csv(PAIRWISE_CSV, dtype=str)
        for _, row in existing.iterrows():
            done_pairs.add((row["text_id_a"], row["text_id_b"]))
        pairwise = existing.to_dict("records")
        print(f"  Resuming: {len(done_pairs)}/{total} pairs done.")

    for i, (id_a, id_b) in enumerate(pairs, 1):
        if (id_a, id_b) in done_pairs or (id_b, id_a) in done_pairs:
            continue

        text_a = text_map.get(id_a, {})
        text_b = text_map.get(id_b, {})

        title_a = text_a.get("title", id_a)[:35]
        title_b = text_b.get("title", id_b)[:35]
        print(f"  [{i}/{total}] {title_a} vs {title_b}")

        prompt = build_pairwise_prompt(text_a, text_b)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw    = call_groq(client, PAIRWISE_SYSTEM, prompt)
                parsed = parse_json(raw)

                harder = parsed.get("harder_text", "")
                if harder not in ("A", "B"):
                    raise ValueError(f"Invalid harder_text: {harder}")

                winner_id = id_a if harder == "A" else id_b
                loser_id  = id_b if harder == "A" else id_a

                pairwise.append({
                    "text_id_a":   id_a,
                    "title_a":     text_a.get("title", ""),
                    "text_id_b":   id_b,
                    "title_b":     text_b.get("title", ""),
                    "harder_text": harder,
                    "winner_id":   winner_id,
                    "loser_id":    loser_id,
                    "confidence":  parsed.get("confidence", ""),
                    "reasoning":   parsed.get("reasoning", ""),
                    "error":       None,
                })
                print(f"         harder={harder}  "
                      f"confidence={parsed.get('confidence','?')}")
                break

            except Exception as e:
                msg = f"Attempt {attempt}: {type(e).__name__}: {e}"
                print(f"    [!] {msg}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)
                else:
                    pairwise.append({
                        "text_id_a":   id_a,
                        "title_a":     text_a.get("title", ""),
                        "text_id_b":   id_b,
                        "title_b":     text_b.get("title", ""),
                        "harder_text": None,
                        "winner_id":   None,
                        "loser_id":    None,
                        "confidence":  None,
                        "reasoning":   None,
                        "error":       msg,
                    })

        # Save after every pair
        pd.DataFrame(pairwise).to_csv(PAIRWISE_CSV, index=False, encoding="utf-8-sig")

        if i < total:
            time.sleep(SLEEP_BETWEEN)

    return pd.DataFrame(pairwise)


# ── Kendall's tau ─────────────────────────────────────────────────────────────

def compute_validation_metrics(scores_df: pd.DataFrame, pairwise_df: pd.DataFrame) -> dict:
    # Compute pairwise win counts → rank
    win_counts = {}
    for _, row in pairwise_df[pairwise_df["winner_id"].notna()].iterrows():
        winner = row["winner_id"]
        win_counts[winner] = win_counts.get(winner, 0) + 1

    sample_ids = list(set(
        pairwise_df["text_id_a"].tolist() +
        pairwise_df["text_id_b"].tolist()
    ))

    pairwise_rank = {tid: win_counts.get(tid, 0) for tid in sample_ids}

    # Anchor-based rank from difficulty scores
    sample_scores = scores_df[scores_df[TEXT_ID_COL].isin(sample_ids)].copy()
    anchor_rank   = dict(zip(
        sample_scores[TEXT_ID_COL],
        sample_scores["difficulty"].astype(float)
    ))

    # Align
    common_ids    = [tid for tid in sample_ids if tid in anchor_rank]
    anchor_vals   = [anchor_rank[tid] for tid in common_ids]
    pairwise_vals = [pairwise_rank[tid] for tid in common_ids]

    tau, p_value = scipy_stats.kendalltau(anchor_vals, pairwise_vals)

    metrics = {
        "n_texts":          len(common_ids),
        "n_pairs":          len(pairwise_df[pairwise_df["winner_id"].notna()]),
        "kendall_tau":      round(float(tau), 4),
        "p_value":          round(float(p_value), 4),
        "interpretation":   (
            "Strong agreement"   if abs(tau) >= 0.7 else
            "Moderate agreement" if abs(tau) >= 0.5 else
            "Weak agreement"
        ),
    }

    print(f"\n{'═'*62}")
    print(f"  VALIDATION METRICS")
    print(f"{'═'*62}")
    print(f"  Texts compared:     {metrics['n_texts']}")
    print(f"  Pairs evaluated:    {metrics['n_pairs']}")
    print(f"  Kendall's τ:        {metrics['kendall_tau']}")
    print(f"  p-value:            {metrics['p_value']}")
    print(f"  Interpretation:     {metrics['interpretation']}")
    print(f"\n  A τ ≥ 0.7 indicates strong agreement between anchor-based")
    print(f"  and pairwise difficulty rankings.")

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Saved → {METRICS_JSON}")

    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Difficulty pipeline for Norwegian children's texts.")
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Skip Stage 1, load existing difficulty_scores.csv, run pairwise validation only.",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore checkpoint and re-run Stage 1 from scratch.",
    )
    args = parser.parse_args()

    load_dotenv(Path(__file__).parent.parent.parent / ".env")
    api_key = "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"
    if not api_key:
        print("Error: GROQ_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    client = Groq(api_key=api_key)
    print(f"  Model: {MODEL_NAME}")

    # ── Load inputs ───────────────────────────────────────────────────────────
    for path in [ANCHORS_CSV, ALL_TEXTS_CSV]:
        if not Path(path).exists():
            print(f"Error: {path} not found.")
            sys.exit(1)

    anchors_df   = pd.read_csv(ANCHORS_CSV, dtype=str)
    all_texts_df = pd.read_csv(ALL_TEXTS_CSV, dtype=str)

    print(f"  Loaded {len(anchors_df)} anchors from {ANCHORS_CSV}")
    print(f"  Loaded {len(all_texts_df)} texts from {ALL_TEXTS_CSV}")

    anchors   = anchors_df.to_dict("records")
    all_texts = all_texts_df.to_dict("records")

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    if args.validate_only:
        if not Path(SCORES_CSV).exists():
            print(f"Error: {SCORES_CSV} not found. Run Stage 1 first.")
            sys.exit(1)
        scores_df = pd.read_csv(SCORES_CSV, dtype=str)
        print(f"\n  Loaded existing scores from {SCORES_CSV} ({len(scores_df)} rows)")
    else:
        if args.fresh and Path(CHECKPOINT_JSON).exists():
            Path(CHECKPOINT_JSON).unlink()
            print("  Cleared checkpoint — starting fresh.")

        results   = run_anchor_comparison(client, all_texts, anchors)
        scores_df = pd.DataFrame(results)
        export_scores(results)

        print(f"\n  Stage 1 complete.")
        print(f"  To run pairwise validation:")
        print(f"  python difficulty_pipeline.py --validate-only")

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    if args.validate_only:
        pairwise_df = run_pairwise_validation(client, scores_df, all_texts)
        compute_validation_metrics(scores_df, pairwise_df)


if __name__ == "__main__":
    main()