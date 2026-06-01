"""
pairwise_pipeline.py

Runs all pairwise comparisons across all 157 Norwegian children's texts
to produce an independent difficulty ranking, then combines with anchor
scores from difficulty_scores_v2.csv.

Total comparisons: 157 +ù 156 / 2 = 12,246 API calls
Estimated time:    ~20 minutes on Groq paid tier (1K RPM)

Usage:
    pip install groq pandas scipy python-dotenv

    # Run full pairwise comparison:
    python pairwise_pipeline.py

    # Skip comparison, compute scores from existing checkpoint:
    python pairwise_pipeline.py --scores-only

    # Combine anchor + pairwise into final scores (no API calls):
    python pairwise_pipeline.py --combine-only

Inputs:
    all_texts_lix_scores_v2.csv    GÇö all 157 texts with full text bodies
    difficulty_scores_v2.csv       GÇö anchor-based scores from Stage 1

Outputs:
    pairwise_checkpoint.json       GÇö auto-saved progress after every pair
    pairwise_results_full.csv      GÇö all 12,246 comparison results
    pairwise_scores.csv            GÇö win counts + normalized 1.0-5.0 per text
    final_difficulty_scores.csv    GÇö combined anchor + pairwise scores
    validation_metrics.json        GÇö Kendall's tau between the two methods
"""

import argparse
import json
import os
import re
import sys
import time
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy import stats as scipy_stats
from groq import Groq
from dotenv import load_dotenv

# GöÇGöÇ Config GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

ALL_TEXTS_CSV     = "all_texts_lix_scores_v2.csv"
ANCHOR_SCORES_CSV = "difficulty_scores_v2.csv"
CHECKPOINT_JSON   = "pairwise_checkpoint.json"
RESULTS_CSV       = "pairwise_results_full.csv"
PAIRWISE_SCORES   = "pairwise_scores.csv"
FINAL_SCORES_CSV  = "final_difficulty_scores.csv"
METRICS_JSON      = "validation_metrics.json"

TEXT_ID_COL       = "sanity_text_id"

MODEL_NAME        = "llama-3.3-70b-versatile"
TEMPERATURE       = 0.1
MAX_RETRIES       = 3
RETRY_DELAY       = 5
SLEEP_BETWEEN     = 0.1     # paid tier GÇö 1K RPM

# GöÇGöÇ Prompt GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

PAIRWISE_SYSTEM = """
You are an expert in Norwegian language education and reading difficulty assessment
for children aged 9-11 (Norwegian primary school grades 4-6).

You will receive two Norwegian texts labeled A and B.
Decide which one is harder to read and understand for a 9-11-year-old Norwegian child.

Consider these dimensions in order of importance:
  1. Vocabulary difficulty    GÇö domain-specific, technical, or uncommon words
  2. Background knowledge     GÇö prior knowledge a 9-11-year-old needs
  3. Abstractness             GÇö concrete vs abstract ideas and concepts
  4. Sentence complexity      GÇö length and syntactic depth
  5. Inferential demand       GÇö how much the reader must infer

Rules:
  - Judge for a 9-11-year-old Norwegian reader, not an adult
  - Do not confuse interesting or engaging topic with high difficulty
  - A text about a familiar everyday topic (sports, animals, food) with
    simple vocabulary is easier than a technical or abstract text even
    if sentence lengths are similar
  - If genuinely equal in all dimensions, choose the one with more
    domain-specific or technical vocabulary

Return ONLY valid JSON. No markdown. No explanation outside the JSON.
{
  "harder_text": "A" or "B",
  "confidence": "high" or "medium" or "low",
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
        f"{'GöÇ'*50}\n\n"
        f"TEXT B:\n"
        f"Title: {text_b.get('title', '')}\n"
        f"{truncate(text_b)}\n\n"
        f"Which text is harder for a 9-11-year-old Norwegian child to read?"
    )


# GöÇGöÇ API helper GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

def call_groq(client: Groq, system: str, user: str) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                max_tokens=200,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"    [!] API error (attempt {attempt}/{MAX_RETRIES}): {e}")
            if "rate_limit" in str(e).lower() or "429" in str(e):
                print("    Rate limited GÇö waiting 60s...")
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


# GöÇGöÇ Checkpoint helpers GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

def load_checkpoint() -> dict:
    if Path(CHECKPOINT_JSON).exists():
        with open(CHECKPOINT_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(results: dict):
    with open(CHECKPOINT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# GöÇGöÇ Stage 1: Pairwise comparisons GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

def run_pairwise(client: Groq, texts: list) -> pd.DataFrame:
    text_map  = {t[TEXT_ID_COL]: t for t in texts}
    all_ids   = [t[TEXT_ID_COL] for t in texts]
    all_pairs = list(combinations(all_ids, 2))
    total     = len(all_pairs)

    checkpoint = load_checkpoint()
    done_pairs = set(tuple(k.split("|||")) for k in checkpoint.keys())
    remaining  = [
        (a, b) for a, b in all_pairs
        if (a, b) not in done_pairs and (b, a) not in done_pairs
    ]

    print(f"\n{'GöÇ'*62}")
    print(f"  Pairwise comparison")
    print(f"  Total pairs:     {total}")
    print(f"  Already done:    {len(done_pairs)}")
    print(f"  Remaining:       {len(remaining)}")
    print(f"  Est. time:       ~{len(remaining) // 600 + 1} minutes")
    print(f"{'GöÇ'*62}")

    for i, (id_a, id_b) in enumerate(remaining, len(done_pairs) + 1):
        text_a  = text_map.get(id_a, {})
        text_b  = text_map.get(id_b, {})
        title_a = str(text_a.get("title", id_a))[:28]
        title_b = str(text_b.get("title", id_b))[:28]

        # Print progress every 200 pairs
        if i % 200 == 0 or i <= 3:
            pct = i * 100 // total
            print(f"  [{i}/{total}] {pct}%  {title_a} vs {title_b}")

        prompt = build_pairwise_prompt(text_a, text_b)
        key    = f"{id_a}|||{id_b}"

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw    = call_groq(client, PAIRWISE_SYSTEM, prompt)
                parsed = parse_json(raw)

                harder = parsed.get("harder_text", "")
                if harder not in ("A", "B"):
                    raise ValueError(f"Invalid harder_text: {harder}")

                winner_id = id_a if harder == "A" else id_b
                loser_id  = id_b if harder == "A" else id_a

                checkpoint[key] = {
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
                }
                break

            except Exception as e:
                msg = f"Attempt {attempt}: {type(e).__name__}: {e}"
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)
                else:
                    checkpoint[key] = {
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
                    }

        # Save checkpoint after every pair
        save_checkpoint(checkpoint)

        if i < total:
            time.sleep(SLEEP_BETWEEN)

    results_df = pd.DataFrame(list(checkpoint.values()))
    results_df.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
    print(f"\n  Comparisons complete GåÆ {RESULTS_CSV}")
    return results_df


# GöÇGöÇ Stage 2: Compute pairwise scores GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

def compute_pairwise_scores(results_df: pd.DataFrame, texts: list) -> pd.DataFrame:
    all_ids = [t[TEXT_ID_COL] for t in texts]

    win_counts = {tid: 0 for tid in all_ids}
    for _, row in results_df[results_df["winner_id"].notna()].iterrows():
        winner = row["winner_id"]
        if winner in win_counts:
            win_counts[winner] += 1

    max_wins = max(win_counts.values()) if win_counts else 1
    min_wins = min(win_counts.values()) if win_counts else 0

    def normalize(wins):
        if max_wins == min_wins:
            return 3.0
        return round(1.0 + (wins - min_wins) / (max_wins - min_wins) * 4.0, 2)

    title_map = {t[TEXT_ID_COL]: t.get("title", "") for t in texts}
    lix_map   = {t[TEXT_ID_COL]: t.get("lix", "") for t in texts}
    wc_map    = {t[TEXT_ID_COL]: t.get("word_count", "") for t in texts}

    rows = []
    for tid in all_ids:
        wins = win_counts.get(tid, 0)
        rows.append({
            TEXT_ID_COL:         tid,
            "title":             title_map.get(tid, ""),
            "wins":              wins,
            "total_comparisons": len(all_ids) - 1,
            "win_rate":          round(wins / max(len(all_ids) - 1, 1), 4),
            "pairwise_score":    normalize(wins),
            "lix":               lix_map.get(tid, ""),
            "word_count":        wc_map.get(tid, ""),
        })

    scores_df = pd.DataFrame(rows).sort_values("pairwise_score", ascending=False)
    scores_df.to_csv(PAIRWISE_SCORES, index=False, encoding="utf-8-sig")

    print(f"\n{'GöÇ'*55}")
    print(f"  PAIRWISE SCORE DISTRIBUTION")
    print(f"{'GöÇ'*55}")
    print(f"  Mean:  {scores_df['pairwise_score'].mean():.2f}")
    print(f"  Std:   {scores_df['pairwise_score'].std():.2f}")
    print(f"  Min:   {scores_df['pairwise_score'].min():.2f}")
    print(f"  Max:   {scores_df['pairwise_score'].max():.2f}")
    print(f"\n  Saved GåÆ {PAIRWISE_SCORES}")
    return scores_df


# GöÇGöÇ Stage 3: Combine anchor + pairwise GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

def combine_scores(anchor_df: pd.DataFrame, pairwise_df: pd.DataFrame) -> pd.DataFrame:
    anchor_slim = anchor_df[[TEXT_ID_COL, "title", "difficulty",
                              "lix", "word_count"]].copy()
    anchor_slim.columns = [TEXT_ID_COL, "title", "anchor_score",
                            "lix", "word_count"]
    anchor_slim["anchor_score"] = pd.to_numeric(
        anchor_slim["anchor_score"], errors="coerce"
    )

    pairwise_slim = pairwise_df[[TEXT_ID_COL, "pairwise_score",
                                  "wins", "win_rate"]].copy()
    pairwise_slim["pairwise_score"] = pd.to_numeric(
        pairwise_slim["pairwise_score"], errors="coerce"
    )

    merged = anchor_slim.merge(pairwise_slim, on=TEXT_ID_COL, how="left")

    # Normalize anchor score to 1-5 range for fair combination
    a_min = merged["anchor_score"].min()
    a_max = merged["anchor_score"].max()
    merged["anchor_score_norm"] = (
        1.0 + (merged["anchor_score"] - a_min) / (a_max - a_min) * 4.0
    ).round(2)

    # Final difficulty = average of both methods
    # Both reported separately so weighting can be adjusted later
    merged["final_difficulty"] = (0.35 * merged["anchor_score_norm"] +0.65 * merged["pairwise_score"]).round(2)

    # Flag short texts as unreliable
    merged["word_count"] = pd.to_numeric(merged["word_count"], errors="coerce")
    merged["reliable"]   = merged["word_count"] >= 100

    merged = merged.sort_values("final_difficulty").reset_index(drop=True)
    merged["difficulty_rank"] = range(1, len(merged) + 1)

    merged.to_csv(FINAL_SCORES_CSV, index=False, encoding="utf-8-sig")

    print(f"\n{'GöÇ'*55}")
    print(f"  FINAL DIFFICULTY DISTRIBUTION")
    print(f"{'GöÇ'*55}")
    print(f"  Mean:       {merged['final_difficulty'].mean():.2f}")
    print(f"  Std:        {merged['final_difficulty'].std():.2f}")
    print(f"  Min:        {merged['final_difficulty'].min():.2f}")
    print(f"  Max:        {merged['final_difficulty'].max():.2f}")
    print(f"  Reliable:   {merged['reliable'].sum()} texts")
    print(f"  Flagged:    {(~merged['reliable']).sum()} texts (< 100 words)")
    print(f"\n  Saved GåÆ {FINAL_SCORES_CSV}")
    return merged


# GöÇGöÇ Kendall's tau GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

def compute_kendall(merged: pd.DataFrame) -> dict:
    valid = merged[
        merged["anchor_score_norm"].notna() &
        merged["pairwise_score"].notna() &
        merged["reliable"]
    ].copy()

    tau, p_value = scipy_stats.kendalltau(
        valid["anchor_score_norm"],
        valid["pairwise_score"]
    )

    metrics = {
        "n_texts":        len(valid),
        "kendall_tau":    round(float(tau), 4),
        "p_value":        round(float(p_value), 6),
        "interpretation": (
            "Strong agreement"   if abs(tau) >= 0.7 else
            "Moderate agreement" if abs(tau) >= 0.5 else
            "Weak agreement"
        ),
    }

    print(f"\n{'GòÉ'*55}")
    print(f"  VALIDATION GÇö KENDALL'S -ä")
    print(f"{'GòÉ'*55}")
    print(f"  Texts (reliable only):  {metrics['n_texts']}")
    print(f"  Kendall's -ä:            {metrics['kendall_tau']}")
    print(f"  p-value:                {metrics['p_value']}")
    print(f"  Interpretation:         {metrics['interpretation']}")
    print(f"\n  -ä GëÑ 0.7 = strong agreement GÇö both methods consistent")
    print(f"  -ä GëÑ 0.5 = moderate agreement GÇö acceptable for thesis")
    print(f"  -ä < 0.5 = weak agreement GÇö investigate discrepancies")

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Saved GåÆ {METRICS_JSON}")
    return metrics


# GöÇGöÇ Main GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

def main():
    parser = argparse.ArgumentParser(description="Full pairwise difficulty pipeline.")
    parser.add_argument(
        "--scores-only", action="store_true",
        help="Skip comparisons, compute scores from existing pairwise_checkpoint.json.",
    )
    parser.add_argument(
        "--combine-only", action="store_true",
        help="Skip comparisons and scoring, just combine anchor + pairwise scores.",
    )
    args = parser.parse_args()

    load_dotenv(Path(__file__).parent.parent.parent / ".env")
    api_key = "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"
    if not api_key:
        print("Error: GROQ_API_KEY not set.")
        sys.exit(1)

    if not Path(ALL_TEXTS_CSV).exists():
        print(f"Error: {ALL_TEXTS_CSV} not found.")
        sys.exit(1)

    all_texts_df = pd.read_csv(ALL_TEXTS_CSV, dtype=str)
    texts        = all_texts_df.to_dict("records")
    print(f"  Loaded {len(texts)} texts")

    # GöÇGöÇ Run comparisons GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
    if args.combine_only:
        if not Path(PAIRWISE_SCORES).exists():
            print(f"Error: {PAIRWISE_SCORES} not found.")
            sys.exit(1)
        pairwise_scores_df = pd.read_csv(PAIRWISE_SCORES, dtype=str)
        pairwise_scores_df["pairwise_score"] = pd.to_numeric(
            pairwise_scores_df["pairwise_score"], errors="coerce"
        )
        print(f"  Loaded pairwise scores from {PAIRWISE_SCORES}")

    elif args.scores_only:
        if not Path(CHECKPOINT_JSON).exists():
            print(f"Error: {CHECKPOINT_JSON} not found.")
            sys.exit(1)
        checkpoint = load_checkpoint()
        results_df = pd.DataFrame(list(checkpoint.values()))
        results_df.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
        print(f"  Loaded {len(results_df)} results from checkpoint.")
        pairwise_scores_df = compute_pairwise_scores(results_df, texts)

    else:
        client             = Groq(api_key=api_key)
        results_df         = run_pairwise(client, texts)
        pairwise_scores_df = compute_pairwise_scores(results_df, texts)

    # GöÇGöÇ Combine GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
    if not Path(ANCHOR_SCORES_CSV).exists():
        print(f"Error: {ANCHOR_SCORES_CSV} not found.")
        print(f"Run difficulty_pipeline.py first.")
        sys.exit(1)

    anchor_df = pd.read_csv(ANCHOR_SCORES_CSV, dtype=str)
    print(f"  Loaded anchor scores from {ANCHOR_SCORES_CSV}")

    merged = combine_scores(anchor_df, pairwise_scores_df)
    compute_kendall(merged)

    print(f"\n  Done. Use final_difficulty_scores.csv for your recommendation system.")


if __name__ == "__main__":
    main()
