"""
difficulty_pipeline.py

Assigns difficulty scores (1-5) to all 157 Norwegian children's texts
using anchor-based LLM comparison, with optional pairwise validation.

Stages:
    1  Anchor comparison GÇö compare each text against 5 anchors (785 API calls)
    2  Pairwise validation GÇö compare 40-text sample pairwise (~780 API calls)
       then compute Kendall's tau between anchor-rank and pairwise-rank

Usage:
    # Run Stage 1 only (anchor comparison):
    python difficulty_pipeline.py

    # Run Stage 2 only (pairwise validation, after Stage 1 is complete):
    python difficulty_pipeline.py --validate-only

    # Re-run Stage 1 from scratch (ignores checkpoint):
    python difficulty_pipeline.py --fresh

Inputs:
    final_anchors_v2.csv          GÇö 5 selected anchor texts
    all_texts_lix_scores_v2.csv   GÇö all 157 texts with full text bodies

Outputs:
    difficulty_scores.csv         GÇö all 157 texts with difficulty 1-5
    difficulty_checkpoint.json    GÇö Stage 1 progress (auto-saved after each text)
    validation_sample.csv         GÇö 40-text sample used for pairwise validation
    pairwise_results.csv          GÇö all pairwise comparison results
    validation_metrics.json       GÇö Kendall's tau and agreement statistics
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

# GöÇGöÇ Config GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

ANCHORS_CSV      = "final_anchors_v2.csv"
ALL_TEXTS_CSV    = "all_texts_lix_scores_v2.csv"
SCORES_CSV       = "difficulty_scores_v2.csv"
CHECKPOINT_JSON  = "difficulty_checkpoint_v2.json"
VALIDATION_CSV   = "validation_sample.csv"
PAIRWISE_CSV     = "pairwise_results.csv"
METRICS_JSON     = "validation_metrics.json"

TEXT_ID_COL      = "sanity_text_id"

MODEL_NAME       = "llama-3.3-70b-versatile"
TEMPERATURE      = 0.1      # low temperature for more consistent scoring
MAX_RETRIES      = 3
RETRY_DELAY      = 5
SLEEP_BETWEEN    = 0.1      # paid tier GÇö 1K RPM allows fast calls

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

# GöÇGöÇ Prompts GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

ANCHOR_COMPARISON_SYSTEM = """
You are an expert in Norwegian language education and reading difficulty assessment
for children aged 9-11 (Norwegian primary school grades 4-6).

You will receive:
1. A target text to score
2. Five anchor texts representing difficulty levels 1 through 5

Your task:
- Read the target text carefully
- Compare it to all five anchor texts
- Assign a difficulty score from 1.0 to 5.0 with ONE decimal place

Difficulty scale:
  1.0 = Identical to or easier than the Level 1 anchor
  2.0 = Identical to the Level 2 anchor
  3.0 = Identical to the Level 3 anchor
  4.0 = Identical to the Level 4 anchor
  5.0 = Identical to or harder than the Level 5 anchor

CRITICAL GÇö USE DECIMALS:
  Most texts fall BETWEEN two anchor levels. You MUST use a decimal score.
  Examples:
    1.3 = slightly harder than anchor 1, clearly easier than anchor 2
    2.7 = clearly harder than anchor 2, approaching anchor 3
    3.5 = exactly halfway between anchor 3 and anchor 4
  Do NOT default to whole numbers (2.0, 3.0, 4.0) unless the text is
  genuinely indistinguishable from that anchor. Whole number scores
  should be rare GÇö most texts deserve a decimal.

When comparing, consider these dimensions in order of importance:
  1. vocab_difficulty     GÇö domain-specific, technical, or uncommon vocabulary
  2. background_knowledge GÇö prior knowledge a 9-11-year-old needs
  3. abstractness         GÇö how abstract vs concrete the concepts are
  4. sentence_complexity  GÇö sentence length and syntactic depth
  5. inferential_demand   GÇö how much the reader must infer

Rules:
  - Judge for a 9-11-year-old Norwegian reader, not an adult
  - A text about a familiar everyday topic (sports, animals, food) with
    simple vocabulary should score LOWER than a text about a technical
    or abstract topic, even if sentence lengths are similar
  - Do not confuse interesting or engaging topic with high difficulty
  - Very short texts (under 80 words) should score conservatively GÇö
    insufficient length makes difficulty hard to assess reliably

Return ONLY valid JSON. No markdown. No explanation outside the JSON.
{
  "difficulty": <float 1.0-5.0 with one decimal>,
  "lower_anchor": <integer, the anchor level just below this text>,
  "upper_anchor": <integer, the anchor level just above this text>,
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
        f"{'GöÇ'*50}\n"
        f"ANCHOR TEXTS (reference scale):\n"
        f"{anchor_section}"
    )


PAIRWISE_SYSTEM = """
You are an expert in Norwegian language education and reading difficulty assessment
for children aged 9-11 (Norwegian primary school grades 4-6).

You will receive two Norwegian texts. Your task is to decide which one is
harder to read and understand for a 9-11-year-old Norwegian child.

Consider:
  1. Vocabulary difficulty GÇö domain-specific or uncommon words
  2. Background knowledge required GÇö assumed familiarity with concepts
  3. Abstractness GÇö concrete vs abstract ideas
  4. Sentence complexity GÇö length and syntactic depth

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
        f"{'GöÇ'*50}\n\n"
        f"TEXT B:\n"
        f"Title: {text_b.get('title', '')}\n"
        f"{truncate(text_b)}\n\n"
        f"Which text is harder for a 9-11-year-old Norwegian child to read?"
    )


# GöÇGöÇ API helper GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

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


# GöÇGöÇ Stage 1: Anchor comparison GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

def run_anchor_comparison(client: Groq, texts: list[dict], anchors: list[dict]) -> list[dict]:
    checkpoint = load_checkpoint()
    results    = dict(checkpoint)
    remaining  = [t for t in texts if t[TEXT_ID_COL] not in results]
    total      = len(texts)
    done       = len(results)

    if done > 0:
        print(f"  Resuming: {done}/{total} done, {len(remaining)} remaining.")

    print(f"\n{'GöÇ'*62}")
    print(f"  Stage 1 GÇö Anchor comparison ({len(remaining)} remaining / {total} total)")
    print(f"{'GöÇ'*62}")

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

                difficulty = float(parsed.get("difficulty", 0))
                if not (1.0 <= difficulty <= 5.0):
                    raise ValueError(f"Invalid difficulty: {difficulty}")
                difficulty = round(difficulty, 1)

                results[text_id] = {
                    TEXT_ID_COL:      text_id,
                    "title":          title,
                    "difficulty":     difficulty,
                    "lower_anchor":   parsed.get("lower_anchor", ""),
                    "upper_anchor":   parsed.get("upper_anchor", ""),
                    "reasoning":      parsed.get("reasoning", ""),
                    "lix":            text.get("lix", ""),
                    "hybrid_score":   text.get("hybrid_score", ""),
                    "word_count":     text.get("word_count", ""),
                    "error":          None,
                }

                print(f"         difficulty={difficulty}/5.0  "
                      f"between anchors {parsed.get('lower_anchor','?')}-{parsed.get('upper_anchor','?')}")
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

        # Save after every text GÇö never lose progress on crash
        save_checkpoint(results)

        if i < total:
            time.sleep(SLEEP_BETWEEN)

    return list(results.values())


# GöÇGöÇ Stage 1 export GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

def export_scores(results: list[dict]):
    df = pd.DataFrame(results)
    df["difficulty"] = pd.to_numeric(df["difficulty"], errors="coerce")

    print(f"\n{'GöÇ'*50}")
    print(f"  DIFFICULTY DISTRIBUTION (1.0-5.0)")
    print(f"{'GöÇ'*50}")
    print(f"  Mean:    {df['difficulty'].mean():.2f}")
    print(f"  Std:     {df['difficulty'].std():.2f}")
    print(f"  Min:     {df['difficulty'].min():.1f}")
    print(f"  Max:     {df['difficulty'].max():.1f}")

    print(f"\n  Binned distribution:")
    bins   = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    labels = ["1.0-1.4","1.5-1.9","2.0-2.4","2.5-2.9",
              "3.0-3.4","3.5-3.9","4.0-4.4","4.5-5.0"]
    df["bin"] = pd.cut(df["difficulty"], bins=bins, labels=labels, include_lowest=True)
    dist = df["bin"].value_counts().sort_index()
    max_count = dist.max() if dist.max() > 0 else 1
    for label, count in dist.items():
        bar = "Gûê" * int(count * 30 / max_count)
        print(f"  {label}  {count:>4} texts  {bar}")

    errors = df[df["error"].notna()]
    if len(errors) > 0:
        print(f"\n  [!] {len(errors)} texts failed:")
        for _, row in errors.iterrows():
            print(f"      - {row['title'][:50]}")

    df.drop(columns=["bin"], errors="ignore").to_csv(
        SCORES_CSV, index=False, encoding="utf-8-sig"
    )
    print(f"\n  Saved GåÆ {SCORES_CSV}")


# GöÇGöÇ Stage 2: Pairwise validation GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

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

    print(f"\n{'GöÇ'*62}")
    print(f"  Stage 2 GÇö Pairwise validation ({len(sample_ids)} texts)")
    print(f"  Total comparisons: {len(sample_ids) * (len(sample_ids)-1) // 2}")
    print(f"{'GöÇ'*62}")

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


# GöÇGöÇ Kendall's tau GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

def compute_validation_metrics(scores_df: pd.DataFrame, pairwise_df: pd.DataFrame) -> dict:
    # Compute pairwise win counts GåÆ rank
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

    print(f"\n{'GòÉ'*62}")
    print(f"  VALIDATION METRICS")
    print(f"{'GòÉ'*62}")
    print(f"  Texts compared:     {metrics['n_texts']}")
    print(f"  Pairs evaluated:    {metrics['n_pairs']}")
    print(f"  Kendall's -ä:        {metrics['kendall_tau']}")
    print(f"  p-value:            {metrics['p_value']}")
    print(f"  Interpretation:     {metrics['interpretation']}")
    print(f"\n  A -ä GëÑ 0.7 indicates strong agreement between anchor-based")
    print(f"  and pairwise difficulty rankings.")

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Saved GåÆ {METRICS_JSON}")

    return metrics


# GöÇGöÇ Main GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

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

    # GöÇGöÇ Load inputs GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
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

    # GöÇGöÇ Stage 1 GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
    if args.validate_only:
        if not Path(SCORES_CSV).exists():
            print(f"Error: {SCORES_CSV} not found. Run Stage 1 first.")
            sys.exit(1)
        scores_df = pd.read_csv(SCORES_CSV, dtype=str)
        print(f"\n  Loaded existing scores from {SCORES_CSV} ({len(scores_df)} rows)")
    else:
        if args.fresh and Path(CHECKPOINT_JSON).exists():
            Path(CHECKPOINT_JSON).unlink()
            print("  Cleared checkpoint GÇö starting fresh.")

        results   = run_anchor_comparison(client, all_texts, anchors)
        scores_df = pd.DataFrame(results)
        export_scores(results)

        print(f"\n  Stage 1 complete.")
        print(f"  To run pairwise validation:")
        print(f"  python difficulty_pipeline.py --validate-only")

    # GöÇGöÇ Stage 2 GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
    if args.validate_only:
        pairwise_df = run_pairwise_validation(client, scores_df, all_texts)
        compute_validation_metrics(scores_df, pairwise_df)


if __name__ == "__main__":
    main()
