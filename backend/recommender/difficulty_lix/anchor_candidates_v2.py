"""
anchor_candidates_v2.py

Preliminary difficulty scoring for Norwegian children's texts.
Generates anchor candidates using a hybrid score combining:
  - LIX (70%) — validated for Norwegian by Björnsson (1968) and Wold et al. (2024)
  - Supplementary NLP features (30%) — addressing LIX's known limitations
    for Norwegian compound words and content-side vocabulary difficulty

Key design decisions:
  - avg_sentence_length and long_word_ratio are EXCLUDED from the supplementary
    score to avoid double-counting with LIX, which already captures both
  - Band assignment uses pd.qcut (corpus-relative equal bands) because all texts
    target the same age group (8-12) and fall within the children's literature
    LIX range (µ=21.57, Wold et al. 2024) — absolute thresholds designed for
    the full Norwegian text spectrum are not applicable within this corpus
  - 3 candidates per band are selected nearest the band median (not extremes)
    to ensure candidates are typical representatives of their band

References:
  Björnsson, C-H. (1968). Läsbarhetsindex. Bokförlaget Liber.
  Wold, S., Mæhlum, P., & Hove, O. (2024). Estimating Lexical Complexity
    from Document-Level Distributions. University of Oslo.
  Dale, E. & Chall, J.S. (1948). A formula for predicting readability.
    Educational Research Bulletin.
  McNamara, D.S. et al. (2014). Automated Evaluation of Text and Discourse
    with Coh-Metrix. Cambridge University Press.
  Vajjala, S. & Meurers, D. (2012). On improving the accuracy of readability
    classification using insights from second language acquisition.
    Proceedings of BEA7.

Usage:
    pip install pandas numpy
    python anchor_candidates_v2.py

Inputs:
    question_texts_texts.csv   — raw corpus CSV

Outputs:
    all_texts_lix_scores_v2.csv         — LIX + hybrid scores for all 157 texts
    anchor_candidates_shortlist_v2.csv  — 15 candidates (3 per band) for LLM evaluation
"""

import re
import math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_CSV   = "../../data/raw/question_texts_texts.csv"
OUTPUT_DIR  = ""

TEXT_ID_COL = "sanity_text_id"
TITLE_COL   = "title"
BODY_COL    = "body"

N_BANDS              = 5
CANDIDATES_PER_BAND  = 3
MIN_WORD_COUNT       = 40   # exclude very short texts from candidacy

# Hybrid score weights
LIX_WEIGHT           = 0.70
SUPPLEMENTARY_WEIGHT = 0.30

# Supplementary feature weights (must sum to 1.0)
# These cover dimensions LIX does not capture:
#   - lexical_diversity: vocabulary richness (McNamara et al. 2014)
#   - rare_word_ratio: unfamiliar vocabulary (Dale & Chall 1948)
#   - subordination_ratio: syntactic complexity (Vajjala & Meurers 2012)
#   - punctuation_complexity: structural complexity proxy
SUPPLEMENTARY_WEIGHTS = {
    "lexical_diversity_z":    0.35,
    "rare_word_ratio_z":      0.35,
    "subordination_ratio_z":  0.15,
    "punctuation_complexity_z": 0.15,
}

BAND_LABELS = {
    "Band_1": "Very Easy",
    "Band_2": "Easy",
    "Band_3": "Medium",
    "Band_4": "Hard",
    "Band_5": "Very Hard",
}

# Norwegian stopwords for lexical measures
NOR_STOPWORDS = {
    "og", "i", "på", "av", "for", "med", "til", "er", "som", "det", "de",
    "en", "et", "ei", "å", "at", "om", "den", "har", "hadde", "var", "ble",
    "blir", "ikke", "så", "fra", "kan", "skal", "vil", "vi", "du", "han",
    "hun", "man", "seg", "sin", "sitt", "sine", "men", "eller", "også",
    "der", "her", "da", "når", "etter", "før", "over", "under", "ut", "inn",
}

# Subordination cue words (proxy for syntactic complexity)
# Vajjala & Meurers (2012): subordinate clauses as readability predictor
SUBORDINATION_CUES = {
    "fordi", "dersom", "hvis", "selv om", "mens", "da", "når",
    "etter at", "siden", "slik at", "selv om",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def clean_markdown(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("Ã¸", "ø").replace("Ã˜", "Ø")
    text = text.replace("Ã¥", "å").replace("Ã…", "Å")
    text = text.replace("Ã¦", "æ").replace("Ã†", "Æ")
    text = re.sub(r'!\[([^\]]*)\]\([^\)]*\)', r'\1', text)
    text = re.sub(r'^\s*#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[([^\]]+)\]\([^\)]*\)', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def split_sentences(text):
    if not text:
        return []
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if s.strip()]


def tokenize(text):
    return re.findall(r"\b[a-zA-ZæøåÆØÅ]+\b", text.lower())


def safe_div(a, b):
    return a / b if b else 0.0


def zscore(series):
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - series.mean()) / std


# ── Load data ─────────────────────────────────────────────────────────────────

ensure_dir(OUTPUT_DIR)

df = pd.read_csv(INPUT_CSV)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

for col in [TITLE_COL, BODY_COL]:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

if TEXT_ID_COL not in df.columns:
    df[TEXT_ID_COL] = [f"text_{i+1}" for i in range(len(df))]

df[TITLE_COL] = df[TITLE_COL].fillna("").astype(str)
df[BODY_COL]  = df[BODY_COL].fillna("").astype(str)

df["full_text_raw"] = (df[TITLE_COL] + " " + df[BODY_COL]).str.strip()
df["full_text"]     = df["full_text_raw"].apply(clean_markdown)

print(f"Loaded {len(df)} texts.")

# ── Build corpus-wide doc frequency for rare word ratio ───────────────────────

doc_freq = Counter()
all_content_tokens = []

for _, row in df.iterrows():
    tokens         = tokenize(row["full_text"])
    content_tokens = [t for t in tokens if t not in NOR_STOPWORDS]
    unique_content = set(content_tokens)
    for tok in unique_content:
        doc_freq[tok] += 1
    all_content_tokens.append(content_tokens)

# ── Compute features ──────────────────────────────────────────────────────────

rows = []

for i, (_, row) in enumerate(df.iterrows()):
    text   = row["full_text"]
    sents  = split_sentences(text)
    tokens = tokenize(text)
    content_tokens = [t for t in tokens if t not in NOR_STOPWORDS]

    word_count = len(tokens)
    sent_count = max(len(sents), 1)

    # ── LIX components ────────────────────────────────────────────────────────
    # Björnsson (1968): LIX = (words/sentences) + (long_words×100/words)
    # "long words" defined as words with more than 6 letters (Wold et al. 2024)
    avg_sentence_length = safe_div(word_count, sent_count)
    long_word_count     = sum(1 for t in tokens if len(t) > 6)
    long_word_ratio     = safe_div(long_word_count, word_count)
    lix                 = avg_sentence_length + (long_word_ratio * 100)

    # ── Supplementary features ────────────────────────────────────────────────
    # These cover dimensions LIX does not capture.
    # avg_sentence_length and long_word_ratio are deliberately EXCLUDED
    # to avoid double-counting with LIX.

    # Lexical diversity (type-token ratio on content words)
    # McNamara et al. (2014): vocabulary richness as readability predictor
    lexical_diversity = safe_div(
        len(set(content_tokens)), max(len(content_tokens), 1)
    )

    # Rare word ratio: words appearing in ≤2 corpus documents
    # Conceptually grounded in Dale & Chall (1948) unfamiliar word ratio
    rare_words    = [t for t in content_tokens if doc_freq[t] <= 2]
    rare_word_ratio = safe_div(len(rare_words), max(len(content_tokens), 1))

    # Subordination ratio: subordinate clause cue words per sentence
    # Vajjala & Meurers (2012): syntactic complexity as readability predictor
    lowered = text.lower()
    subordination_hits  = sum(lowered.count(cue) for cue in SUBORDINATION_CUES)
    subordination_ratio = safe_div(subordination_hits, sent_count)

    # Punctuation complexity: punctuation marks per sentence
    punct_marks         = re.findall(r"[,;:()«»\-]", text)
    punctuation_complexity = safe_div(len(punct_marks), sent_count)

    # ── Structural metadata ───────────────────────────────────────────────────
    paragraph_count = len([
        p for p in re.split(r'\n\s*\n', str(row["full_text_raw"])) if str(p).strip()
    ])
    heading_count   = len(re.findall(
        r'^\s*#+\s+', str(row["full_text_raw"]), flags=re.MULTILINE
    ))
    image_ref_count = len(re.findall(r'!\[.*?\]\(.*?\)', str(row["full_text_raw"])))

    rows.append({
        TEXT_ID_COL:             row[TEXT_ID_COL],
        "title":                 row[TITLE_COL],
        "full_text":             text,
        "word_count":            word_count,
        "sent_count":            sent_count,
        "long_word_count":       long_word_count,
        # LIX components
        "avg_sentence_length":   round(avg_sentence_length, 4),
        "long_word_ratio":       round(long_word_ratio, 4),
        "lix":                   round(lix, 4),
        # Supplementary features (not in LIX)
        "lexical_diversity":     round(lexical_diversity, 4),
        "rare_word_ratio":       round(rare_word_ratio, 4),
        "subordination_ratio":   round(subordination_ratio, 4),
        "punctuation_complexity": round(punctuation_complexity, 4),
        # Metadata
        "paragraph_count":       paragraph_count,
        "heading_count":         heading_count,
        "image_ref_count":       image_ref_count,
    })

feat_df = pd.DataFrame(rows)

# ── Hybrid score ──────────────────────────────────────────────────────────────

# Step 1: z-score LIX
feat_df["lix_z"] = zscore(feat_df["lix"])

# Step 2: z-score supplementary features
for feat in SUPPLEMENTARY_WEIGHTS:
    base = feat.replace("_z", "")
    feat_df[feat] = zscore(feat_df[base])

# Step 3: supplementary score (weighted sum of z-scored features)
feat_df["supplementary_score"] = sum(
    feat_df[feat] * weight
    for feat, weight in SUPPLEMENTARY_WEIGHTS.items()
)
feat_df["supplementary_score_z"] = zscore(feat_df["supplementary_score"])

# Step 4: hybrid score
# 70% LIX (Björnsson 1968, Wold et al. 2024) +
# 30% supplementary (addressing LIX's limitation for Norwegian compound words)
feat_df["hybrid_score"] = (
    LIX_WEIGHT * feat_df["lix_z"] +
    SUPPLEMENTARY_WEIGHT * feat_df["supplementary_score_z"]
)

# Step 5: rank and band
feat_df = feat_df.sort_values("hybrid_score").reset_index(drop=True)
feat_df["difficulty_rank"] = range(1, len(feat_df) + 1)

# Corpus-relative equal bands via pd.qcut
# Rationale: all texts target ages 8-12 and fall within the Norwegian
# children's literature LIX range (µ=21.57, Wold et al. 2024).
# Absolute thresholds designed for the full Norwegian text spectrum
# (children → parliament) are not applicable within this corpus.
feat_df["difficulty_band"] = pd.qcut(
    feat_df["hybrid_score"],
    q=N_BANDS,
    labels=[f"Band_{i}" for i in range(1, N_BANDS + 1)],
)
feat_df["difficulty_band_label"] = feat_df["difficulty_band"].map(BAND_LABELS)

# ── Print corpus LIX summary (reference to Wold et al. 2024) ─────────────────

print(f"\n{'─'*55}")
print(f"  CORPUS LIX SUMMARY")
print(f"  (Wold et al. 2024: Norwegian children's books µ=21.57)")
print(f"{'─'*55}")
print(f"  Mean LIX:    {feat_df['lix'].mean():.2f}")
print(f"  Std LIX:     {feat_df['lix'].std():.2f}")
print(f"  Min LIX:     {feat_df['lix'].min():.2f}")
print(f"  Max LIX:     {feat_df['lix'].max():.2f}")
print(f"\n  Band distribution:")
for band in [f"Band_{i}" for i in range(1, N_BANDS + 1)]:
    band_df  = feat_df[feat_df["difficulty_band"] == band]
    lix_mean = band_df["lix"].mean()
    print(f"    {band} ({BAND_LABELS[band]:<12}) "
          f"{len(band_df):>3} texts  "
          f"mean LIX={lix_mean:.1f}")

# ── Select candidates ─────────────────────────────────────────────────────────

shortlist_rows = []

for band in [f"Band_{i}" for i in range(1, N_BANDS + 1)]:
    band_df = feat_df[feat_df["difficulty_band"] == band].copy()

    if band_df.empty:
        print(f"  [!] No texts in {band}")
        continue

    median_score = band_df["hybrid_score"].median()
    band_df["distance_to_median"] = (
        band_df["hybrid_score"] - median_score
    ).abs()

    # Filter short texts from candidacy
    filtered = band_df[band_df["word_count"] >= MIN_WORD_COUNT].copy()
    if len(filtered) < CANDIDATES_PER_BAND:
        filtered = band_df.copy()

    candidates = filtered.sort_values("distance_to_median").head(CANDIDATES_PER_BAND)

    for rank, (_, r) in enumerate(candidates.iterrows(), 1):
        preview = " ".join(r["full_text"].split()[:80])
        shortlist_rows.append({
            "difficulty_band":              band,
            "difficulty_band_label":        BAND_LABELS[band],
            "candidate_rank_within_band":   rank,
            TEXT_ID_COL:                   r[TEXT_ID_COL],
            "title":                        r["title"],
            # LIX score (primary component, Björnsson 1968)
            "lix":                          round(r["lix"], 4),
            # Hybrid score
            "hybrid_score":                 round(r["hybrid_score"], 4),
            "difficulty_rank":              int(r["difficulty_rank"]),
            # Raw features for reference
            "word_count":                   int(r["word_count"]),
            "avg_sentence_length":          round(r["avg_sentence_length"], 2),
            "long_word_ratio":              round(r["long_word_ratio"], 4),
            "lexical_diversity":            round(r["lexical_diversity"], 4),
            "rare_word_ratio":              round(r["rare_word_ratio"], 4),
            "subordination_ratio":          round(r["subordination_ratio"], 4),
            "punctuation_complexity":       round(r["punctuation_complexity"], 4),
            "heading_count":                int(r["heading_count"]),
            "image_ref_count":              int(r["image_ref_count"]),
            "text_preview":                 preview,
            # LLM evaluation columns (filled by anchor_selector.py)
            "english_summary":              "",
            "estimated_age_fit":            "",
            "manual_vocab_difficulty_1to5": "",
            "manual_sentence_complexity_1to5": "",
            "manual_abstractness_1to5":     "",
            "manual_background_knowledge_1to5": "",
            "manual_cohesion_structure_1to5": "",
            "manual_inferential_demand_1to5": "",
            "overall_difficulty_1to5":      "",
            "overall_anchor_suitability_1to5": "",
            "manual_notes":                 "",
            "final_anchor_pick_yes_no":     "",
        })

shortlist_df = pd.DataFrame(shortlist_rows)

# ── Save ──────────────────────────────────────────────────────────────────────

all_scores_path = Path(OUTPUT_DIR) / "all_texts_lix_scores_v2.csv"
shortlist_path  = Path(OUTPUT_DIR) / "anchor_candidates_shortlist_v2.csv"

feat_df.to_csv(all_scores_path, index=False, encoding="utf-8-sig")
shortlist_df.to_csv(shortlist_path, index=False, encoding="utf-8-sig")

print(f"\n  Saved → {all_scores_path}")
print(f"  Saved → {shortlist_path}")
print(f"\n  Next step:")
print(f"  Run anchor_selector.py with --shortlist {shortlist_path}")

# ── Uncomment after manual/LLM review to finalize anchors ────────────────────
"""
reviewed = pd.read_csv(shortlist_path)
picked   = reviewed[
    reviewed["final_anchor_pick_yes_no"]
    .astype(str).str.strip().str.lower()
    .isin(["yes", "y", "1", "true"])
].copy()

counts = picked.groupby("difficulty_band").size()
bad    = counts[counts != 1]
if not bad.empty:
    print("Each band must have exactly ONE final anchor.")
    print(bad)
else:
    out = Path(OUTPUT_DIR) / "final_anchors_v2.csv"
    picked.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Saved final anchors → {out}")
"""