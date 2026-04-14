import re
import math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict

# =========================================================
# CONFIG
# =========================================================
INPUT_CSV = "../../data/raw/question_texts_texts.csv"
OUTPUT_DIR = ""
TEXT_ID_COL = "sanity_text_id"
TITLE_COL = "title"
BODY_COL = "body"

N_BANDS = 5
CANDIDATES_PER_BAND = 3

# Feature weights for preliminary difficulty score
WEIGHTS = {
    "avg_sentence_length_z": 0.25,
    "avg_word_length_z": 0.15,
    "long_word_ratio_z": 0.15,
    "lexical_diversity_z": 0.10,
    "rare_word_ratio_z": 0.15,
    "text_length_z": 0.10,
    "subordination_ratio_z": 0.05,
    "punctuation_complexity_z": 0.05,
}

# Very small Norwegian stopword starter set for lexical measures
NOR_STOPWORDS = {
    "og", "i", "på", "av", "for", "med", "til", "er", "som", "det", "de", "en",
    "et", "ei", "å", "at", "om", "den", "det", "har", "hadde", "var", "ble",
    "blir", "ikke", "så", "fra", "kan", "skal", "vil", "vi", "du", "han", "hun",
    "man", "seg", "sin", "sitt", "sine", "men", "eller", "også", "der", "her",
    "da", "når", "etter", "før", "over", "under", "ut", "inn"
}

# Some cue words that may hint at subordination / syntactic complexity
SUBORDINATION_CUES = {
    "fordi", "dersom", "hvis", "selv om", "som", "mens", "da", "når", "etter at",
    "før", "siden", "selvfølgelig", "selv om", "slik at"
}

# =========================================================
# HELPERS
# =========================================================
def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)

def clean_markdown(text: str) -> str:
    """Remove some markdown artifacts but keep readable content."""
    if pd.isna(text):
        return ""
    text = str(text)

    # Remove image markdown but keep alt text if available
    text = re.sub(r'!\[([^\]]*)\]\([^\)]*\)', r'\1', text)

    # Remove heading markers
    text = re.sub(r'^\s*#+\s*', '', text, flags=re.MULTILINE)

    # Remove markdown links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^\)]*\)', r'\1', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_sentences(text: str):
    """Simple sentence splitter."""
    if not text:
        return []
    sents = re.split(r'(?<=[.!?])\s+', text)
    sents = [s.strip() for s in sents if s.strip()]
    return sents

def tokenize(text: str):
    """Simple word tokenizer supporting Norwegian letters."""
    return re.findall(r"\b[a-zA-ZæøåÆØÅ]+\b", text.lower())

def safe_div(a, b):
    return a / b if b else 0.0

def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - series.mean()) / std

# =========================================================
# LOAD DATA
# =========================================================
ensure_dir(OUTPUT_DIR)

df = pd.read_csv(INPUT_CSV)

for col in [TITLE_COL, BODY_COL]:
    if col not in df.columns:
        raise ValueError(f"Missing expected column: {col}")

if TEXT_ID_COL not in df.columns:
    # fallback if ID column missing
    df[TEXT_ID_COL] = [f"text_{i+1}" for i in range(len(df))]

df[TITLE_COL] = df[TITLE_COL].fillna("").astype(str)
df[BODY_COL] = df[BODY_COL].fillna("").astype(str)

df["full_text_raw"] = (df[TITLE_COL] + " " + df[BODY_COL]).str.strip()
df["full_text"] = df["full_text_raw"].apply(clean_markdown)

# =========================================================
# FIRST PASS: TOKEN/SENTENCE STATS
# =========================================================
all_doc_tokens = []
doc_freq = Counter()

feature_rows = []

for idx, row in df.iterrows():
    text = row["full_text"]
    sents = split_sentences(text)
    tokens = tokenize(text)

    # content tokens for some measures
    content_tokens = [t for t in tokens if t not in NOR_STOPWORDS]

    # update doc frequency
    unique_content = set(content_tokens)
    for tok in unique_content:
        doc_freq[tok] += 1

    all_doc_tokens.append(content_tokens)

# =========================================================
# SECOND PASS: FEATURES
# =========================================================
for idx, row in df.iterrows():
    text = row["full_text"]
    title = row[TITLE_COL]
    sents = split_sentences(text)
    tokens = tokenize(text)
    content_tokens = [t for t in tokens if t not in NOR_STOPWORDS]

    word_count = len(tokens)
    sent_count = len(sents)
    char_count = len(text)

    avg_sentence_length = safe_div(word_count, sent_count)

    if word_count > 0:
        avg_word_length = np.mean([len(t) for t in tokens])
        long_word_ratio = np.mean([1 if len(t) >= 7 else 0 for t in tokens])
        lexical_diversity = len(set(content_tokens)) / max(len(content_tokens), 1)
    else:
        avg_word_length = 0
        long_word_ratio = 0
        lexical_diversity = 0

    # rare words defined relative to this corpus
    rare_words = [t for t in content_tokens if doc_freq[t] <= 2]
    rare_word_ratio = safe_div(len(rare_words), len(content_tokens))

    # subordination / complexity proxy
    lowered = text.lower()
    subordination_hits = sum(lowered.count(cue) for cue in SUBORDINATION_CUES)
    subordination_ratio = safe_div(subordination_hits, sent_count)

    # punctuation complexity proxy
    punctuation_marks = re.findall(r"[,;:()«»\-]", text)
    punctuation_complexity = safe_div(len(punctuation_marks), sent_count)

    # formatting / structure signals
    paragraph_count = len([p for p in re.split(r'\n\s*\n', row["full_text_raw"]) if str(p).strip()])
    heading_count = len(re.findall(r'^\s*#+\s+', str(row["full_text_raw"]), flags=re.MULTILINE))
    image_ref_count = len(re.findall(r'!\[.*?\]\(.*?\)', str(row["full_text_raw"])))

    feature_rows.append({
        TEXT_ID_COL: row[TEXT_ID_COL],
        "title": title,
        "full_text": text,
        "word_count": word_count,
        "sent_count": sent_count,
        "char_count": char_count,
        "avg_sentence_length": avg_sentence_length,
        "avg_word_length": avg_word_length,
        "long_word_ratio": long_word_ratio,
        "lexical_diversity": lexical_diversity,
        "rare_word_ratio": rare_word_ratio,
        "subordination_ratio": subordination_ratio,
        "punctuation_complexity": punctuation_complexity,
        "paragraph_count": paragraph_count,
        "heading_count": heading_count,
        "image_ref_count": image_ref_count,
    })

feat_df = pd.DataFrame(feature_rows)

# =========================================================
# PRELIMINARY DIFFICULTY SCORE
# =========================================================
# z-score selected features
feat_df["avg_sentence_length_z"] = zscore(feat_df["avg_sentence_length"])
feat_df["avg_word_length_z"] = zscore(feat_df["avg_word_length"])
feat_df["long_word_ratio_z"] = zscore(feat_df["long_word_ratio"])
feat_df["lexical_diversity_z"] = zscore(feat_df["lexical_diversity"])
feat_df["rare_word_ratio_z"] = zscore(feat_df["rare_word_ratio"])
feat_df["text_length_z"] = zscore(feat_df["word_count"])
feat_df["subordination_ratio_z"] = zscore(feat_df["subordination_ratio"])
feat_df["punctuation_complexity_z"] = zscore(feat_df["punctuation_complexity"])

# weighted sum
feat_df["preliminary_difficulty_score"] = 0.0
for feature_name, weight in WEIGHTS.items():
    feat_df["preliminary_difficulty_score"] += feat_df[feature_name] * weight

# rank from easiest to hardest
feat_df = feat_df.sort_values("preliminary_difficulty_score").reset_index(drop=True)
feat_df["difficulty_rank"] = np.arange(1, len(feat_df) + 1)

# =========================================================
# SPLIT INTO 5 BANDS
# =========================================================
# qcut gives roughly equal-sized bands
feat_df["difficulty_band"] = pd.qcut(
    feat_df["preliminary_difficulty_score"],
    q=N_BANDS,
    labels=[f"Band_{i}" for i in range(1, N_BANDS + 1)]
)

# Add readable label meaning
band_name_map = {
    "Band_1": "Very Easy",
    "Band_2": "Easy",
    "Band_3": "Medium",
    "Band_4": "Hard",
    "Band_5": "Very Hard",
}
feat_df["difficulty_band_label"] = feat_df["difficulty_band"].map(band_name_map)

# =========================================================
# SHORTLIST 3 CANDIDATES PER BAND
# =========================================================
# Strategy:
# choose texts nearest the median score of each band
# that avoids picking only the most extreme text
shortlist_rows = []

for band in [f"Band_{i}" for i in range(1, N_BANDS + 1)]:
    band_df = feat_df[feat_df["difficulty_band"] == band].copy()

    if band_df.empty:
        continue

    median_score = band_df["preliminary_difficulty_score"].median()
    band_df["distance_to_band_median"] = (band_df["preliminary_difficulty_score"] - median_score).abs()

    # optional filter to avoid very short texts
    # keep texts with at least 40 words, unless too few remain
    filtered = band_df[band_df["word_count"] >= 40].copy()
    if len(filtered) < CANDIDATES_PER_BAND:
        filtered = band_df.copy()

    candidates = filtered.sort_values("distance_to_band_median").head(CANDIDATES_PER_BAND)

    candidate_rank = 1
    for _, r in candidates.iterrows():
        preview = " ".join(r["full_text"].split()[:80])

        shortlist_rows.append({
            "difficulty_band": band,
            "difficulty_band_label": band_name_map[band],
            "candidate_rank_within_band": candidate_rank,
            TEXT_ID_COL: r[TEXT_ID_COL],
            "title": r["title"],
            "preliminary_difficulty_score": round(r["preliminary_difficulty_score"], 4),
            "difficulty_rank": int(r["difficulty_rank"]),
            "word_count": int(r["word_count"]),
            "avg_sentence_length": round(r["avg_sentence_length"], 2),
            "avg_word_length": round(r["avg_word_length"], 2),
            "long_word_ratio": round(r["long_word_ratio"], 4),
            "lexical_diversity": round(r["lexical_diversity"], 4),
            "rare_word_ratio": round(r["rare_word_ratio"], 4),
            "subordination_ratio": round(r["subordination_ratio"], 4),
            "punctuation_complexity": round(r["punctuation_complexity"], 4),
            "heading_count": int(r["heading_count"]),
            "image_ref_count": int(r["image_ref_count"]),
            "text_preview": preview,

            # manual review columns for Step E
            "manual_vocab_difficulty_1to5": "",
            "manual_sentence_complexity_1to5": "",
            "manual_abstractness_1to5": "",
            "manual_background_knowledge_1to5": "",
            "manual_cohesion_structure_1to5": "",
            "manual_inferential_demand_1to5": "",
            "manual_notes": "",
            "final_anchor_pick_yes_no": "",
        })
        candidate_rank += 1

shortlist_df = pd.DataFrame(shortlist_rows)

# =========================================================
# SAVE OUTPUTS
# =========================================================
all_scores_path = Path(OUTPUT_DIR) / "all_texts_preliminary_difficulty_scores.csv"
shortlist_path = Path(OUTPUT_DIR) / "anchor_candidates_shortlist.csv"

feat_df.to_csv(all_scores_path, index=False, encoding="utf-8-sig")
shortlist_df.to_csv(shortlist_path, index=False, encoding="utf-8-sig")

print(f"Saved full scored file to: {all_scores_path}")
print(f"Saved shortlist file to: {shortlist_path}")

# =========================================================
# OPTIONAL: BUILD FINAL ANCHOR FILE AFTER MANUAL REVIEW
# =========================================================
# After you fill `final_anchor_pick_yes_no` in the shortlist CSV,
# run the block below to create final anchors.
# Uncomment after manual review.

"""
reviewed = pd.read_csv(shortlist_path)

picked = reviewed[
    reviewed["final_anchor_pick_yes_no"].astype(str).str.strip().str.lower().isin(["yes", "y", "1", "true"])
].copy()

# Ensure only one per band
counts = picked.groupby("difficulty_band").size()
bad_bands = counts[counts != 1]

if not bad_bands.empty:
    print("Each band must have exactly ONE final anchor.")
    print(bad_bands)
else:
    final_anchors_path = Path(OUTPUT_DIR) / "final_anchors.csv"
    picked.to_csv(final_anchors_path, index=False, encoding="utf-8-sig")
    print(f"Saved final anchors to: {final_anchors_path}")
"""