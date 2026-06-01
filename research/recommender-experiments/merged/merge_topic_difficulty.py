"""
merge_topic_difficulty.py

Merges final difficulty scores into results.csv from the topic pipeline,
producing a single dataset with topics, text type, and difficulty for
use in the recommendation system.

Usage:
    python merge_difficulty.py

Inputs:
    results.csv                   GÇö from topic_pipeline.py
    final_difficulty_scores.csv   GÇö from pairwise_pipeline.py

Outputs:
    results_with_difficulty.csv   GÇö merged final dataset
"""

import pandas as pd
from pathlib import Path

# GöÇGöÇ Config GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

RESULTS_CSV    = "../topic_groq/results.csv"
DIFFICULTY_CSV = "../difficulty_lix_v2/final_difficulty_scores.csv"
OUTPUT_CSV     = "results_with_topic_difficulty.csv"

RESULTS_ID_COL    = "text_id"
DIFFICULTY_ID_COL = "sanity_text_id"

# GöÇGöÇ Load GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

for path in [RESULTS_CSV, DIFFICULTY_CSV]:
    if not Path(path).exists():
        print(f"Error: {path} not found.")
        exit(1)

results_df    = pd.read_csv(RESULTS_CSV, dtype=str)
difficulty_df = pd.read_csv(DIFFICULTY_CSV, dtype=str)

print(f"Loaded {len(results_df)} rows from {RESULTS_CSV}")
print(f"Loaded {len(difficulty_df)} rows from {DIFFICULTY_CSV}")

# GöÇGöÇ Select difficulty columns to merge GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

diff_cols = [
    DIFFICULTY_ID_COL,
    "anchor_score",         # raw anchor-based score (1.0-5.0)
    "pairwise_score",       # raw pairwise-based score (1.0-5.0)
    "final_difficulty",     # combined score 0.35+ùanchor + 0.65+ùpairwise
    "difficulty_rank",      # rank 1 (easiest) to 157 (hardest)
    "reliable",             # False if text < 100 words
    "wins",                 # raw pairwise win count
    "win_rate",             # wins / 156
]

# Keep only columns that exist
diff_cols     = [c for c in diff_cols if c in difficulty_df.columns]
difficulty_slim = difficulty_df[diff_cols].copy()

# GöÇGöÇ Merge GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

merged = results_df.merge(
    difficulty_slim,
    left_on=RESULTS_ID_COL,
    right_on=DIFFICULTY_ID_COL,
    how="left"
)

missing = merged["final_difficulty"].isna().sum()
if missing > 0:
    print(f"\n  [!] {missing} texts in results.csv have no difficulty score.")
    print(f"      These may be texts added after the difficulty pipeline ran.")
    print(f"      They will have empty difficulty columns in the output.")

# GöÇGöÇ Column order GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

# Put difficulty columns right after text_type for logical grouping
base_cols = [c for c in results_df.columns if c in merged.columns]
diff_only = [c for c in diff_cols if c != DIFFICULTY_ID_COL]

# Insert difficulty columns after text_type if it exists
if "text_type" in base_cols:
    insert_at = base_cols.index("text_type") + 1
    ordered   = base_cols[:insert_at] + diff_only + base_cols[insert_at:]
else:
    ordered   = base_cols + diff_only

# Remove any duplicates while preserving order
seen    = set()
ordered = [c for c in ordered if not (c in seen or seen.add(c))]

merged = merged[ordered]

# GöÇGöÇ Summary GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

merged["final_difficulty"] = pd.to_numeric(
    merged["final_difficulty"], errors="coerce"
)

print(f"\n{'GöÇ'*55}")
print(f"  MERGED DATASET SUMMARY")
print(f"{'GöÇ'*55}")
print(f"  Total texts:         {len(merged)}")
print(f"  With difficulty:     {merged['final_difficulty'].notna().sum()}")
print(f"  Missing difficulty:  {merged['final_difficulty'].isna().sum()}")
print(f"  Reliable scores:     {(merged['reliable'] == 'True').sum()}")
print(f"  Flagged short texts: {(merged['reliable'] == 'False').sum()}")
print(f"\n  Difficulty distribution:")
print(f"    Mean:  {merged['final_difficulty'].mean():.2f}")
print(f"    Std:   {merged['final_difficulty'].std():.2f}")
print(f"    Min:   {merged['final_difficulty'].min():.2f}")
print(f"    Max:   {merged['final_difficulty'].max():.2f}")

print(f"\n  Columns in output:")
for col in merged.columns:
    print(f"    {col}")

# GöÇGöÇ Save GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

merged.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n  Saved GåÆ {OUTPUT_CSV}")
print(f"\n  This file is ready for your recommendation system.")
print(f"  Key columns:")
print(f"    final_difficulty  GÇö use this for recommendation matching")
print(f"    difficulty_rank   GÇö use this for ranking within results")
print(f"    reliable          GÇö filter on True for high-confidence scores")
print(f"    broad_topics      GÇö use this for topic-based filtering")
print(f"    text_type         GÇö fagtekst or fortelling")
