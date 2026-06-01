import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats

FINAL_CSV      = "final_difficulty_scores.csv"
COMPARISON_CSV = "weighting_comparison.csv"
SUMMARY_CSV    = "weighting_summary.csv"

# G��G�� Weightings to compare G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��
# Format: (label, anchor_weight, pairwise_weight)
WEIGHTINGS = [
    ("pairwise_only",    0.00, 1.00),
    ("anchor_35_pw_65",  0.35, 0.65),
    ("equal_50_50",      0.50, 0.50),
    ("anchor_65_pw_35",  0.65, 0.35),
    ("anchor_only",      1.00, 0.00),
]

# G��G�� Load data G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��

if not Path(FINAL_CSV).exists():
    print(f"Error: {FINAL_CSV} not found. Run pairwise_pipeline.py first.")
    exit(1)

df = pd.read_csv(FINAL_CSV, dtype=str)
df["anchor_score_norm"] = pd.to_numeric(df["anchor_score_norm"], errors="coerce")
df["pairwise_score"]    = pd.to_numeric(df["pairwise_score"],    errors="coerce")
df["word_count"]        = pd.to_numeric(df["word_count"],        errors="coerce")
df["reliable"]          = df["word_count"] >= 100

reliable = df[df["reliable"]].copy()
print(f"Loaded {len(df)} texts ({len(reliable)} reliable, "
      f"{len(df) - len(reliable)} flagged short)")

# G��G�� Compute scores for each weighting G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��

result = df[["sanity_text_id", "title", "word_count", "reliable",
             "anchor_score", "anchor_score_norm", "pairwise_score",
             "lix"]].copy()

summary_rows = []

print(f"\n{'G��'*70}")
print(f"  WEIGHTING COMPARISON")
print(f"{'G��'*70}")
print(f"  {'Weighting':<22} {'Mean':>6} {'Std':>6} {'Min':>6} {'Max':>6} "
      f"{'-�(anchor)':>10} {'-�(pairwise)':>12}")
print(f"  {'G��'*68}")

for label, aw, pw in WEIGHTINGS:
    col = f"score_{label}"
    df[col] = (
        aw * df["anchor_score_norm"] +
        pw * df["pairwise_score"]
    ).round(2)

    result[col] = df[col]

    rel = df[df["reliable"]][col]

    # Kendall's tau vs anchor and vs pairwise
    tau_anchor,   _ = scipy_stats.kendalltau(
        df[df["reliable"]]["anchor_score_norm"], rel
    )
    tau_pairwise, _ = scipy_stats.kendalltau(
        df[df["reliable"]]["pairwise_score"], rel
    )

    # Spread G�� how many unique 0.5-step bins are used
    bins   = np.arange(1.0, 5.6, 0.5)
    binned = pd.cut(rel, bins=bins, include_lowest=True)
    n_bins_used = binned.nunique()

    print(f"  {label:<22} {rel.mean():>6.2f} {rel.std():>6.2f} "
          f"{rel.min():>6.2f} {rel.max():>6.2f} "
          f"{tau_anchor:>10.3f} {tau_pairwise:>12.3f}")

    summary_rows.append({
        "weighting":       label,
        "anchor_weight":   aw,
        "pairwise_weight": pw,
        "mean":            round(rel.mean(), 3),
        "std":             round(rel.std(), 3),
        "min":             round(rel.min(), 3),
        "max":             round(rel.max(), 3),
        "spread":          round(rel.max() - rel.min(), 3),
        "n_bins_used":     n_bins_used,
        "kendall_vs_anchor":   round(tau_anchor, 4),
        "kendall_vs_pairwise": round(tau_pairwise, 4),
    })

# G��G�� Distribution per weighting G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��

print(f"\n{'G��'*70}")
print(f"  DISTRIBUTION G�� reliable texts only (n={len(reliable)})")
print(f"{'G��'*70}")

bins   = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.1]
labels = ["1.0-1.4","1.5-1.9","2.0-2.4","2.5-2.9",
          "3.0-3.4","3.5-3.9","4.0-4.4","4.5-5.0"]

for label, aw, pw in WEIGHTINGS:
    col = f"score_{label}"
    print(f"\n  {label} (anchor={aw}, pairwise={pw}):")
    rel_scores = df[df["reliable"]][col]
    binned     = pd.cut(rel_scores, bins=bins, labels=labels, include_lowest=True)
    dist       = binned.value_counts().sort_index()
    max_count  = dist.max() if dist.max() > 0 else 1
    for b, count in dist.items():
        bar = "G��" * int(count * 25 / max_count)
        print(f"    {b}  {count:>4}  {bar}")

# G��G�� Kendall's tau between all pairs of weightings G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��

print(f"\n{'G��'*70}")
print(f"  KENDALL'S -� BETWEEN WEIGHTINGS (how similar are the rankings?)")
print(f"{'G��'*70}")

weighting_cols = [f"score_{label}" for label, _, _ in WEIGHTINGS]
weighting_names = [label for label, _, _ in WEIGHTINGS]

rel_df = df[df["reliable"]].copy()

print(f"  {'':22}", end="")
for name in weighting_names:
    print(f"  {name[:12]:>12}", end="")
print()
print(f"  {'G��'*68}")

for i, (name_i, col_i) in enumerate(zip(weighting_names, weighting_cols)):
    print(f"  {name_i:<22}", end="")
    for j, (name_j, col_j) in enumerate(zip(weighting_names, weighting_cols)):
        if i == j:
            print(f"  {'1.000':>12}", end="")
        else:
            tau, _ = scipy_stats.kendalltau(rel_df[col_i], rel_df[col_j])
            print(f"  {tau:>12.3f}", end="")
    print()

# G��G�� Rank correlation: top 10 and bottom 10 across weightings G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��

print(f"\n{'G��'*70}")
print(f"  TOP 10 HARDEST TEXTS G�� do weightings agree?")
print(f"{'G��'*70}")
print(f"  {'Title':<40}", end="")
for label, _, _ in WEIGHTINGS:
    print(f"  {label[:10]:>10}", end="")
print()
print(f"  {'G��'*68}")

# Rank each weighting
for label, _, _ in WEIGHTINGS:
    col      = f"score_{label}"
    rank_col = f"rank_{label}"
    df[rank_col] = df[col].rank(ascending=False, method="min")

# Show top 10 by pairwise_only
top10 = df[df["reliable"]].nsmallest(10, "rank_pairwise_only")
for _, row in top10.iterrows():
    title = str(row["title"])[:38]
    print(f"  {title:<40}", end="")
    for label, _, _ in WEIGHTINGS:
        print(f"  {row[f'score_{label}']:>10.2f}", end="")
    print()

print(f"\n{'G��'*70}")
print(f"  BOTTOM 10 EASIEST TEXTS G�� do weightings agree?")
print(f"{'G��'*70}")
print(f"  {'Title':<40}", end="")
for label, _, _ in WEIGHTINGS:
    print(f"  {label[:10]:>10}", end="")
print()
print(f"  {'G��'*68}")

bottom10 = df[df["reliable"]].nlargest(10, "rank_pairwise_only")
for _, row in bottom10.iterrows():
    title = str(row["title"])[:38]
    print(f"  {title:<40}", end="")
    for label, _, _ in WEIGHTINGS:
        print(f"  {row[f'score_{label}']:>10.2f}", end="")
    print()

# G��G�� Save outputs G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��

result_cols = (
    ["sanity_text_id", "title", "word_count", "reliable",
     "anchor_score", "anchor_score_norm", "pairwise_score", "lix"] +
    [f"score_{label}" for label, _, _ in WEIGHTINGS]
)
result[result_cols].to_csv(COMPARISON_CSV, index=False, encoding="utf-8-sig")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

print(f"\n  Saved G�� {COMPARISON_CSV}")
print(f"  Saved G�� {SUMMARY_CSV}")
print(f"\n  Recommendation: choose the weighting with:")
print(f"    - Highest spread (max - min)")
print(f"    - Most bins used")
print(f"    - Std closest to 1.0 (good spread without compression)")
