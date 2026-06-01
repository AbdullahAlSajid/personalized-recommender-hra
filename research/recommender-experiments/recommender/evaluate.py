"""
Offline Evaluation for the Text Recommender
============================================
Since no real interaction data exists yet (pilot system), this module
simulates complete sessions over synthetic student profiles and measures:

1. Difficulty MAE      GÇö how well recommendations match the student's level
2. Topic recall        GÇö fraction of interests covered across the session
3. Intra-slate diversity GÇö topic dissimilarity within each 2-text slate
4. Corpus coverage     GÇö what % of corpus texts are reachable
5. Comprehension proxy GÇö simulated comprehension vs difficulty gap
6. Adaptive convergence GÇö how quickly estimated_level approaches true_level

All metrics are computed over a grid of synthetic profiles spanning
the full parameter space (reading levels, interest combinations).

Run:  python evaluate.py results_with_topic_difficulty.csv
"""

from __future__ import annotations

import itertools
import random
from typing import List, Optional

import numpy as np
import pandas as pd

from recommender.engine import (
    BROAD_TOPICS, Corpus, SessionManager, ScoringEngine, SlateBuilder,
)


# GòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉ
# Synthetic profile generator
# GòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉ

def generate_profiles(
    n_profiles: int = 100,
    seed: int = 42,
    min_interests: int = 3,
    max_interests: int = 5,
) -> List[dict]:
    """Generate synthetic student profiles spanning the parameter space."""
    rng = random.Random(seed)
    profiles = []

    for i in range(n_profiles):
        true_level = round(rng.uniform(1.5, 4.5), 1)
        n_int = rng.randint(min_interests, max_interests)
        interests = rng.sample(BROAD_TOPICS, n_int)

        profiles.append({
            "profile_id": f"synth_{i:03d}",
            "true_level": true_level,
            "interests": interests,
        })

    return profiles


def simulate_perceived_difficulty(
    text_difficulty: float, true_level: float, rng: random.Random
) -> int:
    """Simulate a perceived difficulty rating based on gap."""
    gap = text_difficulty - true_level
    raw = 3 + gap * 1.5
    noisy = raw + rng.gauss(0, 0.3)
    return int(max(1, min(5, round(noisy))))


def simulate_comprehension(
    text_difficulty: float, true_level: float, rng: random.Random
) -> float:
    """Simulate comprehension score (0-1) based on gap."""
    gap = text_difficulty - true_level
    if gap <= 0:
        comp = min(1.0, 0.9 + rng.gauss(0, 0.05))
    elif gap < 0.8:
        comp = min(1.0, max(0.0, 0.75 - gap * 0.3 + rng.gauss(0, 0.05)))
    else:
        comp = min(1.0, max(0.0, 0.5 - gap * 0.2 + rng.gauss(0, 0.1)))
    return round(comp * 3) / 3


# GòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉ
# Single session simulation
# GòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉ

def simulate_session(
    corpus: Corpus,
    profile: dict,
    n_rounds: int = 6,
    seed: int = 0,
) -> dict:
    """
    Simulate a complete session for one synthetic profile.

    Returns a dict of per-session metrics.
    """
    rng = random.Random(seed)
    session = SessionManager(corpus, profile["interests"])
    true_level = profile["true_level"]

    # Track per-round metrics
    round_metrics = []
    level_trajectory = []
    all_recommended_ids = set()
    all_topics_seen = set()

    for rnd in range(1, n_rounds + 1):
        slate = session.get_recommendations()

        if slate.empty:
            break

        all_recommended_ids.update(slate["text_id"].tolist())

        # Track topics in this slate
        for _, row in slate.iterrows():
            all_topics_seen.update(row.get("broad_topics_list", []))

        # Intra-slate diversity (1 - Jaccard between the two texts' topics)
        if len(slate) == 2:
            t1 = set(slate.iloc[0].get("broad_topics_list", []))
            t2 = set(slate.iloc[1].get("broad_topics_list", []))
            if t1 | t2:
                slate_diversity = 1 - len(t1 & t2) / len(t1 | t2)
            else:
                slate_diversity = 0.0
        else:
            slate_diversity = np.nan

        # Student picks text 1 (highest score)
        chosen = slate.iloc[0]
        chosen_id = chosen["text_id"]
        chosen_diff = float(chosen["final_difficulty"])
        shown_ids = slate["text_id"].tolist()

        # Simulate feedback
        perceived = simulate_perceived_difficulty(chosen_diff, true_level, rng)
        comprehension = simulate_comprehension(chosen_diff, true_level, rng)
        interest = rng.choice([3, 4, 4, 5])

        # Difficulty match for this round
        diff_error = abs(chosen_diff - true_level)

        round_metrics.append({
            "round": rnd,
            "text_difficulty": chosen_diff,
            "difficulty_error": diff_error,
            "perceived_difficulty": perceived,
            "comprehension": comprehension,
            "interest": interest,
            "composite_score": float(chosen["composite_score"]),
            "slate_diversity": slate_diversity,
        })

        # Record in session
        session.record_reading(
            shown_text_ids=shown_ids,
            chosen_text_id=chosen_id,
            perceived_difficulty=perceived,
            interest_rating=interest,
            comprehension_score=comprehension,
        )

        level_trajectory.append(session.level_estimator.estimated_level)

    # GöÇGöÇ Aggregate session metrics GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

    n_completed = len(round_metrics)
    if n_completed == 0:
        return {"profile_id": profile["profile_id"], "n_completed": 0}

    rm = pd.DataFrame(round_metrics)

    # Topic recall: fraction of student interests seen in recommendations
    interest_set = set(profile["interests"])
    topic_recall = len(all_topics_seen & interest_set) / len(interest_set)

    # Level convergence: |estimated_level - true_level| at end
    final_level = level_trajectory[-1] if level_trajectory else None
    level_error = abs(final_level - true_level) if final_level else None

    # Difficulty MAE across rounds (excluding round 1 which has no difficulty signal)
    adaptive_rounds = rm[rm["round"] > 1]
    diff_mae_adaptive = (
        adaptive_rounds["difficulty_error"].mean()
        if len(adaptive_rounds) > 0 else np.nan
    )

    return {
        "profile_id": profile["profile_id"],
        "true_level": true_level,
        "n_interests": len(profile["interests"]),
        "n_completed": n_completed,
        "diff_mae_all": rm["difficulty_error"].mean(),
        "diff_mae_adaptive": diff_mae_adaptive,
        "diff_mae_round1": rm.iloc[0]["difficulty_error"],
        "topic_recall": topic_recall,
        "mean_slate_diversity": rm["slate_diversity"].mean(),
        "mean_comprehension": rm["comprehension"].mean(),
        "mean_interest": rm["interest"].mean(),
        "mean_composite_score": rm["composite_score"].mean(),
        "final_estimated_level": final_level,
        "level_convergence_error": level_error,
        "n_unique_texts": len(all_recommended_ids),
        "level_trajectory": level_trajectory,
    }


# GòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉ
# Full evaluation
# GòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉ

def evaluate(
    corpus: Corpus,
    n_profiles: int = 100,
    n_rounds: int = 6,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run offline evaluation over synthetic profiles.

    Returns DataFrame with one row per profile and all metrics.
    """
    profiles = generate_profiles(n_profiles, seed=seed)
    results = []

    all_recommended_ids = set()

    for i, profile in enumerate(profiles):
        result = simulate_session(
            corpus, profile, n_rounds=n_rounds, seed=seed + i
        )
        results.append(result)

        if result["n_completed"] > 0:
            # Collect for coverage
            session = SessionManager(corpus, profile["interests"])
            for _ in range(n_rounds):
                slate = session.get_recommendations()
                if slate.empty:
                    break
                all_recommended_ids.update(slate["text_id"].tolist())
                chosen_id = slate.iloc[0]["text_id"]
                shown_ids = slate["text_id"].tolist()
                session.record_reading(
                    shown_text_ids=shown_ids,
                    chosen_text_id=chosen_id,
                    perceived_difficulty=3,
                    interest_rating=4,
                    comprehension_score=0.67,
                )

    results_df = pd.DataFrame(results)

    # Corpus coverage
    reliable_ids = set(corpus.get_reliable_texts()["text_id"])
    coverage = len(all_recommended_ids & reliable_ids) / len(reliable_ids)

    if verbose:
        m = results_df[results_df["n_completed"] > 0]

        print("\n" + "=" * 62)
        print("  OFFLINE EVALUATION RESULTS")
        print("=" * 62)
        print(f"  Profiles evaluated        : {n_profiles}")
        print(f"  Rounds per session        : {n_rounds}")
        print(f"  Profiles with results     : {len(m)}")

        print(f"\n  GöÇGöÇ Corpus coverage GöÇGöÇ")
        print(f"  Texts reachable           : {coverage:.1%}  "
              f"({len(all_recommended_ids & reliable_ids)}/{len(reliable_ids)})")

        print(f"\n  GöÇGöÇ Difficulty matching GöÇGöÇ")
        print(f"  MAE (all rounds)          : {m['diff_mae_all'].mean():.3f}  "
              f"(-¦{m['diff_mae_all'].std():.3f})")
        print(f"  MAE (round 1 only)        : {m['diff_mae_round1'].mean():.3f}  "
              f"(-¦{m['diff_mae_round1'].std():.3f})  [no difficulty signal]")
        print(f"  MAE (rounds 2+, adaptive) : {m['diff_mae_adaptive'].mean():.3f}  "
              f"(-¦{m['diff_mae_adaptive'].std():.3f})  [with difficulty signal]")

        print(f"\n  GöÇGöÇ Topic matching GöÇGöÇ")
        print(f"  Topic recall              : {m['topic_recall'].mean():.3f}  "
              f"(-¦{m['topic_recall'].std():.3f})")

        print(f"\n  GöÇGöÇ Slate diversity GöÇGöÇ")
        print(f"  Mean intra-slate diversity: {m['mean_slate_diversity'].mean():.3f}  "
              f"(-¦{m['mean_slate_diversity'].std():.3f})  [1.0 = max variety]")

        print(f"\n  GöÇGöÇ Level convergence GöÇGöÇ")
        print(f"  Level error at session end: {m['level_convergence_error'].mean():.3f}  "
              f"(-¦{m['level_convergence_error'].std():.3f})")

        print(f"\n  GöÇGöÇ Simulated outcomes GöÇGöÇ")
        print(f"  Mean comprehension        : {m['mean_comprehension'].mean():.1%}")
        print(f"  Mean interest rating      : {m['mean_interest'].mean():.2f}/5")

        # Breakdown by true level band
        print(f"\n  GöÇGöÇ Difficulty MAE by true reading level GöÇGöÇ")
        m_copy = m.copy()
        m_copy["level_band"] = pd.cut(
            m_copy["true_level"],
            bins=[1.0, 2.0, 3.0, 4.0, 5.0],
            labels=["1-2 (lett)", "2-3 (middels)", "3-4 (vanskelig)", "4-5 (avansert)"],
        )
        band_stats = (
            m_copy.groupby("level_band", observed=True)
            .agg(
                MAE_all=("diff_mae_all", "mean"),
                MAE_adaptive=("diff_mae_adaptive", "mean"),
                Level_error=("level_convergence_error", "mean"),
                N=("profile_id", "count"),
            )
            .round(3)
        )
        print(band_stats.to_string(index=True))

        # Adaptive improvement: compare round 1 MAE vs rounds 2+ MAE
        improvement = m["diff_mae_round1"].mean() - m["diff_mae_adaptive"].mean()
        print(f"\n  GöÇGöÇ Adaptive improvement GöÇGöÇ")
        print(f"  MAE reduction (R1 GåÆ R2+)  : {improvement:.3f}  "
              f"({'improved' if improvement > 0 else 'no improvement'})")
        if improvement > 0:
            pct = improvement / m["diff_mae_round1"].mean() * 100
            print(f"  Relative improvement      : {pct:.1f}%")

        print("=" * 62)

    return results_df


if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "results_with_topic_difficulty.csv"
    corpus = Corpus(data_path)
    results = evaluate(corpus, n_profiles=200, n_rounds=6)
