"""
Personalized Text Recommender for Norwegian Children's Texts
============================================================
Master's Thesis — University of Stavanger
Target: Children aged 9–11
Corpus: 157 texts with topic labels and difficulty scores (1.0–5.0)

Architecture
------------
Session-based content recommender with two scoring components:
  1. Topic match  — recall overlap between student interests and text broad_topics
  2. Difficulty match — Gaussian kernel targeting the student's growth zone

Weights transition across rounds as difficulty signal accumulates:
  Round 1:  topic 1.00 | difficulty 0.00  (cold start, no signal)
  Round 2:  topic 0.70 | difficulty 0.30  (one rating, still noisy)
  Round 3+: topic 0.50 | difficulty 0.50  (two+ ratings, estimate stabilises)

Slate diversity uses MMR (Carbonell & Goldstein, 1998) with broad_topic
and sub_topic overlap to ensure the 2 shown texts cover different themes.

Reading level is estimated from perceived difficulty ratings:
  implied_level = text_difficulty + (3 - perceived_difficulty) × 0.5
  estimated_level = mean(implied_levels)

Research grounding
------------------
- Content-based cold start:  Adomavicius & Tuzhilin (2005)
- Interest + difficulty:     Hsu, Hwang & Chang (2013), Wu & Huang (2023)
- ZPD growth-zone targeting: Vygotsky (1978)
- MMR diversity:             Carbonell & Goldstein (1998), Ziegler et al. (2005)
- Adaptive level update:     Corbett & Anderson (1995)
- Children's reading RS:     Kucirkova (2019, UiS), Walgermo et al. (2024, UiS)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════

BROAD_TOPICS: List[str] = [
    "Dyr", "Vitenskap", "Natur", "Kultur", "Historie", "Teknologi",
    "Idrett", "Mat", "Helse", "Samfunn", "Kunst", "Matematikk", "Fortelling",
]

# Weight schedules per round
WEIGHT_SCHEDULE: Dict[int, Tuple[float, float]] = {
    1: (1.00, 0.00),   # cold start — topic only
    2: (0.70, 0.30),   # one difficulty rating — topic dominant
    3: (0.50, 0.50),   # two+ ratings — balanced
}


def _get_weights(round_number: int) -> Tuple[float, float]:
    """Return (w_topic, w_difficulty) for the given round."""
    if round_number <= 1:
        return WEIGHT_SCHEDULE[1]
    elif round_number == 2:
        return WEIGHT_SCHEDULE[2]
    else:
        return WEIGHT_SCHEDULE[3]


# ════════════════════════════════════════════════════════════
# Corpus Loader
# ════════════════════════════════════════════════════════════

class Corpus:
    """
    Loads and prepares the merged dataset (results_with_topic_difficulty.csv).
    Parses pipe-separated topic fields into lists and normalises types.
    """

    def __init__(self, data_path: str):
        df = pd.read_csv(data_path, encoding="utf-8-sig")

        # Parse pipe-separated topic strings into lists
        df["broad_topics_list"] = df["broad_topics"].apply(self._parse_pipe)
        df["sub_topics_list"] = df["sub_topics"].apply(self._parse_pipe)

        # Normalise reliable column
        if "reliable" in df.columns:
            df["reliable"] = df["reliable"].apply(
                lambda x: str(x).strip().lower() == "true" if pd.notna(x) else True
            )
        else:
            df["reliable"] = True

        # Ensure final_difficulty is numeric
        df["final_difficulty"] = pd.to_numeric(df["final_difficulty"], errors="coerce")

        self.df = df

        n_total = len(df)
        n_reliable = int(df["reliable"].sum())
        print(f"[Corpus] Loaded {n_total} texts ({n_reliable} reliable, "
              f"{n_total - n_reliable} flagged unreliable).")

    @staticmethod
    def _parse_pipe(val) -> List[str]:
        if pd.isna(val):
            return []
        return [t.strip() for t in str(val).split("|") if t.strip()]

    def get_reliable_texts(self) -> pd.DataFrame:
        """Return only texts flagged as reliable."""
        return self.df[self.df["reliable"] == True].copy()

    def stats(self) -> dict:
        """Summary statistics about the corpus."""
        df = self.df
        topic_counts = {}
        for topic in BROAD_TOPICS:
            topic_counts[topic] = int(
                df["broad_topics_list"].apply(lambda t: topic in t).sum()
            )
        return {
            "total_texts": len(df),
            "reliable_texts": int(df["reliable"].sum()),
            "mean_difficulty": round(df["final_difficulty"].mean(), 3),
            "std_difficulty": round(df["final_difficulty"].std(), 3),
            "min_difficulty": round(df["final_difficulty"].min(), 3),
            "max_difficulty": round(df["final_difficulty"].max(), 3),
            "topic_counts": topic_counts,
        }


# ════════════════════════════════════════════════════════════
# Level Estimator
# ════════════════════════════════════════════════════════════

class LevelEstimator:
    """
    Estimates student reading level from perceived difficulty ratings.

    Formula (per text):
        implied_level = text_difficulty + (3 - perceived_difficulty) × 0.5

    The midpoint 3 means "about right for me."
    Scaling factor 0.5 maps the full perceived range (1–5) to ±1.0.

    Session estimate = simple mean of all implied levels.
    No exponential weighting — with 5–7 observations, every data point matters.

    Research basis: simplified Bayesian Knowledge Tracing
    (Corbett & Anderson, 1995).
    """

    def __init__(self):
        self.implied_levels: List[float] = []

    def update(self, text_difficulty: float, perceived_difficulty: int) -> float:
        """
        Record one observation and return the updated estimated level.

        Parameters
        ----------
        text_difficulty     : The text's final_difficulty (1.0–5.0)
        perceived_difficulty: Student's rating of "how hard did you find it?" (1–5)

        Returns
        -------
        Updated estimated reading level.
        """
        if not (1 <= perceived_difficulty <= 5):
            raise ValueError(f"perceived_difficulty must be 1–5, got {perceived_difficulty}")

        implied = text_difficulty + (3 - perceived_difficulty) * 0.5
        implied = float(np.clip(implied, 1.0, 5.0))
        self.implied_levels.append(implied)
        return self.estimated_level

    @property
    def estimated_level(self) -> Optional[float]:
        """Current estimated reading level, or None if no observations."""
        if not self.implied_levels:
            return None
        return float(np.mean(self.implied_levels))

    @property
    def n_observations(self) -> int:
        return len(self.implied_levels)

    def summary(self) -> str:
        if not self.implied_levels:
            return "LevelEstimator: no observations yet"
        return (
            f"LevelEstimator: {self.n_observations} observations, "
            f"estimated_level = {self.estimated_level:.2f}, "
            f"implied_levels = {[round(x, 2) for x in self.implied_levels]}"
        )


# ════════════════════════════════════════════════════════════
# Scoring Engine
# ════════════════════════════════════════════════════════════

class ScoringEngine:
    """
    Computes composite scores for candidate texts.

    Two components:
        topic_score     = |student_interests ∩ text_broad_topics| / |student_interests|
        difficulty_score = Gaussian(text_diff; μ=estimated_level+0.2, σ=0.8)

    The +0.2 growth-zone shift targets texts slightly above the student's
    level, grounded in Vygotsky's ZPD (1978).

    Parameters
    ----------
    difficulty_sigma : Gaussian tolerance. 0.8 means texts ±0.8 from level
                       still score ~0.61.
    growth_shift     : How much above estimated_level to target. Default 0.2.
    """

    def __init__(self, difficulty_sigma: float = 0.8, growth_shift: float = 0.2):
        self.difficulty_sigma = difficulty_sigma
        self.growth_shift = growth_shift

    def topic_score(
        self, text_broad_topics: List[str], student_interests: List[str]
    ) -> float:
        """
        Recall-based topic overlap.
        How many of the student's interests does this text cover?
        """
        if not student_interests:
            return 0.0
        matched = len(set(text_broad_topics) & set(student_interests))
        return matched / len(student_interests)

    def difficulty_score(
        self, text_difficulty: float, estimated_level: float
    ) -> float:
        """
        Gaussian kernel centred on estimated_level + growth_shift.
        Returns 0.0 for NaN difficulty.
        """
        if math.isnan(text_difficulty):
            return 0.0
        target = estimated_level + self.growth_shift
        # DB may provide Decimal for numeric columns; normalize to float.
        delta = float(text_difficulty) - float(target)
        return math.exp(-0.5 * (delta / self.difficulty_sigma) ** 2)

    def score_candidates(
        self,
        candidates: pd.DataFrame,
        student_interests: List[str],
        estimated_level: Optional[float],
        round_number: int,
    ) -> pd.DataFrame:
        """
        Score all candidate texts and return with score columns added.

        Returns DataFrame sorted by composite_score descending.
        """
        df = candidates.copy()
        w_topic, w_diff = _get_weights(round_number)

        # Topic scores
        df["score_topic"] = df["broad_topics_list"].apply(
            lambda topics: self.topic_score(topics, student_interests)
        )

        # Difficulty scores
        if estimated_level is not None and w_diff > 0:
            df["score_difficulty"] = df["final_difficulty"].apply(
                lambda d: self.difficulty_score(d, estimated_level)
            )
        else:
            df["score_difficulty"] = 0.0

        # Composite
        df["composite_score"] = (
            w_topic * df["score_topic"]
            + w_diff * df["score_difficulty"]
        )

        # Record weights used
        df["w_topic"] = w_topic
        df["w_difficulty"] = w_diff

        return df.sort_values("composite_score", ascending=False)


# ════════════════════════════════════════════════════════════
# Slate Builder (MMR Diversity)
# ════════════════════════════════════════════════════════════

class SlateBuilder:
    """
    Builds a slate of 2 texts using MMR-based diversity.

    Text 1: highest composite_score.
    Text 2: maximises  λ × composite_score − (1−λ) × similarity_to_text_1

    Similarity uses both broad_topic and sub_topic Jaccard overlap,
    as recommended by the topic pipeline documentation for finer
    diversification.

    Parameters
    ----------
    diversity_lambda : Trade-off. 1.0 = pure relevance, 0.0 = pure diversity.
                       Default 0.65.
    broad_weight     : Weight of broad_topic similarity. Default 0.6.
    sub_weight       : Weight of sub_topic similarity. Default 0.4.
    """

    def __init__(
        self,
        diversity_lambda: float = 0.65,
        broad_weight: float = 0.6,
        sub_weight: float = 0.4,
    ):
        self.diversity_lambda = diversity_lambda
        self.broad_weight = broad_weight
        self.sub_weight = sub_weight

    @staticmethod
    def _jaccard(a: List[str], b: List[str]) -> float:
        if not a or not b:
            return 0.0
        sa, sb = set(a), set(b)
        intersection = len(sa & sb)
        union = len(sa | sb)
        return intersection / union if union > 0 else 0.0

    def _combined_similarity(self, row_a: dict, row_b: dict) -> float:
        broad_sim = self._jaccard(
            row_a.get("broad_topics_list", []),
            row_b.get("broad_topics_list", []),
        )
        sub_sim = self._jaccard(
            row_a.get("sub_topics_list", []),
            row_b.get("sub_topics_list", []),
        )
        return self.broad_weight * broad_sim + self.sub_weight * sub_sim

    def build_slate(
        self, scored_df: pd.DataFrame, slate_size: int = 2
    ) -> pd.DataFrame:
        """
        Select a slate of texts using MMR.

        Parameters
        ----------
        scored_df  : DataFrame sorted by composite_score descending.
        slate_size : Number of texts to select (default 2).

        Returns
        -------
        DataFrame with the selected slate texts.
        """
        if len(scored_df) == 0:
            return pd.DataFrame()

        if len(scored_df) <= slate_size:
            return scored_df.reset_index(drop=True)

        # Consider top candidates for diversity pool
        pool_size = min(3 * slate_size + 4, len(scored_df))
        pool = scored_df.head(pool_size).to_dict("records")

        # Text 1: highest composite score
        selected = [pool[0]]
        remaining = pool[1:]

        # Text 2+: MMR selection
        for _ in range(slate_size - 1):
            if not remaining:
                break

            best_mmr = -float("inf")
            best_idx = 0

            for i, candidate in enumerate(remaining):
                relevance = candidate["composite_score"]
                max_sim = max(
                    self._combined_similarity(candidate, sel)
                    for sel in selected
                )
                mmr = (
                    self.diversity_lambda * relevance
                    - (1 - self.diversity_lambda) * max_sim
                )
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return pd.DataFrame(selected).reset_index(drop=True)


# ════════════════════════════════════════════════════════════
# Session Manager
# ════════════════════════════════════════════════════════════

@dataclass
class ReadingEvent:
    """Record of one completed reading within a session."""
    text_id: str
    text_difficulty: float
    perceived_difficulty: int        # 1–5
    interest_rating: int             # 1–5 (evaluation only)
    comprehension_score: float       # 0.0–1.0 (evaluation only)
    round_number: int


@dataclass
class SlateEvent:
    """Record of one slate shown to the student."""
    round_number: int
    shown_text_ids: List[str]
    chosen_text_id: Optional[str]    # None if refresh
    was_refresh: bool


class SessionManager:
    """
    Manages one anonymous session from start to finish.

    Orchestrates:
        1. Filter candidates (reliable, unseen, topic-matching)
        2. Score candidates (topic + difficulty, round-dependent weights)
        3. Build slate of 2 (MMR diversity with broad + sub topics)
        4. Record feedback and update reading level

    Parameters
    ----------
    corpus         : Corpus instance with the loaded dataset.
    interests      : Student's selected interests (3+ broad topics).
    scoring_engine : ScoringEngine instance (optional, creates default).
    slate_builder  : SlateBuilder instance (optional, creates default).
    """

    def __init__(
        self,
        corpus: Corpus,
        interests: List[str],
        scoring_engine: Optional[ScoringEngine] = None,
        slate_builder: Optional[SlateBuilder] = None,
    ):
        unknown = [t for t in interests if t not in BROAD_TOPICS]
        if unknown:
            raise ValueError(f"Unknown topics: {unknown}. Valid: {BROAD_TOPICS}")
        if len(interests) < 1:
            raise ValueError("At least 1 interest required.")

        self.interests = interests
        self.corpus = corpus
        self.scoring = scoring_engine or ScoringEngine()
        self.slate_builder = slate_builder or SlateBuilder()
        self.level_estimator = LevelEstimator()

        # Session state
        self.round_number: int = 1
        self.seen_text_ids: set = set()
        self.reading_history: List[ReadingEvent] = []
        self.slate_history: List[SlateEvent] = []

    # ── Candidate filtering ───────────────────────────────

    def _get_candidates(self) -> pd.DataFrame:
        """
        Filter corpus to eligible candidates:
          - reliable == True
          - not already seen in this session
          - overlaps with at least 1 student interest
        """
        df = self.corpus.get_reliable_texts()

        if self.seen_text_ids:
            df = df[~df["text_id"].isin(self.seen_text_ids)]

        interest_set = set(self.interests)
        df = df[df["broad_topics_list"].apply(
            lambda topics: bool(set(topics) & interest_set)
        )]

        return df

    # ── Slate generation ──────────────────────────────────

    def get_recommendations(self) -> pd.DataFrame:
        """
        Generate a slate of 2 recommended texts for the current round.

        Returns DataFrame with 2 rows (or fewer if pool exhausted).
        """
        candidates = self._get_candidates()

        if len(candidates) == 0:
            print("[Session] No candidates remaining.")
            return pd.DataFrame()

        scored = self.scoring.score_candidates(
            candidates=candidates,
            student_interests=self.interests,
            estimated_level=self.level_estimator.estimated_level,
            round_number=self.round_number,
        )

        slate = self.slate_builder.build_slate(scored, slate_size=2)
        return slate

    # ── Refresh ───────────────────────────────────────────

    def handle_refresh(self, shown_text_ids: List[str]) -> pd.DataFrame:
        """
        Mark shown texts as seen, log the refresh, return new slate.
        No profile update — refresh signal is ambiguous.
        """
        for tid in shown_text_ids:
            self.seen_text_ids.add(tid)

        self.slate_history.append(SlateEvent(
            round_number=self.round_number,
            shown_text_ids=list(shown_text_ids),
            chosen_text_id=None,
            was_refresh=True,
        ))

        return self.get_recommendations()

    # ── Record reading ────────────────────────────────────

    def record_reading(
        self,
        shown_text_ids: List[str],
        chosen_text_id: str,
        perceived_difficulty: int,
        interest_rating: int,
        comprehension_score: float,
    ) -> None:
        """
        Record that the student read a text and provided feedback.

        perceived_difficulty : "How hard did you find it?" (1–5) → used for adaptation
        interest_rating      : "How interesting was it?" (1–5) → evaluation only
        comprehension_score  : MCQ/TF proportion correct (0.0–1.0) → evaluation only
        """
        if not (1 <= perceived_difficulty <= 5):
            raise ValueError(f"perceived_difficulty must be 1–5, got {perceived_difficulty}")
        if not (1 <= interest_rating <= 5):
            raise ValueError(f"interest_rating must be 1–5, got {interest_rating}")
        if not (0.0 <= comprehension_score <= 1.0):
            raise ValueError(f"comprehension_score must be 0.0–1.0, got {comprehension_score}")

        text_row = self.corpus.df[self.corpus.df["text_id"] == chosen_text_id]
        if text_row.empty:
            raise ValueError(f"text_id '{chosen_text_id}' not found in corpus.")
        text_difficulty = float(text_row.iloc[0]["final_difficulty"])

        # Mark all shown texts as seen
        for tid in shown_text_ids:
            self.seen_text_ids.add(tid)

        # Log events
        self.slate_history.append(SlateEvent(
            round_number=self.round_number,
            shown_text_ids=list(shown_text_ids),
            chosen_text_id=chosen_text_id,
            was_refresh=False,
        ))

        self.reading_history.append(ReadingEvent(
            text_id=chosen_text_id,
            text_difficulty=text_difficulty,
            perceived_difficulty=perceived_difficulty,
            interest_rating=interest_rating,
            comprehension_score=comprehension_score,
            round_number=self.round_number,
        ))

        # Update reading level (only signal: perceived difficulty)
        self.level_estimator.update(text_difficulty, perceived_difficulty)

        # Advance round
        self.round_number += 1

    # ── Explanation (Norwegian) ───────────────────────────

    def explain(self, text_row: dict) -> str:
        """Norwegian-language explanation for a recommendation."""
        title = text_row.get("title", "")
        broad_raw = text_row.get("broad_topics", "")
        difficulty = float(text_row.get("final_difficulty", 3.0))

        broad_list = [t.strip() for t in str(broad_raw).split("|") if t.strip()]
        matched = set(broad_list) & set(self.interests)

        lines = [f"  «{title}»"]

        if matched:
            lines.append(f"    Emner: {', '.join(sorted(matched))}")

        level = self.level_estimator.estimated_level
        if level is not None:
            diff_gap = difficulty - level
            if abs(diff_gap) < 0.4:
                lines.append(f"    Passer godt til nivået ditt")
            elif diff_gap > 0:
                lines.append(f"    Litt mer utfordrende — bra for å strekke seg!")
            else:
                lines.append(f"    Litt lettere — god for flytsone-lesing")

        return "\n".join(lines)

    # ── Session summary ───────────────────────────────────

    def session_summary(self) -> dict:
        """Return a full summary dict for logging/evaluation."""
        readings = self.reading_history
        slates = self.slate_history

        n_refreshes = sum(1 for s in slates if s.was_refresh)

        avg_interest = (
            np.mean([r.interest_rating for r in readings]) if readings else None
        )
        avg_comprehension = (
            np.mean([r.comprehension_score for r in readings]) if readings else None
        )
        avg_perceived = (
            np.mean([r.perceived_difficulty for r in readings]) if readings else None
        )

        return {
            "interests": self.interests,
            "n_rounds": self.round_number - 1,
            "n_readings": len(readings),
            "n_refreshes": n_refreshes,
            "n_texts_seen": len(self.seen_text_ids),
            "estimated_level": self.level_estimator.estimated_level,
            "implied_levels": [round(x, 2) for x in self.level_estimator.implied_levels],
            "avg_interest_rating": round(avg_interest, 2) if avg_interest else None,
            "avg_comprehension": round(avg_comprehension, 2) if avg_comprehension else None,
            "avg_perceived_difficulty": round(avg_perceived, 2) if avg_perceived else None,
            "reading_history": [
                {
                    "text_id": r.text_id,
                    "difficulty": r.text_difficulty,
                    "perceived": r.perceived_difficulty,
                    "interest": r.interest_rating,
                    "comprehension": r.comprehension_score,
                    "round": r.round_number,
                }
                for r in readings
            ],
            "slate_history": [
                {
                    "round": s.round_number,
                    "shown": s.shown_text_ids,
                    "chosen": s.chosen_text_id,
                    "refresh": s.was_refresh,
                }
                for s in slates
            ],
        }

    def print_summary(self) -> None:
        """Print formatted session summary."""
        s = self.session_summary()
        print("\n" + "=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)
        print(f"  Interesser        : {', '.join(s['interests'])}")
        print(f"  Runder fullført   : {s['n_rounds']}")
        print(f"  Tekster lest      : {s['n_readings']}")
        print(f"  Oppfriskninger    : {s['n_refreshes']}")
        print(f"  Tekster sett      : {s['n_texts_seen']}")
        level_str = f"{s['estimated_level']:.2f}" if s['estimated_level'] else "(ingen)"
        print(f"  Estimert lesenivå : {level_str}")
        print(f"  Snitt interesse   : {s['avg_interest_rating']}")
        print(f"  Snitt forståelse  : {s['avg_comprehension']}")
        print(f"  Snitt opplevd vanskelighet: {s['avg_perceived_difficulty']}")

        if s["reading_history"]:
            print("\n  Lesehistorikk:")
            for r in s["reading_history"]:
                print(f"    Runde {r['round']}: {r['text_id']} "
                      f"(diff={r['difficulty']:.2f}, "
                      f"opplevd={r['perceived']}, "
                      f"interesse={r['interest']}, "
                      f"forståelse={r['comprehension']:.0%})")
        print("=" * 60)
