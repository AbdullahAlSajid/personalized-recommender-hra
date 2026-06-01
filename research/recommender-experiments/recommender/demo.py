"""
Demo GÇö Simulated Session Walk-through
======================================
Simulates a full 6-round session for three different student profiles,
showing how the recommender adapts across rounds.

Run:  python demo.py results_with_topic_difficulty.csv
"""

import sys
import random

from recommender.engine import Corpus, SessionManager, BROAD_TOPICS


DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "results_with_topic_difficulty.csv"


def print_header(text: str, width: int = 65) -> None:
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def simulate_feedback(text_difficulty: float, true_level: float) -> dict:
    """
    Simulate realistic student feedback based on the gap between
    the text's difficulty and the student's 'true' reading level.

    Returns dict with perceived_difficulty, interest_rating, comprehension_score.
    """
    gap = text_difficulty - true_level

    # Perceived difficulty: if text is harder than true level, student rates higher
    raw_perceived = 3 + gap * 1.5  # scale gap to 1-5 range
    perceived = int(max(1, min(5, round(raw_perceived + random.gauss(0, 0.3)))))

    # Comprehension: higher when text is at or below true level
    if gap <= 0:
        comp = min(1.0, 0.9 + random.gauss(0, 0.05))
    elif gap < 0.8:
        comp = min(1.0, max(0.33, 0.75 - gap * 0.3 + random.gauss(0, 0.05)))
    else:
        comp = min(1.0, max(0.0, 0.5 - gap * 0.2 + random.gauss(0, 0.1)))

    # Round to nearest third (3 questions GåÆ 0/3, 1/3, 2/3, 3/3)
    comp = round(comp * 3) / 3

    # Interest: random 3-5 (assuming topic-matched texts are generally interesting)
    interest = random.choice([3, 4, 4, 5])

    return {
        "perceived_difficulty": perceived,
        "interest_rating": interest,
        "comprehension_score": comp,
    }


def run_session(corpus, interests, true_level, label, n_rounds=6):
    """Simulate a full session and print each step."""
    print_header(f"{label}")
    print(f"  Interesser  : {', '.join(interests)}")
    print(f"  Sant niv+Ñ   : {true_level:.1f} (ukjent for systemet)")
    print(f"  Runder      : {n_rounds}")

    random.seed(hash(label) % 2**32)
    session = SessionManager(corpus, interests)

    for rnd in range(1, n_rounds + 1):
        # Get recommendations
        slate = session.get_recommendations()

        if slate.empty:
            print(f"\n  Runde {rnd}: Ingen flere kandidater!")
            break

        # Get weight info
        w_topic = slate.iloc[0].get("w_topic", "?")
        w_diff = slate.iloc[0].get("w_difficulty", "?")
        level_str = (
            f"{session.level_estimator.estimated_level:.2f}"
            if session.level_estimator.estimated_level is not None
            else "ukjent"
        )

        print(f"\n  GöÇGöÇ Runde {rnd} "
              f"(vekter: emne={w_topic}, vanskelighet={w_diff}, "
              f"niv+Ñ={level_str}) GöÇGöÇ")

        # Show the slate
        for i, (_, row) in enumerate(slate.iterrows()):
            marker = "GåÆ" if i == 0 else " "
            print(f"  {marker} {row['title']}")
            print(f"      Emner: {row['broad_topics']}  |  "
                  f"Vanskelighet: {row['final_difficulty']:.2f}  |  "
                  f"Score: {row['composite_score']:.3f} "
                  f"(emne={row['score_topic']:.2f}, "
                  f"vansk={row['score_difficulty']:.2f})")

        # Simulate: student picks text 1 (highest scored)
        chosen_id = slate.iloc[0]["text_id"]
        shown_ids = slate["text_id"].tolist()
        chosen_diff = float(slate.iloc[0]["final_difficulty"])

        # Simulate: maybe refresh once (20% chance)
        if random.random() < 0.2 and rnd > 1:
            print(f"    Gå+ Studenten trykker oppfrisk")
            slate = session.handle_refresh(shown_ids)
            if slate.empty:
                print(f"    Ingen flere kandidater etter oppfrisk!")
                break
            chosen_id = slate.iloc[0]["text_id"]
            shown_ids = slate["text_id"].tolist()
            chosen_diff = float(slate.iloc[0]["final_difficulty"])
            for i, (_, row) in enumerate(slate.iterrows()):
                marker = "GåÆ" if i == 0 else " "
                print(f"  {marker} {row['title']}")
                print(f"      Emner: {row['broad_topics']}  |  "
                      f"Vanskelighet: {row['final_difficulty']:.2f}  |  "
                      f"Score: {row['composite_score']:.3f}")

        # Generate feedback
        fb = simulate_feedback(chosen_diff, true_level)

        print(f"    G£ô Valgt: {chosen_id}")
        print(f"      Opplevd vanskelighet: {fb['perceived_difficulty']}/5  |  "
              f"Interesse: {fb['interest_rating']}/5  |  "
              f"Forst+Ñelse: {fb['comprehension_score']:.0%}")

        # Record
        session.record_reading(
            shown_text_ids=shown_ids,
            chosen_text_id=chosen_id,
            perceived_difficulty=fb["perceived_difficulty"],
            interest_rating=fb["interest_rating"],
            comprehension_score=fb["comprehension_score"],
        )

        # Show explanation
        explanation = session.explain(slate.iloc[0].to_dict())
        print(explanation)

    # Print session summary
    session.print_summary()


def main():
    print_header("PERSONALISERT TEKSTANBEFALING GÇö DEMO", width=65)
    print("  Simulerer fullstendige ++kter med syntetiske elevprofiler")
    print("  Viser hvordan anbefalinger tilpasses gjennom ++kten")

    corpus = Corpus(DATA_PATH)

    # Show corpus stats
    stats = corpus.stats()
    print(f"\n  Korpus: {stats['total_texts']} tekster "
          f"({stats['reliable_texts']} p+Ñlitelige)")
    print(f"  Vanskelighet: snitt={stats['mean_difficulty']}, "
          f"std={stats['std_difficulty']}, "
          f"range=[{stats['min_difficulty']}, {stats['max_difficulty']}]")
    print(f"\n  Tekster per emne:")
    for topic, count in sorted(stats["topic_counts"].items(), key=lambda x: -x[1]):
        bar = "Gûê" * (count // 2)
        print(f"    {topic:<14}: {count:3d}  {bar}")

    # GöÇGöÇ Three student profiles GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

    run_session(
        corpus,
        interests=["Dyr", "Natur", "Fortelling"],
        true_level=2.2,
        label="Elev A GÇö Lett leser, naturinteressert",
        n_rounds=6,
    )

    run_session(
        corpus,
        interests=["Vitenskap", "Teknologi", "Matematikk"],
        true_level=4.0,
        label="Elev B GÇö Avansert leser, STEM-interessert",
        n_rounds=6,
    )

    run_session(
        corpus,
        interests=["Idrett", "Kultur", "Historie", "Samfunn"],
        true_level=3.2,
        label="Elev C GÇö Middels leser, bredt interessert",
        n_rounds=6,
    )

    print("\n[Ferdig]\n")


if __name__ == "__main__":
    main()
