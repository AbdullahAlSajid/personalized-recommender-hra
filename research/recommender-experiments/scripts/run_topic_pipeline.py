from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from recommender.topics.models import TopicConfig
from recommender.topics.pipeline import run_topic_pipeline


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    input_csv = PROJECT_ROOT / "data" / "raw" / "question_texts_texts.csv"
    output_dir = PROJECT_ROOT / "outputs" / "topics"

    config = TopicConfig(
        max_words_per_chunk=1200,
        min_words_per_chunk=200,
        max_secondary_topics=2,
        candidate_topics_per_text=5,
        language="no",
        llm_model_name="qwen3:4b",
        taxonomy_version="v1",
    )

    result = run_topic_pipeline(
        input_csv=input_csv,
        output_dir=output_dir,
        config=config,
        custom_topic_map=None,
        taxonomy_descriptions=None,
        aggregation_method="majority_vote",
        min_taxonomy_coverage=1,
        max_taxonomy_topics=16,
    )

    print("\nTopic pipeline finished successfully.")
    print(f"Records processed: {result['n_records']}")
    print(f"Normalized candidate results: {result['n_normalized_results']}")
    print(f"Final taxonomy topics: {result['n_taxonomy_topics']}")
    print(f"Final assignments: {result['n_assignments']}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
