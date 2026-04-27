"""
Norwegian Reading Text Topic Taxonomy + Student Recommender
===========================================================
Fully local pipeline — no data leaves your machine.

This version fixes qwen3 thinking-mode behavior in Ollama by:
- disabling thinking explicitly with "think": false
- increasing generation budget
- accepting JSON or simple key:value fallback output

Requirements:
    pip install sentence-transformers pandas requests numpy scikit-learn
"""

import json
import re
import time
import warnings

import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
CSV_PATH = r"..\data\raw\question_texts_texts.csv"

# Ollama
OLLAMA_MODEL = "llama3"   # change to "llama3" if you want
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
OLLAMA_TIMEOUT = 180
OLLAMA_RETRIES = 2
OLLAMA_KEEP_ALIVE = "30m"
PRINT_DEBUG_ONCE = True

# Embeddings for recommendation
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Topic settings
CANDIDATE_TOPICS_PER_TEXT = 3   # 1 main + up to 2 secondary
MAX_TEXT_CHARS = 1200
SAVE_EVERY_N = 10
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

ALLOWED_TOPICS = [
    "Dyr",
    "Natur",
    "Miljø",
    "Verdensrom",
    "Vitenskap",
    "Teknologi",
    "Skole",
    "Vennskap",
    "Familie",
    "Kultur",
    "Historie",
    "Samfunn",
    "Helse",
    "Følelser",
    "Eventyr",
    "Media",
    "Kunst",
    "Mat",
    "Kropp",
    "Språk",
    "Geografi",
    "Politikk",
    "Idrett",
    "Transport",
    "Arbeidsliv",
    "Religion",
    "Fritid",
    "Kommunikasjon"
]
# ─────────────────────────────────────────


# ── UTILITIES ────────────────────────────
def normalize_text(text):
    text = "" if pd.isna(text) else str(text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_truncate(text, max_chars=1200):
    text = normalize_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip() + " ..."


def clean_topic(topic):
    if not topic:
        return None

    topic = str(topic).strip()
    topic = topic.strip("\"'` ")
    topic = re.sub(r"\s+", " ", topic)

    for allowed in ALLOWED_TOPICS:
        if topic.lower() == allowed.lower():
            return allowed

    return None


def normalize_secondary_topics(topics):
    if not isinstance(topics, list):
        return []

    cleaned = []
    seen = set()

    for t in topics:
        ct = clean_topic(t)
        if ct and ct not in seen:
            cleaned.append(ct)
            seen.add(ct)

    return cleaned


def clamp_confidence(value):
    try:
        value = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, value))


def extract_json_object(text):
    if not text:
        return None

    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    if start == -1:
        return None

    brace_count = 0
    end = -1
    for i in range(start, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                end = i
                break

    if end != -1:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return None


def parse_key_value_response(text):
    if not text:
        return None

    result = {}
    for line in str(text).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip().lower()] = value.strip()

    if not result:
        return None

    secondary = result.get("secondary_topics", "")
    secondary_topics = [s.strip() for s in secondary.split(",") if s.strip()]

    return {
        "main_topic": result.get("main_topic", ""),
        "secondary_topics": secondary_topics,
        "english_gloss": result.get("english_gloss", ""),
        "confidence": result.get("confidence", 0.0)
    }


# ── DATA LOADING ─────────────────────────
def load_data(path):
    print(f"\n📂 Loading data from: {path}")
    df = pd.read_csv(path)

    required_columns = ["serialNumber", "title", "body"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    df = df[["serialNumber", "title", "body"]].copy()
    df["title"] = df["title"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)

    df = df[df["body"].str.strip() != ""].copy()
    df.reset_index(drop=True, inplace=True)

    df["title_clean"] = df["title"].apply(normalize_text)
    df["body_clean"] = df["body"].apply(normalize_text)
    df["full_text"] = (df["title_clean"] + ". " + df["body_clean"]).str.strip()
    df["full_text"] = df["full_text"].str.replace(r"\s+", " ", regex=True)

    print(f"   ✅ Loaded {len(df)} passages")
    return df


# ── OLLAMA HELPERS ───────────────────────
def check_ollama():
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"   ❌ Ollama is not reachable: {e}")
        return False


def warmup_ollama():
    print(f"\n🔥 Warming up Ollama model ({OLLAMA_MODEL})...")
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": 'Return exactly this text: {"status":"ready"}',
                "stream": False,
                "think": False,
                "keep_alive": OLLAMA_KEEP_ALIVE,
                "options": {
                    "temperature": 0,
                    "num_predict": 50
                }
            },
            timeout=120
        )
        response.raise_for_status()
        print("   ✅ Ollama model is ready")
        return True
    except Exception as e:
        print(f"   ⚠️  Ollama warm-up failed: {e}")
        return False


_printed_debug = False


def query_ollama_structured(prompt, timeout=OLLAMA_TIMEOUT, retries=OLLAMA_RETRIES):
    global _printed_debug

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": 0,
            "num_predict": 220
        }
    }

    for attempt in range(retries + 1):
        try:
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            outer = response.json()

            if PRINT_DEBUG_ONCE and not _printed_debug:
                print("\n--- FULL OLLAMA OUTER RESPONSE SAMPLE ---")
                print(json.dumps(outer, ensure_ascii=False, indent=2)[:3000])
                print("--- END OUTER SAMPLE ---\n")
                _printed_debug = True

            raw = outer.get("response", "")
            if raw:
                parsed = extract_json_object(raw)
                if parsed is not None:
                    return parsed

                parsed = parse_key_value_response(raw)
                if parsed is not None:
                    return parsed

            print(f"   ⚠️  Empty or non-structured response on attempt {attempt + 1}/{retries + 1}")

        except Exception as e:
            print(f"   ⚠️  Ollama error (attempt {attempt + 1}/{retries + 1}): {e}")

    return None


# ── PROMPTING ────────────────────────────
def build_topic_prompt(title, text, candidate_topics_per_text=3):
    n_secondary = max(0, candidate_topics_per_text - 1)
    allowed = ", ".join(ALLOWED_TOPICS)

    return f"""
Choose broad school topics for this Norwegian text.

Use ONLY topics from this list:
{allowed}

Return valid JSON exactly like this:
{{
  "main_topic": "one topic from list",
  "secondary_topics": ["up to {n_secondary} topics from list"],
  "english_gloss": "short English summary",
  "confidence": 0.0
}}

If you cannot produce JSON, return exactly these 4 lines:
main_topic: <one topic from list>
secondary_topics: <comma separated topics from list, max {n_secondary}>
english_gloss: <short English summary>
confidence: <number between 0.0 and 1.0>

Rules:
- use only topics from the list
- do not use names, titles, characters, organizations, or specific nouns from the text
- choose broad reusable school topics
- output only the structured result
- no markdown
- no explanations

Title: {title}
Text: {text}
""".strip()


def validate_topic_result(result):
    if not isinstance(result, dict):
        return {
            "main_topic": "Samfunn",
            "secondary_topics": [],
            "english_gloss": "General Norwegian reading text.",
            "confidence": 0.0
        }

    main_topic = clean_topic(result.get("main_topic"))
    secondary_topics = normalize_secondary_topics(result.get("secondary_topics", []))
    english_gloss = normalize_text(result.get("english_gloss", ""))
    confidence = clamp_confidence(result.get("confidence", 0.0))

    if not main_topic:
        if secondary_topics:
            main_topic = secondary_topics[0]
            secondary_topics = secondary_topics[1:]
        else:
            main_topic = "Samfunn"

    secondary_topics = [t for t in secondary_topics if t != main_topic]
    secondary_topics = secondary_topics[:max(0, CANDIDATE_TOPICS_PER_TEXT - 1)]

    if not english_gloss:
        english_gloss = "General Norwegian reading text."

    return {
        "main_topic": main_topic,
        "secondary_topics": secondary_topics,
        "english_gloss": english_gloss,
        "confidence": confidence
    }


# ── TOPIC CLASSIFICATION ─────────────────
def classify_single_text(title, body):
    prompt = build_topic_prompt(
        title=normalize_text(title)[:160],
        text=safe_truncate(body, MAX_TEXT_CHARS),
        candidate_topics_per_text=CANDIDATE_TOPICS_PER_TEXT
    )
    raw_result = query_ollama_structured(prompt)
    return validate_topic_result(raw_result)


def classify_topics(df):
    print(f"\n🤖 Classifying topics with Ollama ({OLLAMA_MODEL})...")

    if not check_ollama():
        raise RuntimeError("Ollama is not reachable on localhost:11434")

    warmup_ollama()

    results = []
    total = len(df)

    for i, row in df.iterrows():
        result = classify_single_text(row["title_clean"], row["body_clean"])
        results.append(result)

        secondary_str = ", ".join(result["secondary_topics"]) if result["secondary_topics"] else "-"
        print(
            f"   [{i + 1:>3}/{total}] "
            f"{result['main_topic']} | {secondary_str} | conf={result['confidence']:.2f}"
        )

        if (i + 1) % SAVE_EVERY_N == 0:
            temp_df = df.iloc[: i + 1].copy()
            temp_df["main_topic"] = [r["main_topic"] for r in results]
            temp_df["secondary_topics"] = [json.dumps(r["secondary_topics"], ensure_ascii=False) for r in results]
            temp_df["english_gloss"] = [r["english_gloss"] for r in results]
            temp_df["topic_confidence"] = [r["confidence"] for r in results]
            temp_df.to_csv("topic_classification_checkpoint.csv", index=False, encoding="utf-8-sig")

        time.sleep(0.05)

    return results


# ── EMBEDDINGS FOR RECOMMENDATION ────────
def embed_passages(df):
    print(f"\n🔢 Embedding passages with '{EMBED_MODEL}' (runs locally)...")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(
        df["full_text"].tolist(),
        show_progress_bar=True,
        batch_size=32
    )
    embeddings = np.array(embeddings)
    print(f"   ✅ Embeddings shape: {embeddings.shape}")
    return embeddings, model


# ── BUILD OUTPUT ─────────────────────────
def build_output(df, topic_results):
    df = df.copy()
    df["main_topic"] = [r["main_topic"] for r in topic_results]
    df["secondary_topics"] = [json.dumps(r["secondary_topics"], ensure_ascii=False) for r in topic_results]
    df["english_gloss"] = [r["english_gloss"] for r in topic_results]
    df["topic_confidence"] = [r["confidence"] for r in topic_results]
    return df


# ── RECOMMENDATION ───────────────────────
def parse_secondary_topics(value):
    try:
        data = json.loads(value) if isinstance(value, str) else value
        return data if isinstance(data, list) else []
    except Exception:
        return []


def recommend_for_student(
    interest_text,
    embed_model,
    df_out,
    embeddings,
    top_n=5,
    preferred_topics=None
):
    interest_vec = embed_model.encode([interest_text])
    sims = cosine_similarity(interest_vec, embeddings)[0]

    result_df = df_out.copy()
    result_df["similarity"] = sims

    if preferred_topics:
        preferred_topics = set(preferred_topics)

        def has_preferred_topic(row):
            secondaries = parse_secondary_topics(row["secondary_topics"])
            topics = {row["main_topic"], *secondaries}
            return len(topics.intersection(preferred_topics)) > 0

        preferred_mask = result_df.apply(has_preferred_topic, axis=1)
        preferred_df = result_df[preferred_mask].copy()

        if not preferred_df.empty:
            preferred_df = preferred_df.sort_values(
                ["similarity", "topic_confidence"],
                ascending=[False, False]
            )
            return preferred_df[
                ["serialNumber", "title", "main_topic", "secondary_topics", "similarity", "topic_confidence"]
            ].head(top_n)

    result_df = result_df.sort_values(
        ["similarity", "topic_confidence"],
        ascending=[False, False]
    )
    return result_df[
        ["serialNumber", "title", "main_topic", "secondary_topics", "similarity", "topic_confidence"]
    ].head(top_n)


def demo_recommendations(embed_model, df_out, embeddings):
    print("\n🎓 Demo: Student Recommendations")
    print("─" * 60)

    demo_students = [
        {
            "name": "Student A",
            "interest": "Jeg liker natur, dyr og miljø",
            "preferred_topics": ["Natur", "Dyr", "Miljø"]
        },
        {
            "name": "Student B",
            "interest": "Jeg er interessert i historie og samfunn",
            "preferred_topics": ["Historie", "Samfunn", "Politikk"]
        },
        {
            "name": "Student C",
            "interest": "Jeg liker kroppen, helse og vitenskap",
            "preferred_topics": ["Helse", "Kropp", "Vitenskap"]
        }
    ]

    for student in demo_students:
        print(f'\n👤 {student["name"]}: "{student["interest"]}"')
        recs = recommend_for_student(
            interest_text=student["interest"],
            embed_model=embed_model,
            df_out=df_out,
            embeddings=embeddings,
            top_n=5,
            preferred_topics=student["preferred_topics"]
        )

        for _, row in recs.iterrows():
            print(
                f"   → [{row['serialNumber']}] {row['title']} | "
                f"{row['main_topic']} | sim={row['similarity']:.3f} | "
                f"conf={row['topic_confidence']:.2f}"
            )


# ── MAIN ─────────────────────────────────
def main():
    print("=" * 55)
    print("  Norwegian Topic Taxonomy + Student Recommender")
    print("  Fully Local | No data leaves your machine")
    print("=" * 55)

    df = load_data(CSV_PATH)
    topic_results = classify_topics(df)
    df_out = build_output(df, topic_results)

    out_path = "categorized_passages.csv"
    df_out[
        ["serialNumber", "title", "main_topic", "secondary_topics", "english_gloss", "topic_confidence"]
    ].to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n💾 Saved categorized passages → {out_path}")

    embeddings, embed_model = embed_passages(df_out)
    demo_recommendations(embed_model, df_out, embeddings)

    print("\n✅ Done! Summary:")
    print(
        df_out.groupby("main_topic")["serialNumber"]
        .count()
        .rename("passage_count")
        .sort_values(ascending=False)
        .to_string()
    )


if __name__ == "__main__":
    main()