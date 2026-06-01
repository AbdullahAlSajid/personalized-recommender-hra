"""
Norwegian Reading Text Difficulty Scorer  v2
=============================================
Fully local pipeline GÇö no data leaves your machine.

Fixes from v1:
  - Passage comes FIRST in prompt (model attends to it properly)
  - Two-step chain-of-thought: analyse first, then score
  - Shortened, flatter prompt (less template-locking)
  - temperature=0.3 so scores can vary across passages
  - SCORE_REPEATS=1 by default (re-enable to 3 after confirming scores vary)
  - num_predict reduced (less runaway generation)
  - Raw LLM output always printed for first passage (easy debugging)
  - Three-layer JSON fallback + free-text score extraction

Requirements:
    pip install pandas requests numpy
"""

import json
import re
import time
import warnings

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")


# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
# CONFIGURATION
# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
CSV_PATH = r"..\data\raw\question_texts_texts.csv"

# Ollama GÇö swap model if llama3 still collapses:
#   "mistral"      GåÉ great structured output
#   "gemma2:9b"   GåÉ strong instruction following
#   "llama3.1:8b" GåÉ slightly better than llama3
OLLAMA_MODEL = "llama3"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
OLLAMA_TIMEOUT = 180
OLLAMA_RETRIES = 2
OLLAMA_KEEP_ALIVE = "30m"

# Scoring
MAX_TEXT_CHARS = 1200
SCORE_REPEATS = 1       # keep at 1 until scores look realistic; raise to 3 after
TEMPERATURE = 0.3       # >0 allows variation; 0 = locked-in template responses
SAVE_EVERY_N = 10
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# GöÇGöÇ CEFR scale GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
DIFFICULTY_LEVELS = {
    1: "A1",
    2: "A2",
    3: "B1",
    4: "B2",
    5: "C1",
    6: "C2",
}

# GöÇGöÇ Anchor examples GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
ANCHORS = {
    1: "Jeg har en katt. Den er svart. Katten heter Leo. Leo liker +Ñ sove.",
    2: "Vi har mange fag p+Ñ skolen. Jeg liker best norsk og gym. L+ªreren v+Ñr er veldig snill.",
    3: "Mange barn i Norge bruker mye tid foran skjermen. Foreldre er bekymret for dette.",
    4: "Klimaendringer har f++rt til hyppigere ekstremv+ªr og stigende havniv+Ñer de siste ti+Ñrene.",
    5: "Kvantemekanikken utfordrer v+Ñr intuitive forst+Ñelse av kausalitet og lokalitet i fysikkens lover.",
    6: "Den fenomenologiske tradisjonen insisterer p+Ñ at bevisstheten alltid er intensjonalt rettet mot verden.",
}
# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ


# GöÇGöÇ UTILITIES GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
def normalize_text(text):
    text = "" if pd.isna(text) else str(text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_truncate(text, max_chars=MAX_TEXT_CHARS):
    text = normalize_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip() + " ..."


def clamp_score(value):
    try:
        value = float(value)
    except Exception:
        return None
    return max(1.0, min(6.0, value))


def clamp_confidence(value):
    try:
        value = float(value)
    except Exception:
        return 0.5
    return max(0.0, min(1.0, value))


def score_to_label(score):
    level = int(round(max(1, min(6, float(score)))))
    return DIFFICULTY_LEVELS[level]


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

    brace_count, end = 0, -1
    for i in range(start, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                end = i
                break

    if end != -1:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return None


def parse_key_value_fallback(text):
    """Fallback: parse plain  key: value  lines."""
    if not text:
        return None
    result = {}
    for line in str(text).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip().lower().replace(" ", "_")] = value.strip()
    if not result:
        return None
    return {
        "difficulty_score": result.get("difficulty_score", result.get("score", "")),
        "difficulty_label": result.get("difficulty_label", result.get("label", "")),
        "confidence": result.get("confidence", 0.5),
    }


def extract_score_from_free_text(text):
    """Last resort: scan raw text for any score signal."""
    if not text:
        return None
    score_match = re.search(r"\bscore[:\s]+([1-6])\b", text, re.IGNORECASE)
    label_match = re.search(r"\b(A1|A2|B1|B2|C1|C2)\b", text, re.IGNORECASE)

    if not score_match and not label_match:
        return None

    label_map = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}
    if score_match:
        score = float(score_match.group(1))
    else:
        score = float(label_map.get(label_match.group(1).upper(), 3))

    label = label_match.group(1).upper() if label_match else score_to_label(score)
    return {"difficulty_score": score, "difficulty_label": label, "confidence": 0.4}


# GöÇGöÇ DATA LOADING GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
def load_data(path):
    print(f"\n=ƒôé Loading data from: {path}")
    df = pd.read_csv(path)

    required = ["serialNumber", "title", "body"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[required].copy()
    df["title"] = df["title"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    df = df[df["body"].str.strip() != ""].copy()
    df.reset_index(drop=True, inplace=True)
    df["title_clean"] = df["title"].apply(normalize_text)
    df["body_clean"] = df["body"].apply(normalize_text)

    print(f"   G£à Loaded {len(df)} passages")
    return df


# GöÇGöÇ OLLAMA HELPERS GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
def check_ollama():
    try:
        requests.get(OLLAMA_TAGS_URL, timeout=10).raise_for_status()
        return True
    except Exception as e:
        print(f"   G¥î Ollama not reachable: {e}")
        return False


def warmup_ollama():
    print(f"\n=ƒöÑ Warming up {OLLAMA_MODEL}...")
    try:
        requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": 'Say exactly: {"status":"ready"}',
                "stream": False,
                "think": False,
                "keep_alive": OLLAMA_KEEP_ALIVE,
                "options": {"temperature": 0, "num_predict": 30}
            },
            timeout=120
        ).raise_for_status()
        print("   G£à Model ready")
        return True
    except Exception as e:
        print(f"   GÜán+Å  Warm-up failed: {e}")
        return False


def call_ollama(prompt, num_predict=200, timeout=OLLAMA_TIMEOUT, retries=OLLAMA_RETRIES):
    """Raw Ollama call GÇö returns the response string or None."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": num_predict,
        }
    }
    for attempt in range(retries + 1):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json().get("response", "")
        except Exception as e:
            print(f"   GÜán+Å  Ollama error (attempt {attempt + 1}/{retries + 1}): {e}")
    return None


# GöÇGöÇ PROMPTS GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
def build_analysis_prompt(title, text):
    return f"""Analyze this Norwegian text. Be precise and specific.

Title: {title}
Text: {text}

Answer each question with a concrete observation about THIS text:

1. SENTENCE LENGTH: What is the approximate average word count per sentence? (count a few and give a number)
2. HARDEST 3 WORDS: List the 3 most difficult or uncommon Norwegian words in this text.
3. GRAMMAR: List any complex structures present: passive voice / conditional / indirect speech / embedded clauses / none
4. TOPIC: Is the topic concrete-everyday / familiar-general / requires background knowledge / abstract-academic / highly specialised?
5. FIGURATIVE LANGUAGE: none / some idioms / heavy metaphor or irony

Answer only these 5 points. One line each.
""".strip()


def build_score_prompt(title, analysis):
    anchor_lines = "\n".join(
        f'  {score} ({label}): "{ANCHORS[score]}"'
        for score, label in DIFFICULTY_LEVELS.items()
    )
    return f"""A Norwegian reading passage was analyzed:

Title: {title}
{analysis}

Score its reading difficulty using these fixed anchors as your reference:
{anchor_lines}

Scale:
  1=A1  avg <8 words/sentence, only basic words, present tense, concrete everyday
  2=A2  avg 8-12 words, familiar words, simple past/future, few clauses
  3=B1  avg 12-17 words, general vocabulary, multiple tenses, some clauses
  4=B2  avg 17-22 words, less common words, passive/conditional, abstract content
  5=C1  avg 22+ words, formal/academic vocabulary, complex embedded clauses, dense
  6=C2  highly complex syntax, rare/specialised words, literary or technical

IMPORTANT: Use the full scale. If average sentence length is above 18 words, score is at least 4.
If hardest words are basic everyday words, score is at most 2.

Reply ONLY with JSON GÇö no markdown:
{{"difficulty_score": <integer 1-6>, "difficulty_label": "<A1|A2|B1|B2|C1|C2>", "confidence": <0.0-1.0>}}
""".strip()


# GöÇGöÇ SINGLE PASSAGE SCORING GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
_debug_printed = False


def score_single_passage(title, body):
    global _debug_printed

    title_clean = normalize_text(title)[:160]
    text_clean = safe_truncate(body)

    scores, confidences, analyses = [], [], []

    for run in range(SCORE_REPEATS):

        # GöÇGöÇ Step 1: analyse GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
        analysis_raw = call_ollama(
            build_analysis_prompt(title_clean, text_clean),
            num_predict=180
        )
        if not analysis_raw or len(analysis_raw.strip()) < 20:
            analysis_raw = "No analysis returned."

        if not _debug_printed:
            print("\n" + "GöÇ" * 55)
            print("DEBUG GÇö Step 1 (analysis) raw output:")
            print(repr(analysis_raw[:500]))
            print("GöÇ" * 55)

        # GöÇGöÇ Step 2: score GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
        score_raw = call_ollama(
            build_score_prompt(title_clean, analysis_raw.strip()[:400]),
            num_predict=80
        )

        if not _debug_printed:
            print("DEBUG GÇö Step 2 (score) raw output:")
            print(repr(score_raw[:300]))
            print("GöÇ" * 55 + "\n")
            _debug_printed = True

        # Parse GÇö three layers of fallback
        parsed = extract_json_object(score_raw)
        if parsed is None:
            parsed = parse_key_value_fallback(score_raw)
        if parsed is None:
            parsed = extract_score_from_free_text(score_raw)

        if parsed is None:
            print(f"   GÜán+Å  Could not parse score (run {run + 1}): {repr((score_raw or '')[:120])}")
            scores.append(3.0)
            confidences.append(0.0)
            analyses.append(analysis_raw.strip()[:300])
            continue

        score = clamp_score(parsed.get("difficulty_score", 3)) or 3.0
        conf = clamp_confidence(parsed.get("confidence", 0.5))

        scores.append(score)
        confidences.append(conf)
        analyses.append(analysis_raw.strip()[:300])
        time.sleep(0.02)

    avg_score = float(np.mean(scores))
    avg_conf = float(np.mean(confidences))
    score_std = float(np.std(scores))
    final_score = clamp_score(round(avg_score))
    final_label = score_to_label(final_score)
    best_idx = int(np.argmin(np.abs(np.array(scores) - avg_score)))

    return {
        "difficulty_score": final_score,
        "difficulty_score_raw": round(avg_score, 3),
        "difficulty_label": final_label,
        "difficulty_score_std": round(score_std, 3),
        "analysis": analyses[best_idx],
        "confidence": round(avg_conf, 3),
        "scores_per_run": scores,
    }


# GöÇGöÇ BATCH SCORING GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
def score_passages(df):
    print(
        f"\n=ƒñû Scoring {len(df)} passages | "
        f"model={OLLAMA_MODEL} | repeats={SCORE_REPEATS} | temp={TEMPERATURE}"
    )

    if not check_ollama():
        raise RuntimeError("Ollama not reachable on localhost:11434")
    warmup_ollama()

    results = []
    total = len(df)

    for i, row in df.iterrows():
        result = score_single_passage(row["title_clean"], row["body_clean"])
        results.append(result)

        print(
            f"   [{i + 1:>3}/{total}] "
            f"Score={result['difficulty_score']} ({result['difficulty_label']}) | "
            f"raw={result['difficulty_score_raw']:.2f} | "
            f"std={result['difficulty_score_std']:.2f} | "
            f"conf={result['confidence']:.2f} | "
            f"runs={result['scores_per_run']}"
        )

        if (i + 1) % SAVE_EVERY_N == 0:
            _save_checkpoint(df, results, i + 1)

        time.sleep(0.05)

    return results


def _save_checkpoint(df, results, up_to):
    temp = df.iloc[:up_to].copy()
    for col in ["difficulty_score", "difficulty_score_raw", "difficulty_label",
                "difficulty_score_std", "analysis", "confidence"]:
        temp[col] = [r[col] for r in results]
    temp.to_csv("difficulty_checkpoint.csv", index=False, encoding="utf-8-sig")
    print(f"   =ƒÆ+ Checkpoint saved ({up_to} passages)")


# GöÇGöÇ OUTPUT GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
def build_output(df, results):
    df = df.copy()
    for col in ["difficulty_score", "difficulty_score_raw", "difficulty_label",
                "difficulty_score_std", "analysis", "confidence"]:
        df[col] = [r[col] for r in results]
    return df


def print_distribution(df_out):
    print("\n=ƒôè Difficulty Distribution:")
    print("GöÇ" * 50)
    dist = (
        df_out.groupby(["difficulty_label", "difficulty_score"])["serialNumber"]
        .count().rename("count").reset_index().sort_values("difficulty_score")
    )
    for _, row in dist.iterrows():
        bar = "Gûê" * int(row["count"])
        print(f"  {row['difficulty_label']} ({int(row['difficulty_score'])}): {bar} {int(row['count'])}")

    high_var = df_out[df_out["difficulty_score_std"] >= 1.0]
    if not high_var.empty:
        print(f"\nGÜán+Å  {len(high_var)} high-variance passage(s) (std GëÑ 1.0) GÇö consider manual review:")
        for _, row in high_var.iterrows():
            print(
                f"   [{row['serialNumber']}] {row['title_clean'][:55]} | "
                f"score={row['difficulty_score']} std={row['difficulty_score_std']}"
            )

    print(f"\n   Mean  : {df_out['difficulty_score'].mean():.2f}")
    print(f"   Median: {df_out['difficulty_score'].median():.2f}")
    print(f"   Range : {df_out['difficulty_score'].min():.0f} GÇô {df_out['difficulty_score'].max():.0f}")


# GöÇGöÇ MAIN GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
def main():
    print("=" * 55)
    print("  Norwegian Difficulty Scorer  v2")
    print("  Two-step CoT | Passage-first prompts")
    print(f"  Model: {OLLAMA_MODEL} | Temp: {TEMPERATURE} | Repeats: {SCORE_REPEATS}")
    print("=" * 55)

    df = load_data(CSV_PATH)
    results = score_passages(df)
    df_out = build_output(df, results)

    out_path = "difficulty_scored_passages.csv"
    df_out[[
        "serialNumber", "title", "difficulty_score", "difficulty_score_raw",
        "difficulty_label", "difficulty_score_std", "analysis", "confidence"
    ]].to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n=ƒÆ+ Saved GåÆ {out_path}")
    print_distribution(df_out)
    print("\nG£à Done!")


if __name__ == "__main__":
    main()
