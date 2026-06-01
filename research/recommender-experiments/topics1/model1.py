import json
import time
from typing import List, Dict, Any

import pandas as pd
import requests

# =========================
# CONFIG
# =========================
INPUT_CSV = "../data/question_texts_texts.csv"
OUTPUT_CSV = "../data/processed/labeled_passages.csv"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3:4b"   # change if needed, e.g. llama3.1:8b, mistral, gemma3

# Which columns to combine as the input passage
TEXT_COLUMNS = ["title", "body"]   # adjust to your CSV
ID_COLUMN = None                   # e.g. "id" if you have one, else None

# Allowed labels
LABELS = [
    "health",
    "technology",
    "nature",
    "sports",
    "education",
    "culture",
    "politics",
    "economy",
    "history",
    "science",
    "other"
]

# Inference behavior
TEMPERATURE = 0
MAX_RETRIES = 3
REQUEST_TIMEOUT = 180

# =========================
# HELPERS
# =========================
def build_text(row: pd.Series, text_columns: List[str]) -> str:
    parts = []
    for col in text_columns:
        if col in row and pd.notna(row[col]):
            value = str(row[col]).strip()
            if value:
                parts.append(value)
    return "\n\n".join(parts).strip()


def clean_text(text: str) -> str:
    """Light cleanup only. Keeps original content mostly intact."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()


def warm_up_model(model_name: str) -> None:
    """
    Optional warmup call so the first real request is less slow.
    Ollama allows empty/preload-style requests and keep_alive settings. :contentReference[oaicite:1]{index=1}
    """
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": " "}],
        "stream": False,
        "keep_alive": "10m"
    }
    try:
        requests.post(OLLAMA_URL, json=payload, timeout=60)
    except Exception:
        pass


def classify_passage(text: str, model_name: str, labels: List[str]) -> Dict[str, Any]:
    """
    Calls local Ollama and forces structured JSON output.
    """
    schema = {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "enum": labels
            },
            "confidence": {
                "type": "number"
            },
            "reason": {
                "type": "string"
            }
        },
        "required": ["label", "confidence", "reason"],
        "additionalProperties": False
    }

    system_prompt = (
        "You are a text topic classifier.\n"
        "Choose the SINGLE best label for the passage.\n"
        "Return only valid JSON matching the schema.\n"
        "Do not invent new labels.\n"
        "Base the label on the overall main topic of the passage.\n"
        "If no label fits well, use 'other'."
    )

    user_prompt = f"""
Allowed labels:
{", ".join(labels)}

Task:
Classify the following passage into exactly one best label.

Rules:
- Pick the main topic, not every possible subtopic.
- If the passage is mostly about animals, environment, biology, ecosystems, or wildlife, choose "nature".
- If the passage is about school, reading tasks, learning, teaching, or students, choose "education".
- If uncertain, choose the closest best match.
- Confidence should be between 0 and 1.
- Keep the reason short.

Passage:
\"\"\"
{text}
\"\"\"
""".strip()

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": schema,       # structured outputs supported by Ollama :contentReference[oaicite:2]{index=2}
        "options": {
            "temperature": TEMPERATURE
        },
        "keep_alive": "10m"
    }

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()

            # Ollama chat response returns content in message.content. :contentReference[oaicite:3]{index=3}
            content = data["message"]["content"]

            parsed = json.loads(content)

            label = parsed.get("label", "other")
            confidence = parsed.get("confidence", 0.0)
            reason = parsed.get("reason", "")

            if label not in labels:
                label = "other"

            try:
                confidence = float(confidence)
            except Exception:
                confidence = 0.0

            confidence = max(0.0, min(1.0, confidence))

            return {
                "label": label,
                "confidence": confidence,
                "reason": reason
            }

        except Exception as e:
            last_error = str(e)
            time.sleep(1.5 * attempt)

    return {
        "label": "other",
        "confidence": 0.0,
        "reason": f"Failed after retries: {last_error}"
    }


# =========================
# MAIN
# =========================
def main():
    print("Loading CSV...")
    df = pd.read_csv(INPUT_CSV)

    missing_cols = [c for c in TEXT_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}")

    print(f"Loaded {len(df)} rows")
    print("Warming up Ollama model...")
    warm_up_model(MODEL_NAME)

    combined_texts = []
    labels_out = []
    confidences_out = []
    reasons_out = []

    for idx, row in df.iterrows():
        text = build_text(row, TEXT_COLUMNS)
        text = clean_text(text)

        combined_texts.append(text)

        if not text:
            labels_out.append("other")
            confidences_out.append(0.0)
            reasons_out.append("Empty text")
            continue

        print(f"[{idx + 1}/{len(df)}] Classifying...")

        result = classify_passage(text=text, model_name=MODEL_NAME, labels=LABELS)

        labels_out.append(result["label"])
        confidences_out.append(result["confidence"])
        reasons_out.append(result["reason"])

    df["combined_text"] = combined_texts
    df["predicted_label"] = labels_out
    df["label_confidence"] = confidences_out
    df["label_reason"] = reasons_out

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Done. Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
