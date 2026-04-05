"""
topic_pipeline.py
=================
Pipeline for extracting and clustering topics from Norwegian children's texts.
Uses Groq API (llama-3.3-70b-versatile).

STAGES:
  1. Load & preprocess TSV
  2. Per-text topic extraction  (with schema validation + string normalization)
  3. Draft taxonomy proposal    (model proposes categories from normalized labels only)
     → PAUSE: researcher reviews/edits taxonomy_draft.json → saves as taxonomy_final.json
  4. Text assignment            (assign each text to the finalized taxonomy)
  5. Export CSV + JSON artifacts

USAGE:
  pip install groq pandas
  export GROQ_API_KEY=gsk_your_key_here

  # Full run (first time):
  python topic_pipeline.py --input texts.tsv

  # After editing taxonomy_final.json, re-run assignment only:
  python topic_pipeline.py --input texts.tsv --assign-only

  # Re-propose taxonomy from scratch (keep extraction, redo clustering):
  python topic_pipeline.py --input texts.tsv --skip-extraction

OUTPUT FILES:
  extracted_topics.json   — per-text main_topic + sub_topics
  taxonomy_draft.json     — model-proposed broad categories (edit this → taxonomy_final.json)
  taxonomy_final.json     — finalized taxonomy used for assignment
  assignments.json        — text_id → broad_topics mapping
  results.csv             — everything merged into one CSV
"""

import argparse
import json
import os
import re
import time
import sys
from pathlib import Path
from collections import Counter

from groq import Groq
import pandas as pd


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MODEL           = "llama-3.3-70b-versatile"
BATCH_SIZE      = 10
CHECKPOINT_FILE = "topics_checkpoint.json"
MAX_RETRIES     = 3
RETRY_DELAY     = 5

# Output artifact paths
EXTRACTED_JSON  = "extracted_topics.json"
DRAFT_JSON      = "taxonomy_draft.json"
FINAL_JSON      = "taxonomy_final.json"
ASSIGNMENTS_JSON = "assignments.json"
OUTPUT_CSV      = "results.csv"


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def fix_encoding(text: str) -> str:
    """Fix UTF-8 text mis-decoded as Latin-1. e.g. 'Ã˜rner' → 'Ørner'"""
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def normalize_topic(topic: str) -> str:
    """
    Normalize a topic string for consistent clustering:
    - Strip leading/trailing whitespace
    - Collapse internal whitespace
    - Title-case (e.g. 'natur og dyr' → 'Natur og dyr')
    - Remove trailing punctuation
    """
    if not isinstance(topic, str):
        return ""
    topic = topic.strip()
    topic = re.sub(r'\s+', ' ', topic)
    topic = topic.rstrip('.,;:!?')
    # Title-case only the first letter, preserve rest
    if topic:
        topic = topic[0].upper() + topic[1:]
    return topic


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# SCHEMA VALIDATION
# ─────────────────────────────────────────────

class SchemaError(Exception):
    pass


def validate_extracted_topics(raw: dict) -> dict:
    """
    Validate and coerce the LLM's extraction response into the expected schema.

    Expected:
      {
        "main_topic": str (2-5 words),
        "sub_topics": list[str] (2-4 items)
      }

    Raises SchemaError if the structure is unrecoverable.
    Returns a clean, normalized dict if valid (with coercion where safe).
    """
    if not isinstance(raw, dict):
        raise SchemaError(f"Expected dict, got {type(raw).__name__}")

    # --- main_topic ---
    main = raw.get("main_topic", "")
    if not isinstance(main, str):
        raise SchemaError(f"main_topic must be a string, got {type(main).__name__}")
    main = normalize_topic(main)
    if not main:
        raise SchemaError("main_topic is empty")

    # --- sub_topics ---
    subs = raw.get("sub_topics", [])

    # Coerce: if model returned a string instead of a list, split on commas
    if isinstance(subs, str):
        subs = [s.strip() for s in subs.split(",") if s.strip()]

    if not isinstance(subs, list):
        raise SchemaError(f"sub_topics must be a list, got {type(subs).__name__}")

    # Coerce: filter out non-string items
    subs = [normalize_topic(s) for s in subs if isinstance(s, str)]
    subs = [s for s in subs if s]  # drop empty strings after normalization

    if len(subs) < 1:
        raise SchemaError("sub_topics has no valid entries")

    return {
        "main_topic": main,
        "sub_topics": subs,
    }


def validate_taxonomy(taxonomy: list) -> list:
    """
    Validate the taxonomy list (broad_topics).
    Returns normalized list or raises SchemaError.
    """
    if not isinstance(taxonomy, list):
        raise SchemaError(f"Taxonomy must be a list, got {type(taxonomy).__name__}")
    normalized = [normalize_topic(t) for t in taxonomy if isinstance(t, str)]
    normalized = [t for t in normalized if t]
    if not (12 <= len(normalized) <= 15):
        raise SchemaError(f"Taxonomy must have 12-15 topics, got {len(normalized)}")
    return normalized


def validate_assignments(assignments: dict, taxonomy: list, index_to_id: dict) -> dict:
    """
    Validate assignment dict and remap short numeric keys → real text_ids.
    Coerces any assigned topic not in taxonomy to the closest match (or drops it).
    """
    taxonomy_set = set(taxonomy)
    real_assignments = {}

    for key, categories in assignments.items():
        real_id = index_to_id.get(str(key))
        if not real_id:
            continue

        if not isinstance(categories, list):
            categories = [categories] if isinstance(categories, str) else []

        # Normalize and filter to only valid taxonomy categories
        clean_cats = []
        for cat in categories:
            norm = normalize_topic(cat)
            if norm in taxonomy_set:
                clean_cats.append(norm)

        if clean_cats:
            real_assignments[real_id] = clean_cats

    return real_assignments


# ─────────────────────────────────────────────
# STAGE 1 — LOAD & PREPROCESS
# ─────────────────────────────────────────────

def strip_markdown(text: str) -> str:
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]*\)', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[*_]{1,3}([^*_]+)[*_]{1,3}', r'\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_subheadings(body: str) -> list[str]:
    return re.findall(r'^#{2,6}\s+(.+)$', body, flags=re.MULTILINE)


def is_empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        import math
        return math.isnan(value)
    return str(value).strip().lower() in ("", "nan", "none")


def preprocess_row(row: pd.Series) -> dict:
    title = fix_encoding(str(row.get("title", "") or ""))
    body  = fix_encoding(str(row.get("body",  "") or ""))

    subheadings = extract_subheadings(body)
    clean_body  = strip_markdown(body)

    lines = clean_body.splitlines()
    if lines and lines[0].strip().lower() == title.strip().lower():
        clean_body = "\n".join(lines[1:]).strip()

    return {
        "text_id":     str(row.get("sanity_text_id", row.get("text_id", ""))),
        "serial":      str(row.get("serialnumber",   row.get("serial_number", ""))),
        "title":       title,
        "subheadings": subheadings,
        "clean_body":  clean_body,
    }


def load_and_preprocess(filepath: str) -> list[dict]:
    path = Path(filepath)
    sep  = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","

    df = None
    for enc in ("utf-8", "latin-1", "utf-8-sig"):
        try:
            df = pd.read_csv(filepath, sep=sep, encoding=enc, dtype=str)
            print(f"  Read {len(df)} raw rows with encoding={enc}")
            break
        except Exception:
            continue

    if df is None:
        raise ValueError(f"Could not read file: {filepath}")

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    before    = len(df)
    title_col = "title" if "title" in df.columns else None
    body_col  = "body"  if "body"  in df.columns else None

    if title_col and body_col:
        df = df[~(df[title_col].apply(is_empty) & df[body_col].apply(is_empty))]
    elif title_col:
        df = df[~df[title_col].apply(is_empty)]

    df      = df.reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} empty rows → {len(df)} valid rows remaining.")

    texts = [preprocess_row(row) for _, row in df.iterrows()]
    print(f"  Preprocessed {len(texts)} texts.")
    return texts


# ─────────────────────────────────────────────
# STAGE 2 — PER-TEXT TOPIC EXTRACTION
# ─────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """Du er en ekspert på å klassifisere norske barnetekster.
For hver tekst du mottar, skal du returnere KUN et JSON-objekt – ingen forklaring, ingen markdown-blokker.

Format:
{
  "main_topic": "ett kortfattet norsk emne (2-5 ord)",
  "sub_topics": ["emne1", "emne2", "emne3"]
}

Regler:
- Skriv alle emner på norsk bokmål
- main_topic skal fange tekstens primære tema
- sub_topics: 2-4 underkategorier eller relaterte temaer
- Vær spesifikk nok til at emnene er meningsfulle, men ikke for smale
- Eksempel for en tekst om ørner: {"main_topic": "Rovfugler", "sub_topics": ["Natur og dyreliv", "Norsk fauna", "Kultur og symbolikk"]}"""


def build_extraction_prompt(text: dict) -> str:
    subheadings_str = ""
    if text["subheadings"]:
        subheadings_str = "\nDeltitler: " + " | ".join(text["subheadings"])
    body_preview = text["clean_body"][:1500]
    if len(text["clean_body"]) > 1500:
        body_preview += "\n[...]"
    return f"""Tittel: {text["title"]}{subheadings_str}\n\nTekst:\n{body_preview}"""


def extract_topics_for_text(client: Groq, text: dict) -> dict:
    prompt = build_extraction_prompt(text)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=300,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r'^```[a-z]*\n?', '', raw)
            raw = re.sub(r'\n?```$', '', raw)

            parsed   = json.loads(raw)
            validated = validate_extracted_topics(parsed)

            return {
                **text,
                "main_topic":       validated["main_topic"],
                "sub_topics":       validated["sub_topics"],
                "extraction_error": None,
            }

        except json.JSONDecodeError as e:
            print(f"    [!] JSON parse error for '{text['title']}' (attempt {attempt}): {e}")
        except SchemaError as e:
            print(f"    [!] Schema error for '{text['title']}' (attempt {attempt}): {e}")
        except Exception as e:
            print(f"    [!] API error for '{text['title']}' (attempt {attempt}): {e}")
            if "rate_limit" in str(e).lower() or "429" in str(e):
                print(f"    Rate limit — waiting 60s...")
                time.sleep(60)
                continue

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * attempt)

    return {
        **text,
        "main_topic":       "",
        "sub_topics":       [],
        "extraction_error": "Failed after retries",
    }


def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(results: dict):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def run_extraction(client: Groq, texts: list[dict]) -> list[dict]:
    checkpoint = load_checkpoint()
    results    = {tid: data for tid, data in checkpoint.items()}

    remaining = [t for t in texts if t["text_id"] not in results]
    total     = len(texts)
    done      = len(results)

    if done > 0:
        print(f"  Resuming: {done}/{total} done, {len(remaining)} remaining.")

    for batch_start in range(0, len(remaining), BATCH_SIZE):
        batch         = remaining[batch_start: batch_start + BATCH_SIZE]
        batch_num     = (batch_start // BATCH_SIZE) + 1
        total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"  Batch {batch_num}/{total_batches} — {len(batch)} texts...")

        for text in batch:
            result = extract_topics_for_text(client, text)
            results[text["text_id"]] = result
            done  += 1
            status = "✓" if not result["extraction_error"] else "✗"
            print(f"    [{done}/{total}] {status} {text['title'][:50]}")

        save_checkpoint(results)
        print(f"  Checkpoint saved.")

        if batch_start + BATCH_SIZE < len(remaining):
            time.sleep(2)

    extracted = list(results.values())

    # Save extracted topics as JSON artifact
    save_json(EXTRACTED_JSON, [
        {
            "text_id":    t["text_id"],
            "title":      t["title"],
            "main_topic": t.get("main_topic", ""),
            "sub_topics": t.get("sub_topics", []),
            "error":      t.get("extraction_error"),
        }
        for t in extracted
    ])
    print(f"  Saved extraction results → {EXTRACTED_JSON}")

    return extracted


# ─────────────────────────────────────────────
# STAGE 3 — DRAFT TAXONOMY PROPOSAL
# ─────────────────────────────────────────────
#
# The model only sees normalized main_topic labels + their frequency —
# NOT the full text context. This keeps the prompt small and focused.
#
# The model proposes 12-15 broad categories based purely on what topics
# actually appear in the data. The researcher then reviews and finalizes.
#
# ─────────────────────────────────────────────

TAXONOMY_SYSTEM_PROMPT = """Du er en ekspert på å lage emnestruktur for norske barnetekster.

Du vil motta en liste over emner hentet fra barnetekster, med hvor mange tekster hvert emne dekker.
Basert på dette skal du foreslå 12-15 overordnede kategorier som dekker alle emnene.

Regler:
- NØYAKTIG 12-15 kategorier
- Kategoriene skal være TYDELIG FORSKJELLIGE — ingen overlapp
  (f.eks. IKKE både "Natur og dyr" og "Dyreliv og natur")
- Kategoriene skal dekke bredden i tekstsamlingen
- Breie nok til å være meningsfulle, men ikke så brede at alt havner i én kategori
- På norsk bokmål, passende for barn
- Hvert emne fra listen skal kunne plasseres i minst én kategori

Returner KUN et JSON-objekt:
{
  "broad_topics": ["Kategori1", "Kategori2", ...],
  "rationale": {
    "Kategori1": "Kort begrunnelse for denne kategorien",
    ...
  }
}"""


def build_taxonomy_prompt(extracted: list[dict]) -> str:
    """
    Build a compact prompt from normalized main_topic labels and their frequency.
    Does NOT include full text body — only the extracted topic labels.
    """
    # Count frequency of each normalized main_topic
    topic_counts = Counter(
        t.get("main_topic", "")
        for t in extracted
        if t.get("main_topic")
    )

    # Sort by frequency descending
    sorted_topics = sorted(topic_counts.items(), key=lambda x: -x[1])

    lines = [f"- {topic} ({count} tekster)" for topic, count in sorted_topics]

    return (
        f"Her er alle {len(extracted)} tekstenes main_topic med frekvens:\n\n"
        + "\n".join(lines)
        + "\n\nForeslå 12-15 overordnede kategorier som dekker disse emnene."
    )


def propose_taxonomy(client: Groq, extracted: list[dict]) -> dict:
    """
    Ask the model to propose a draft taxonomy based on normalized topic labels only.
    Returns {"broad_topics": [...], "rationale": {...}}
    """
    prompt = build_taxonomy_prompt(extracted)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=2000,
                messages=[
                    {"role": "system", "content": TAXONOMY_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r'^```[a-z]*\n?', '', raw)
            raw = re.sub(r'\n?```$', '', raw)

            result   = json.loads(raw)
            taxonomy = validate_taxonomy(result.get("broad_topics", []))
            result["broad_topics"] = taxonomy

            return result

        except (json.JSONDecodeError, SchemaError) as e:
            print(f"  [!] Taxonomy proposal error (attempt {attempt}): {e}")
        except Exception as e:
            print(f"  [!] API error (attempt {attempt}): {e}")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * attempt)

    raise RuntimeError("Taxonomy proposal failed after all retries.")


# ─────────────────────────────────────────────
# STAGE 4 — TEXT ASSIGNMENT
# ─────────────────────────────────────────────
#
# Given a finalized taxonomy (from taxonomy_final.json),
# assign each text to 1-3 broad topics.
# Uses short numeric indexes to keep response size small.
# ─────────────────────────────────────────────

ASSIGNMENT_SYSTEM_PROMPT = """Du er en ekspert på å klassifisere norske barnetekster.

Du vil motta:
1. En godkjent liste over brede kategorier
2. En liste over tekster (nummerert 1, 2, 3...) med tittel og hovedemne

Tildel hver tekst 1-3 kategorier fra den godkjente listen.

Regler:
- Bruk KUN kategorier fra den godkjente listen — ikke oppfinn nye
- Hver tekst skal ha 1-3 kategorier
- Velg kategoriene som best beskriver tekstens innhold
- Bruk de korte nummerne (1, 2, 3...) som nøkler

Returner KUN et JSON-objekt:
{
  "assignments": {
    "1": ["Kategori A", "Kategori B"],
    "2": ["Kategori C"],
    ...
  }
}"""


def build_assignment_prompt(taxonomy: list, extracted: list[dict], index_to_id: dict) -> str:
    id_to_index = {v: k for k, v in index_to_id.items()}

    taxonomy_str = "\n".join(f"- {t}" for t in taxonomy)

    text_lines = []
    for t in extracted:
        idx = id_to_index.get(t["text_id"], "?")
        text_lines.append(
            f'{idx}. Tittel: {t["title"]} | Hovedemne: {t.get("main_topic", "")}'
        )

    return (
        f"Godkjente kategorier:\n{taxonomy_str}\n\n"
        f"Tekster:\n" + "\n".join(text_lines)
        + "\n\nTildel hver tekst 1-3 kategorier fra listen ovenfor."
    )


def assign_texts(client: Groq, taxonomy: list, extracted: list[dict]) -> dict:
    """Assign each text to 1-3 broad topics from the finalized taxonomy."""
    index_to_id = {str(i): t["text_id"] for i, t in enumerate(extracted, start=1)}
    prompt      = build_assignment_prompt(taxonomy, extracted, index_to_id)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=6000,
                messages=[
                    {"role": "system", "content": ASSIGNMENT_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r'^```[a-z]*\n?', '', raw)
            raw = re.sub(r'\n?```$', '', raw)

            result      = json.loads(raw)
            assignments = validate_assignments(
                result.get("assignments", {}), taxonomy, index_to_id
            )

            # Warn about unassigned texts
            assigned_ids = set(assignments.keys())
            all_ids      = {t["text_id"] for t in extracted}
            missing      = all_ids - assigned_ids
            if missing:
                print(f"  [!] {len(missing)} texts were not assigned — they will have empty broad_topics.")

            return assignments

        except (json.JSONDecodeError, SchemaError) as e:
            print(f"  [!] Assignment error (attempt {attempt}): {e}")
        except Exception as e:
            print(f"  [!] API error (attempt {attempt}): {e}")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * attempt)

    raise RuntimeError("Assignment failed after all retries.")


# ─────────────────────────────────────────────
# STAGE 5 — EXPORT
# ─────────────────────────────────────────────

def export_results(extracted: list[dict], taxonomy: list, assignments: dict, output_path: str):
    rows = []
    for t in extracted:
        broad = assignments.get(t["text_id"], [])
        rows.append({
            "text_id":          t["text_id"],
            "serial_number":    t["serial"],
            "title":            t["title"],
            "main_topic":       t.get("main_topic", ""),
            "sub_topics":       " | ".join(t.get("sub_topics", [])),
            "broad_topics":     " | ".join(broad),
            "extraction_error": t.get("extraction_error", ""),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  CSV → {output_path}")

    # Save assignments JSON artifact
    save_json(ASSIGNMENTS_JSON, {
        "taxonomy":    taxonomy,
        "assignments": assignments,
    })
    print(f"  Assignments → {ASSIGNMENTS_JSON}")

    # Print summary
    print(f"\n{'─'*56}")
    print(f"  FINAL BROAD TOPICS ({len(taxonomy)}):")
    print(f"{'─'*56}")
    for topic in taxonomy:
        count = sum(1 for r in rows if topic in r["broad_topics"].split(" | "))
        print(f"  {topic:44s}  ({count} texts)")

    errors = [r for r in rows if r["extraction_error"]]
    if errors:
        print(f"\n  [!] {len(errors)} extraction errors:")
        for e in errors:
            print(f"      - {e['title']}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract and cluster topics from Norwegian children's texts."
    )
    parser.add_argument("--input",           required=True, help="Path to input TSV/CSV")
    parser.add_argument("--output",          default=OUTPUT_CSV, help=f"Output CSV (default: {OUTPUT_CSV})")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip extraction, load from checkpoint. Re-proposes taxonomy.")
    parser.add_argument("--assign-only",     action="store_true",
                        help="Skip extraction + taxonomy proposal. Loads taxonomy_final.json and assigns texts.")
    args = parser.parse_args()

    api_key = "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"
    if not api_key:
        print("Error: GROQ_API_KEY not set.")
        print("  export GROQ_API_KEY=gsk_your_key_here")
        sys.exit(1)

    client = Groq(api_key=api_key)
    print(f"  Model: {MODEL}")

    # ── Stage 1: Load & Preprocess ───────────────────────────
    print("\n[Stage 1] Loading and preprocessing...")
    texts = load_and_preprocess(args.input)

    # ── Stage 2: Extraction ──────────────────────────────────
    if args.skip_extraction or args.assign_only:
        print("\n[Stage 2] Loading extracted topics from checkpoint...")
        if os.path.exists(EXTRACTED_JSON):
            raw_extracted = load_json(EXTRACTED_JSON)
            # Merge clean_body back from preprocessing (needed for assignment prompt)
            text_map  = {t["text_id"]: t for t in texts}
            extracted = []
            for e in raw_extracted:
                tid = e.get("text_id", "")
                if tid in text_map:
                    extracted.append({**text_map[tid], **e})
        else:
            checkpoint = load_checkpoint()
            valid_ids  = {t["text_id"] for t in texts}
            extracted  = [v for k, v in checkpoint.items() if k in valid_ids]
        print(f"  Loaded {len(extracted)} texts.")
    else:
        print(f"\n[Stage 2] Extracting topics ({len(texts)} texts)...")
        extracted = run_extraction(client, texts)

    # ── Stage 3: Taxonomy Proposal ───────────────────────────
    if args.assign_only:
        # Load finalized taxonomy directly
        if not os.path.exists(FINAL_JSON):
            print(f"\nError: {FINAL_JSON} not found.")
            print(f"  Run without --assign-only first to generate a draft taxonomy.")
            print(f"  Then copy {DRAFT_JSON} → {FINAL_JSON} and edit as needed.")
            sys.exit(1)
        final_data = load_json(FINAL_JSON)
        taxonomy   = final_data.get("broad_topics", [])
        print(f"\n[Stage 3] Loaded finalized taxonomy ({len(taxonomy)} categories) from {FINAL_JSON}")

    else:
        print(f"\n[Stage 3] Proposing draft taxonomy from normalized topic labels...")
        draft = propose_taxonomy(client, extracted)

        # Save draft for researcher review
        save_json(DRAFT_JSON, draft)

        print(f"\n  Draft taxonomy saved → {DRAFT_JSON}")
        print(f"\n  {'─'*54}")
        print(f"  DRAFT BROAD TOPICS ({len(draft['broad_topics'])}):")
        print(f"  {'─'*54}")
        for topic in draft["broad_topics"]:
            rationale = draft.get("rationale", {}).get(topic, "")
            print(f"  • {topic}")
            if rationale:
                print(f"    → {rationale}")

        print(f"""
  ╔══════════════════════════════════════════════════════╗
  ║  RESEARCHER REVIEW STEP                              ║
  ║                                                      ║
  ║  1. Open: {DRAFT_JSON:<42s}║
  ║  2. Review the proposed broad_topics list            ║
  ║     - Merge overlapping categories                   ║
  ║     - Rename categories to better fit your thesis    ║
  ║     - Add or remove categories as needed             ║
  ║  3. Save your edited version as: {FINAL_JSON:<19s}║
  ║  4. Re-run with:                                     ║
  ║       python topic_pipeline.py \\                     ║
  ║         --input {Path(args.input).name:<38s}║
  ║         --assign-only                                ║
  ╚══════════════════════════════════════════════════════╝
""")
        print("  Pipeline paused. Edit the taxonomy and re-run with --assign-only.")
        return  # ← deliberate pause

    # ── Stage 4: Assignment ──────────────────────────────────
    print(f"\n[Stage 4] Assigning texts to finalized taxonomy...")
    assignments = assign_texts(client, taxonomy, extracted)
    print(f"  Assigned {len(assignments)}/{len(extracted)} texts.")

    # ── Stage 5: Export ──────────────────────────────────────
    print(f"\n[Stage 5] Exporting results...")
    export_results(extracted, taxonomy, assignments, args.output)

    print("\nDone! ✓")


if __name__ == "__main__":
    main()