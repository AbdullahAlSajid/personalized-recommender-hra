"""
topic_pipeline.py
=================
Pipeline for extracting and clustering topics from Norwegian children's texts.
Uses Groq API (llama-3.3-70b-versatile).

STAGES:
  1.   Load & preprocess
  2.   Per-text topic extraction
  2.5  Extraction quality check  ← samples 30 texts for manual review
  3.   Draft taxonomy proposal
  3.5  Gap analysis              ← flags missing/overlapping categories
       [PAUSE — researcher edits taxonomy_draft.json → taxonomy_final.json]
  4.   Text assignment
  4.5  Assignment quality check  ← flags thin/bloated categories, spot checks
       [PAUSE — researcher reviews, edits assignments.json if needed]
  5.   Export CSV + JSON artifacts

EVALUATION (--evaluate):
  Exports evaluation_sample.csv for manual labeling.
  After manual labeling, run --evaluate --compare to compute metrics.

USAGE:
  pip install groq pandas python-dotenv

  # Full run:
  python topic_pipeline.py --input texts.tsv

  # After editing taxonomy_final.json:
  python topic_pipeline.py --input texts.tsv --assign-only

  # Re-propose taxonomy without re-extracting:
  python topic_pipeline.py --input texts.tsv --skip-extraction

  # Export evaluation sample:
  python topic_pipeline.py --input texts.tsv --evaluate

  # Compute metrics after manual labeling:
  python topic_pipeline.py --input texts.tsv --evaluate --compare

OUTPUT FILES:
  extracted_topics.json      — per-text extraction results
  taxonomy_draft.json        — model-proposed taxonomy
  taxonomy_final.json        — researcher-finalized taxonomy
  assignments.json           — text_id → broad_topics mapping
  results.csv                — final merged output
  evaluation_sample.csv      — sample for manual review/labeling
  evaluation_metrics.json    — computed metrics (after --compare)
"""

import argparse
import json
import os
import re
import time
import sys
import random
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv

from groq import Groq
import pandas as pd


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MODEL            = "llama-3.3-70b-versatile"
BATCH_SIZE       = 10
CHECKPOINT_FILE  = "topics_checkpoint.json"
MAX_RETRIES      = 3
RETRY_DELAY      = 5

EXTRACTED_JSON   = "extracted_topics.json"
DRAFT_JSON       = "taxonomy_draft.json"
FINAL_JSON       = "taxonomy_final.json"
ASSIGNMENTS_JSON = "assignments.json"
OUTPUT_CSV       = "results.csv"
EVAL_SAMPLE_CSV  = "evaluation_sample.csv"
EVAL_METRICS_JSON = "evaluation_metrics.json"
ASSIGNMENT_EVAL_CSV  = "assignment_spot_check.csv"
EVAL_EXTRACTION_METRICS_JSON = "evaluation_extraction_metrics.json"
EVAL_ASSIGNMENT_METRICS_JSON = "evaluation_assignment_metrics.json"


EXTRACTION_SAMPLE_SIZE = 30   # texts sampled for extraction quality check
CATEGORY_SPOT_SIZE     = 5    # texts sampled per category for assignment check
THIN_CATEGORY_THRESHOLD  = 5  # categories with fewer texts flagged as thin
BLOATED_CATEGORY_THRESHOLD = 40  # categories with more texts flagged as bloated


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def fix_encoding(text: str) -> str:
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def normalize_topic(topic: str) -> str:
    if not isinstance(topic, str):
        return ""
    topic = topic.strip()
    topic = re.sub(r'\s+', ' ', topic)
    topic = topic.rstrip('.,;:!?')
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
    if not isinstance(raw, dict):
        raise SchemaError(f"Expected dict, got {type(raw).__name__}")

    main = raw.get("main_topic", "")
    if not isinstance(main, str):
        raise SchemaError(f"main_topic must be string, got {type(main).__name__}")
    main = normalize_topic(main)
    if not main:
        raise SchemaError("main_topic is empty")

    subs = raw.get("sub_topics", [])
    if isinstance(subs, str):
        subs = [s.strip() for s in subs.split(",") if s.strip()]
    if not isinstance(subs, list):
        raise SchemaError(f"sub_topics must be list, got {type(subs).__name__}")
    subs = [normalize_topic(s) for s in subs if isinstance(s, str)]
    subs = [s for s in subs if s]
    if len(subs) < 1:
        raise SchemaError("sub_topics has no valid entries")

    text_type = raw.get("text_type", "fagtekst")
    if text_type not in ("fagtekst", "fortelling"):
        text_type = "fagtekst"

    return {
        "main_topic": main,
        "sub_topics": subs,
        "text_type":  text_type,
    }


def validate_taxonomy(taxonomy: list) -> list:
    if not isinstance(taxonomy, list):
        raise SchemaError(f"Taxonomy must be list, got {type(taxonomy).__name__}")
    normalized = [normalize_topic(t) for t in taxonomy if isinstance(t, str)]
    normalized = [t for t in normalized if t]
    if not (12 <= len(normalized) <= 15):
        raise SchemaError(f"Taxonomy must have 12-15 topics, got {len(normalized)}")
    return normalized


def validate_assignments(assignments: dict, taxonomy: list, index_to_id: dict) -> dict:
    taxonomy_set     = set(taxonomy)
    real_assignments = {}

    for key, categories in assignments.items():
        real_id = index_to_id.get(str(key))
        if not real_id:
            continue
        if not isinstance(categories, list):
            categories = [categories] if isinstance(categories, str) else []
        clean_cats = [normalize_topic(cat) for cat in categories
                      if normalize_topic(cat) in taxonomy_set]
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
  "sub_topics": ["emne1", "emne2", "emne3"],
  "text_type": "fagtekst" eller "fortelling"
}

Regler:
- Skriv alle emner på norsk bokmål
- main_topic skal fange tekstens primære tema
- sub_topics: 2-4 underkategorier eller relaterte temaer
- text_type skal være NØYAKTIG ett av disse to:
    "fortelling" — hvis teksten er en historie, novelle, eller narrativ fiksjon med karakterer og handling
    "fagtekst"   — hvis teksten er informativ, faktabasert eller forklarende
- Eksempel fagtekst: {"main_topic": "Rovfugler", "sub_topics": ["Natur og dyreliv", "Norsk fauna"], "text_type": "fagtekst"}
- Eksempel fortelling: {"main_topic": "Fortelling om katt", "sub_topics": ["Familieliv", "Dyr som husdyr"], "text_type": "fortelling"}"""


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

            parsed    = json.loads(raw)
            validated = validate_extracted_topics(parsed)

            return {
                **text,
                "main_topic":       validated["main_topic"],
                "sub_topics":       validated["sub_topics"],
                "text_type":        validated["text_type"],
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
        "text_type":        "fagtekst",
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

    save_json(EXTRACTED_JSON, [
        {
            "text_id":    t["text_id"],
            "title":      t["title"],
            "main_topic": t.get("main_topic", ""),
            "sub_topics": t.get("sub_topics", []),
            "text_type":  t.get("text_type", "fagtekst"),
            "error":      t.get("extraction_error"),
        }
        for t in extracted
    ])
    print(f"  Saved extraction results → {EXTRACTED_JSON}")

    return extracted


# ─────────────────────────────────────────────
# STAGE 2.5 — EXTRACTION QUALITY CHECK
# ─────────────────────────────────────────────
#
# Samples EXTRACTION_SAMPLE_SIZE texts and prints them for researcher review.
# Saves the sample to evaluation_sample.csv with empty columns for manual labels.
# This lets the researcher spot problems before they poison the taxonomy.
# ─────────────────────────────────────────────

def run_extraction_quality_check(extracted: list[dict]) -> list[dict]:
    """
    Sample texts and display for researcher review.
    Returns the sampled texts for use in evaluation.
    """
    n      = min(EXTRACTION_SAMPLE_SIZE, len(extracted))
    sample = random.sample(extracted, n)

    print(f"\n  {'─'*60}")
    print(f"  EXTRACTION QUALITY CHECK — {n} randomly sampled texts:")
    print(f"  {'─'*60}")
    print(f"  {'#':<4} {'Title':<40} {'Type':<12} {'Main Topic':<30} Sub Topics")
    print(f"  {'─'*60}")

    for i, t in enumerate(sample, 1):
        subs  = ", ".join(t.get("sub_topics", []))
        ttype = t.get("text_type", "?")
        print(f"  {i:<4} {t['title'][:38]:<40} {ttype:<12} "
              f"{t.get('main_topic', '')[:28]:<30} {subs[:50]}")

    # Save sample for manual review
    rows = []
    for t in sample:
        rows.append({
            "text_id":               t["text_id"],
            "title":                 t["title"],
            "text_type_llm":         t.get("text_type", ""),
            "main_topic_llm":        t.get("main_topic", ""),
            "sub_topics_llm":        " | ".join(t.get("sub_topics", [])),
            # Empty columns for researcher to fill in
            "text_type_manual":      "",
            "main_topic_manual":     "",
            "sub_topics_manual":     "",
            "correct":               "",   # yes/no
            "notes":                 "",
        })

    df = pd.DataFrame(rows)
    df.to_csv(EVAL_SAMPLE_CSV, index=False, encoding="utf-8-sig")

    print(f"\n  Sample saved → {EVAL_SAMPLE_CSV}")
    print(f"  Review the sample above. If extractions look wrong,")
    print(f"  note which texts need fixing before continuing.")

    return sample


# ─────────────────────────────────────────────
# STAGE 3 — DRAFT TAXONOMY PROPOSAL
# ─────────────────────────────────────────────

TAXONOMY_SYSTEM_PROMPT = """Du er en ekspert på å lage emnestruktur for norske barnetekster.

Du vil motta en liste over emner hentet fra barnetekster med frekvens, samt antall fortellinger vs fagtekster.
Basert på dette skal du foreslå 12-15 overordnede kategorier.

Regler:
- NØYAKTIG 12-15 kategorier
- Hvert kategorinavn skal være ET ENKELT ORD på norsk bokmål
- Kategoriene skal være brede nok til å dekke mange tekster
- Ingen overlapp mellom kategorier
- Ordene skal være enkle og gjenkjennelige for barn mellom 8-12 år
- Kategoriene skal fungere som interessevalg i et anbefalingssystem
- Hvis mer enn 10 tekster er fortellinger skal "Fortelling" være en egen kategori

Returner KUN et JSON-objekt:
{
  "broad_topics": ["Dyr", "Vitenskap", ...],
  "rationale": {
    "Dyr": "Kort begrunnelse",
    ...
  }
}"""


def build_taxonomy_prompt(extracted: list[dict]) -> str:
    topic_counts  = Counter(
        t.get("main_topic", "") for t in extracted if t.get("main_topic")
    )
    sorted_topics = sorted(topic_counts.items(), key=lambda x: -x[1])
    lines         = [f"- {topic} ({count} tekster)" for topic, count in sorted_topics]

    fortelling_count = sum(1 for t in extracted if t.get("text_type") == "fortelling")
    fagtekst_count   = sum(1 for t in extracted if t.get("text_type") == "fagtekst")

    return (
        f"Tekstsamlingen inneholder {len(extracted)} tekster:\n"
        f"- Fagtekster: {fagtekst_count}\n"
        f"- Fortellinger: {fortelling_count}\n\n"
        f"Her er alle tekstenes main_topic med frekvens:\n\n"
        + "\n".join(lines)
        + "\n\nForeslå 12-15 overordnede kategorier som dekker disse emnene."
    )


def propose_taxonomy(client: Groq, extracted: list[dict]) -> dict:
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
# STAGE 3.5 — GAP ANALYSIS
# ─────────────────────────────────────────────

GAP_ANALYSIS_SYSTEM_PROMPT = """Du er en ekspert på tekstklassifisering.

Du vil motta:
1. En liste over foreslåtte brede kategorier
2. En liste over ekstraherte main_topics fra tekstene med frekvens
3. Antall fortellinger i samlingen

Din oppgave er å finne main_topics som IKKE passer naturlig inn i noen av de foreslåtte kategoriene.

Returner KUN et JSON-objekt:
{
  "unmatched": [
    {
      "topics": ["topic1", "topic2"],
      "reason": "Hvorfor disse ikke passer i noen eksisterende kategori",
      "suggested_category": "Forslag til ny kategori (ett enkelt ord på norsk)"
    }
  ],
  "warnings": [
    "Advarsel om mulige overlapp eller tvetydigheter i kategoriene"
  ],
  "coverage": "En kort vurdering av hvor godt kategoriene dekker tekstsamlingen"
}

Hvis alle topics passer inn, returner unmatched som tom liste [].
Vær spesielt oppmerksom på:
- Matematikk-relaterte topics som kan forveksles med Mat (mat og drikke)
- Teknologi vs Vitenskap — disse er forskjellige
- Fortellinger som trenger sin egen kategori hvis det er mange av dem"""


def run_gap_analysis(client: Groq, extracted: list[dict], taxonomy: list) -> dict:
    topic_counts  = Counter(
        t.get("main_topic", "") for t in extracted if t.get("main_topic")
    )
    sorted_topics    = sorted(topic_counts.items(), key=lambda x: -x[1])
    topics_str       = "\n".join(f"- {topic} ({count} tekster)"
                                  for topic, count in sorted_topics)
    taxonomy_str     = "\n".join(f"- {t}" for t in taxonomy)
    fortelling_count = sum(1 for t in extracted if t.get("text_type") == "fortelling")

    prompt = (
        f"Foreslåtte kategorier:\n{taxonomy_str}\n\n"
        f"Ekstraherte main_topics med frekvens:\n{topics_str}\n\n"
        f"Antall fortellinger i samlingen: {fortelling_count}\n\n"
        f"Finn topics som ikke passer inn i noen av de foreslåtte kategoriene, "
        f"og gi advarsler om mulige problemer."
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=2000,
                messages=[
                    {"role": "system", "content": GAP_ANALYSIS_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r'^```[a-z]*\n?', '', raw)
            raw = re.sub(r'\n?```$', '', raw)
            return json.loads(raw)

        except (json.JSONDecodeError, Exception) as e:
            print(f"  [!] Gap analysis error (attempt {attempt}): {e}")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * attempt)

    return {"unmatched": [], "warnings": [], "coverage": "Gap analysis failed"}


def print_gap_analysis(gaps: dict):
    print(f"\n  {'─'*54}")
    print(f"  GAP ANALYSIS REPORT:")
    print(f"  {'─'*54}")
    print(f"  Coverage: {gaps.get('coverage', 'N/A')}")

    warnings = gaps.get("warnings", [])
    if warnings:
        print(f"\n  ⚠ Warnings:")
        for w in warnings:
            print(f"    - {w}")

    unmatched = gaps.get("unmatched", [])
    if unmatched:
        print(f"\n  [!] {len(unmatched)} gap(s) found — consider adding these categories:")
        for gap in unmatched:
            topics    = ", ".join(gap.get("topics", []))
            reason    = gap.get("reason", "")
            suggested = gap.get("suggested_category", "")
            print(f"\n    Topics without a home: {topics}")
            print(f"    Reason:                {reason}")
            print(f"    Suggested category:    '{suggested}'")
    else:
        print(f"\n  ✓ All topics are covered by the proposed taxonomy.")


# ─────────────────────────────────────────────
# STAGE 4 — TEXT ASSIGNMENT
# ─────────────────────────────────────────────

ASSIGNMENT_SYSTEM_PROMPT = """Du er en ekspert på å klassifisere norske barnetekster.

Du vil motta:
1. En godkjent liste over brede kategorier
2. En liste over tekster (nummerert 1, 2, 3...) med tittel, teksttype, hovedemne og underemner

Tildel hver tekst 1-3 kategorier fra den godkjente listen.

Regler:
- Bruk KUN kategorier fra den godkjente listen — ikke oppfinn nye
- Hver tekst skal ha 1-3 kategorier
- VIKTIG: Hvis teksttype er "fortelling" skal "Fortelling" ALLTID være én av kategoriene,
  i tillegg til relevante tematiske kategorier
  Eksempel: en fortelling om en katt → ["Dyr", "Fortelling"]
- VIKTIG: "Mat" gjelder KUN mat, drikke og kosthold
- VIKTIG: "Matematikk" gjelder tall, regning, geometri og matematiske begreper
  — IKKE forveksle Mat og Matematikk
- VIKTIG: "Vitenskap" = naturvitenskapelige fenomener (astronomi, biologi, kjemi, fysikk)
- VIKTIG: "Teknologi" = oppfinnelser, maskiner, ingeniørkunst, dataspill, digitale verktøy
- Bruk de korte nummerne (1, 2, 3...) som nøkler i assignments

Returner KUN et JSON-objekt:
{
  "assignments": {
    "1": ["Kategori A", "Fortelling"],
    "2": ["Kategori B", "Kategori C"],
    ...
  }
}"""


def build_assignment_prompt(taxonomy: list, extracted: list[dict], index_to_id: dict) -> str:
    id_to_index  = {v: k for k, v in index_to_id.items()}
    taxonomy_str = "\n".join(f"- {t}" for t in taxonomy)

    text_lines = []
    for t in extracted:
        idx        = id_to_index.get(t["text_id"], "?")
        text_type  = t.get("text_type", "fagtekst")
        sub_topics = ", ".join(t.get("sub_topics", []))
        text_lines.append(
            f'{idx}. [{text_type}] Tittel: {t["title"]} '
            f'| Hovedemne: {t.get("main_topic", "")} '
            f'| Underemner: {sub_topics}'
        )

    return (
        f"Godkjente kategorier:\n{taxonomy_str}\n\n"
        f"Tekster:\n" + "\n".join(text_lines)
        + "\n\nTildel hver tekst 1-3 kategorier. "
        + "Husk: alle fortellinger skal alltid ha 'Fortelling' som en av kategoriene."
    )


def assign_texts(client: Groq, taxonomy: list, extracted: list[dict]) -> dict:
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

            assigned_ids = set(assignments.keys())
            all_ids      = {t["text_id"] for t in extracted}
            missing      = all_ids - assigned_ids
            if missing:
                print(f"  [!] {len(missing)} texts were not assigned.")

            fortelling_texts   = [t for t in extracted if t.get("text_type") == "fortelling"]
            missing_fortelling = [
                t["title"] for t in fortelling_texts
                if "Fortelling" not in assignments.get(t["text_id"], [])
            ]
            if missing_fortelling:
                print(f"  [!] {len(missing_fortelling)} fortelling texts missing 'Fortelling' tag:")
                for title in missing_fortelling[:5]:
                    print(f"      - {title}")
                if len(missing_fortelling) > 5:
                    print(f"      ... and {len(missing_fortelling) - 5} more")

            return assignments

        except (json.JSONDecodeError, SchemaError) as e:
            print(f"  [!] Assignment error (attempt {attempt}): {e}")
        except Exception as e:
            print(f"  [!] API error (attempt {attempt}): {e}")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * attempt)

    raise RuntimeError("Assignment failed after all retries.")


# ─────────────────────────────────────────────
# STAGE 4.5 — ASSIGNMENT QUALITY CHECK
# ─────────────────────────────────────────────
#
# Flags thin and bloated categories.
# Samples CATEGORY_SPOT_SIZE texts per category for spot checking.
# Saves spot check to evaluation_sample.csv for manual review.
# ─────────────────────────────────────────────

def run_assignment_quality_check(
    extracted: list[dict],
    taxonomy: list,
    assignments: dict,
    output_path: str
) -> dict:
    """
    Check assignment distribution and sample texts per category for spot checking.
    Returns a dict of category → sampled texts.
    """
    text_map = {t["text_id"]: t for t in extracted}

    # Build category → texts mapping
    category_texts = {cat: [] for cat in taxonomy}
    for text_id, cats in assignments.items():
        t = text_map.get(text_id)
        if not t:
            continue
        for cat in cats:
            if cat in category_texts:
                category_texts[cat].append(t)

    print(f"\n  {'─'*60}")
    print(f"  ASSIGNMENT QUALITY CHECK:")
    print(f"  {'─'*60}")

    thin_categories    = []
    bloated_categories = []
    spot_check_rows    = []

    for cat in taxonomy:
        texts = category_texts[cat]
        count = len(texts)
        flag  = ""

        if count < THIN_CATEGORY_THRESHOLD:
            flag = " ⚠ TOO THIN"
            thin_categories.append(cat)
        elif count > BLOATED_CATEGORY_THRESHOLD:
            flag = " ⚠ TOO BROAD"
            bloated_categories.append(cat)

        print(f"  {cat:<20} {count:>4} texts{flag}")

        # Sample texts for spot check
        sample_size   = min(CATEGORY_SPOT_SIZE, count)
        sampled_texts = random.sample(texts, sample_size)

        for t in sampled_texts:
            spot_check_rows.append({
                "category":            cat,
                "text_id":             t["text_id"],
                "title":               t["title"],
                "text_type":           t.get("text_type", ""),
                "main_topic":          t.get("main_topic", ""),
                "sub_topics":          " | ".join(t.get("sub_topics", [])),
                "assigned_categories": " | ".join(assignments.get(t["text_id"], [])),
                # Researcher fills this in
                "relevance":           "",  # primary / secondary / wrong
                "notes":               "",
            })

    # Print warnings
    if thin_categories:
        print(f"\n  ⚠ Thin categories (< {THIN_CATEGORY_THRESHOLD} texts) — consider merging:")
        for cat in thin_categories:
            print(f"    - {cat} ({len(category_texts[cat])} texts)")

    if bloated_categories:
        print(f"\n  ⚠ Broad categories (> {BLOATED_CATEGORY_THRESHOLD} texts) — consider splitting:")
        for cat in bloated_categories:
            print(f"    - {cat} ({len(category_texts[cat])} texts)")

    # Save spot check CSV
    df = pd.DataFrame(spot_check_rows)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n  Spot check sample saved → {output_path}")
    print(f"  Open this file, review each row, fill in 'relevance':")
    print(f"    primary   — best category for this text")
    print(f"    secondary — acceptable but not the main category")
    print(f"    wrong     — this category does not fit at all")
    print(f"  Then run --evaluate --compare to compute metrics.")

    return {
        "category_counts":     {cat: len(texts) for cat, texts in category_texts.items()},
        "thin_categories":     thin_categories,
        "bloated_categories":  bloated_categories,
        "spot_check_rows":     spot_check_rows,
    }


# ─────────────────────────────────────────────
# EVALUATION — METRICS COMPUTATION
# ─────────────────────────────────────────────

def compute_extraction_metrics(eval_df: pd.DataFrame) -> dict:
    """
    Compute extraction quality metrics from manually labeled evaluation CSV.
    Expects columns: main_topic_llm, main_topic_manual, text_type_llm,
                     text_type_manual, correct (yes/no)
    """
    # Filter rows that have been manually labeled
    labeled = eval_df[eval_df["correct"].str.strip().str.lower().isin(["yes", "no"])]

    if len(labeled) == 0:
        return {"error": "No labeled rows found in evaluation CSV"}

    total   = len(labeled)
    correct = len(labeled[labeled["correct"].str.strip().str.lower() == "yes"])
    accuracy = correct / total

    # Text type accuracy if manually labeled
    type_metrics = {}
    if "text_type_manual" in labeled.columns:
        type_labeled = labeled[labeled["text_type_manual"].str.strip() != ""]
        if len(type_labeled) > 0:
            type_correct = (
                type_labeled["text_type_llm"].str.strip() ==
                type_labeled["text_type_manual"].str.strip()
            ).sum()
            type_metrics = {
                "text_type_accuracy": type_correct / len(type_labeled),
                "text_type_total":    len(type_labeled),
            }

    return {
        "extraction_accuracy": round(accuracy, 3),
        "correct":             correct,
        "total":               total,
        **type_metrics,
    }


def compute_assignment_metrics(spot_df: pd.DataFrame) -> dict:
    """
    Compute assignment quality metrics from spot check CSV.
    Uses three-way relevance rating: primary / secondary / wrong
    """
    labeled = spot_df[
        spot_df["relevance"].str.strip().str.lower().isin(
            ["primary", "secondary", "wrong"]
        )
    ]

    if len(labeled) == 0:
        return {"error": "No labeled rows found. Fill in 'relevance' column first."}

    total     = len(labeled)
    primary   = (labeled["relevance"].str.strip().str.lower() == "primary").sum()
    secondary = (labeled["relevance"].str.strip().str.lower() == "secondary").sum()
    wrong     = (labeled["relevance"].str.strip().str.lower() == "wrong").sum()

    # Primary precision — how often is the category the best fit
    primary_precision = primary / total

    # Acceptable precision — primary + secondary (not wrong)
    acceptable_precision = (primary + secondary) / total

    # Per-category breakdown
    per_category = {}
    for cat, group in labeled.groupby("category"):
        g_primary   = (group["relevance"].str.strip().str.lower() == "primary").sum()
        g_secondary = (group["relevance"].str.strip().str.lower() == "secondary").sum()
        g_wrong     = (group["relevance"].str.strip().str.lower() == "wrong").sum()
        g_total     = len(group)

        per_category[cat] = {
            "primary":            int(g_primary),
            "secondary":          int(g_secondary),
            "wrong":              int(g_wrong),
            "total":              g_total,
            "primary_precision":  round(g_primary / g_total, 3),
            "acceptable_precision": round((g_primary + g_secondary) / g_total, 3),
        }

    return {
        "total_rated":            total,
        "primary":                int(primary),
        "secondary":              int(secondary),
        "wrong":                  int(wrong),
        "primary_precision":      round(primary_precision, 3),
        "acceptable_precision":   round(acceptable_precision, 3),
        "per_category":           per_category,
    }


def run_evaluation(compare: bool = False):
    """
    Export evaluation sample or compute metrics from labeled sample.
    """
    if not compare:
        print("\n[Evaluation] Fill in evaluation files and re-run with --compare")
        return

    metrics = {}

    # Extraction evaluation — always from evaluation_sample.csv
    if os.path.exists(EVAL_SAMPLE_CSV):
        eval_df = pd.read_csv(EVAL_SAMPLE_CSV, dtype=str).fillna("")
        if "correct" in eval_df.columns:
            print("\n[Evaluation] Computing extraction metrics...")
            metrics["extraction"] = compute_extraction_metrics(eval_df)
            m = metrics["extraction"]
            if "error" not in m:
                print(f"  Extraction accuracy: {m.get('extraction_accuracy', 'N/A')}")
                print(f"  ({m.get('correct', 0)}/{m.get('total', 0)} correct)")
                if "text_type_accuracy" in m:
                    print(f"  Text type accuracy:  {m['text_type_accuracy']}")
        else:
            print("\n[Evaluation] evaluation_sample.csv found but no 'correct' column — skipping extraction metrics.")
    else:
        print(f"\n[Evaluation] {EVAL_SAMPLE_CSV} not found — skipping extraction metrics.")

    # Assignment evaluation — always from assignment_spot_check.csv
    ASSIGNMENT_SPOT_CSV = "assignment_spot_check.csv"
    if os.path.exists(ASSIGNMENT_SPOT_CSV):
        spot_df = pd.read_csv(ASSIGNMENT_SPOT_CSV, dtype=str).fillna("")
        if "relevance" in spot_df.columns:
            print("\n[Evaluation] Computing assignment metrics...")
            metrics["assignment"] = compute_assignment_metrics(spot_df)
            m = metrics["assignment"]
            if "error" in m:
                print(f"  Error: {m['error']}")
            else:
                print(f"  Total rated:          {m['total_rated']}")
                print(f"  Primary precision:    {m['primary_precision']:.1%}  "
                      f"({m['primary']} texts where category is the best fit)")
                print(f"  Acceptable precision: {m['acceptable_precision']:.1%}  "
                      f"({m['primary'] + m['secondary']} texts where category is acceptable)")
                print(f"  Wrong:                {m['wrong']} texts")
                print(f"\n  Per-category breakdown:")
                print(f"  {'Category':<20} {'Primary':>8} {'Secondary':>10} {'Wrong':>6} {'Acceptable':>11}")
                print(f"  {'─'*58}")
                for cat, cm in m.get("per_category", {}).items():
                    print(f"  {cat:<20} {cm['primary']:>8} {cm['secondary']:>10} "
                          f"{cm['wrong']:>6} {cm['acceptable_precision']:>10.0%}")

                # Show secondary assignments
                secondary_rows = spot_df[
                    spot_df["relevance"].str.strip().str.lower() == "secondary"
                ]
                if len(secondary_rows) > 0:
                    print(f"\n  Secondary assignments (consider taxonomy adjustment):")
                    for _, row in secondary_rows.iterrows():
                        correct = row.get("correct_category(if not primary)", "")
                        if str(correct).strip():
                            print(f"    {row['category']:<20} → "
                                  f"{row['title'][:35]} → better: {correct}")
        else:
            print("\n[Evaluation] assignment_spot_check.csv found but no 'relevance' column — skipping.")
    else:
        print(f"\n[Evaluation] assignment_spot_check.csv not found — skipping assignment metrics.")

    # Save metrics
    save_json(EVAL_METRICS_JSON, metrics)
    print(f"\n  Metrics saved → {EVAL_METRICS_JSON}")

    if "extraction" in metrics:
        save_json("evaluation_extraction_metrics.json", metrics["extraction"])
        print(f"  Extraction metrics → evaluation_extraction_metrics.json")

    if "assignment" in metrics:
        save_json("evaluation_assignment_metrics.json", metrics["assignment"])
        print(f"  Assignment metrics → evaluation_assignment_metrics.json")


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
            "text_type":        t.get("text_type", ""),
            "main_topic":       t.get("main_topic", ""),
            "sub_topics":       " | ".join(t.get("sub_topics", [])),
            "broad_topics":     " | ".join(broad),
            "extraction_error": t.get("extraction_error", ""),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  CSV → {output_path}")

    save_json(ASSIGNMENTS_JSON, {
        "taxonomy":    taxonomy,
        "assignments": assignments,
    })
    print(f"  Assignments → {ASSIGNMENTS_JSON}")

    print(f"\n{'─'*56}")
    print(f"  FINAL BROAD TOPICS ({len(taxonomy)}):")
    print(f"{'─'*56}")
    for topic in taxonomy:
        count = sum(1 for r in rows if topic in r["broad_topics"].split(" | "))
        print(f"  {topic:44s}  ({count} texts)")

    fortelling_rows = [r for r in rows if r.get("text_type") == "fortelling"]
    tagged          = [r for r in fortelling_rows
                       if "Fortelling" in r["broad_topics"].split(" | ")]
    print(f"\n  Fortelling coverage: {len(tagged)}/{len(fortelling_rows)} story texts tagged correctly")

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
    parser.add_argument("--input",           required=True,
                        help="Path to input TSV/CSV")
    parser.add_argument("--output",          default=OUTPUT_CSV,
                        help=f"Output CSV (default: {OUTPUT_CSV})")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip extraction, load from checkpoint. Re-proposes taxonomy.")
    parser.add_argument("--assign-only",     action="store_true",
                        help="Skip extraction + taxonomy. Loads taxonomy_final.json and assigns.")
    parser.add_argument("--evaluate",        action="store_true",
                        help="Export evaluation sample or compute metrics.")
    parser.add_argument("--compare",         action="store_true",
                        help="Used with --evaluate: compute metrics from labeled sample.")
    args = parser.parse_args()

    load_dotenv(Path(__file__).parent.parent.parent / ".env")

    api_key = ""
    if not api_key:
        print("Error: GROQ_API_KEY not set.")
        print("  Add GROQ_API_KEY=gsk_... to your .env file")
        sys.exit(1)

    # Evaluation-only mode
    if args.evaluate:
        run_evaluation(compare=args.compare)
        return

    client = Groq(api_key=api_key)
    print(f"  Model: {MODEL}")

    # ── Stage 1 ──────────────────────────────────────────────
    print("\n[Stage 1] Loading and preprocessing...")
    texts = load_and_preprocess(args.input)

    # ── Stage 2 ──────────────────────────────────────────────
    if args.skip_extraction or args.assign_only:
        print("\n[Stage 2] Loading extracted topics...")
        if os.path.exists(EXTRACTED_JSON):
            raw_extracted = load_json(EXTRACTED_JSON)
            text_map      = {t["text_id"]: t for t in texts}
            extracted     = []
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

        # ── Stage 2.5: Extraction Quality Check ─────────────
        print(f"\n[Stage 2.5] Extraction quality check...")
        run_extraction_quality_check(extracted)
        print(f"\n  Review the sample above.")
        print(f"  If extractions look good, continue.")
        print(f"  If not, fix the extraction prompt and re-run.")

    # ── Stage 3 ──────────────────────────────────────────────
    if args.assign_only:
        if not os.path.exists(FINAL_JSON):
            print(f"\nError: {FINAL_JSON} not found.")
            sys.exit(1)
        final_data = load_json(FINAL_JSON)
        taxonomy   = final_data.get("broad_topics", [])
        print(f"\n[Stage 3] Loaded finalized taxonomy ({len(taxonomy)} categories) from {FINAL_JSON}")

    else:
        print(f"\n[Stage 3] Proposing draft taxonomy...")
        draft = propose_taxonomy(client, extracted)
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

        # ── Stage 3.5: Gap Analysis ──────────────────────────
        print(f"\n[Stage 3.5] Running gap analysis...")
        gaps = run_gap_analysis(client, extracted, draft["broad_topics"])
        print_gap_analysis(gaps)

        print(f"""
  ╔══════════════════════════════════════════════════════╗
  ║  RESEARCHER REVIEW STEP 1 — TAXONOMY                 ║
  ║                                                      ║
  ║  1. Open: {DRAFT_JSON:<42s}║
  ║  2. Review proposed categories AND the gap report    ║
  ║     - Add suggested categories for any gaps found    ║
  ║     - Fix any warnings flagged above                 ║
  ║     - Merge overlapping categories                   ║
  ║     - Rename to fit your thesis                      ║
  ║  3. Save your edited version as: {FINAL_JSON:<19s}║
  ║  4. Re-run with:                                     ║
  ║       python topic_pipeline.py \\                     ║
  ║         --input {Path(args.input).name:<38s}║
  ║         --assign-only                                ║
  ╚══════════════════════════════════════════════════════╝
""")
        print("  Pipeline paused. Edit taxonomy, then re-run with --assign-only.")
        return

    # ── Stage 4 ──────────────────────────────────────────────
    print(f"\n[Stage 4] Assigning texts to finalized taxonomy...")
    assignments = assign_texts(client, taxonomy, extracted)
    print(f"  Assigned {len(assignments)}/{len(extracted)} texts.")

    # ── Stage 4.5: Assignment Quality Check ─────────────────
    print(f"\n[Stage 4.5] Assignment quality check...")
    quality = run_assignment_quality_check(
        extracted, taxonomy, assignments, ASSIGNMENT_EVAL_CSV
    )

    thin     = quality["thin_categories"]
    bloated  = quality["bloated_categories"]

    if thin or bloated:
        print(f"""
  ╔══════════════════════════════════════════════════════╗
  ║  RESEARCHER REVIEW STEP 2 — ASSIGNMENTS              ║
  ║                                                      ║
  ║  Issues were found in the distribution above.        ║
  ║  Options:                                            ║
  ║  A) Edit taxonomy_final.json and re-run              ║
  ║       --assign-only                                  ║
  ║  B) Manually edit assignments in results.csv         ║
  ║  C) Accept as-is and proceed                         ║
  ║                                                      ║
  ║  After reviewing spot check CSV:                     ║
  ║    python topic_pipeline.py \\                        ║
  ║      --input {Path(args.input).name:<40s}║
  ║      --evaluate --compare                            ║
  ╚══════════════════════════════════════════════════════╝
""")
    else:
        print(f"\n  ✓ Distribution looks good. Proceeding to export.")

    # ── Stage 5 ──────────────────────────────────────────────
    print(f"\n[Stage 5] Exporting results...")
    export_results(extracted, taxonomy, assignments, args.output)

    print(f"""
  Next steps:
  1. Open assignment_spot_check.csv and fill in 'relevance' (yes/no)
  2. Run: python topic_pipeline.py --input {Path(args.input).name} --evaluate --compare
  3. Review evaluation_metrics.json for precision scores
""")


if __name__ == "__main__":
    main()