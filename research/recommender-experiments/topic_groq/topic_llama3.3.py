"""
topic_pipeline.py
=================
Pipeline for extracting and clustering topics from Norwegian children's texts.
Uses Groq API (llama-3.3-70b-versatile) GÇö free tier, no credit card required.
Sign up at: https://console.groq.com

Stages:
  1. Load & preprocess TSV (fix encoding, strip markdown, drop empty rows)
  2. Per-text topic extraction (batched Groq API calls, with checkpointing)
  3. Broad topic clustering (1-2 API calls)
  4. Export final CSV

Usage:
  pip install groq pandas
  export GROQ_API_KEY=gsk_your_key_here
  python topic_pipeline.py --input texts.tsv --output results.csv

Checkpointing:
  Intermediate results are saved to topics_checkpoint.json after each batch.
  If the script is interrupted, re-running it will resume from where it left off.
"""

import argparse
import json
import os
import re
import time
import sys
from pathlib import Path

from groq import Groq
import pandas as pd


# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
# CONFIG
# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

MODEL = "llama-3.3-70b-versatile"
BATCH_SIZE = 10
CHECKPOINT_FILE = "topics_checkpoint.json"
MAX_RETRIES = 3
RETRY_DELAY = 5


# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
# STAGE 1 GÇö LOAD & PREPROCESS
# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

def fix_encoding(text: str) -> str:
    """Fix UTF-8 text mis-decoded as Latin-1. e.g. '+â-£rner' GåÆ '+ÿrner'"""
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def strip_markdown(text: str) -> str:
    """Strip markdown syntax, keep plain text content."""
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)           # remove images
    text = re.sub(r'\[([^\]]+)\]\([^\)]*\)', r'\1', text) # links GåÆ text
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)  # headings
    text = re.sub(r'[*_]{1,3}([^*_]+)[*_]{1,3}', r'\1', text)  # bold/italic
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_subheadings(body: str) -> list[str]:
    """Extract subheadings (##, ###) from markdown body."""
    return re.findall(r'^#{2,6}\s+(.+)$', body, flags=re.MULTILINE)


def is_empty(value) -> bool:
    """Check if a cell value is empty / NaN."""
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

    # Remove first line if it's just the title repeated
    lines = clean_body.splitlines()
    if lines and lines[0].strip().lower() == title.strip().lower():
        clean_body = "\n".join(lines[1:]).strip()

    return {
        "text_id":    str(row.get("sanity_text_id", row.get("text_id", ""))),
        "serial":     str(row.get("serialnumber",   row.get("serial_number", ""))),
        "title":      title,
        "subheadings": subheadings,
        "clean_body": clean_body,
    }


def load_and_preprocess(filepath: str) -> list[dict]:
    """Load TSV/CSV, drop empty rows, preprocess all valid rows."""
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

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # GöÇGöÇ Drop empty rows GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
    # A row is considered empty if both 'title' and 'body' are missing/NaN.
    # This handles Excel exports that pad files with blank rows.
    before = len(df)
    title_col = "title" if "title" in df.columns else None
    body_col  = "body"  if "body"  in df.columns else None

    if title_col and body_col:
        df = df[~(df[title_col].apply(is_empty) & df[body_col].apply(is_empty))]
    elif title_col:
        df = df[~df[title_col].apply(is_empty)]

    df = df.reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} empty rows GåÆ {len(df)} valid rows remaining.")

    texts = [preprocess_row(row) for _, row in df.iterrows()]
    print(f"  Preprocessed {len(texts)} texts.")
    return texts


# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
# STAGE 2 GÇö PER-TEXT TOPIC EXTRACTION
# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

EXTRACTION_SYSTEM_PROMPT = """Du er en ekspert p+Ñ +Ñ klassifisere norske barnetekster.
For hver tekst du mottar, skal du returnere KUN et JSON-objekt GÇô ingen forklaring, ingen markdown-blokker.

Format:
{
  "main_topic": "ett kortfattet norsk emne (2-5 ord)",
  "sub_topics": ["emne1", "emne2", "emne3"]
}

Regler:
- Skriv alle emner p+Ñ norsk bokm+Ñl
- main_topic skal fange tekstens prim+ªre tema
- sub_topics: 2-4 underkategorier eller relaterte temaer
- V+ªr spesifikk nok til at emnene er meningsfulle, men ikke for smale
- Eksempel for en tekst om ++rner: {"main_topic": "Rovfugler", "sub_topics": ["Natur og dyreliv", "Norsk fauna", "Kultur og symbolikk"]}"""


def build_extraction_prompt(text: dict) -> str:
    subheadings_str = ""
    if text["subheadings"]:
        subheadings_str = "\nDeltitler: " + " | ".join(text["subheadings"])

    body_preview = text["clean_body"][:1500]
    if len(text["clean_body"]) > 1500:
        body_preview += "\n[...]"

    return f"""Tittel: {text["title"]}{subheadings_str}

Tekst:
{body_preview}"""


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

            parsed = json.loads(raw)
            return {
                **text,
                "main_topic":      parsed.get("main_topic", ""),
                "sub_topics":      parsed.get("sub_topics", []),
                "extraction_error": None,
            }

        except json.JSONDecodeError as e:
            print(f"    [!] JSON parse error for '{text['title']}' (attempt {attempt}): {e}")
        except Exception as e:
            print(f"    [!] API error for '{text['title']}' (attempt {attempt}): {e}")
            if "rate_limit" in str(e).lower() or "429" in str(e):
                print(f"    Rate limit hit GÇö waiting 60s...")
                time.sleep(60)
                continue

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * attempt)

    return {
        **text,
        "main_topic":      "",
        "sub_topics":      [],
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
        batch        = remaining[batch_start: batch_start + BATCH_SIZE]
        batch_num    = (batch_start // BATCH_SIZE) + 1
        total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"  Batch {batch_num}/{total_batches} GÇö {len(batch)} texts...")

        for text in batch:
            result = extract_topics_for_text(client, text)
            results[text["text_id"]] = result
            done  += 1
            status = "G£ô" if not result["extraction_error"] else "G£ù"
            print(f"    [{done}/{total}] {status} {text['title'][:50]}")

        save_checkpoint(results)
        print(f"  Checkpoint saved.")

        if batch_start + BATCH_SIZE < len(remaining):
            time.sleep(2)

    return list(results.values())


# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
# STAGE 3 GÇö BROAD TOPIC CLUSTERING
# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

CLUSTERING_SYSTEM_PROMPT = """Du er en ekspert p+Ñ +Ñ lage emnestruktur for norske barnetekster.
Du vil motta en liste med tekster og deres emner.

Oppgaven din er +Ñ:
1. Foresl+Ñ N+ÿYAKTIG 12-15 brede overordnede kategorier for barnetekster
2. Tildele hver tekst 1-3 av disse kategoriene

Viktige regler for kategoriene:
- Lag N+ÿYAKTIG 12-15 kategorier GÇö ikke f+ªrre, ikke flere
- Kategoriene skal v+ªre TYDELIG FORSKJELLIGE fra hverandre GÇö unng+Ñ overlapp
  (f.eks. IKKE ha b+Ñde "Natur og dyr" og "Dyreliv og ++kologi" GÇö sl+Ñ dem sammen til +¬n)
- Hver kategori skal dekke minst 3-4 tekster for +Ñ v+ªre meningsfull
- Kategoriene skal v+ªre p+Ñ norsk bokm+Ñl og passe for aldersgruppen (barn)
- Unng+Ñ for spesifikke kategorier som bare dekker 1-2 tekster

Returner KUN et JSON-objekt GÇö ingen forklaring, ingen markdown:
{
  "broad_topics": ["Kategori1", "Kategori2", ...],
  "assignments": {
    "TEXT_ID": ["Kategori1", "Kategori3"],
    ...
  }
}"""


def run_clustering(client: Groq, extracted: list[dict]) -> dict:
    summaries = []
    for t in extracted:
        sub = ", ".join(t.get("sub_topics", []))
        summaries.append(
            f'- ID: {t["text_id"]} | Tittel: {t["title"]} '
            f'| Hovedemne: {t.get("main_topic", "")} | Underemner: {sub}'
        )

    prompt  = "Her er alle tekstene med emner:\n\n" + "\n".join(summaries)
    prompt += (
        "\n\nLag N+ÿYAKTIG 12-15 brede, ikke-overlappende kategorier og "
        "tildel hver tekst 1-3 kategorier."
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=4000,
                messages=[
                    {"role": "system", "content": CLUSTERING_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r'^```[a-z]*\n?', '', raw)
            raw = re.sub(r'\n?```$', '', raw)
            result = json.loads(raw)

            n = len(result.get("broad_topics", []))
            if not (12 <= n <= 15):
                print(f"  [!] Got {n} broad topics (expected 12-15) GÇö retrying...")
                raise ValueError(f"Expected 12-15 broad topics, got {n}")

            return result

        except (json.JSONDecodeError, ValueError) as e:
            print(f"  [!] Clustering error (attempt {attempt}): {e}")
        except Exception as e:
            print(f"  [!] API error (attempt {attempt}): {e}")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * attempt)

    raise RuntimeError("Clustering failed after all retries.")


# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
# STAGE 4 GÇö EXPORT
# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

def export_results(extracted: list[dict], clustering: dict, output_path: str):
    assignments      = clustering.get("assignments", {})
    broad_topics_list = clustering.get("broad_topics", [])

    rows = []
    for t in extracted:
        broad = assignments.get(t["text_id"], [])
        rows.append({
            "text_id":         t["text_id"],
            "serial_number":   t["serial"],
            "title":           t["title"],
            "main_topic":      t.get("main_topic", ""),
            "sub_topics":      " | ".join(t.get("sub_topics", [])),
            "broad_topics":    " | ".join(broad),
            "extraction_error": t.get("extraction_error", ""),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n  Results written to: {output_path}")

    print(f"\n{'GöÇ'*54}")
    print(f"  BROAD TOPICS GENERATED ({len(broad_topics_list)}):")
    print(f"{'GöÇ'*54}")
    for topic in broad_topics_list:
        count = sum(1 for r in rows if topic in r["broad_topics"].split(" | "))
        print(f"  {topic:42s}  ({count} texts)")

    errors = [r for r in rows if r["extraction_error"]]
    if errors:
        print(f"\n  [!] {len(errors)} texts had extraction errors:")
        for e in errors:
            print(f"      - {e['title']}")


# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
# MAIN
# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

def main():
    parser = argparse.ArgumentParser(
        description="Extract and cluster topics from Norwegian children's texts using Groq."
    )
    parser.add_argument("--input",  required=True,  help="Path to input TSV/CSV file")
    parser.add_argument("--output", default="results.csv", help="Output CSV (default: results.csv)")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip Stage 2, load from checkpoint (re-run clustering only)")
    args = parser.parse_args()

    api_key = "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"
    if not api_key:
        print("Error: GROQ_API_KEY environment variable not set.")
        print("  Get your free key at: https://console.groq.com/keys")
        print("  Then run: export GROQ_API_KEY=gsk_your_key_here")
        sys.exit(1)

    client = Groq(api_key=api_key)
    print(f"  Model: {MODEL} (Groq free tier)")

    # GöÇGöÇ Stage 1 GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
    print("\n[Stage 1] Loading and preprocessing...")
    texts = load_and_preprocess(args.input)

    # GöÇGöÇ Stage 2 GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
    if args.skip_extraction:
        print("\n[Stage 2] Skipping extraction, loading from checkpoint...")
        checkpoint = load_checkpoint()
        if not checkpoint:
            print("  Error: no checkpoint found. Run without --skip-extraction first.")
            sys.exit(1)
        # Only keep entries that match actual texts (not stale empty rows)
        valid_ids = {t["text_id"] for t in texts}
        extracted = [v for k, v in checkpoint.items() if k in valid_ids]
        print(f"  Loaded {len(extracted)} texts from checkpoint.")
    else:
        print(f"\n[Stage 2] Extracting topics ({len(texts)} texts, batch size {BATCH_SIZE})...")
        extracted = run_extraction(client, texts)

    # GöÇGöÇ Stage 3 GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
    print("\n[Stage 3] Clustering into broad topics...")
    clustering = run_clustering(client, extracted)
    print(f"  Generated {len(clustering.get('broad_topics', []))} broad topics.")

    # GöÇGöÇ Stage 4 GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
    print(f"\n[Stage 4] Exporting to {args.output}...")
    export_results(extracted, clustering, args.output)

    print("\nDone! G£ô")


if __name__ == "__main__":
    main()
