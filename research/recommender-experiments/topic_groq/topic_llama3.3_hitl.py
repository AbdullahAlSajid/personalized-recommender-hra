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


# G��G�� Configuration G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��

MODEL           = "llama-3.3-70b-versatile"
BATCH_SIZE      = 10
MAX_RETRIES     = 3
RETRY_DELAY     = 5

CHECKPOINT_FILE  = "topics_checkpoint.json"
EXTRACTED_JSON   = "extracted_topics.json"
DRAFT_JSON       = "taxonomy_draft.json"
FINAL_JSON       = "taxonomy_final.json"
ASSIGNMENTS_JSON = "assignments.json"
OUTPUT_CSV       = "results.csv"

EVAL_SAMPLE_CSV              = "evaluation_sample.csv"
ASSIGNMENT_EVAL_CSV          = "assignment_spot_check.csv"
EVAL_METRICS_JSON            = "evaluation_metrics.json"
EVAL_EXTRACTION_METRICS_JSON = "evaluation_extraction_metrics.json"
EVAL_ASSIGNMENT_METRICS_JSON = "evaluation_assignment_metrics.json"

EXTRACTION_SAMPLE_SIZE      = 30
CATEGORY_SPOT_SIZE          = 5
THIN_CATEGORY_THRESHOLD     = 5
BLOATED_CATEGORY_THRESHOLD  = 40


# G��G�� Utilities G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��

def fix_encoding(text: str) -> str:
    """Recover Norwegian characters from Latin-1 mis-decoded UTF-8."""
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def normalize_topic(topic: str) -> str:
    """Standardize topic string: strip, collapse whitespace, capitalize first letter."""
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


# G��G�� Schema validation G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��

class SchemaError(Exception):
    pass


def validate_extracted_topics(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise SchemaError(f"Expected dict, got {type(raw).__name__}")

    main = raw.get("main_topic", "")
    if not isinstance(main, str):
        raise SchemaError(f"main_topic must be str, got {type(main).__name__}")
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
    if not subs:
        raise SchemaError("sub_topics is empty after normalization")

    text_type = raw.get("text_type", "fagtekst")
    if text_type not in ("fagtekst", "fortelling"):
        text_type = "fagtekst"

    return {"main_topic": main, "sub_topics": subs, "text_type": text_type}


def validate_taxonomy(taxonomy: list) -> list:
    if not isinstance(taxonomy, list):
        raise SchemaError(f"Expected list, got {type(taxonomy).__name__}")
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
        clean = [normalize_topic(c) for c in categories if normalize_topic(c) in taxonomy_set]
        if clean:
            real_assignments[real_id] = clean
    return real_assignments


# G��G�� Stage 1: Load and preprocess G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��

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
        "serial":      str(row.get("serialnumber", row.get("serial_number", ""))),
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
            print(f"  Read {len(df)} rows ({enc})")
            break
        except Exception:
            continue

    if df is None:
        raise ValueError(f"Could not read: {filepath}")

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
        print(f"  Dropped {dropped} empty rows G�� {len(df)} remaining.")

    texts = [preprocess_row(row) for _, row in df.iterrows()]
    print(f"  Preprocessed {len(texts)} texts.")
    return texts


# G��G�� Stage 2: Topic extraction G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��

EXTRACTION_SYSTEM_PROMPT = """Du er en ekspert p+� +� klassifisere norske barnetekster for elever i alderen 8-12 +�r.
For hver tekst du mottar, skal du returnere KUN et JSON-objekt G�� ingen forklaring, ingen markdown-blokker.

Format:
{
  "main_topic": "ett kortfattet norsk emne (2-5 ord)",
  "sub_topics": ["emne1", "emne2", "emne3"],
  "text_type": "fagtekst" eller "fortelling"
}

Regler:
- Skriv alle emner p+� norsk bokm+�l
- main_topic skal fange tekstens prim+�re tema
- sub_topics: 2-4 underkategorier eller relaterte temaer
- text_type skal v+�re N+�YAKTIG ett av disse to:
    "fortelling" G�� hvis teksten er en historie, novelle, eller narrativ fiksjon med karakterer og handling
    "fagtekst"   G�� hvis teksten er informativ, faktabasert eller forklarende
- Eksempel fagtekst: {"main_topic": "Rovfugler", "sub_topics": ["Natur og dyreliv", "Norsk fauna"], "text_type": "fagtekst"}
- Eksempel fortelling: {"main_topic": "Fortelling om katt", "sub_topics": ["Familieliv", "Dyr som husdyr"], "text_type": "fortelling"}"""


def build_extraction_prompt(text: dict) -> str:
    subheadings_str = ""
    if text["subheadings"]:
        subheadings_str = "\nDeltitler: " + " | ".join(text["subheadings"])
    body_preview = text["clean_body"][:1500]
    if len(text["clean_body"]) > 1500:
        body_preview += "\n[...]"
    return f'Tittel: {text["title"]}{subheadings_str}\n\nTekst:\n{body_preview}'


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

            validated = validate_extracted_topics(json.loads(raw))
            return {
                **text,
                "main_topic":       validated["main_topic"],
                "sub_topics":       validated["sub_topics"],
                "text_type":        validated["text_type"],
                "extraction_error": None,
            }

        except json.JSONDecodeError as e:
            print(f"    [!] JSON error for '{text['title']}' (attempt {attempt}): {e}")
        except SchemaError as e:
            print(f"    [!] Schema error for '{text['title']}' (attempt {attempt}): {e}")
        except Exception as e:
            print(f"    [!] API error for '{text['title']}' (attempt {attempt}): {e}")
            if "rate_limit" in str(e).lower() or "429" in str(e):
                print("    Rate limited G�� waiting 60s...")
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
    remaining  = [t for t in texts if t["text_id"] not in results]
    total      = len(texts)
    done       = len(results)

    if done > 0:
        print(f"  Resuming: {done}/{total} done, {len(remaining)} remaining.")

    for batch_start in range(0, len(remaining), BATCH_SIZE):
        batch         = remaining[batch_start: batch_start + BATCH_SIZE]
        batch_num     = (batch_start // BATCH_SIZE) + 1
        total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} texts)...")

        for text in batch:
            result = extract_topics_for_text(client, text)
            results[text["text_id"]] = result
            done  += 1
            status = "G��" if not result["extraction_error"] else "G��"
            print(f"    [{done}/{total}] {status} {text['title'][:50]}")

        save_checkpoint(results)

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
    print(f"  Saved G�� {EXTRACTED_JSON}")
    return extracted


# G��G�� Stage 2.5: Extraction quality check G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��

def run_extraction_quality_check(extracted: list[dict]) -> list[dict]:
    n      = min(EXTRACTION_SAMPLE_SIZE, len(extracted))
    sample = random.sample(extracted, n)

    print(f"\n  {'G��'*60}")
    print(f"  EXTRACTION SAMPLE G�� {n} texts:")
    print(f"  {'G��'*60}")
    print(f"  {'#':<4} {'Title':<40} {'Type':<12} {'Main Topic':<30} Sub Topics")
    print(f"  {'G��'*60}")

    for i, t in enumerate(sample, 1):
        subs = ", ".join(t.get("sub_topics", []))
        print(
            f"  {i:<4} {t['title'][:38]:<40} {t.get('text_type', '?'):<12} "
            f"{t.get('main_topic', '')[:28]:<30} {subs[:50]}"
        )

    if not os.path.exists(EVAL_SAMPLE_CSV):
        rows = [
            {
                "text_id":           t["text_id"],
                "title":             t["title"],
                "text_type_llm":     t.get("text_type", ""),
                "main_topic_llm":    t.get("main_topic", ""),
                "sub_topics_llm":    " | ".join(t.get("sub_topics", [])),
                "text_type_manual":  "",
                "main_topic_manual": "",
                "sub_topics_manual": "",
                "correct":           "",
                "notes":             "",
            }
            for t in sample
        ]
        pd.DataFrame(rows).to_csv(EVAL_SAMPLE_CSV, index=False, encoding="utf-8-sig")
        print(f"\n  Sample saved G�� {EVAL_SAMPLE_CSV}")
        print(f"  Fill in 'correct' (yes/no) for each row, then run --evaluate --compare.")
    else:
        print(f"\n  {EVAL_SAMPLE_CSV} already exists G�� skipping overwrite.")

    return sample


# G��G�� Stage 3: Taxonomy proposal G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��

TAXONOMY_SYSTEM_PROMPT = """Du er en ekspert p+� +� lage emnestruktur for norske barnetekster for elever i alderen 8-12 +�r.

Du vil motta en liste over emner hentet fra barnetekster med frekvens, samt antall fortellinger vs fagtekster.
Basert p+� dette skal du foresl+� 12-15 overordnede kategorier.

Regler:
- N+�YAKTIG 12-15 kategorier
- Hvert kategorinavn skal v+�re ET ENKELT ORD p+� norsk bokm+�l
- Kategoriene skal v+�re brede nok til +� dekke mange tekster
- Ingen overlapp mellom kategorier
- Ordene skal v+�re enkle og gjenkjennelige for barn mellom 8-12 +�r
- Kategoriene skal fungere som interessevalg i et anbefalingssystem
- Hvis mer enn 10 tekster er fortellinger skal "Fortelling" v+�re en egen kategori

Returner KUN et JSON-objekt:
{
  "broad_topics": ["Dyr", "Vitenskap", ...],
  "rationale": {
    "Dyr": "Kort begrunnelse",
    ...
  }
}"""


def build_taxonomy_prompt(extracted: list[dict]) -> str:
    topic_counts  = Counter(t.get("main_topic", "") for t in extracted if t.get("main_topic"))
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
        + "\n\nForesl+� 12-15 overordnede kategorier som dekker disse emnene."
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
            print(f"  [!] Taxonomy error (attempt {attempt}): {e}")
        except Exception as e:
            print(f"  [!] API error (attempt {attempt}): {e}")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * attempt)

    raise RuntimeError("Taxonomy proposal failed.")


# G��G�� Stage 3.5: Gap analysis G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��

GAP_ANALYSIS_SYSTEM_PROMPT = """Du er en ekspert p+� tekstklassifisering.

Du vil motta:
1. En liste over foresl+�tte brede kategorier
2. En liste over ekstraherte main_topics fra tekstene med frekvens
3. Antall fortellinger i samlingen

Finn main_topics som IKKE passer naturlig inn i noen av de foresl+�tte kategoriene.

Returner KUN et JSON-objekt:
{
  "unmatched": [
    {
      "topics": ["topic1", "topic2"],
      "reason": "Hvorfor disse ikke passer i noen eksisterende kategori",
      "suggested_category": "Forslag til ny kategori (ett enkelt ord p+� norsk)"
    }
  ],
  "warnings": ["Advarsel om mulige overlapp eller tvetydigheter"],
  "coverage": "Kort vurdering av dekning"
}

Returner unmatched som tom liste [] hvis alle topics er dekket.
V+�r s+�rlig oppmerksom p+�:
- Matematikk-relaterte topics som kan forveksles med Mat
- Teknologi vs Vitenskap
- Fortellinger som kan trenge egen kategori"""


def run_gap_analysis(client: Groq, extracted: list[dict], taxonomy: list) -> dict:
    topic_counts     = Counter(t.get("main_topic", "") for t in extracted if t.get("main_topic"))
    sorted_topics    = sorted(topic_counts.items(), key=lambda x: -x[1])
    topics_str       = "\n".join(f"- {t} ({c} tekster)" for t, c in sorted_topics)
    taxonomy_str     = "\n".join(f"- {t}" for t in taxonomy)
    fortelling_count = sum(1 for t in extracted if t.get("text_type") == "fortelling")

    prompt = (
        f"Foresl+�tte kategorier:\n{taxonomy_str}\n\n"
        f"Ekstraherte main_topics:\n{topics_str}\n\n"
        f"Antall fortellinger: {fortelling_count}\n\n"
        f"Finn topics uten kategori og gi advarsler om mulige problemer."
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

        except Exception as e:
            print(f"  [!] Gap analysis error (attempt {attempt}): {e}")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * attempt)

    return {"unmatched": [], "warnings": [], "coverage": "Gap analysis failed"}


def print_gap_analysis(gaps: dict):
    print(f"\n  {'G��'*54}")
    print(f"  GAP ANALYSIS")
    print(f"  {'G��'*54}")
    print(f"  {gaps.get('coverage', '')}")

    for w in gaps.get("warnings", []):
        print(f"\n  G�� {w}")

    unmatched = gaps.get("unmatched", [])
    if unmatched:
        print(f"\n  {len(unmatched)} gap(s) found:")
        for gap in unmatched:
            print(f"\n  Topics:    {', '.join(gap.get('topics', []))}")
            print(f"  Reason:    {gap.get('reason', '')}")
            print(f"  Suggested: {gap.get('suggested_category', '')}")
    else:
        print("\n  All topics covered.")


# G��G�� Stage 4: Text assignment G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��

ASSIGNMENT_SYSTEM_PROMPT = """Du er en ekspert p+� +� klassifisere norske barnetekster for elever i alderen 8-12 +�r.

Du vil motta:
1. En godkjent liste over brede kategorier
2. En liste over tekster (nummerert 1, 2, 3...) med tittel, teksttype, hovedemne og underemner

Tildel hver tekst 1-3 kategorier fra den godkjente listen.

Regler:
- Bruk KUN kategorier fra den godkjente listen
- Hver tekst skal ha 1-3 kategorier
- Hvis teksttype er "fortelling" skal "Fortelling" ALLTID v+�re +�n av kategoriene,
  i tillegg til relevante tematiske kategorier (f.eks. en fortelling om en katt G�� ["Dyr", "Fortelling"])
- "Mat" gjelder KUN mat, drikke og kosthold
- "Matematikk" gjelder tall, regning, geometri og matematiske begreper G�� ikke forveksle med Mat
- "Vitenskap" = naturvitenskapelige fenomener (astronomi, biologi, kjemi, fysikk)
- "Teknologi" = oppfinnelser, maskiner, ingeni++rkunst, dataspill, digitale verkt++y
- Bruk de korte nummerne (1, 2, 3...) som n++kler

Returner KUN et JSON-objekt:
{
  "assignments": {
    "1": ["Kategori A", "Fortelling"],
    "2": ["Kategori B", "Kategori C"]
  }
}"""


def build_assignment_prompt(taxonomy: list, extracted: list[dict], index_to_id: dict) -> str:
    id_to_index  = {v: k for k, v in index_to_id.items()}
    taxonomy_str = "\n".join(f"- {t}" for t in taxonomy)

    lines = []
    for t in extracted:
        idx  = id_to_index.get(t["text_id"], "?")
        subs = ", ".join(t.get("sub_topics", []))
        lines.append(
            f'{idx}. [{t.get("text_type", "fagtekst")}] '
            f'Tittel: {t["title"]} | '
            f'Hovedemne: {t.get("main_topic", "")} | '
            f'Underemner: {subs}'
        )

    return (
        f"Godkjente kategorier:\n{taxonomy_str}\n\n"
        f"Tekster:\n" + "\n".join(lines)
        + "\n\nTildel hver tekst 1-3 kategorier. "
        + "Fortellinger skal alltid ha 'Fortelling' som en av kategoriene."
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

            assignments = validate_assignments(
                json.loads(raw).get("assignments", {}), taxonomy, index_to_id
            )

            missing = {t["text_id"] for t in extracted} - set(assignments.keys())
            if missing:
                print(f"  [!] {len(missing)} texts not assigned.")

            missing_tag = [
                t["title"] for t in extracted
                if t.get("text_type") == "fortelling"
                and "Fortelling" not in assignments.get(t["text_id"], [])
            ]
            if missing_tag:
                print(f"  [!] {len(missing_tag)} stories missing Fortelling tag:")
                for title in missing_tag[:5]:
                    print(f"      - {title}")
                if len(missing_tag) > 5:
                    print(f"      ... and {len(missing_tag) - 5} more")

            return assignments

        except (json.JSONDecodeError, SchemaError) as e:
            print(f"  [!] Assignment error (attempt {attempt}): {e}")
        except Exception as e:
            print(f"  [!] API error (attempt {attempt}): {e}")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * attempt)

    raise RuntimeError("Assignment failed.")


# G��G�� Stage 4.5: Assignment quality check G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��

def run_assignment_quality_check(
    extracted: list[dict],
    taxonomy: list,
    assignments: dict,
    output_path: str,
) -> dict:
    text_map       = {t["text_id"]: t for t in extracted}
    category_texts = {cat: [] for cat in taxonomy}

    for text_id, cats in assignments.items():
        t = text_map.get(text_id)
        if not t:
            continue
        for cat in cats:
            if cat in category_texts:
                category_texts[cat].append(t)

    print(f"\n  {'G��'*50}")
    print(f"  ASSIGNMENT DISTRIBUTION")
    print(f"  {'G��'*50}")

    thin    = []
    bloated = []
    rows    = []

    for cat in taxonomy:
        texts = category_texts[cat]
        count = len(texts)
        flag  = ""

        if count < THIN_CATEGORY_THRESHOLD:
            flag = "  G�� thin"
            thin.append(cat)
        elif count > BLOATED_CATEGORY_THRESHOLD:
            flag = "  G�� broad"
            bloated.append(cat)

        print(f"  {cat:<20} {count:>4} texts{flag}")

        sample = random.sample(texts, min(CATEGORY_SPOT_SIZE, count))
        for t in sample:
            rows.append({
                "category":            cat,
                "text_id":             t["text_id"],
                "title":               t["title"],
                "text_type":           t.get("text_type", ""),
                "main_topic":          t.get("main_topic", ""),
                "sub_topics":          " | ".join(t.get("sub_topics", [])),
                "assigned_categories": " | ".join(assignments.get(t["text_id"], [])),
                "relevence":           "",
                "correct_category(if not primary)": "",
                "notes":               "",
            })

    if thin:
        print(f"\n  Thin (< {THIN_CATEGORY_THRESHOLD}): {', '.join(thin)}")
    if bloated:
        print(f"  Broad (> {BLOATED_CATEGORY_THRESHOLD}): {', '.join(bloated)}")

    if not os.path.exists(output_path):
        pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n  Spot check saved G�� {output_path}")
        print(f"  Fill in 'relevence' column: primary / secondary / wrong")
        print(f"  Then run --evaluate --compare")
    else:
        print(f"\n  {output_path} already exists G�� skipping overwrite.")

    return {
        "category_counts":    {cat: len(texts) for cat, texts in category_texts.items()},
        "thin_categories":    thin,
        "bloated_categories": bloated,
    }


# G��G�� Evaluation G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��

def compute_extraction_metrics(df: pd.DataFrame) -> dict:
    labeled = df[df["correct"].str.strip().str.lower().isin(["yes", "no"])]
    if len(labeled) == 0:
        return {"error": "No labeled rows found"}

    total    = len(labeled)
    correct  = (labeled["correct"].str.strip().str.lower() == "yes").sum()
    accuracy = round(correct / total, 3)

    result = {"extraction_accuracy": accuracy, "correct": int(correct), "total": total}

    if "text_type_manual" in labeled.columns:
        type_labeled = labeled[labeled["text_type_manual"].str.strip() != ""]
        if len(type_labeled) > 0:
            type_correct = (
                type_labeled["text_type_llm"].str.strip() ==
                type_labeled["text_type_manual"].str.strip()
            ).sum()
            result["text_type_accuracy"] = round(type_correct / len(type_labeled), 3)
            result["text_type_total"]    = len(type_labeled)

    return result


def compute_assignment_metrics(df: pd.DataFrame) -> dict:
    col     = "relevence"
    labeled = df[df[col].str.strip().str.lower().isin(["primary", "secondary", "wrong"])]

    if len(labeled) == 0:
        return {"error": "No labeled rows found G�� fill in 'relevence' column first"}

    total     = len(labeled)
    primary   = int((labeled[col].str.strip().str.lower() == "primary").sum())
    secondary = int((labeled[col].str.strip().str.lower() == "secondary").sum())
    wrong     = int((labeled[col].str.strip().str.lower() == "wrong").sum())

    per_category = {}
    for cat, group in labeled.groupby("category"):
        gp = int((group[col].str.strip().str.lower() == "primary").sum())
        gs = int((group[col].str.strip().str.lower() == "secondary").sum())
        gw = int((group[col].str.strip().str.lower() == "wrong").sum())
        gt = len(group)
        per_category[cat] = {
            "primary":              gp,
            "secondary":            gs,
            "wrong":                gw,
            "total":                gt,
            "primary_precision":    round(gp / gt, 3),
            "acceptable_precision": round((gp + gs) / gt, 3),
        }

    return {
        "total_rated":          total,
        "primary":              primary,
        "secondary":            secondary,
        "wrong":                wrong,
        "primary_precision":    round(primary / total, 3),
        "acceptable_precision": round((primary + secondary) / total, 3),
        "per_category":         per_category,
    }


def run_evaluation(compare: bool = False):
    if not compare:
        print("\nFill in evaluation files and re-run with --compare")
        return

    metrics = {}

    if os.path.exists(EVAL_SAMPLE_CSV):
        df = pd.read_csv(EVAL_SAMPLE_CSV, dtype=str).fillna("")
        if "correct" in df.columns:
            print("\n[Evaluation] Extraction metrics")
            metrics["extraction"] = compute_extraction_metrics(df)
            m = metrics["extraction"]
            if "error" not in m:
                print(f"  Accuracy:        {m['extraction_accuracy']:.1%}  ({m['correct']}/{m['total']})")
                if "text_type_accuracy" in m:
                    print(f"  Text type:       {m['text_type_accuracy']:.1%}")
        else:
            print(f"\n  {EVAL_SAMPLE_CSV}: no 'correct' column found")
    else:
        print(f"\n  {EVAL_SAMPLE_CSV} not found")

    if os.path.exists(ASSIGNMENT_EVAL_CSV):
        df = pd.read_csv(ASSIGNMENT_EVAL_CSV, dtype=str).fillna("")
        if "relevence" in df.columns:
            print("\n[Evaluation] Assignment metrics")
            metrics["assignment"] = compute_assignment_metrics(df)
            m = metrics["assignment"]
            if "error" in m:
                print(f"  {m['error']}")
            else:
                print(f"  Total rated:          {m['total_rated']}")
                print(f"  Primary precision:    {m['primary_precision']:.1%}  ({m['primary']} texts)")
                print(f"  Acceptable precision: {m['acceptable_precision']:.1%}  ({m['primary'] + m['secondary']} texts)")
                print(f"  Wrong:                {m['wrong']} texts")
                print(f"\n  {'Category':<20} {'Primary':>8} {'Secondary':>10} {'Wrong':>6} {'Acceptable':>11}")
                print(f"  {'G��'*58}")
                for cat, cm in m.get("per_category", {}).items():
                    print(
                        f"  {cat:<20} {cm['primary']:>8} {cm['secondary']:>10} "
                        f"{cm['wrong']:>6} {cm['acceptable_precision']:>10.0%}"
                    )

                secondary_rows = df[df["relevence"].str.strip().str.lower() == "secondary"]
                if len(secondary_rows) > 0:
                    corrections = [
                        (row["category"], row["title"], row.get("correct_category(if not primary)", ""))
                        for _, row in secondary_rows.iterrows()
                        if str(row.get("correct_category(if not primary)", "")).strip()
                    ]
                    if corrections:
                        print(f"\n  Secondary assignments with suggested corrections:")
                        for cat, title, better in corrections:
                            print(f"    {cat:<20} {title[:35]} G�� {better}")
        else:
            print(f"\n  {ASSIGNMENT_EVAL_CSV}: no 'relevence' column found")
    else:
        print(f"\n  {ASSIGNMENT_EVAL_CSV} not found")

    save_json(EVAL_METRICS_JSON, metrics)
    if "extraction" in metrics:
        save_json(EVAL_EXTRACTION_METRICS_JSON, metrics["extraction"])
    if "assignment" in metrics:
        save_json(EVAL_ASSIGNMENT_METRICS_JSON, metrics["assignment"])

    print(f"\n  Saved G�� {EVAL_METRICS_JSON}")


# G��G�� Stage 5: Export G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��

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

    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  CSV G�� {output_path}")

    save_json(ASSIGNMENTS_JSON, {"taxonomy": taxonomy, "assignments": assignments})
    print(f"  Assignments G�� {ASSIGNMENTS_JSON}")

    print(f"\n  {'G��'*50}")
    print(f"  BROAD TOPICS ({len(taxonomy)})")
    print(f"  {'G��'*50}")
    for topic in taxonomy:
        count = sum(1 for r in rows if topic in r["broad_topics"].split(" | "))
        print(f"  {topic:<25} {count} texts")

    stories = [r for r in rows if r.get("text_type") == "fortelling"]
    tagged  = [r for r in stories if "Fortelling" in r["broad_topics"].split(" | ")]
    print(f"\n  Fortelling coverage: {len(tagged)}/{len(stories)}")

    errors = [r for r in rows if r["extraction_error"]]
    if errors:
        print(f"\n  Extraction errors ({len(errors)}):")
        for e in errors:
            print(f"    - {e['title']}")


# G��G�� Main G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��

def main():
    parser = argparse.ArgumentParser(description="Topic pipeline for Norwegian children's texts.")
    parser.add_argument("--input",           required=True)
    parser.add_argument("--output",          default=OUTPUT_CSV)
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip extraction, reload from checkpoint, re-propose taxonomy.")
    parser.add_argument("--assign-only",     action="store_true",
                        help="Skip extraction and taxonomy. Load taxonomy_final.json and assign.")
    parser.add_argument("--evaluate",        action="store_true",
                        help="Compute evaluation metrics from labeled CSV files.")
    parser.add_argument("--compare",         action="store_true",
                        help="Use with --evaluate to compute and print metrics.")
    args = parser.parse_args()

    load_dotenv(Path(__file__).parent.parent.parent / ".env")

    if args.evaluate:
        run_evaluation(compare=args.compare)
        return

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    client = Groq(api_key=api_key)
    print(f"  Model: {MODEL}\n")

    # Stage 1
    print("[Stage 1] Preprocessing...")
    texts = load_and_preprocess(args.input)

    # Stage 2
    if args.skip_extraction or args.assign_only:
        print("\n[Stage 2] Loading extracted topics...")
        if os.path.exists(EXTRACTED_JSON):
            raw  = load_json(EXTRACTED_JSON)
            tmap = {t["text_id"]: t for t in texts}
            extracted = [{**tmap[e["text_id"]], **e} for e in raw if e.get("text_id") in tmap]
        else:
            ckpt      = load_checkpoint()
            valid_ids = {t["text_id"] for t in texts}
            extracted = [v for k, v in ckpt.items() if k in valid_ids]
        print(f"  Loaded {len(extracted)} texts.")
    else:
        print(f"\n[Stage 2] Extracting topics ({len(texts)} texts)...")
        extracted = run_extraction(client, texts)
        print(f"\n[Stage 2.5] Extraction quality check...")
        run_extraction_quality_check(extracted)

    # Stage 3
    if args.assign_only:
        if not os.path.exists(FINAL_JSON):
            print(f"\nError: {FINAL_JSON} not found.")
            sys.exit(1)
        taxonomy = load_json(FINAL_JSON).get("broad_topics", [])
        print(f"\n[Stage 3] Loaded taxonomy ({len(taxonomy)} categories) from {FINAL_JSON}")
    else:
        print(f"\n[Stage 3] Proposing taxonomy...")
        draft = propose_taxonomy(client, extracted)
        save_json(DRAFT_JSON, draft)

        print(f"\n  Saved G�� {DRAFT_JSON}")
        print(f"\n  {'G��'*50}")
        print(f"  DRAFT TAXONOMY ({len(draft['broad_topics'])} categories)")
        print(f"  {'G��'*50}")
        for topic in draft["broad_topics"]:
            rationale = draft.get("rationale", {}).get(topic, "")
            print(f"  G�� {topic}")
            if rationale:
                print(f"    {rationale}")

        print(f"\n[Stage 3.5] Gap analysis...")
        gaps = run_gap_analysis(client, extracted, draft["broad_topics"])
        print_gap_analysis(gaps)

        print(f"""
  Next steps:
  1. Open {DRAFT_JSON}
  2. Review categories and gaps above
     - Add missing categories
     - Remove overlapping ones
     - Rename as needed
  3. Save as {FINAL_JSON}
  4. Run: python {Path(__file__).name} --input {Path(args.input).name} --assign-only
""")
        return

    # Stage 4
    print(f"\n[Stage 4] Assigning texts...")
    assignments = assign_texts(client, taxonomy, extracted)
    print(f"  {len(assignments)}/{len(extracted)} texts assigned.")

    # Stage 4.5
    print(f"\n[Stage 4.5] Assignment quality check...")
    quality = run_assignment_quality_check(extracted, taxonomy, assignments, ASSIGNMENT_EVAL_CSV)

    if quality["thin_categories"] or quality["bloated_categories"]:
        print(f"""
  Options:
  A) Edit {FINAL_JSON} and re-run --assign-only
  B) Manually fix assignments in results.csv
  C) Accept as-is
""")

    # Stage 5
    print(f"\n[Stage 5] Exporting...")
    export_results(extracted, taxonomy, assignments, args.output)

    print(f"""
  Fill in {ASSIGNMENT_EVAL_CSV} (relevence: primary/secondary/wrong)
  then run: python {Path(__file__).name} --input {Path(args.input).name} --evaluate --compare
""")


if __name__ == "__main__":
    main()
