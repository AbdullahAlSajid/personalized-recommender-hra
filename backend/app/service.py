"""
Recommender Service
====================
Stateless bridge between FastAPI endpoints and the recommender engine.
Texts are loaded once and cached. Session state is reconstructed from DB per request.
"""

from __future__ import annotations

import math
import uuid
import html
import re
from pathlib import Path
from urllib.parse import quote
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text as sql_text
from sqlalchemy import bindparam
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as DBSession

from app.models import (
    BroadTopic,
    SessionInterest,
    SlateEvent,
    ReadingEvent,
    SessionEvent,
    ReadingQuestionResponse,
    SessionFeedback,
)
from recommender.recommender.engine import (
    LevelEstimator,
    ScoringEngine,
    SlateBuilder,
    _get_weights,
)


# ════════════════════════════════════════════════════════════
# Text cache
# ════════════════════════════════════════════════════════════

_texts_cache: Optional[pd.DataFrame] = None
_images_dir = Path(__file__).resolve().parents[2] / "data" / "images"
_thumbs_dir = _images_dir / "thumbs"


def _normalize_preview_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _normalize_title_like_text(value: str) -> str:
    return (
        str(value)
        .replace("\ufeff", "")
        .strip()
        .strip("\"'“”‘’")
        .replace("\u00a0", " ")
        .strip()
        .lower()
    )


def _strip_leading_title_heading_markdown(markdown: str, title: str) -> str:
    if not markdown:
        return ""
    normalized_title = _normalize_title_like_text(title)
    if not normalized_title:
        return markdown

    lines = re.split(r"\r?\n", markdown.replace("\ufeff", ""))
    first_non_empty = next((i for i, ln in enumerate(lines) if ln.strip()), -1)
    if first_non_empty == -1:
        return markdown

    m = re.match(r"^(#{1,6})\s+(.*)$", lines[first_non_empty])
    if not m:
        return markdown

    heading_text = _normalize_title_like_text(m.group(2) or "")
    if heading_text != normalized_title:
        return markdown

    del lines[first_non_empty]
    return "\n".join(lines).lstrip("\n").lstrip()


def _remove_empty_markdown_headings(markdown: str) -> str:
    if not markdown:
        return ""
    return re.sub(r"^#{1,6}\s*$", "", markdown, flags=re.MULTILINE)


def _strip_markdown_images(markdown: str) -> str:
    if not markdown:
        return ""
    # Drop markdown image syntax entirely: ![alt](url)
    return re.sub(r"!\[[^\]]*\]\(\s*[^)]*\)", " ", markdown)


def _strip_html_to_text(value: str) -> str:
    if not value:
        return ""
    text = value
    # Remove scripts/styles
    text = re.sub(r"<script\b[\s\S]*?</script>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<style\b[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    # Remove images
    text = re.sub(r"<img\b[^>]*>", " ", text, flags=re.IGNORECASE)
    # Replace common block separators with spaces/newlines before stripping tags
    text = re.sub(r"<(br|br\s*/)>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(p|div|h[1-6]|li)>", "\n", text, flags=re.IGNORECASE)
    # Strip remaining tags
    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(text)


def _derive_preview_text(
    *,
    title: Any,
    body: Any,
    max_chars: int = 250,
) -> str | None:
    if body is None:
        return None

    text = str(body).replace("\r\n", "\n").strip()
    if not text:
        return None

    t = str(title).strip() if title is not None else ""
    if t:
        # Mirror Reading page: if the first non-empty line is a markdown heading
        # matching the title, drop it.
        text = _strip_leading_title_heading_markdown(text, t)

        # Also drop a plain leading title line / prefix if present.
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if lines and _normalize_title_like_text(lines[0]) == _normalize_title_like_text(t):
            lines = lines[1:]
            text = "\n".join(lines)
        elif _normalize_title_like_text(text[: len(t)]) == _normalize_title_like_text(t):
            text = text[len(t) :]

    # Remove empty markdown headings and images to avoid junky previews.
    text = _remove_empty_markdown_headings(text)
    text = _strip_markdown_images(text)

    # If content looks like HTML, strip tags/images to plain text for preview.
    if re.search(r"</?[a-z][\s\S]*>", text, flags=re.IGNORECASE):
        text = _strip_html_to_text(text)

    normalized = _normalize_preview_text(text)
    if not normalized:
        return None

    if len(normalized) > max_chars:
        return normalized[:max_chars].rstrip() + "...."
    return normalized


def _get_text_content_column(db: DBSession) -> str | None:
    candidate_columns = [
        "content_html",
        "content",
        "body",
        "text",
        "html",
        "clean_body",
        "cleaned_body",
    ]
    cols = db.execute(
        sql_text(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'texts'
            """
        )
    ).fetchall()
    existing = {r[0] for r in cols}
    return next((c for c in candidate_columns if c in existing), None)


def _fetch_text_bodies(db: DBSession, text_ids: list[str]) -> dict[str, str]:
    if not text_ids:
        return {}

    chosen = _get_text_content_column(db)
    content_select = "NULL::text AS content"
    if chosen is not None:
        content_select = f't."{chosen}"::text AS content'

    query = (
        sql_text(
            f"""
            SELECT
                t.id::text AS text_id,
                {content_select}
            FROM texts t
            WHERE t.id::text IN :text_ids
            """
        )
        .bindparams(bindparam("text_ids", expanding=True))
    )
    rows = db.execute(query, {"text_ids": text_ids}).mappings().all()
    return {str(r.get("text_id")): (r.get("content") or "") for r in rows}


def _to_safe_image_name(value: object) -> str | None:
    """Normalize DB image file values to a safe string for URL quoting."""
    if value is None:
        return None

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    elif isinstance(value, memoryview):
        value = value.tobytes().decode("utf-8", errors="ignore")

    text = str(value).strip()
    return text or None


def _build_image_url(name: str, *, prefer_thumbnail: bool = False) -> str:
    encoded = quote(name)
    if prefer_thumbnail:
      return f"/images/thumbs/{encoded}"
    return f"/images/{encoded}"


def _build_preferred_image_url(name: str) -> str:
    if (_thumbs_dir / name).is_file():
        return _build_image_url(name, prefer_thumbnail=True)
    return _build_image_url(name)


def _choose_preview_image_url(values: object) -> str | None:
    candidates = [
        name for name in (_to_safe_image_name(value) for value in (values or [])) if name
    ]
    if not candidates:
        return None

    for name in candidates:
        if (_thumbs_dir / name).is_file():
            return _build_preferred_image_url(name)

    return _build_image_url(candidates[0])


def load_texts(db: DBSession) -> pd.DataFrame:
    """Load reliable texts from DB into a cached DataFrame."""
    global _texts_cache
    if _texts_cache is not None:
        return _texts_cache

    ti_cols = db.execute(
        sql_text(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'text_images'
            """
        )
    ).fetchall()
    ti_existing = {r[0] for r in ti_cols}
    ti_order_by = (
        "ti.image_order, ti.file_name" if "image_order" in ti_existing else "ti.file_name"
    )

    query = sql_text(
        f"""
        SELECT
            t.id AS text_id,
            t.title,
            t.serial_number,
            td.final_difficulty,
            td.reliable,
            ARRAY_AGG(bt.name ORDER BY tta.is_primary DESC, bt.name)
                AS broad_topics_array,
            (
                SELECT ARRAY_AGG(ti.file_name ORDER BY {ti_order_by})
                FROM text_images ti
                WHERE ti.text_id = t.id
            ) AS image_file_names_array,
            (
                SELECT ti.file_name
                FROM text_images ti
                WHERE ti.text_id = t.id
                ORDER BY {ti_order_by}
                LIMIT 1
            ) AS first_image_file_name
        FROM texts t
        JOIN text_difficulty td ON t.id = td.text_id
        JOIN text_topic_assignments tta ON t.id = tta.text_id
        JOIN broad_topics bt ON tta.broad_topic_id = bt.id
        WHERE td.reliable = TRUE
        AND t.is_active = TRUE
        GROUP BY t.id, t.title, t.serial_number,
                td.final_difficulty, td.reliable
        ORDER BY t.title;
        """
    )
    result = db.execute(query)
    rows = result.fetchall()
    columns = result.keys()

    df = pd.DataFrame(rows, columns=columns)
    df["text_id"] = df["text_id"].astype(str)

    # DB numeric columns may come back as Decimal; recommender math assumes float.
    if "final_difficulty" in df.columns:
        df["final_difficulty"] = pd.to_numeric(df["final_difficulty"], errors="coerce").astype(float)

    df["broad_topics_list"] = df["broad_topics_array"].apply(
        lambda x: list(x) if x else []
    )
    df["broad_topics"] = df["broad_topics_list"].apply(
        lambda x: " | ".join(x) if x else ""
    )
    df["image_file_names_list"] = df["image_file_names_array"].apply(
        lambda x: list(x) if x else []
    )
    df["first_image_url"] = df["image_file_names_list"].apply(_choose_preview_image_url)
    df["sub_topics_list"] = [[] for _ in range(len(df))]
    if "broad_topics_array" in df.columns:
        df = df.drop(columns=["broad_topics_array"])
    if "image_file_names_array" in df.columns:
        df = df.drop(columns=["image_file_names_array"])

    _texts_cache = df
    return df


def get_text_detail(db: DBSession, text_id: str) -> dict | None:
    """Fetch one text's details (title + content + images) by id."""

    chosen = _get_text_content_column(db)

    ti_cols = db.execute(
        sql_text(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'text_images'
            """
        )
    ).fetchall()
    ti_existing = {r[0] for r in ti_cols}
    ti_order_by = (
        "ti.image_order, ti.file_name" if "image_order" in ti_existing else "ti.file_name"
    )

    content_select = "NULL::text AS content"
    if chosen is not None:
        # chosen comes from a fixed allowlist, so identifier injection isn't possible here.
        content_select = f't."{chosen}"::text AS content'

    query = sql_text(
        f"""
        SELECT
            t.id::text AS text_id,
            t.title,
            {content_select},
            (
                SELECT ARRAY_AGG(ti.file_name ORDER BY {ti_order_by})
                FROM text_images ti
                WHERE ti.text_id = t.id
            ) AS image_file_names_array,
            (
                SELECT ti.file_name
                FROM text_images ti
                WHERE ti.text_id = t.id
                ORDER BY {ti_order_by}
                LIMIT 1
            ) AS first_image_file_name
        FROM texts t
        WHERE t.id::text = :text_id
        LIMIT 1
        """
    )

    row = db.execute(query, {"text_id": text_id}).mappings().first()
    if not row:
        return None

    image_file_names = list(row.get("image_file_names_array") or [])
    safe_names = [
        s for s in (
            _to_safe_image_name(x) for x in image_file_names
        )
        if s
    ]
    image_urls = [_build_preferred_image_url(name) for name in safe_names]

    first_image_url = None
    if (safe_first := _to_safe_image_name(row.get("first_image_file_name"))):
        first_image_url = _build_preferred_image_url(safe_first)

    return {
        "text_id": row.get("text_id"),
        "title": row.get("title"),
        "body": row.get("content") or "",
        "content": row.get("content") or "",
        "image_urls": image_urls,
        "first_image_url": first_image_url,
    }


def mark_slate_choice(db: DBSession, session_id: str, chosen_text_id: str) -> bool:
    """Mark a pending slate as having been chosen.

    This is a lightweight way to reflect the user's selection as soon as they
    open the reading view (before submitting the full reading feedback).
    """
    try:
        chosen_uuid = _to_uuid(chosen_text_id)
        session_uuid = _to_uuid(session_id)
    except (ValueError, TypeError):
        return False

    round_number = _get_round_number(db, session_id)

    pending = (
        db.query(SlateEvent)
        .filter(SlateEvent.session_id == session_uuid)
        .filter(SlateEvent.round_number == round_number)
        .filter(SlateEvent.was_refresh == False)
        .filter(SlateEvent.chosen_text_id.is_(None))
        .filter(SlateEvent.shown_text_ids.any(chosen_uuid))
        .order_by(SlateEvent.created_at.desc(), SlateEvent.slate_id.desc())
        .first()
    )
    if pending is None:
        return False

    pending.chosen_text_id = chosen_uuid
    db.add(
        SessionEvent(
            session_id=session_uuid,
            slate_id=pending.slate_id,
            text_id=chosen_uuid,
            event_type="dashboard_text_selected",
            event_metadata={"source": "text_open"},
        )
    )
    db.commit()
    return True


def log_session_event(
    db: DBSession,
    session_id: str,
    event_type: str,
    slate_id: Optional[int] = None,
    text_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> None:
    session_uuid = _to_uuid(session_id)
    text_uuid = _to_uuid(text_id) if text_id else None

    if slate_id is None and isinstance(metadata, dict):
        shown_ids = metadata.get("shown_text_ids")
        if isinstance(shown_ids, list):
            shown_uuids = []
            for value in shown_ids:
                try:
                    shown_uuids.append(_to_uuid(value))
                except (TypeError, ValueError):
                    continue

            if shown_uuids:
                slate_q = (
                    db.query(SlateEvent)
                    .filter(SlateEvent.session_id == session_uuid)
                )
                if len(shown_uuids) == 1:
                    slate_q = slate_q.filter(SlateEvent.shown_text_ids.any(shown_uuids[0]))
                else:
                    slate_q = slate_q.filter(SlateEvent.shown_text_ids.contains(shown_uuids))

                match = slate_q.order_by(
                    SlateEvent.created_at.desc(),
                    SlateEvent.slate_id.desc(),
                ).first()
                if match is not None:
                    slate_id = match.slate_id

    db.add(
        SessionEvent(
            session_id=session_uuid,
            slate_id=slate_id,
            text_id=text_uuid,
            event_type=event_type,
            event_metadata=metadata,
        )
    )
    db.commit()


def _get_table_columns(db: DBSession, table_name: str) -> set[str]:
    rows = db.execute(
        sql_text(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = :table_name
            """
        ),
        {"table_name": table_name},
    ).fetchall()
    return {r[0] for r in rows}


def _is_free_text_question_type(value: Any) -> bool:
    if value is None:
        return False
    normalized = str(value).strip().lower().replace("_", " ")
    return normalized in {"freetext", "free text"}


def _get_global_post_reading_questions() -> list[dict[str, Any]]:
    """Questions shown for every text (not stored in DB).

    Returned shape matches TextQuestionItem/TextQuestionsResponse.
    """

    def make_option(value: str, body: str, order: int) -> dict[str, Any]:
        return {
            "option_id": value,
            "body": body,
            "sanity_answer_key": None,
            "is_correct": None,
            "display_order": order,
        }

    return [
        {
            "question_id": "global:perceived_difficulty",
            "body": "Hvor vanskelig synes du teksten var?",
            "question_type": "rating_1_5",
            "display_order": None,
            "options": [
                make_option("1", "Veldig lett", 1),
                make_option("2", "Ganske lett", 2),
                make_option("3", "Passe", 3),
                make_option("4", "Ganske vanskelig", 4),
                make_option("5", "Veldig vanskelig", 5),
            ],
        },
        {
            "question_id": "global:interest_rating",
            "body": "Hvor interessant synes du teksten var?",
            "question_type": "rating_1_5",
            "display_order": None,
            "options": [
                make_option("1", "Ikke interessant i det hele tatt", 1),
                make_option("2", "Litt interessant", 2),
                make_option("3", "Passe interessant", 3),
                make_option("4", "Ganske interessant", 4),
                make_option("5", "Veldig interessant", 5),
            ],
        },
        {
            "question_id": "global:recommend_to_friend",
            "body": "Ville du anbefalt denne teksten til en venn?",
            "question_type": "single_choice",
            "display_order": None,
            "options": [
                make_option("yes", "Ja", 1),
                make_option("maybe", "Kanskje", 2),
                make_option("no", "Nei", 3),
            ],
        },
    ]


def get_text_questions(db: DBSession, text_id: str) -> list[dict[str, Any]]:
    """Fetch all questions (and their options) for a given text.

    This uses raw SQL and light schema introspection so it can tolerate
    minor schema differences across environments.
    """

    q_cols = _get_table_columns(db, "questions")
    if not q_cols:
        return []

    qt_cols = _get_table_columns(db, "question_types")
    qo_cols = _get_table_columns(db, "question_options")

    q_text_fk = next((c for c in ["text_id", "sanity_text_id"] if c in q_cols), None)
    if q_text_fk is None:
        return []

    q_body_col = next((c for c in ["body", "question", "text", "prompt"] if c in q_cols), None)
    q_order_col = next((c for c in ["display_order", "order", "sort_order"] if c in q_cols), None)

    qt_name_col = next((c for c in ["name", "type", "label"] if c in qt_cols), None)
    q_qt_fk = next(
        (c for c in ["question_type_id", "question_type", "type_id"] if c in q_cols),
        None,
    )
    q_type_inline_col = next((c for c in ["question_type", "type"] if c in q_cols), None)

    qo_body_col = next((c for c in ["body", "option", "text", "label"] if c in qo_cols), None)
    qo_order_col = next((c for c in ["display_order", "order", "sort_order"] if c in qo_cols), None)
    qo_q_fk = next((c for c in ["question_id", "question_key_id"] if c in qo_cols), None)
    qo_sanity_key_col = next(
        (c for c in ["sanity_answer_key", "sanity_option_key", "answer_key"] if c in qo_cols),
        None,
    )
    qo_correct_col = next((c for c in ["is_correct", "correct"] if c in qo_cols), None)

    question_body_select = "NULL::text AS question_body"
    if q_body_col is not None:
        question_body_select = f'q."{q_body_col}"::text AS question_body'

    question_order_select = "NULL::int AS question_display_order"
    question_order_orderby = "q.id"
    if q_order_col is not None:
        question_order_select = f'q."{q_order_col}"::int AS question_display_order'
        question_order_orderby = f'q."{q_order_col}", q.id'

    option_selects: list[str] = [
        "qo.id::text AS option_id",
        "NULL::text AS option_body",
        "NULL::text AS option_sanity_answer_key",
        "NULL::boolean AS option_is_correct",
        "NULL::int AS option_display_order",
    ]
    option_orderby = "qo.id"
    if qo_cols and qo_body_col is not None:
        option_selects[1] = f'qo."{qo_body_col}"::text AS option_body'
    if qo_cols and qo_sanity_key_col is not None:
        option_selects[2] = f'qo."{qo_sanity_key_col}"::text AS option_sanity_answer_key'
    if qo_cols and qo_correct_col is not None:
        option_selects[3] = f'qo."{qo_correct_col}"::boolean AS option_is_correct'
    if qo_cols and qo_order_col is not None:
        option_selects[4] = f'qo."{qo_order_col}"::int AS option_display_order'
        option_orderby = f'qo."{qo_order_col}", qo.id'

    use_qt_join = bool(qt_cols) and qt_name_col is not None and q_qt_fk is not None
    if use_qt_join:
        question_type_select = f'qt."{qt_name_col}"::text AS question_type'
        question_type_join = f'JOIN question_types qt ON qt.id = q."{q_qt_fk}"'
    else:
        question_type_join = ""
        if q_type_inline_col is not None:
            question_type_select = f'q."{q_type_inline_col}"::text AS question_type'
        else:
            question_type_select = "'unknown'::text AS question_type"

    use_qo_join = bool(qo_cols) and qo_q_fk is not None
    if use_qo_join:
        options_join = f'LEFT JOIN question_options qo ON qo."{qo_q_fk}" = q.id'
    else:
        # Still include the option columns as NULLs.
        options_join = "LEFT JOIN (SELECT NULL::text AS id) qo ON FALSE"

    query = sql_text(
        f"""
        SELECT
            q.id::text AS question_id,
            {question_body_select},
            {question_type_select},
            {question_order_select},
            {", ".join(option_selects)}
        FROM questions q
        {question_type_join}
        {options_join}
        WHERE q."{q_text_fk}"::text = :text_id
        ORDER BY {question_order_orderby}, {option_orderby}
        """
    )

    rows = db.execute(query, {"text_id": text_id}).mappings().all()
    if not rows:
        return []

    questions: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        qid = row.get("question_id")
        if not qid:
            continue

        if _is_free_text_question_type(row.get("question_type")):
            continue

        if qid not in questions:
            questions[qid] = {
                "question_id": qid,
                "body": row.get("question_body") or "",
                "question_type": row.get("question_type") or "unknown",
                "display_order": row.get("question_display_order"),
                "options": [],
            }
            order.append(qid)

        option_id = row.get("option_id")
        option_body = row.get("option_body")
        if option_id and option_body:
            questions[qid]["options"].append(
                {
                    "option_id": option_id,
                    "body": option_body,
                    "sanity_answer_key": row.get("option_sanity_answer_key"),
                    "is_correct": row.get("option_is_correct"),
                    "display_order": row.get("option_display_order"),
                }
            )

    text_specific = [questions[qid] for qid in order]
    return text_specific + _get_global_post_reading_questions()


def get_texts(db: DBSession) -> pd.DataFrame:
    global _texts_cache
    if _texts_cache is None:
        return load_texts(db)
    return _texts_cache


# ════════════════════════════════════════════════════════════
# Interests (via session_interests table)
# ════════════════════════════════════════════════════════════

def get_session_interests(db: DBSession, session_id: str) -> List[str]:
    """Get interest topic names for a session."""
    rows = (
        db.query(BroadTopic.name)
        .join(SessionInterest, SessionInterest.broad_topic_id == BroadTopic.id)
        .filter(SessionInterest.session_id == session_id)
        .order_by(BroadTopic.name)
        .all()
    )
    return [r[0] for r in rows]


def set_session_interests(
    db: DBSession, session_id: str, topic_names: List[str]
) -> List[str]:
    """Replace all interests for a session. Returns the saved list."""
    # Delete existing
    db.query(SessionInterest).filter(
        SessionInterest.session_id == session_id
    ).delete()

    # Look up topic IDs
    topics = db.query(BroadTopic).filter(BroadTopic.name.in_(topic_names)).all()
    topic_map = {t.name: t.id for t in topics}

    # Insert new
    for name in topic_names:
        tid = topic_map.get(name)
        if tid:
            db.add(SessionInterest(session_id=session_id, broad_topic_id=tid))

    db.commit()
    return get_session_interests(db, session_id)


# ════════════════════════════════════════════════════════════
# Session state reconstruction
# ════════════════════════════════════════════════════════════

def _get_seen_text_ids(db: DBSession, session_id: str) -> set:
    slates = (
        db.query(SlateEvent)
        .filter(SlateEvent.session_id == session_id)
        .all()
    )
    seen = set()
    for slate in slates:
        if slate.shown_text_ids:
            for tid in slate.shown_text_ids:
                seen.add(str(tid))
    return seen


def _get_round_number(db: DBSession, session_id: str) -> int:
    n_readings = (
        db.query(ReadingEvent)
        .filter(ReadingEvent.session_id == session_id)
        .count()
    )
    return n_readings + 1


def _rebuild_level_estimator(db: DBSession, session_id: str) -> LevelEstimator:
    readings = (
        db.query(ReadingEvent)
        .filter(ReadingEvent.session_id == session_id)
        .order_by(ReadingEvent.round_number)
        .all()
    )
    estimator = LevelEstimator()
    for r in readings:
        estimator.update(float(r.text_difficulty), int(r.perceived_difficulty))
    return estimator


def _to_uuid(value: str | uuid.UUID) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


def _to_uuid_list(values: List[str | uuid.UUID]) -> List[uuid.UUID]:
    return [_to_uuid(v) for v in values]


def _pending_slate_event(
    db: DBSession,
    session_id: str,
    round_number: int,
    shown_text_ids: List[str] | None = None,
) -> SlateEvent | None:
    query = (
        db.query(SlateEvent)
        .filter(SlateEvent.session_id == _to_uuid(session_id))
        .filter(SlateEvent.round_number == round_number)
        .filter(SlateEvent.was_refresh == False)
        .filter(SlateEvent.chosen_text_id.is_(None))
        .order_by(SlateEvent.created_at.desc(), SlateEvent.slate_id.desc())
    )
    if shown_text_ids is not None:
        query = query.filter(SlateEvent.shown_text_ids == _to_uuid_list(shown_text_ids))
    return query.first()


def _build_recommendations_payload_from_ids(
    db: DBSession,
    session_id: str,
    interests: List[str],
    shown_ids: List[uuid.UUID],
) -> dict:
    texts_df = get_texts(db)
    round_number = _get_round_number(db, session_id)
    level_estimator = _rebuild_level_estimator(db, session_id)
    w_topic, w_diff = _get_weights(round_number)

    shown_strs = [str(x) for x in shown_ids]
    subset = texts_df[texts_df["text_id"].isin(shown_strs)].copy()
    if len(subset) == 0:
        return {
            "round_number": round_number,
            "w_topic": w_topic,
            "w_difficulty": w_diff,
            "estimated_level": level_estimator.estimated_level,
            "texts": [],
        }

    scoring = ScoringEngine()
    scored = scoring.score_candidates(
        candidates=subset,
        student_interests=interests,
        estimated_level=level_estimator.estimated_level,
        round_number=round_number,
    )
    scored_by_id = {str(r["text_id"]): r for _, r in scored.iterrows()}
    rows_by_id = {str(r["text_id"]): r for _, r in subset.iterrows()}

    bodies_by_id = _fetch_text_bodies(db, shown_strs)

    texts_out: list[dict[str, Any]] = []
    for tid in shown_strs:
        if tid not in rows_by_id:
            continue
        row = rows_by_id[tid]
        score_row = scored_by_id.get(tid)
        preview_text = _derive_preview_text(
            title=row.get("title"),
            body=bodies_by_id.get(tid) or "",
        )
        texts_out.append(
            {
                "text_id": tid,
                "title": row.get("title"),
                "broad_topics": row.get("broad_topics_list", []),
                "first_image_url": row.get("first_image_url"),
                "preview_text": preview_text,
                "final_difficulty": float(row["final_difficulty"]),
                "composite_score": float(score_row["composite_score"]) if score_row is not None else 0.0,
                "score_topic": float(score_row["score_topic"]) if score_row is not None else 0.0,
                "score_difficulty": float(score_row["score_difficulty"]) if score_row is not None else 0.0,
            }
        )

    return {
        "round_number": round_number,
        "w_topic": w_topic,
        "w_difficulty": w_diff,
        "estimated_level": level_estimator.estimated_level,
        "texts": texts_out,
    }


def get_current_recommendations(
    db: DBSession,
    session_id: str,
    interests: List[str],
) -> dict:
    """Return the current pending slate for this round, or compute+log a new one."""
    round_number = _get_round_number(db, session_id)

    pending = _pending_slate_event(db, session_id=session_id, round_number=round_number)
    if pending is not None:
        return _build_recommendations_payload_from_ids(
            db=db,
            session_id=session_id,
            interests=interests,
            shown_ids=list(pending.shown_text_ids or []),
        )

    result = get_recommendations(db=db, session_id=session_id, interests=interests)

    shown_text_ids = [t.get("text_id") for t in result.get("texts", [])]
    shown_text_ids = [t for t in shown_text_ids if t]
    if shown_text_ids:
        slate_event = SlateEvent(
            session_id=_to_uuid(session_id),
            round_number=result["round_number"],
            shown_text_ids=_to_uuid_list(shown_text_ids),
            chosen_text_id=None,
            was_refresh=False,
            w_topic=result.get("w_topic"),
            w_difficulty=result.get("w_difficulty"),
            estimated_level=result.get("estimated_level"),
        )
        db.add(slate_event)
        try:
            db.flush()
            db.add(
                SessionEvent(
                    session_id=_to_uuid(session_id),
                    slate_id=slate_event.slate_id,
                    event_type="dashboard_slate_shown",
                    event_metadata={
                        "round_number": result.get("round_number"),
                        "shown_text_ids": shown_text_ids,
                        "w_topic": result.get("w_topic"),
                        "w_difficulty": result.get("w_difficulty"),
                        "estimated_level": result.get("estimated_level"),
                    },
                )
            )
            db.commit()
        except IntegrityError:
            # Another concurrent request already inserted the slate for this
            # round. Roll back, then return that winning slate instead.
            db.rollback()
            winner = _pending_slate_event(
                db, session_id=session_id, round_number=result["round_number"]
            )
            if winner is not None:
                return _build_recommendations_payload_from_ids(
                    db=db,
                    session_id=session_id,
                    interests=interests,
                    shown_ids=list(winner.shown_text_ids or []),
                )

    return result


# ════════════════════════════════════════════════════════════
# Core recommendation logic
# ════════════════════════════════════════════════════════════

def get_recommendations(
    db: DBSession,
    session_id: str,
    interests: List[str],
) -> dict:
    texts_df = get_texts(db)
    seen_ids = _get_seen_text_ids(db, session_id)
    round_number = _get_round_number(db, session_id)
    level_estimator = _rebuild_level_estimator(db, session_id)

    candidates = texts_df.copy()
    if seen_ids:
        candidates = candidates[~candidates["text_id"].isin(seen_ids)]

    interest_set = set(interests)
    candidates = candidates[candidates["broad_topics_list"].apply(
        lambda topics: bool(set(topics) & interest_set)
    )]

    w_topic, w_diff = _get_weights(round_number)

    if len(candidates) == 0:
        return {
            "round_number": round_number,
            "w_topic": w_topic,
            "w_difficulty": w_diff,
            "estimated_level": level_estimator.estimated_level,
            "texts": [],
        }

    scoring = ScoringEngine()
    scored = scoring.score_candidates(
        candidates=candidates,
        student_interests=interests,
        estimated_level=level_estimator.estimated_level,
        round_number=round_number,
    )

    builder = SlateBuilder()
    slate = builder.build_slate(scored, slate_size=2)

    slate_ids = [str(x) for x in slate["text_id"].tolist()]
    bodies_by_id = _fetch_text_bodies(db, slate_ids)

    texts_out = []
    for _, row in slate.iterrows():
        tid = str(row["text_id"])
        texts_out.append({
            "text_id": tid,
            "title": row["title"],
            "broad_topics": row.get("broad_topics_list", []),
            "first_image_url": row.get("first_image_url"),
            "preview_text": _derive_preview_text(
                title=row.get("title"),
                body=bodies_by_id.get(tid) or "",
            ),
            "final_difficulty": float(row["final_difficulty"]),
            "composite_score": float(row["composite_score"]),
            "score_topic": float(row["score_topic"]),
            "score_difficulty": float(row["score_difficulty"]),
        })

    return {
        "round_number": round_number,
        "w_topic": w_topic,
        "w_difficulty": w_diff,
        "estimated_level": level_estimator.estimated_level,
        "texts": texts_out,
    }


def handle_refresh(
    db: DBSession,
    session_id: str,
    interests: List[str],
    shown_text_ids: List[str],
) -> dict:
    round_number = _get_round_number(db, session_id)
    level_estimator = _rebuild_level_estimator(db, session_id)
    w_topic, w_diff = _get_weights(round_number)

    pending = _pending_slate_event(
        db,
        session_id=session_id,
        round_number=round_number,
        shown_text_ids=shown_text_ids,
    )
    if pending is None:
        # Fallback: tolerate ordering differences from the client.
        candidate = _pending_slate_event(
            db,
            session_id=session_id,
            round_number=round_number,
            shown_text_ids=None,
        )
        if candidate is not None:
            payload_set = set(_to_uuid_list(shown_text_ids))
            candidate_set = set(candidate.shown_text_ids or [])
            if payload_set == candidate_set:
                pending = candidate
    if pending is not None:
        pending.was_refresh = True
        pending.w_topic = pending.w_topic if pending.w_topic is not None else w_topic
        pending.w_difficulty = pending.w_difficulty if pending.w_difficulty is not None else w_diff
        pending.estimated_level = (
            pending.estimated_level
            if pending.estimated_level is not None
            else level_estimator.estimated_level
        )
    else:
        refresh_event = SlateEvent(
            session_id=_to_uuid(session_id),
            round_number=round_number,
            shown_text_ids=_to_uuid_list(shown_text_ids),
            chosen_text_id=None,
            was_refresh=True,
            w_topic=w_topic,
            w_difficulty=w_diff,
            estimated_level=level_estimator.estimated_level,
        )
        db.add(refresh_event)

        # log the click even if we couldn't match an existing pending slate
        db.flush()
        db.add(
            SessionEvent(
                session_id=_to_uuid(session_id),
                slate_id=refresh_event.slate_id,
                event_type="dashboard_refresh_clicked",
                event_metadata={"shown_text_ids": shown_text_ids},
            )
        )

    db.commit()

    if pending is not None:
        db.add(
            SessionEvent(
                session_id=_to_uuid(session_id),
                slate_id=pending.slate_id,
                event_type="dashboard_refresh_clicked",
                event_metadata={"shown_text_ids": shown_text_ids},
            )
        )
        db.commit()

    # Compute the next slate (excluding seen ids, incl. this refreshed slate)
    result = get_recommendations(db, session_id, interests)
    next_ids = [t.get("text_id") for t in result.get("texts", [])]
    next_ids = [t for t in next_ids if t]
    if next_ids:
        next_event = SlateEvent(
            session_id=_to_uuid(session_id),
            round_number=result["round_number"],
            shown_text_ids=_to_uuid_list(next_ids),
            chosen_text_id=None,
            was_refresh=False,
            w_topic=result.get("w_topic"),
            w_difficulty=result.get("w_difficulty"),
            estimated_level=result.get("estimated_level"),
        )
        db.add(next_event)
        db.flush()
        db.add(
            SessionEvent(
                session_id=_to_uuid(session_id),
                slate_id=next_event.slate_id,
                event_type="dashboard_slate_shown",
                event_metadata={
                    "round_number": result.get("round_number"),
                    "shown_text_ids": next_ids,
                    "w_topic": result.get("w_topic"),
                    "w_difficulty": result.get("w_difficulty"),
                    "estimated_level": result.get("estimated_level"),
                    "source": "refresh",
                },
            )
        )
        db.commit()

    return result


def record_reading(
    db: DBSession,
    session_id: str,
    shown_text_ids: List[str],
    chosen_text_id: str,
    perceived_difficulty: int,
    interest_rating: int,
    comprehension_score: float,
    comprehension_detail: dict = None,
) -> dict:
    texts_df = get_texts(db)
    round_number = _get_round_number(db, session_id)
    level_estimator = _rebuild_level_estimator(db, session_id)

    text_row = texts_df[texts_df["text_id"] == str(chosen_text_id)]
    if text_row.empty:
        raise ValueError(f"text_id '{chosen_text_id}' not found in corpus.")
    text_difficulty = float(text_row.iloc[0]["final_difficulty"])
    if math.isnan(text_difficulty):
        raise ValueError(
            f"Text '{chosen_text_id}' has no valid difficulty value; cannot update level estimate."
        )

    w_topic, w_diff = _get_weights(round_number)

    pending = _pending_slate_event(
        db,
        session_id=session_id,
        round_number=round_number,
        shown_text_ids=shown_text_ids,
    )
    if pending is None:
        # Fallback: tolerate ordering differences from the client.
        candidate = _pending_slate_event(
            db,
            session_id=session_id,
            round_number=round_number,
            shown_text_ids=None,
        )
        if candidate is not None:
            payload_set = set(_to_uuid_list(shown_text_ids))
            candidate_set = set(candidate.shown_text_ids or [])
            if payload_set == candidate_set and _to_uuid(chosen_text_id) in candidate_set:
                pending = candidate

    # If the slate was already marked as chosen when the reading page opened
    # (see mark_slate_choice), the pending lookup above won't find it because
    # it filters on chosen_text_id IS NULL.
    slate_event = None
    if pending is None:
        already_chosen = (
            db.query(SlateEvent)
            .filter(SlateEvent.session_id == _to_uuid(session_id))
            .filter(SlateEvent.round_number == round_number)
            .filter(SlateEvent.was_refresh == False)
            .filter(SlateEvent.shown_text_ids == _to_uuid_list(shown_text_ids))
            .filter(SlateEvent.chosen_text_id == _to_uuid(chosen_text_id))
            .order_by(SlateEvent.created_at.desc(), SlateEvent.slate_id.desc())
            .first()
        )
        if already_chosen is not None:
            slate_event = already_chosen

    if pending is not None:
        pending.chosen_text_id = _to_uuid(chosen_text_id)
        slate_event = pending
    elif slate_event is None:
        slate_event = SlateEvent(
            session_id=_to_uuid(session_id),
            round_number=round_number,
            shown_text_ids=_to_uuid_list(shown_text_ids),
            chosen_text_id=_to_uuid(chosen_text_id),
            was_refresh=False,
            w_topic=w_topic,
            w_difficulty=w_diff,
            estimated_level=level_estimator.estimated_level,
        )
        db.add(slate_event)
        db.flush()

    implied_level = level_estimator.update(text_difficulty, perceived_difficulty)
    estimated_level = level_estimator.estimated_level

    reading_event = ReadingEvent(
        session_id=session_id,
        slate_id=slate_event.slate_id,
        text_id=chosen_text_id,
        round_number=round_number,
        text_difficulty=text_difficulty,
        perceived_difficulty=perceived_difficulty,
        interest_rating=interest_rating,
        comprehension_score=comprehension_score,
        comprehension_detail=comprehension_detail,
        implied_level=implied_level,
        estimated_level=estimated_level,
    )
    db.add(reading_event)
    db.commit()
    db.refresh(reading_event)

    return {
        "reading_id": reading_event.reading_id,
        "slate_id": slate_event.slate_id,
        "round_number": round_number,
        "implied_level": implied_level,
        "estimated_level": estimated_level,
    }


def get_session_summary(
    db: DBSession,
    session_id: str,
) -> dict:
    interests = get_session_interests(db, session_id)
    seen_ids = _get_seen_text_ids(db, session_id)
    round_number = _get_round_number(db, session_id)
    level_estimator = _rebuild_level_estimator(db, session_id)

    n_readings = (
        db.query(ReadingEvent)
        .filter(ReadingEvent.session_id == session_id)
        .count()
    )
    n_refreshes = (
        db.query(SlateEvent)
        .filter(SlateEvent.session_id == session_id, SlateEvent.was_refresh == True)
        .count()
    )

    return {
        "session_id": session_id,
        "round_number": round_number,
        "interests": interests,
        "n_texts_seen": len(seen_ids),
        "estimated_level": level_estimator.estimated_level,
        "n_readings": n_readings,
        "n_refreshes": n_refreshes,
    }


# ════════════════════════════════════════════════════════════
# Session feedback helpers
# ════════════════════════════════════════════════════════════

def get_session_feedback_texts(db: DBSession, session_id: str) -> list[dict[str, str]]:
    """Return text titles the student interacted with in this session.

    We currently use session_events (dashboard_text_selected) because it exists
    even before we persist full reading responses.
    """

    try:
        session_uuid = _to_uuid(session_id)
    except (TypeError, ValueError):
        return []

    ti_cols = _get_table_columns(db, "text_images")
    ti_order_by = (
        "ti.image_order, ti.file_name" if "image_order" in ti_cols else "ti.file_name"
    )

    query = sql_text(
        f"""
        SELECT
            se.text_id::text AS text_id,
            MAX(t.title)::text AS title,
            (
                SELECT ti.file_name
                FROM text_images ti
                WHERE ti.text_id = se.text_id
                ORDER BY {ti_order_by}
                LIMIT 1
            )::text AS first_image_file_name,
            MIN(se.created_at) AS first_at
        FROM session_events se
        JOIN texts t ON t.id = se.text_id
        WHERE se.session_id = :session_id
          AND se.event_type = 'dashboard_text_selected'
          AND se.text_id IS NOT NULL
        GROUP BY se.text_id
        ORDER BY first_at ASC;
        """
    )

    rows = db.execute(query, {"session_id": session_uuid}).mappings().all()
    out: list[dict[str, str]] = []
    for r in rows:
        tid = r.get("text_id")
        title = r.get("title")
        if not tid or not title:
            continue

        first_image_url = None
        if (safe_name := _to_safe_image_name(r.get("first_image_file_name"))):
            first_image_url = _build_preferred_image_url(safe_name)

        out.append(
            {
                "text_id": str(tid),
                "title": str(title),
                "first_image_url": first_image_url,
            }
        )
    return out


# ════════════════════════════════════════════════════════════
# Persist question answers
# ════════════════════════════════════════════════════════════

def _normalize_question_type(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower().replace("_", "")


def _infer_shown_text_ids_for_reading(
    db: DBSession,
    *,
    session_id: str,
    text_id: str,
) -> list[str]:
    """Best-effort inference of shown_text_ids for the current round.

    The frontend currently doesn't carry shown_text_ids into the reading page.
    We infer it from the latest slate event that has this chosen_text_id.
    """

    try:
        session_uuid = _to_uuid(session_id)
        text_uuid = _to_uuid(text_id)
    except (TypeError, ValueError):
        return [text_id]

    round_number = _get_round_number(db, session_id)
    slate = (
        db.query(SlateEvent)
        .filter(SlateEvent.session_id == session_uuid)
        .filter(SlateEvent.round_number == round_number)
        .filter(SlateEvent.was_refresh == False)
        .filter(SlateEvent.chosen_text_id == text_uuid)
        .order_by(SlateEvent.created_at.desc(), SlateEvent.slate_id.desc())
        .first()
    )

    if slate and slate.shown_text_ids:
        return [str(x) for x in slate.shown_text_ids]

    # Fallback: any slate where the text was shown.
    slate2 = (
        db.query(SlateEvent)
        .filter(SlateEvent.session_id == session_uuid)
        .filter(SlateEvent.round_number == round_number)
        .filter(SlateEvent.shown_text_ids.any(text_uuid))
        .order_by(SlateEvent.created_at.desc(), SlateEvent.slate_id.desc())
        .first()
    )
    if slate2 and slate2.shown_text_ids:
        return [str(x) for x in slate2.shown_text_ids]

    return [text_id]


def _compute_comprehension_from_answers(
    *,
    questions: list[dict[str, Any]],
    answers: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Compute a comprehension score from raw answers.

    Uses `options[].is_correct` when present.
    Global questions (question_id starting with "global:") are excluded.
    """

    total_items = 0
    correct_items = 0
    detail: dict[str, Any] = {}

    for q in questions or []:
        qid = str(q.get("question_id") or "")
        if not qid or qid.startswith("global:"):
            continue

        q_type = _normalize_question_type(q.get("question_type"))
        opts = q.get("options") or []
        if not isinstance(opts, list) or len(opts) == 0:
            # Free text questions are filtered out by get_text_questions, but keep safe.
            continue

        if q_type == "trueorfalse":
            per_opt: dict[str, Any] = {}
            for opt in opts:
                opt_id = str(opt.get("option_id") or "")
                if not opt_id:
                    continue

                is_correct_flag = opt.get("is_correct")
                if is_correct_flag is None:
                    continue

                key = f"{qid}:{opt_id}"
                raw = answers.get(key)
                if raw not in ("true", "false", True, False):
                    per_opt[opt_id] = {"answered": False}
                    continue

                answered_true = raw is True or str(raw).lower() == "true"
                expected_true = bool(is_correct_flag)
                ok = answered_true == expected_true
                per_opt[opt_id] = {
                    "answered": True,
                    "value": "true" if answered_true else "false",
                    "is_correct": ok,
                }
                total_items += 1
                if ok:
                    correct_items += 1

            if per_opt:
                detail[qid] = {"type": "trueorfalse", "statements": per_opt}
            continue

        if q_type in {"checkbox", "checkboxes"}:
            selected = answers.get(qid)
            selected_ids = selected if isinstance(selected, list) else []
            selected_set = {str(x) for x in selected_ids if str(x)}

            correct_option_ids = [
                str(opt.get("option_id"))
                for opt in opts
                if opt.get("is_correct") is True and opt.get("option_id")
            ]
            known_flags = [opt.get("is_correct") for opt in opts]
            can_score = any(v is True for v in known_flags) and all(
                v in (True, False, None) for v in known_flags
            )

            ok = None
            if can_score and correct_option_ids:
                ok = selected_set == set(correct_option_ids)
                total_items += 1
                if ok:
                    correct_items += 1

            detail[qid] = {
                "type": "checkboxes",
                "selected": sorted(selected_set),
                "is_correct": ok,
            }
            continue

        # Default: single choice
        selected = answers.get(qid)
        selected_id = str(selected) if isinstance(selected, str) else ""
        chosen_opt = next((o for o in opts if str(o.get("option_id")) == selected_id), None)

        ok = None
        if chosen_opt is not None and chosen_opt.get("is_correct") is not None:
            ok = bool(chosen_opt.get("is_correct"))
            total_items += 1
            if ok:
                correct_items += 1

        detail[qid] = {
            "type": "single_choice",
            "selected": selected_id or None,
            "is_correct": ok,
        }

    score = float(correct_items / total_items) if total_items > 0 else 0.0
    return score, detail


def submit_reading_answers(
    db: DBSession,
    *,
    session_id: str,
    text_id: str,
    answers: dict[str, Any],
) -> dict[str, Any]:
    """Persist per-question answers for a reading and record the reading summary.

    - Extracts global post-reading answers from the raw answer dict.
    - Computes comprehension_score from correct flags.
    - Calls record_reading() to persist the main reading_event row.
    - Persists all raw answers into reading_question_responses.
    """

    questions = get_text_questions(db, text_id)
    qtype_by_id = {
        str(q.get("question_id")): (q.get("question_type") or None)
        for q in (questions or [])
        if q.get("question_id")
    }

    perceived_raw = answers.get("global:perceived_difficulty")
    interest_raw = answers.get("global:interest_rating")

    if perceived_raw is None or interest_raw is None:
        raise ValueError("Missing required global answers (difficulty/interest).")

    try:
        perceived_difficulty = int(perceived_raw)
        interest_rating = int(interest_raw)
    except (TypeError, ValueError):
        raise ValueError("Invalid global answers (difficulty/interest).")

    comprehension_score, comprehension_detail = _compute_comprehension_from_answers(
        questions=questions,
        answers=answers or {},
    )

    shown_text_ids = _infer_shown_text_ids_for_reading(
        db,
        session_id=session_id,
        text_id=text_id,
    )

    result = record_reading(
        db=db,
        session_id=session_id,
        shown_text_ids=shown_text_ids,
        chosen_text_id=text_id,
        perceived_difficulty=perceived_difficulty,
        interest_rating=interest_rating,
        comprehension_score=comprehension_score,
        comprehension_detail=comprehension_detail,
    )

    reading = (
        db.query(ReadingEvent)
        .filter(ReadingEvent.reading_id == result["reading_id"])
        .first()
    )
    if reading is None:
        return result

    # Store all raw answers (including global and TF statement keys) as JSON.
    try:
        session_uuid = _to_uuid(session_id)
        text_uuid = _to_uuid(text_id)
    except (TypeError, ValueError):
        return result

    rows: list[ReadingQuestionResponse] = []
    for key, value in (answers or {}).items():
        if key is None:
            continue
        qid = str(key)
        if not qid:
            continue

        q_type = qtype_by_id.get(qid)
        if q_type is None and ":" in qid:
            parent = qid.split(":", 1)[0]
            if _normalize_question_type(qtype_by_id.get(parent)) == "trueorfalse":
                q_type = "trueorfalse_statement"

        rows.append(
            ReadingQuestionResponse(
                reading_id=int(reading.reading_id),
                session_id=session_uuid,
                text_id=text_uuid,
                question_id=qid,
                question_type=q_type,
                answer=value,
            )
        )

    if rows:
        db.add_all(rows)
        db.commit()

    return result


def submit_session_feedback(
    db: DBSession,
    *,
    session_id: str,
    q1: int,
    q2: str,
    favorite_text_id: str | None = None,
    favorite_why: str | None = None,
) -> None:
    """Upsert end-of-session feedback for a session."""

    session_uuid = _to_uuid(session_id)

    fav_uuid = None
    if favorite_text_id:
        try:
            fav_uuid = _to_uuid(favorite_text_id)
        except (TypeError, ValueError):
            fav_uuid = None

    existing = (
        db.query(SessionFeedback)
        .filter(SessionFeedback.session_id == session_uuid)
        .first()
    )

    if existing is None:
        existing = SessionFeedback(session_id=session_uuid)
        db.add(existing)

    existing.q1_system_wanted_texts = int(q1) if q1 is not None else None
    existing.q2_level_fit = str(q2) if q2 is not None else None
    existing.favorite_text_id = fav_uuid
    existing.favorite_why = favorite_why

    db.commit()