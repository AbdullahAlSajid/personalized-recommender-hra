"""
Migration: add consent_given back to student_sessions.

Run once:
    .venv/Scripts/python migrate_add_session_consent.py
"""
from app.db import engine
from sqlalchemy import text


with engine.begin() as conn:
    conn.execute(
        text(
            "ALTER TABLE student_sessions "
            "ADD COLUMN IF NOT EXISTS consent_given BOOLEAN NOT NULL DEFAULT FALSE"
        )
    )

    result = conn.execute(text(
        "SELECT column_name, data_type, is_nullable, column_default "
        "FROM information_schema.columns "
        "WHERE table_name = 'student_sessions' "
        "ORDER BY ordinal_position"
    ))
    print("\nUpdated student_sessions schema:")
    for row in result:
        print(" ", row)