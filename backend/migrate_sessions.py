"""
Migration: slim student_sessions to (session_id, started_at, ended_at)

Run once:
    .venv/Scripts/python migrate_sessions.py
"""
from app.db import engine
from sqlalchemy import text

DROP_COLUMNS = [
    "pilot_run_id",
    "school_id",
    "access_code_id",
    "session_token",
    "status",
    "device_type",
    "browser_info",
    "consent_given",
    "total_texts_completed",
    "final_feedback_submitted",
]

with engine.begin() as conn:
    for col in DROP_COLUMNS:
        try:
            conn.execute(text(f"ALTER TABLE student_sessions DROP COLUMN IF EXISTS {col}"))
            print(f"  dropped: {col}")
        except Exception as e:
            print(f"  SKIP {col}: {e}")

    # Verify result
    result = conn.execute(text(
        "SELECT column_name, data_type, is_nullable "
        "FROM information_schema.columns "
        "WHERE table_name = 'student_sessions' "
        "ORDER BY ordinal_position"
    ))
    print("\nFinal student_sessions schema:")
    for row in result:
        print(" ", row)
