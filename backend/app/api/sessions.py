import os
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import StudentSession
from app.schemas import (
    ValidatePasscodeRequest,
    ValidatePasscodeResponse,
    CreateSessionResponse,
    EndSessionResponse,
    SessionStatusResponse,
)

router = APIRouter()

COMMON_PASSCODE = os.getenv("COMMON_PASSCODE", "123456")


@router.get("/status", response_model=SessionStatusResponse)
def session_status(
    x_session_id: str | None = Header(default=None),
    db: Session = Depends(get_db),
):
    """
    Return whether the X-Session-Id header maps to an active (not-yet-ended)
    session. No DB write. Safe to call on every app load.
    """
    if not x_session_id:
        return SessionStatusResponse(active=False)

    session = (
        db.query(StudentSession)
        .filter(StudentSession.session_id == x_session_id)
        .first()
    )

    if not session or session.ended_at is not None:
        return SessionStatusResponse(active=False)

    return SessionStatusResponse(active=True)


@router.post("/validate", response_model=ValidatePasscodeResponse)
def validate_passcode(payload: ValidatePasscodeRequest):
    """Check the access code. No DB write — consent page is shown next."""
    if payload.passcode != COMMON_PASSCODE:
        raise HTTPException(status_code=401, detail="Ugyldig kode. Prøv igjen.")
    return ValidatePasscodeResponse(valid=True)


@router.post("/start", response_model=CreateSessionResponse)
def start_session(
    x_session_id: str | None = Header(default=None),
    db: Session = Depends(get_db),
):
    """
    Create a session row. Idempotent: if the request already carries a valid
    active session ID, return the existing session. Only call after consent.
    """
    if x_session_id:
        existing = (
            db.query(StudentSession)
            .filter(StudentSession.session_id == x_session_id)
            .first()
        )
        if existing and not existing.ended_at:
            return CreateSessionResponse(
                session_id=str(existing.session_id),
                started_at=existing.started_at,
            )

    session = StudentSession()
    db.add(session)
    db.commit()
    db.refresh(session)
    return CreateSessionResponse(
        session_id=str(session.session_id),
        started_at=session.started_at,
    )


@router.post("/end", response_model=EndSessionResponse)
def end_session(
    x_session_id: str | None = Header(default=None),
    db: Session = Depends(get_db),
):
    """
    Mark the session as ended. Idempotent: if already ended, returns 200.
    """
    if not x_session_id:
        raise HTTPException(status_code=400, detail="X-Session-Id header missing.")

    session = (
        db.query(StudentSession)
        .filter(StudentSession.session_id == x_session_id)
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    if not session.ended_at:
        session.ended_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(session)

    return EndSessionResponse(
        session_id=str(session.session_id),
        ended_at=session.ended_at,
    )
