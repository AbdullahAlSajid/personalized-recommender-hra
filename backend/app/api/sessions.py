import os
import uuid
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Cookie, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import StudentSession
from app.schemas import (
    ValidatePasscodeRequest,
    ValidatePasscodeResponse,
    StartSessionRequest,
    CreateSessionResponse,
    EndSessionResponse,
    SessionStatusResponse,
)

router = APIRouter()

COMMON_PASSCODE = os.getenv("COMMON_PASSCODE", "123456")
COOKIE_NAME = "recsys_session_id"
COOKIE_MAX_AGE = 60 * 60  # 1 hour
TOKEN_TTL = timedelta(hours=1)

# In-memory one-time token store: { token_str: expiry_datetime }
# Tokens are deleted immediately on first use.
_consent_tokens: dict[str, datetime] = {}


def _issue_token() -> str:
    token = str(uuid.uuid4())
    _consent_tokens[token] = datetime.now(timezone.utc) + TOKEN_TTL
    return token


def _consume_token(token: str) -> bool:
    """Returns True and deletes the token if it is valid and unexpired."""
    expiry = _consent_tokens.get(token)
    if not expiry:
        return False
    # Always delete — one-time use regardless of expiry
    del _consent_tokens[token]
    if datetime.now(timezone.utc) > expiry:
        return False
    return True


def _set_session_cookie(response: Response, session_id: str) -> None:
    response.set_cookie(
        key=COOKIE_NAME,
        value=session_id,
        httponly=True,
        samesite="lax",
        secure=False,   # Set True in production (HTTPS)
        max_age=COOKIE_MAX_AGE,
        path="/",
    )


def _clear_session_cookie(response: Response) -> None:
    response.delete_cookie(key=COOKIE_NAME, path="/", samesite="lax", httponly=True)


@router.get("/status", response_model=SessionStatusResponse)
def session_status(
    recsys_session_id: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
):
    """
    Return whether the browser's session cookie maps to an active session.
    JS never receives the session_id — only a boolean.
    No DB write. Safe to call on every app load.
    """
    if not recsys_session_id:
        return SessionStatusResponse(active=False)

    session = (
        db.query(StudentSession)
        .filter(StudentSession.session_id == recsys_session_id)
        .first()
    )

    if not session or session.ended_at is not None:
        return SessionStatusResponse(active=False)

    return SessionStatusResponse(active=True)


@router.get("/authorize")
def authorize_session(
    recsys_session_id: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
):
    """Return 204 only when the session cookie maps to an active session.

    This is intentionally tiny and status-code based so it can be used from
    Nginx `auth_request` for protecting static assets.
    """
    if not recsys_session_id:
        raise HTTPException(status_code=401, detail="No active session.")

    session = (
        db.query(StudentSession)
        .filter(StudentSession.session_id == recsys_session_id)
        .first()
    )

    if not session or session.ended_at is not None:
        raise HTTPException(status_code=401, detail="No active session.")

    return Response(status_code=204)


@router.post("/validate", response_model=ValidatePasscodeResponse)
def validate_passcode(payload: ValidatePasscodeRequest):
    """
    Check the access code. No DB write, no cookie.
    Returns a one-time token (5-min TTL) to be sent to /start after consent.
    The passcode never needs to leave the client again after this call.
    """
    if payload.passcode != COMMON_PASSCODE:
        raise HTTPException(status_code=401, detail="Ugyldig kode. Prøv igjen.")
    return ValidatePasscodeResponse(token=_issue_token())


@router.post("/start", response_model=CreateSessionResponse)
def start_session(
    payload: StartSessionRequest,
    response: Response,
    recsys_session_id: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
):
    """
    Consume the one-time token, then create a session row and set the cookie.
    Token must be valid and unexpired — rejects without it (no DB write).
    Idempotent: if the cookie already maps to an active session, the token is
    still consumed and the existing session is returned.
    """
    if not _consume_token(payload.token):
        raise HTTPException(status_code=401, detail="Ugyldig eller utløpt token.")

    if recsys_session_id:
        existing = (
            db.query(StudentSession)
            .filter(StudentSession.session_id == recsys_session_id)
            .first()
        )
        if existing and not existing.ended_at:
            _set_session_cookie(response, str(existing.session_id))
            return CreateSessionResponse(
                session_id=str(existing.session_id),
                started_at=existing.started_at,
            )

    session = StudentSession()
    db.add(session)
    db.commit()
    db.refresh(session)
    _set_session_cookie(response, str(session.session_id))
    return CreateSessionResponse(
        session_id=str(session.session_id),
        started_at=session.started_at,
    )


@router.post("/end", response_model=EndSessionResponse)
def end_session(
    response: Response,
    recsys_session_id: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
):
    """
    Mark the session as ended and clear the cookie.
    Idempotent: if already ended, clears cookie and returns 200.
    """
    if not recsys_session_id:
        raise HTTPException(status_code=400, detail="No active session cookie.")

    session = (
        db.query(StudentSession)
        .filter(StudentSession.session_id == recsys_session_id)
        .first()
    )
    if not session:
        _clear_session_cookie(response)
        raise HTTPException(status_code=404, detail="Session not found.")

    if not session.ended_at:
        session.ended_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(session)

    _clear_session_cookie(response)
    return EndSessionResponse(
        session_id=str(session.session_id),
        ended_at=session.ended_at,
    )
