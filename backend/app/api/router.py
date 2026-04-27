"""
Recommendation Router
======================
Endpoints between /session/start and /session/end.
All use the session cookie for context.
"""

from fastapi import APIRouter, Cookie, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import StudentSession, BroadTopic
from app.schemas import (
    SelectInterestsRequest,
    SelectInterestsResponse,
    BroadTopicItem,
    BroadTopicsResponse,
    RecommendedText,
    RecommendationsResponse,
    RefreshRequest,
    RecordReadingRequest,
    RecordReadingResponse,
    SessionSummaryResponse,
    TextQuestionsResponse,
    LogSessionEventRequest,
    LogSessionEventResponse,
    SessionFeedbackTextsResponse,
    SubmitReadingAnswersRequest,
    SubmitReadingAnswersResponse,
    SubmitSessionFeedbackRequest,
    SubmitSessionFeedbackResponse,
)
from app import service

router = APIRouter()

COOKIE_NAME = "recsys_session_id"


# ── Dependency: get active session ────────────────────────

def _get_active_session(
    recsys_session_id: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
) -> StudentSession:
    if not recsys_session_id:
        raise HTTPException(status_code=401, detail="No active session.")

    session = (
        db.query(StudentSession)
        .filter(StudentSession.session_id == recsys_session_id)
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    if session.ended_at is not None:
        raise HTTPException(status_code=400, detail="Session already ended.")

    return session


# ── GET /topics ───────────────────────────────────────────

@router.get("/topics", response_model=BroadTopicsResponse)
def get_topics(db: Session = Depends(get_db)):
    """Return all broad topics for the interest picker UI."""
    topics = db.query(BroadTopic).order_by(BroadTopic.name).all()
    return BroadTopicsResponse(
        topics=[BroadTopicItem(id=t.id, name=t.name) for t in topics]
    )


# ── POST /interests ──────────────────────────────────────

@router.post("/interests", response_model=SelectInterestsResponse)
def select_interests(
    payload: SelectInterestsRequest,
    db: Session = Depends(get_db),
    session: StudentSession = Depends(_get_active_session),
):
    """Set the student's interests for this session."""
    valid_names = {t.name for t in db.query(BroadTopic).all()}
    invalid = [t for t in payload.interests if t not in valid_names]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown topics: {invalid}. Valid: {sorted(valid_names)}",
        )

    if len(payload.interests) < 3:
        raise HTTPException(status_code=400, detail="At least 3 interests required.")

    saved = service.set_session_interests(
        db, str(session.session_id), payload.interests
    )

    service.load_texts(db)

    return SelectInterestsResponse(
        session_id=str(session.session_id),
        interests=saved,
    )


# ── GET /recommendations ─────────────────────────────────

@router.get("/recommendations", response_model=RecommendationsResponse)
def get_recommendations(
    db: Session = Depends(get_db),
    session: StudentSession = Depends(_get_active_session),
):
    """Get a slate of 2 recommended texts."""
    interests = service.get_session_interests(db, str(session.session_id))
    if not interests:
        raise HTTPException(status_code=400, detail="Interests not set")

    result = service.get_current_recommendations(
        db=db,
        session_id=str(session.session_id),
        interests=interests,
    )

    return RecommendationsResponse(
        round_number=result["round_number"],
        w_topic=result["w_topic"],
        w_difficulty=result["w_difficulty"],
        estimated_level=result["estimated_level"],
        texts=[RecommendedText(**t) for t in result["texts"]],
    )


# ── POST /refresh ────────────────────────────────────────

@router.post("/refresh", response_model=RecommendationsResponse)
def refresh_slate(
    payload: RefreshRequest,
    db: Session = Depends(get_db),
    session: StudentSession = Depends(_get_active_session),
):
    """Reject current slate, get 2 new texts."""
    interests = service.get_session_interests(db, str(session.session_id))
    if not interests:
        raise HTTPException(status_code=400, detail="Interests not set.")

    try:
        result = service.handle_refresh(
            db=db,
            session_id=str(session.session_id),
            interests=interests,
            shown_text_ids=payload.shown_text_ids,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return RecommendationsResponse(
        round_number=result["round_number"],
        w_topic=result["w_topic"],
        w_difficulty=result["w_difficulty"],
        estimated_level=result["estimated_level"],
        texts=[RecommendedText(**t) for t in result["texts"]],
    )


# ── POST /reading ────────────────────────────────────────

@router.post("/reading", response_model=RecordReadingResponse)
def record_reading(
    payload: RecordReadingRequest,
    db: Session = Depends(get_db),
    session: StudentSession = Depends(_get_active_session),
):
    """Record completed reading with feedback."""
    interests = service.get_session_interests(db, str(session.session_id))
    if not interests:
        raise HTTPException(status_code=400, detail="Interests not set.")

    try:
        result = service.record_reading(
            db=db,
            session_id=str(session.session_id),
            shown_text_ids=payload.shown_text_ids,
            chosen_text_id=payload.chosen_text_id,
            perceived_difficulty=payload.perceived_difficulty,
            interest_rating=payload.interest_rating,
            comprehension_score=payload.comprehension_score,
            comprehension_detail=payload.comprehension_detail,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return RecordReadingResponse(**result)


# ── GET /summary ─────────────────────────────────────────

@router.get("/summary", response_model=SessionSummaryResponse)
def session_summary(
    db: Session = Depends(get_db),
    session: StudentSession = Depends(_get_active_session),
):
    """Session summary for debugging."""
    result = service.get_session_summary(
        db=db,
        session_id=str(session.session_id),
    )
    return SessionSummaryResponse(**result)


# ── GET /text/{text_id} ──────────────────────────────────
@router.get("/text/{text_id}")
def get_text_by_id(
    text_id: str,
    db: Session = Depends(get_db),
    session: StudentSession = Depends(_get_active_session),
):
    # Treat opening the reading page as selecting a text from the shown slate.
    # This updates slate_events.chosen_text_id even before the user submits
    # the full reading feedback.
    _ = service.mark_slate_choice(db, str(session.session_id), text_id)
    detail = service.get_text_detail(db, text_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Text not found")
    return detail


# ── GET /text/{text_id}/questions ───────────────────────

@router.get("/text/{text_id}/questions", response_model=TextQuestionsResponse)
def get_text_questions(
    text_id: str,
    db: Session = Depends(get_db),
    session: StudentSession = Depends(_get_active_session),
):
    _ = session  # force session validation (cookie-based)
    questions = service.get_text_questions(db, text_id)
    return {"text_id": text_id, "questions": questions}


# ── POST /events ───────────────────────────────────────

@router.post("/events", response_model=LogSessionEventResponse)
def log_session_event(
    payload: LogSessionEventRequest,
    db: Session = Depends(get_db),
    session: StudentSession = Depends(_get_active_session),
):
    service.log_session_event(
        db=db,
        session_id=str(session.session_id),
        event_type=payload.event_type,
        slate_id=payload.slate_id,
        text_id=payload.text_id,
        metadata=payload.metadata,
    )
    return LogSessionEventResponse(ok=True)


# ── GET /feedback/texts ─────────────────────────────────

@router.get("/feedback/texts", response_model=SessionFeedbackTextsResponse)
def get_session_feedback_texts(
    db: Session = Depends(get_db),
    session: StudentSession = Depends(_get_active_session),
):
    texts = service.get_session_feedback_texts(db, str(session.session_id))
    return {"texts": texts}


# ── POST /reading/submit ─────────────────────────────────

@router.post("/reading/submit", response_model=SubmitReadingAnswersResponse)
def submit_reading_answers(
    payload: SubmitReadingAnswersRequest,
    db: Session = Depends(get_db),
    session: StudentSession = Depends(_get_active_session),
):
    try:
        result = service.submit_reading_answers(
            db=db,
            session_id=str(session.session_id),
            text_id=payload.text_id,
            answers=payload.answers or {},
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return SubmitReadingAnswersResponse(**result)


# ── POST /feedback ───────────────────────────────────────

@router.post("/feedback", response_model=SubmitSessionFeedbackResponse)
def submit_session_feedback(
    payload: SubmitSessionFeedbackRequest,
    db: Session = Depends(get_db),
    session: StudentSession = Depends(_get_active_session),
):
    try:
        service.submit_session_feedback(
            db=db,
            session_id=str(session.session_id),
            q1=payload.q1,
            q2=payload.q2,
            favorite_text_id=payload.favorite_text_id,
            favorite_why=payload.favorite_why,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return SubmitSessionFeedbackResponse(ok=True)