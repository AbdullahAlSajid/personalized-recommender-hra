from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any, Dict, List, Optional


class ValidatePasscodeRequest(BaseModel):
    passcode: str = Field(..., min_length=1)


class ValidatePasscodeResponse(BaseModel):
    token: str  # one-time token to be sent to /start


class CreateSessionResponse(BaseModel):
    session_id: str
    started_at: datetime


class EndSessionResponse(BaseModel):
    session_id: str
    ended_at: datetime


class SessionStatusResponse(BaseModel):
    active: bool


class StartSessionRequest(BaseModel):
    token: str = Field(..., min_length=1)
    consent_given: bool


"""
Pydantic schemas for recommendation endpoints.
Add these to your existing app/schemas.py or import from here.
"""

# ── Interests ─────────────────────────────────────────────

class SelectInterestsRequest(BaseModel):
    interests: List[str] = Field(
        ..., min_length=3,
        description="At least 3 broad topics from the 13 available",
    )


class SelectInterestsResponse(BaseModel):
    session_id: str
    interests: List[str]


# ── Topics (for frontend to display) ─────────────────────

class BroadTopicItem(BaseModel):
    id: int
    name: str


class BroadTopicsResponse(BaseModel):
    topics: List[BroadTopicItem]


# ── Recommendations ──────────────────────────────────────

class RecommendedText(BaseModel):
    text_id: str
    title: str
    broad_topics: List[str]
    first_image_url: Optional[str] = None
    preview_text: Optional[str] = None
    final_difficulty: float
    composite_score: float
    score_topic: float
    score_difficulty: float


class RecommendationsResponse(BaseModel):
    round_number: int
    w_topic: float
    w_difficulty: float
    estimated_level: Optional[float]
    texts: List[RecommendedText]


# ── Refresh ──────────────────────────────────────────────

class RefreshRequest(BaseModel):
    shown_text_ids: List[str] = Field(
        ..., min_length=1, max_length=2,
        description="The text_ids from the current slate",
    )


# RefreshResponse reuses RecommendationsResponse


# ── Record reading ───────────────────────────────────────

class RecordReadingRequest(BaseModel):
    shown_text_ids: List[str] = Field(
        ..., min_length=1, max_length=2,
        description="The text_ids that were in the slate",
    )
    chosen_text_id: str = Field(
        ..., description="The text_id the student chose to read",
    )
    perceived_difficulty: int = Field(
        ..., ge=1, le=5,
        description="How hard did you find it? 1=veldig lett, 5=veldig vanskelig",
    )
    interest_rating: int = Field(
        ..., ge=1, le=5,
        description="How interesting was it? 1=kjedelig, 5=veldig interessant",
    )
    comprehension_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="MCQ/TF score as proportion correct (e.g. 2/3 = 0.67)",
    )
    comprehension_detail: Optional[dict] = Field(
        default=None,
        description='Per-question results, e.g. {"q1": true, "q2": false, "q3": true}',
    )


class RecordReadingResponse(BaseModel):
    reading_id: int
    slate_id: int
    round_number: int
    implied_level: float
    estimated_level: float


# ── Session summary (for debugging / admin) ──────────────

class SessionSummaryResponse(BaseModel):
    session_id: str
    round_number: int
    interests: List[str]
    n_texts_seen: int
    estimated_level: Optional[float]
    n_readings: int
    n_refreshes: int


# ── Text questions (quiz) ────────────────────────────────

class QuestionOptionItem(BaseModel):
    option_id: str
    body: str
    sanity_answer_key: Optional[str] = None
    is_correct: Optional[bool] = None
    display_order: Optional[int] = None


class TextQuestionItem(BaseModel):
    question_id: str
    body: str
    question_type: str
    display_order: Optional[int] = None
    options: List[QuestionOptionItem] = []


class TextQuestionsResponse(BaseModel):
    text_id: str
    questions: List[TextQuestionItem]


# ── Session events (analytics) ───────────────────────────

class LogSessionEventRequest(BaseModel):
    event_type: str = Field(..., min_length=1, max_length=64)
    slate_id: Optional[int] = None
    text_id: Optional[str] = None
    metadata: Optional[dict] = None


class LogSessionEventResponse(BaseModel):
    ok: bool = True


# ── Session feedback (end-of-session) ─────────────────────

class SessionFeedbackTextItem(BaseModel):
    text_id: str
    title: str
    first_image_url: Optional[str] = None


class SessionFeedbackTextsResponse(BaseModel):
    texts: List[SessionFeedbackTextItem]


# ── Persist reading answers (per-question) ───────────────

class SubmitReadingAnswersRequest(BaseModel):
    text_id: str = Field(..., min_length=1)
    answers: Dict[str, Any] = Field(default_factory=dict)


class SubmitReadingAnswersResponse(RecordReadingResponse):
    pass


# ── Persist session feedback ─────────────────────────────

class SubmitSessionFeedbackRequest(BaseModel):
    q1: int = Field(..., ge=1, le=5)
    q2: str = Field(..., min_length=1, max_length=64)
    favorite_text_id: Optional[str] = None
    favorite_why: Optional[str] = None


class SubmitSessionFeedbackResponse(BaseModel):
    ok: bool = True
