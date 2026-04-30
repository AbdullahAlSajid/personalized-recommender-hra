import uuid
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime, timezone
from sqlalchemy import (
    BigInteger, Boolean, Column, DateTime, Float, ForeignKey, Integer, String, func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import relationship

from app.db import Base


class StudentSession(Base):
    __tablename__ = "student_sessions"

    session_id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    started_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    consent_given = Column(Boolean, nullable=False, server_default="false")
    ended_at = Column(DateTime(timezone=True), nullable=True)


class Text(Base):
    __tablename__ = "texts"

    id = Column(UUID(as_uuid=True), primary_key=True)

class BroadTopic(Base):
    __tablename__ = "broad_topics"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, nullable=False)


class SessionInterest(Base):
    __tablename__ = "session_interests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("student_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
    )
    broad_topic_id = Column(
        BigInteger,
        ForeignKey("broad_topics.id", ondelete="CASCADE"),
        nullable=False,
    )

    session = relationship("StudentSession", backref="session_interests")
    broad_topic = relationship("BroadTopic")

"""
SQLAlchemy models for the recommender tables.

Requires: slate_events and reading_events tables
  (run: python -m data.database migrate)
"""

class SlateEvent(Base):
    __tablename__ = "slate_events"

    slate_id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("student_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
    )
    round_number = Column(Integer, nullable=False)
    shown_text_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=False)
    chosen_text_id = Column(UUID(as_uuid=True), nullable=True)  # NULL if refresh
    was_refresh = Column(Boolean, default=False)
    w_topic = Column(Float, nullable=True)
    w_difficulty = Column(Float, nullable=True)
    estimated_level = Column(Float, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    session = relationship("StudentSession", backref="slate_events")


class ReadingEvent(Base):
    __tablename__ = "reading_events"

    reading_id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("student_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
    )
    slate_id = Column(
        Integer,
        ForeignKey("slate_events.slate_id"),
        nullable=True,
    )
    text_id = Column(
        UUID(as_uuid=True),
        ForeignKey("texts.id", ondelete="CASCADE"),
        nullable=False,
    )
    round_number = Column(Integer, nullable=False)
    text_difficulty = Column(Float, nullable=False)
    perceived_difficulty = Column(Integer, nullable=False)  # 1-5
    interest_rating = Column(Integer, nullable=False)  # 1-5
    comprehension_score = Column(Float, nullable=False)  # 0.0-1.0
    comprehension_detail = Column(JSONB, nullable=True)
    implied_level = Column(Float, nullable=True)
    estimated_level = Column(Float, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    session = relationship("StudentSession", backref="reading_events")
    slate = relationship("SlateEvent", backref="readings")


class SessionEvent(Base):
    __tablename__ = "session_events"

    event_id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("student_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
    )
    slate_id = Column(
        Integer,
        ForeignKey("slate_events.slate_id"),
        nullable=True,
    )
    text_id = Column(
        UUID(as_uuid=True),
        ForeignKey("texts.id", ondelete="CASCADE"),
        nullable=True,
    )
    event_type = Column(String, nullable=False)
    event_metadata = Column("metadata", JSONB, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    session = relationship("StudentSession", backref="session_events")
    slate = relationship("SlateEvent")


class ReadingQuestionResponse(Base):
    __tablename__ = "reading_question_responses"

    response_id = Column(BigInteger, primary_key=True, autoincrement=True)
    reading_id = Column(
        Integer,
        ForeignKey("reading_events.reading_id", ondelete="CASCADE"),
        nullable=False,
    )
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("student_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
    )
    text_id = Column(
        UUID(as_uuid=True),
        ForeignKey("texts.id", ondelete="CASCADE"),
        nullable=False,
    )

    # For text-specific questions: question_id is typically the DB id from the `questions` table.
    # For global questions (not stored in DB): question_id uses stable keys like `global:interest_rating`.
    # For True/False statements: question_id can be `question_id:option_id` (matches frontend keys).
    question_id = Column(String, nullable=False)
    question_type = Column(String, nullable=True)
    answer = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    reading = relationship("ReadingEvent", backref="question_responses")
    session = relationship("StudentSession")


class SessionFeedback(Base):
    __tablename__ = "session_feedback"

    # One feedback row per session.
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("student_sessions.session_id", ondelete="CASCADE"),
        primary_key=True,
    )

    q1_system_wanted_texts = Column(Integer, nullable=True)  # 1-5
    q2_level_fit = Column(String, nullable=True)
    favorite_text_id = Column(UUID(as_uuid=True), ForeignKey("texts.id"), nullable=True)
    favorite_why = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    session = relationship("StudentSession", backref="session_feedback")
