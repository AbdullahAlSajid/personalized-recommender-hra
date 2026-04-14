from sqlalchemy import Column, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from app.db import Base


class StudentSession(Base):
    __tablename__ = "student_sessions"

    session_id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    started_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    ended_at = Column(DateTime(timezone=True), nullable=True)
