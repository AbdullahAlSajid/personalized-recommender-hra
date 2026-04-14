from pydantic import BaseModel, Field
from datetime import datetime


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
