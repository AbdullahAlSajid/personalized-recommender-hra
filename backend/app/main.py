from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db import Base, engine
from app.api.sessions import router as sessions_router

app = FastAPI(title="HRA Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_methods=["*"],
    allow_headers=["*", "X-Session-Id"],
)

Base.metadata.create_all(bind=engine)

app.include_router(sessions_router, prefix="/sessions", tags=["sessions"])


@app.get("/health")
def health():
    return {"status": "ok"}