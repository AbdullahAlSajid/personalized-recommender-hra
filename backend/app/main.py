from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from app.db import Base, engine
from app.api.sessions import router as sessions_router
from app.api.router import router as recommendation_router


app = FastAPI(title="HRA Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://44.192.27.80:3001",
        "http://localhost",
        "http://44.192.27.80",
        "http://localhost:3000",
        "http://127.0.0.1:3000",

    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

app.include_router(sessions_router, prefix="/sessions", tags=["sessions"])
app.include_router(recommendation_router, prefix="/session", tags=["recommendations"])

images_dir = Path(__file__).resolve().parents[2] / "data" / "images"
if images_dir.exists():
    app.mount("/images", StaticFiles(directory=str(images_dir)), name="images")


@app.get("/health")
def health():
    return {"status": "ok"}