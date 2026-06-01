# Personalized Recommender System for Student Reading Tasks

This repository contains the final application and the archived research work for the MSc thesis project:
Developing a personalized recommender system for student reading tasks
as part of the Human Reading Assessment (HRA) project.

---

## Repository Layout

- `frontend/`: production React application.
- `backend/`: production FastAPI API, database models, and deployment Dockerfile.
- `data/`: source datasets and static assets used by the application.
- `research/`: archived experiments, notebooks, checkpoints, and earlier recommender approaches.

The production system now lives in `frontend/` and `backend/`.
The `research/` directory is kept for thesis traceability and is not part of the runtime backend.

## Runtime Surface

- The backend entrypoint is `backend/app/main.py`.
- The backend recommender logic used at runtime is `backend/app/recommender_engine.py`.
- The frontend routes live under `frontend/src/`.

## Research Archive

Archived recommender experiments are kept under `research/recommender-experiments/`.
This includes:

- notebooks and exploratory scripts
- topic and difficulty pipelines
- intermediate checkpoints and evaluation outputs
- the legacy Flask prototype and earlier recommender package

## Timeline

The project timeline is documented here:  
[View Timeline](https://docs.google.com/document/d/1mo0pmYI1wCPc2Bba-3mc47PL-45-e1B2jN9Hs8HqUIg/)
