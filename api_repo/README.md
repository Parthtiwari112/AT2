# AT2 API Repository (placeholder)

This repository contains the FastAPI application for serving the trained models for the AT2 assignment.

Structure:
- app/main.py : FastAPI application (endpoints per brief)
- models/ : place trained joblib models here (rain_class_baseline.joblib and precip_reg_baseline.joblib)
- requirements.txt, Dockerfile

Quick start (local):
1. Create virtualenv and install:
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2. Run the API:
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Endpoints:
- GET / -> project overview
- GET /health/ -> health check
- GET /predict/rain/?date=YYYY-MM-DD -> classification prediction for date+7
- GET /predict/precipitation/fall?date=YYYY-MM-DD -> precipitation prediction (next 3 days)

Deployment:
- Build Docker image:
  docker build -t at2-api .
- Run:
  docker run -p 8000:8000 at2-api

Notes:
- Place trained models into models/ before deploying for real inference.
- This template returns deterministic placeholders if models are not present.
