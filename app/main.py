# app/main.py
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import joblib, os, datetime, requests

app = FastAPI(title="Open Meteo ML - Demo API", version="1.0")

# Models directory relative to this file
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_model(name: str):
    """Load a trained model if it exists in models/"""
    path = os.path.join(MODELS_DIR, name)
    if os.path.exists(path):
        return joblib.load(path)
    return None


# Attempt to load pre-trained models (if present)
rain_model = load_model("rain_model.pkl")
precip_model = load_model("precipitation_model.pkl")


@app.get("/")
def root():
    return {
        "project": "Open Meteo ML - Rain (+7 days) & Precipitation (next 3 days)",
        "endpoints": [
            "/health/",
            "/predict/rain/",
            "/predict/precipitation/fall/",
        ],
        "note": "Place trained models into app/models/. If absent, API falls back to Open-Meteo forecast.",
    }


@app.get("/health/")
def health():
    return {"status": "ok", "message": "API is healthy."}


def fetch_forecast_precip(lat: float, lon: float, start_date: str, end_date: str):
    """Call Open-Meteo API to get precipitation and rain forecast."""
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&daily=precipitation_sum,rain_sum"
        f"&start_date={start_date}&end_date={end_date}&timezone=UTC"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        days = data.get("daily", {}).get("time", [])
        precip = data.get("daily", {}).get("precipitation_sum", [])
        rain = data.get("daily", {}).get("rain_sum", [])
        return [
            {
                "date": d,
                "precipitation_sum": float(p or 0.0),
                "rain_sum": float(ra or 0.0),
            }
            for d, p, ra in zip(days, precip, rain)
        ]
    except Exception:
        return None


# ---------- Prediction Endpoints ----------

class RainFeatures(BaseModel):
    features: list[float]


@app.post("/predict/rain/")
def predict_rain(features: RainFeatures):
    """Predict rain (yes/no) using the trained classifier"""
    if rain_model is None:
        raise HTTPException(status_code=500, detail="Rain model not available.")
    try:
        pred = rain_model.predict([features.features])[0]
        return {"will_rain": bool(pred)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


class PrecipFeatures(BaseModel):
    features: list[float]


@app.post("/predict/precipitation/fall/")
def predict_precipitation(features: PrecipFeatures):
    """Predict precipitation amount using the trained regressor"""
    if precip_model is None:
        raise HTTPException(status_code=500, detail="Precipitation model not available.")
    try:
        pred = precip_model.predict([features.features])[0]
        return {"precipitation_fall": float(pred)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
