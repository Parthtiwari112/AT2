# app/main_impl.py
from fastapi import FastAPI, HTTPException, Query
import pickle, os, datetime, requests
from typing import Optional

app = FastAPI(title="Open Meteo ML - Demo API", version="1.0")

# Models directory relative to this file
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def load_model(name: str):
    path = os.path.join(MODELS_DIR, name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

# Attempt to load pre-trained models (if present)
rain_model = load_model("rain_model.pkl")
precip_model = load_model("precipitation_model.pkl")

@app.get("/")
def root():
    return {
        "project": "Open Meteo ML - Rain (+7 days) & Precipitation (next 3 days)",
        "endpoints": ["/health/", "/predict/rain?date=YYYY-MM-DD", "/predict/precipitation/fall?date=YYYY-MM-DD"],
        "note": "Place trained models into app/models/ (rain_model.pkl, precipitation_model.pkl). If absent, the API falls back to Open-Meteo forecast."
    }

@app.get("/health/")
def health():
    return {"status": "ok", "message": "API is healthy."}

def fetch_forecast_precip(lat: float, lon: float, start_date: str, end_date: str):
    """
    Call Open-Meteo forecast endpoint to get daily precipitation_sum and rain_sum.
    """
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
        return [{"date": d, "precipitation_sum": float(p or 0.0), "rain_sum": float(ra or 0.0)} for d, p, ra in zip(days, precip, rain)]
    except Exception:
        return None

@app.get("/predict/rain/")
def predict_rain(date: str = Query(..., regex="^\\d{4}-\\d{2}-\\d{2}$")):
    """
    Return:
    {
      "input_date": "2023-01-01",
      "prediction": {
         "date": "2023-01-08",
         "will_rain": true
      }
    }
    """
    try:
        input_date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    target_date = input_date + datetime.timedelta(days=7)

    if rain_model is None:
        lat, lon = -33.8678, 151.2073
        res = fetch_forecast_precip(lat, lon, target_date.isoformat(), target_date.isoformat())
        if res is None or len(res) == 0:
            raise HTTPException(status_code=500, detail="Could not fetch forecast data.")
        will_rain = res[0].get("rain_sum", 0.0) > 0.0
        return {"input_date": date, "prediction": {"date": target_date.isoformat(), "will_rain": bool(will_rain)}}
    else:
        # If model exists and you want model-based predictions, you must ensure the endpoint computes the engineered features
        # (or accept them in the request). For now, return 501 indicating client modification required.
        raise HTTPException(status_code=501, detail="Model-based prediction not enabled in this endpoint. Use retrain.py and adapt client or API to provide features.")

@app.get("/predict/precipitation/fall")
def predict_precipitation(date: str = Query(..., regex="^\\d{4}-\\d{2}-\\d{2}$")):
    """
    Return:
    {
      "input_date": "2023-01-01",
      "prediction": {
        "start_date": "2023-01-02",
        "end_date_date": "2023-01-04",
        "precipitation_fall": 28.2
      }
    }
    """
    try:
        input_date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    start = (input_date + datetime.timedelta(days=1)).isoformat()
    end = (input_date + datetime.timedelta(days=3)).isoformat()

    if precip_model is None:
        lat, lon = -33.8678, 151.2073
        res = fetch_forecast_precip(lat, lon, start, end)
        if res is None:
            raise HTTPException(status_code=500, detail="Could not fetch forecast data.")
        total = sum([r.get("precipitation_sum", 0.0) for r in res])
        return {"input_date": date, "prediction": {"start_date": start, "end_date_date": end, "precipitation_fall": round(float(total), 2)}}
    else:
        raise HTTPException(status_code=501, detail="Model-based prediction not enabled in this endpoint. Use retrain.py and adapt client or API to provide features.")
