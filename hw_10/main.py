from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "real_estate_model.joblib"

FEATURES = ["total_square", "rooms", "floor", "lat", "lon",
            "city", "district", "settlement"]

app = FastAPI(
    title="Real Estate Price Prediction API",
    description="API для прогнозирования стоимости квартир в Москве и Московской области.",
    version="1.0.0"
)

model = None

@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)

class Property(BaseModel):
    total_square: float = Field(gt=0, description="Общая площадь квартиры, м²")
    rooms: float | None = Field(default=None, ge=0, description="Количество комнат")
    floor: float = Field(ge=0, description="Этаж")
    lat: float = Field(description="Широта")
    lon: float = Field(description="Долгота")
    city: str | None = Field(default=None, description="Город")
    district: str | None = Field(default=None, description="Район")
    settlement: str | None = Field(default=None, description="Населённый пункт")

def make_prediction(property_data: dict) -> float:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Модель не загружена. Сначала выполните python train.py"
        )
    row = pd.DataFrame([property_data], columns=FEATURES)
    prediction = model.predict(row)[0]
    return round(float(prediction), 2)

@app.get("/predict_get")
def predict_get(
    total_square: float = Query(gt=0),
    rooms: float | None = None,
    floor: float = 0,
    lat: float = 55.7558,
    lon: float = 37.6173,
    city: str | None = "Москва",
    district: str | None = None,
    settlement: str | None = None
):
    data = {
        "total_square": total_square,
        "rooms": rooms,
        "floor": floor,
        "lat": lat,
        "lon": lon,
        "city": city,
        "district": district,
        "settlement": settlement
    }
    return {"predicted_price": make_prediction(data), "currency": "RUB"}

@app.post("/predict_post")
def predict_post(property_data: Property):
    prediction = make_prediction(property_data.dict())
    return {"predicted_price": prediction, "currency": "RUB"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }
