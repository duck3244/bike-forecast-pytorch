"""FastAPI entrypoint for the bike demand forecasting app."""

from __future__ import annotations

import json
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Make sibling modules importable when running as ``uvicorn app.main:app``.
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from bike_forecast_pytorch import BikeDataProcessor, BikeForecaster, LSTMForecaster  # noqa: E402
from utils import load_checkpoint  # noqa: E402

OUTPUT_DIR = BACKEND_DIR / 'outputs'
MODEL_DIR = OUTPUT_DIR / 'models'
PREDICTIONS_PATH = OUTPUT_DIR / 'predictions' / 'mlp_predictions.json'
TRAIN_CSV = BACKEND_DIR / 'data' / 'train.csv'
UPLOAD_DIR = OUTPUT_DIR / 'uploads'

NUMERICAL_COLS = ['temp', 'atemp', 'humidity', 'windspeed', 'temp_humidity', 'temp_windspeed']
MODEL_CLASS_MAP = {'BikeForecaster': BikeForecaster, 'LSTMForecaster': LSTMForecaster}

STATE: dict = {}
STATE_LOCK = threading.Lock()


def reload_inference_model() -> None:
    """Swap the cached inference model atomically. Called after training."""
    try:
        model, sidecar = load_checkpoint(MODEL_DIR, MODEL_CLASS_MAP)
        model.to('cpu').eval()
        processor = BikeDataProcessor()
        processor.scalers = sidecar['scalers']
        with STATE_LOCK:
            STATE['model'] = model
            STATE['processor'] = processor
            STATE['feature_cols'] = sidecar['feature_cols']
            STATE['metadata'] = sidecar.get('metadata', {})
    except FileNotFoundError:
        # No trained model yet — keep the API usable for training/upload.
        with STATE_LOCK:
            STATE.pop('model', None)
            STATE.pop('processor', None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    reload_inference_model()
    yield
    STATE.clear()


app = FastAPI(title='Bike Demand Forecast API', version='0.2.0', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
    allow_credentials=False,
    allow_methods=['GET', 'POST', 'DELETE'],
    allow_headers=['*'],
)

# Routers registered after STATE is defined so they can import from app.main.
from app.routers import datasets as datasets_router  # noqa: E402
from app.routers import training as training_router  # noqa: E402

app.include_router(datasets_router.router)
app.include_router(training_router.router)


# ---------- Schemas ----------

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_class: str | None = None
    input_size: int | None = None


class Metrics(BaseModel):
    r2: float
    rmse: float
    mae: float
    mse: float
    mape: float | None = None


class ModelMetricsResponse(BaseModel):
    model_type: str
    metrics: Metrics
    training_time: float | None = None
    n_samples: int


class PredictRequest(BaseModel):
    datetime: str = Field(..., description='ISO datetime string, e.g. 2011-07-15T08:00:00')
    season: int = Field(..., ge=1, le=4)
    holiday: int = Field(..., ge=0, le=1)
    workingday: int = Field(..., ge=0, le=1)
    weather: int = Field(..., ge=1, le=4)
    temp: float = Field(..., ge=-20, le=50)
    atemp: float = Field(..., ge=-30, le=60)
    humidity: float = Field(..., ge=0, le=100)
    windspeed: float = Field(..., ge=0, le=100)


class PredictResponse(BaseModel):
    count: float
    count_rounded: int


# ---------- Routes ----------

@app.get('/api/health', response_model=HealthResponse)
async def health() -> HealthResponse:
    with STATE_LOCK:
        model = STATE.get('model')
        feature_cols = STATE.get('feature_cols', [])
    return HealthResponse(
        status='ok',
        model_loaded=model is not None,
        model_class=model.__class__.__name__ if model else None,
        input_size=len(feature_cols) or None,
    )


@app.get('/api/models/current/metrics', response_model=ModelMetricsResponse)
async def current_metrics() -> ModelMetricsResponse:
    # Prefer metrics from the most recent training job (matches the live model).
    from app.services import training as training_service
    latest = next(
        (j for j in training_service.list_jobs()
         if j.status.value == 'completed' and j.metrics),
        None,
    )
    if latest is not None:
        return ModelMetricsResponse(
            model_type=latest.model_type,
            metrics=Metrics(**{k: latest.metrics.get(k) for k in ['r2', 'rmse', 'mae', 'mse', 'mape']}),
            training_time=(latest.finished_at or 0) - (latest.started_at or 0) or None,
            n_samples=0,
        )

    # Fallback: read from the pre-trained prediction artifact.
    if not PREDICTIONS_PATH.exists():
        raise HTTPException(404, f'No prediction artifacts at {PREDICTIONS_PATH.name}')
    data = json.loads(PREDICTIONS_PATH.read_text())
    meta = data.get('metadata', {}) or {}
    metrics = meta.get('metrics') or data.get('metrics') or {}
    return ModelMetricsResponse(
        model_type=meta.get('model_type', 'mlp'),
        metrics=Metrics(**{k: metrics.get(k) for k in ['r2', 'rmse', 'mae', 'mse', 'mape']}),
        training_time=meta.get('training_time'),
        n_samples=len(data.get('y_true', [])),
    )


@app.post('/api/predict', response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    with STATE_LOCK:
        model = STATE.get('model')
        processor = STATE.get('processor')
        feature_cols = STATE.get('feature_cols')
    if model is None or processor is None or not feature_cols:
        raise HTTPException(503, 'Model not loaded — train a model first')

    row = req.model_dump()
    row['datetime'] = pd.to_datetime(row['datetime'])
    df = pd.DataFrame([row])

    engineered = processor.engineer_features(df)
    scaler = processor.scalers['numerical']
    engineered[NUMERICAL_COLS] = scaler.transform(engineered[NUMERICAL_COLS])

    X = engineered[feature_cols].values.astype('float32')
    tensor = torch.from_numpy(X)
    if isinstance(model, LSTMForecaster):
        # LSTM expects (batch, seq, features); MLP expects (batch, features)
        tensor = tensor.unsqueeze(1)

    with torch.no_grad():
        y = model(tensor).squeeze(-1).item()

    y = max(0.0, float(y))
    return PredictResponse(count=y, count_rounded=int(round(y)))
