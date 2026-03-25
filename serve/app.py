"""
FastAPI Model Serving Application
===================================
Exposes the trained model as a REST API.

Endpoints:
  GET  /health   → health check (used by load balancers, K8s probes)
  GET  /info     → model metadata
  POST /predict  → return house price prediction

MLOps Concept: Model Serving
- The model is loaded from MLflow Model Registry at startup (stage: Production)
- Falls back to a local file if the registry is unavailable
- In production: containerized with Docker, scaled with Kubernetes/ECS
- Zero-downtime deploys: new container starts, health check passes, old stops

Run locally:  make serve
Run in Docker: make docker-compose-up

Interactive docs (auto-generated): http://localhost:8000/docs
"""

import logging
import os
from contextlib import asynccontextmanager

import joblib
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from serve.schemas import HealthResponse, PredictionRequest, PredictionResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model state — populated at startup, read during requests
model_state: dict = {}

FEATURE_ORDER = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
]


def load_model() -> tuple:
    """Load model from MLflow Registry or fall back to a local file."""
    model_name = os.getenv("MLFLOW_MODEL_NAME", "house-price-model")
    model_stage = os.getenv("MLFLOW_MODEL_STAGE", "Production")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # Try MLflow Model Registry first
    try:
        model_uri = f"models:/{model_name}/{model_stage}"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"✅ Model loaded from MLflow Registry: {model_uri}")
        return model, {"source": "mlflow_registry", "name": model_name, "stage": model_stage}
    except Exception as e:
        logger.warning(f"Could not load from MLflow Registry: {e}")

    # Fallback: local file
    local_path = os.getenv("MODEL_PATH", "models/model.pkl")
    if os.path.exists(local_path):
        model = joblib.load(local_path)
        logger.info(f"✅ Model loaded from local file: {local_path}")
        return model, {"source": "local_file", "path": local_path}

    raise RuntimeError(
        "No model found. Run 'make pipeline' to train a model first, "
        "then promote it to Production in MLflow Registry."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("Starting model server...")
    model, info = load_model()
    model_state["model"] = model
    model_state["info"] = info
    logger.info(f"Model info: {info}")
    yield
    # ── Shutdown ─────────────────────────────────────────────────────────────
    model_state.clear()
    logger.info("Model server stopped.")


app = FastAPI(
    title="House Price Predictor API",
    description=(
        "MLOps learning project — predicts California median house prices.\n\n"
        "The model is a RandomForest trained with scikit-learn and tracked via MLflow."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Operations"])
def health():
    """Health check — used by load balancers and Kubernetes liveness probes."""
    return HealthResponse(status="ok", model_loaded="model" in model_state)


@app.get("/info", tags=["Operations"])
def info():
    """Return metadata about the loaded model."""
    if "info" not in model_state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_state["info"]


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest):
    """
    Predict median house value.

    Returns the predicted price in both $100,000 units and USD.
    Example: predicted_price=2.5 means predicted_price_usd=$250,000.
    """
    if "model" not in model_state:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model = model_state["model"]

    # Compute engineered features (same logic as build_features.py)
    rooms_per_household = request.ave_rooms / max(request.house_age, 1)
    bedrooms_per_room = request.ave_bedrms / max(request.ave_rooms, 1)
    population_per_household = request.population / max(request.ave_occup, 1)

    features = np.array(
        [[
            request.med_inc,
            request.house_age,
            request.ave_rooms,
            request.ave_bedrms,
            request.population,
            request.ave_occup,
            request.latitude,
            request.longitude,
            rooms_per_household,
            bedrooms_per_room,
            population_per_household,
        ]]
    )

    prediction = float(model.predict(features)[0])

    return PredictionResponse(
        predicted_price=round(prediction, 4),
        predicted_price_usd=round(prediction * 100_000, 2),
    )
