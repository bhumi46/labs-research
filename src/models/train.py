"""
Model Training — Stage 4
==========================
Trains a model and logs everything to MLflow.

MLOps Concept: Experiment Tracking
- Every run logs: parameters, metrics, and the trained model artifact
- You can compare runs in the MLflow UI (make mlflow-ui)
- The model is automatically registered in the MLflow Model Registry

MLOps Concept: Model Registry
- After training, the model goes to stage "None" in the registry
- A human (or automated CI) reviews metrics and promotes it:
    None → Staging → Production
- The serving API loads from "Production" stage

View runs: mlflow ui → http://localhost:5000
"""

import json
import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

TARGET_COLUMN = "MedHouseVal"

# Registry: add new model types here without changing the rest of the code
MODEL_REGISTRY = {
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train(params: dict) -> None:
    data_params = params["data"]
    model_params = params["model"]
    mlflow_params = params["mlflow"]

    # Load training features
    train_df = pd.read_csv(data_params["train_features_path"])
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    logger.info(f"Training on {X_train.shape[0]} samples, {X_train.shape[1]} features")
    logger.info(f"Model: {model_params['type']}")

    # Point MLflow to the tracking server (local by default)
    if mlflow_params.get("tracking_uri"):
        mlflow.set_tracking_uri(mlflow_params["tracking_uri"])

    mlflow.set_experiment(mlflow_params["experiment_name"])

    with mlflow.start_run():
        # ── Log hyperparameters ──────────────────────────────────────────────
        mlflow.log_params(
            {
                "model_type": model_params["type"],
                "n_estimators": model_params["n_estimators"],
                "max_depth": model_params["max_depth"],
                "random_state": model_params["random_state"],
                "test_size": data_params["test_size"],
            }
        )

        # ── Train ────────────────────────────────────────────────────────────
        ModelClass = MODEL_REGISTRY[model_params["type"]]
        model = ModelClass(
            n_estimators=model_params["n_estimators"],
            max_depth=model_params["max_depth"],
            random_state=model_params["random_state"],
        )
        model.fit(X_train, y_train)
        logger.info("Training complete ✅")

        # ── Log train metrics ────────────────────────────────────────────────
        train_preds = model.predict(X_train)
        train_metrics = compute_metrics(y_train, train_preds)
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        logger.info(f"Train metrics: {train_metrics}")

        # ── Log model artifact + register ────────────────────────────────────
        # MLflow auto-registers the model version in the Model Registry
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=mlflow_params["model_name"],
        )

        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run ID: {run_id}")

        # ── Save metrics for DVC tracking ────────────────────────────────────
        metrics_path = Path("metrics/train_metrics.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(train_metrics, f, indent=2)

    logger.info("👉  Run  'make mlflow-ui'  to view this run in MLflow")


if __name__ == "__main__":
    params = load_params()
    train(params)
