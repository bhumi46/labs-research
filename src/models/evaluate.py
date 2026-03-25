"""
Model Evaluation — Stage 5
============================
Evaluates the trained model on the held-out test set.

MLOps Concept: Quality Gates
- This stage acts as an automated gate in the CI/CD pipeline
- If R² < min_r2 OR RMSE > max_rmse (from params.yaml), the pipeline FAILS
- This prevents a degraded model from being deployed to production
- In real projects, you might also compare against the current Production model

MLOps Concept: The Test Set is Sacred
- Test data was split BEFORE any feature engineering
- The model has NEVER seen this data → honest evaluation
"""

import json
import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

TARGET_COLUMN = "MedHouseVal"


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def get_latest_model_uri(experiment_name: str, model_name: str) -> str:
    """Get the URI of the most recently registered model version."""
    client = mlflow.tracking.MlflowClient()

    try:
        versions = client.get_latest_versions(
            model_name, stages=["None", "Staging", "Production"]
        )
        if versions:
            latest = max(versions, key=lambda v: int(v.version))
            uri = f"models:/{model_name}/{latest.version}"
            logger.info(f"Loading from Model Registry: {uri}")
            return uri
    except Exception as exc:
        logger.warning(f"Registry lookup failed ({exc}), falling back to latest run")

    # Fallback: load from the latest MLflow run
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
        )
        if not runs.empty:
            run_id = runs.iloc[0]["run_id"]
            uri = f"runs:/{run_id}/model"
            logger.info(f"Loading from latest run: {uri}")
            return uri

    raise ValueError(
        f"No trained model found for experiment '{experiment_name}'. "
        "Run 'make train' first."
    )


def evaluate(params: dict) -> None:
    data_params = params["data"]
    mlflow_params = params["mlflow"]
    thresholds = params.get("evaluation", {})

    if mlflow_params.get("tracking_uri"):
        mlflow.set_tracking_uri(mlflow_params["tracking_uri"])

    # Load test features
    test_df = pd.read_csv(data_params["test_features_path"])
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    logger.info(f"Evaluating on {X_test.shape[0]} test samples")

    # Load model from MLflow
    model_uri = get_latest_model_uri(
        mlflow_params["experiment_name"], mlflow_params["model_name"]
    )
    model = mlflow.sklearn.load_model(model_uri)

    # Predict and compute metrics
    preds = model.predict(X_test)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
    }
    logger.info(f"Test metrics: {metrics}")

    # ── Quality Gates ────────────────────────────────────────────────────────
    min_r2 = thresholds.get("min_r2", 0.0)
    max_rmse = thresholds.get("max_rmse", float("inf"))

    gate_failed = False
    if metrics["r2"] < min_r2:
        logger.error(f"❌ QUALITY GATE FAILED: R²={metrics['r2']:.4f} < threshold={min_r2}")
        gate_failed = True
    if metrics["rmse"] > max_rmse:
        logger.error(
            f"❌ QUALITY GATE FAILED: RMSE={metrics['rmse']:.4f} > threshold={max_rmse}"
        )
        gate_failed = True

    if not gate_failed:
        logger.info("✅ All quality gates passed!")

    # Log to MLflow
    mlflow.set_experiment(mlflow_params["experiment_name"])
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

    # Save metrics for DVC
    metrics_path = Path("metrics/test_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics saved → {metrics_path}")

    if gate_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    params = load_params()
    evaluate(params)
