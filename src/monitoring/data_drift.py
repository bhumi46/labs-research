"""
Data Drift Monitoring
======================
Detects when the distribution of incoming data shifts away from training data.

MLOps Concept: Why Models Fail in Production
- A model is trained on historical data (reference distribution)
- Over time, the real world changes → new data has a different distribution
- This is called "data drift" or "concept drift"
- The model's predictions become less accurate without retraining

This script:
1. Loads reference data (training set)
2. Loads current data (simulated here with test set)
3. Generates an Evidently HTML report showing which features drifted

In production, run this on a schedule (daily/weekly) against real incoming data.
If drift is detected → trigger retraining pipeline.

Run: make monitor
View: open reports/drift_report.html
"""

import logging
from pathlib import Path

import pandas as pd
import yaml
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

TARGET_COLUMN = "MedHouseVal"


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def detect_drift(
    reference_path: str,
    current_path: str,
    report_path: str,
) -> dict:
    """
    Generate an Evidently data drift report.

    Args:
        reference_path: Training data — the "expected" distribution
        current_path:   New/incoming data — what the model sees now
        report_path:    Where to save the HTML report

    Returns:
        dict with drift summary
    """
    logger.info(f"Loading reference data (training) from {reference_path}")
    reference_data = pd.read_csv(reference_path)

    logger.info(f"Loading current data from {current_path}")
    current_data = pd.read_csv(current_path)

    # Drop target column — we're monitoring input features, not predictions
    ref_features = reference_data.drop(columns=[TARGET_COLUMN], errors="ignore")
    cur_features = current_data.drop(columns=[TARGET_COLUMN], errors="ignore")

    # Build Evidently report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_features, current_data=cur_features)

    # Save HTML report (open in browser)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(report_path)

    # Extract summary
    result = report.as_dict()
    drift_detected = (
        result.get("metrics", [{}])[0]
        .get("result", {})
        .get("dataset_drift", False)
    )
    drifted_features = (
        result.get("metrics", [{}])[0]
        .get("result", {})
        .get("number_of_drifted_columns", 0)
    )

    summary = {
        "drift_detected": drift_detected,
        "drifted_features": drifted_features,
        "report_path": report_path,
    }

    if drift_detected:
        logger.warning(f"⚠️  Data drift detected! {drifted_features} features drifted.")
        logger.warning(f"   Action: review report and consider retraining.")
    else:
        logger.info("✅ No significant data drift detected.")

    logger.info(f"Report saved → {report_path}")
    return summary


if __name__ == "__main__":
    params = load_params()
    result = detect_drift(
        reference_path=params["data"]["train_features_path"],
        current_path=params["data"]["test_features_path"],  # simulated current data
        report_path="reports/drift_report.html",
    )
    print(f"\nDrift summary: {result}")
