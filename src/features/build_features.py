"""
Feature Engineering — Stage 3
================================
Builds domain-driven features and scales them for the model.

MLOps Concept: Preventing Data Leakage
- StandardScaler is FIT only on training data
- Then APPLIED (transform only) to test data
- The fitted scaler is saved as an artifact so inference uses the same scale

MLOps Concept: Feature Store
- In production, engineered features would be stored in a feature store
  (e.g., Feast, Tecton, Hopsworks) for reuse across models
- For this demo, we save transformed CSVs tracked by DVC

Engineered Features (domain knowledge for real estate):
- rooms_per_household:      larger rooms-per-age → spacious newer homes
- bedrooms_per_room:        lower ratio → more living space (better value)
- population_per_household: higher → denser neighborhood
"""

import logging
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

TARGET_COLUMN = "MedHouseVal"


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-driven engineered features to the DataFrame."""
    df = df.copy()
    df["rooms_per_household"] = df["AveRooms"] / df["HouseAge"].clip(lower=1)
    df["bedrooms_per_room"] = df["AveBedrms"] / df["AveRooms"].clip(lower=1)
    df["population_per_household"] = df["Population"] / df["AveOccup"].clip(lower=1)
    return df


def build_features(
    train_path: str,
    test_path: str,
    train_out: str,
    test_out: str,
    scaler_path: str,
) -> None:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Add engineered features
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

    feature_cols = [c for c in train_df.columns if c != TARGET_COLUMN]

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    # Fit scaler on train data ONLY — prevents data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # transform only, no fit

    # Save scaler artifact (needed at inference time)
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info(f"✅ Scaler saved → {scaler_path}")

    # Reconstruct DataFrames with scaled features
    train_out_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
    train_out_df[TARGET_COLUMN] = train_df[TARGET_COLUMN].values

    test_out_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
    test_out_df[TARGET_COLUMN] = test_df[TARGET_COLUMN].values

    Path(train_out).parent.mkdir(parents=True, exist_ok=True)
    train_out_df.to_csv(train_out, index=False)
    test_out_df.to_csv(test_out, index=False)

    logger.info(f"✅ Train features: {train_out_df.shape} → {train_out}")
    logger.info(f"✅ Test features:  {test_out_df.shape} → {test_out}")
    logger.info(f"   Feature list: {feature_cols}")


if __name__ == "__main__":
    params = load_params()
    build_features(
        train_path=params["data"]["train_path"],
        test_path=params["data"]["test_path"],
        train_out=params["data"]["train_features_path"],
        test_out=params["data"]["test_features_path"],
        scaler_path=params["model"]["scaler_path"],
    )
