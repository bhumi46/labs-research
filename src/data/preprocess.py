"""
Data Preprocessing — Stage 2
==============================
Cleans raw data and splits into train/test sets.

MLOps Concept: Reproducible Preprocessing
- All parameters (test_size, random_state) come from params.yaml
- DVC tracks: inputs (raw CSV) + outputs (train/test CSVs)
- Changing params.yaml → DVC re-runs this stage automatically

Key principle: The test set must NEVER be seen during training.
  Split early so nothing from test leaks into feature engineering.
"""

import logging
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def preprocess(
    raw_path: str,
    train_path: str,
    test_path: str,
    test_size: float,
    random_state: int,
) -> None:
    logger.info(f"Loading raw data from {raw_path}...")
    df = pd.read_csv(raw_path)

    logger.info(f"Dataset shape: {df.shape}")
    missing = df.isnull().sum()
    if missing.any():
        logger.warning(f"Missing values found:\n{missing[missing > 0]}")
        df = df.dropna()
        logger.info(f"Shape after dropping NAs: {df.shape}")

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    Path(train_path).parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"✅ Train set: {train_df.shape} → {train_path}")
    logger.info(f"✅ Test set:  {test_df.shape} → {test_path}")


if __name__ == "__main__":
    params = load_params()
    preprocess(
        raw_path=params["data"]["raw_path"],
        train_path=params["data"]["train_path"],
        test_path=params["data"]["test_path"],
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"],
    )
