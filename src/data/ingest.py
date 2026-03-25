"""
Data Ingestion — Stage 1
========================
Downloads the California Housing dataset and saves it as a raw CSV.

MLOps Concept: Data Versioning
- The output (data/raw/housing.csv) is tracked by DVC
- DVC stores a hash of the file — like git for data
- Run: dvc push   → upload data to remote storage (S3/GCS/etc.)
- Run: dvc pull   → download data on a new machine

In a real project, this stage would:
- Download from S3 / GCS / Azure Blob
- Query a database (Postgres, Snowflake)
- Call a data platform API
"""

import logging
from pathlib import Path

import pandas as pd
import yaml
from sklearn.datasets import fetch_california_housing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def ingest_data(output_path: str) -> None:
    """Fetch California Housing data from sklearn and save as CSV."""
    logger.info("Fetching California Housing dataset from sklearn...")

    housing = fetch_california_housing(as_frame=True)
    df = housing.frame  # DataFrame with feature columns + target 'MedHouseVal'

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    logger.info(f"✅ Raw data saved → {output_path}  shape={df.shape}")
    logger.info(f"   Columns: {list(df.columns)}")
    logger.info(f"   Target 'MedHouseVal': median house value in $100,000s")


if __name__ == "__main__":
    params = load_params()
    ingest_data(output_path=params["data"]["raw_path"])
