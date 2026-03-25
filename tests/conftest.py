"""
Shared pytest fixtures used across all test modules.

fixtures here are auto-discovered by pytest — no imports needed in test files.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """100-row DataFrame mimicking the California Housing raw dataset."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "MedInc": np.random.uniform(1.0, 10.0, n),
            "HouseAge": np.random.uniform(1.0, 52.0, n),
            "AveRooms": np.random.uniform(3.0, 10.0, n),
            "AveBedrms": np.random.uniform(0.8, 3.0, n),
            "Population": np.random.uniform(100.0, 3000.0, n),
            "AveOccup": np.random.uniform(1.0, 5.0, n),
            "Latitude": np.random.uniform(32.0, 42.0, n),
            "Longitude": np.random.uniform(-124.0, -114.0, n),
            "MedHouseVal": np.random.uniform(0.5, 5.0, n),
        }
    )
