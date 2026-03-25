"""Tests for feature engineering stage."""

import numpy as np
import pandas as pd
import pytest


def test_engineer_features_adds_three_columns(sample_raw_data):
    """Feature engineering should add exactly 3 derived columns."""
    from src.features.build_features import engineer_features

    original_cols = set(sample_raw_data.columns)
    result = engineer_features(sample_raw_data)

    new_cols = set(result.columns) - original_cols
    assert new_cols == {"rooms_per_household", "bedrooms_per_room", "population_per_household"}


def test_engineer_features_preserves_original_columns(sample_raw_data):
    """All original columns should still be present after feature engineering."""
    from src.features.build_features import engineer_features

    result = engineer_features(sample_raw_data)

    for col in sample_raw_data.columns:
        assert col in result.columns


def test_engineer_features_does_not_mutate_input(sample_raw_data):
    """engineer_features should not modify the input DataFrame."""
    from src.features.build_features import engineer_features

    original_cols = list(sample_raw_data.columns)
    engineer_features(sample_raw_data)

    assert list(sample_raw_data.columns) == original_cols


def test_bedrooms_per_room_is_positive(sample_raw_data):
    """bedrooms_per_room should be positive for all realistic inputs."""
    from src.features.build_features import engineer_features

    result = engineer_features(sample_raw_data)
    assert (result["bedrooms_per_room"] > 0).all()


def test_build_features_saves_scaler(sample_raw_data, tmp_path):
    """build_features should save a scaler artifact."""
    import joblib
    from src.features.build_features import build_features

    train_path = str(tmp_path / "train.csv")
    test_path = str(tmp_path / "test.csv")
    train_out = str(tmp_path / "train_features.csv")
    test_out = str(tmp_path / "test_features.csv")
    scaler_path = str(tmp_path / "scaler.pkl")

    # Split sample data into train/test
    split = int(len(sample_raw_data) * 0.8)
    sample_raw_data.iloc[:split].to_csv(train_path, index=False)
    sample_raw_data.iloc[split:].to_csv(test_path, index=False)

    build_features(train_path, test_path, train_out, test_out, scaler_path)

    assert (tmp_path / "scaler.pkl").exists()
    scaler = joblib.load(scaler_path)
    assert hasattr(scaler, "transform")  # is a valid sklearn scaler
