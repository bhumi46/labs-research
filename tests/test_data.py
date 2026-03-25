"""Tests for data ingestion and preprocessing stages."""

import tempfile

import pandas as pd
import pytest


def test_ingest_produces_csv_with_expected_shape():
    """Data ingestion should produce a valid CSV with California Housing columns."""
    from src.data.ingest import ingest_data

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = f"{tmpdir}/housing.csv"
        ingest_data(output_path=output_path)

        df = pd.read_csv(output_path)

        assert not df.empty
        assert "MedHouseVal" in df.columns
        assert df.shape[0] > 1000  # California dataset has ~20k rows
        assert df.shape[1] == 9    # 8 features + 1 target


def test_ingest_has_no_negative_values():
    """Key columns should not have negative values."""
    from src.data.ingest import ingest_data

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = f"{tmpdir}/housing.csv"
        ingest_data(output_path=output_path)
        df = pd.read_csv(output_path)

        assert (df["MedHouseVal"] > 0).all()
        assert (df["Population"] > 0).all()


def test_preprocess_creates_correct_splits(sample_raw_data):
    """Preprocessing should split data preserving total row count."""
    from src.data.preprocess import preprocess

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = f"{tmpdir}/raw.csv"
        train_path = f"{tmpdir}/train.csv"
        test_path = f"{tmpdir}/test.csv"

        sample_raw_data.to_csv(raw_path, index=False)
        preprocess(
            raw_path=raw_path,
            train_path=train_path,
            test_path=test_path,
            test_size=0.2,
            random_state=42,
        )

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        assert len(train_df) + len(test_df) == len(sample_raw_data)
        assert len(test_df) == pytest.approx(len(sample_raw_data) * 0.2, abs=2)


def test_preprocess_preserves_columns(sample_raw_data):
    """Train and test splits should have the same columns as raw data."""
    from src.data.preprocess import preprocess

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = f"{tmpdir}/raw.csv"
        train_path = f"{tmpdir}/train.csv"
        test_path = f"{tmpdir}/test.csv"

        sample_raw_data.to_csv(raw_path, index=False)
        preprocess(raw_path, train_path, test_path, test_size=0.2, random_state=42)

        train_df = pd.read_csv(train_path)
        assert set(train_df.columns) == set(sample_raw_data.columns)
