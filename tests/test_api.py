"""Tests for the FastAPI serving application."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


SAMPLE_PAYLOAD = {
    "med_inc": 3.5,
    "house_age": 20.0,
    "ave_rooms": 5.0,
    "ave_bedrms": 1.0,
    "population": 1000.0,
    "ave_occup": 3.0,
    "latitude": 37.88,
    "longitude": -122.23,
}


@pytest.fixture
def client():
    """
    FastAPI TestClient with a mocked model.

    We patch load_model so the lifespan startup doesn't need real MLflow.
    """
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([2.5])

    with patch("serve.app.load_model", return_value=(mock_model, {"source": "mock"})):
        from fastapi.testclient import TestClient
        from serve.app import app

        with TestClient(app) as c:
            yield c


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_predict_returns_200(client):
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200


def test_predict_response_has_required_fields(client):
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    data = response.json()
    assert "predicted_price" in data
    assert "predicted_price_usd" in data


def test_predict_usd_equals_price_times_100k(client):
    """predicted_price_usd should be predicted_price × 100,000."""
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    data = response.json()
    assert data["predicted_price_usd"] == pytest.approx(
        data["predicted_price"] * 100_000, rel=1e-3
    )


def test_predict_with_mocked_model_returns_250k(client):
    """Mock returns 2.5 → $250,000."""
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    data = response.json()
    assert data["predicted_price"] == pytest.approx(2.5, rel=1e-3)
    assert data["predicted_price_usd"] == pytest.approx(250_000.0, rel=1e-3)


def test_predict_rejects_missing_field(client):
    """Request missing required fields should return 422 Unprocessable Entity."""
    incomplete = {k: v for k, v in SAMPLE_PAYLOAD.items() if k != "med_inc"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_info_endpoint(client):
    response = client.get("/info")
    assert response.status_code == 200
    assert response.json()["source"] == "mock"
