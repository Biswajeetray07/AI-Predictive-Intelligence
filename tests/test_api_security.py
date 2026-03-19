import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.api.app import app

client = TestClient(app)

def test_missing_api_key():
    response = client.post(
        "/predict", 
        json={"ticker": "AAPL", "date": "2026-03-12"}
    )
    assert response.status_code == 403 or response.status_code == 401

def test_invalid_api_key():
    response = client.post(
        "/predict", 
        headers={"X-API-Key": "wrong_password"},
        json={"ticker": "AAPL", "date": "2026-03-12"}
    )
    assert response.status_code == 401

def test_valid_api_key_invalid_body():
    # AAPL1 is invalid length, should fail Pydantic validation
    response = client.post(
        "/predict", 
        headers={"X-API-Key": "predictive_intel_dev_key_2026"},
        json={"ticker": "AAPL12", "date": "2026/03/12"}
    )
    assert response.status_code == 422 # Unprocessable Entity
    assert "ticker" in response.text.lower() or "date" in response.text.lower()

