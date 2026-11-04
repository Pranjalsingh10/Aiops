import sys
from pathlib import Path

import pytest

# Add the project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from main import create_app


class TestModel:
    def predict(self, features):
        return [0]


@pytest.fixture
def client():
    app = create_app(
        model_loader=TestModel,
        version_loader=lambda: "test",
    )
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_version": "test"}


def test_valid_iris_prediction(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.6,
        "petal_width": 0.5
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "class_name" in result

    assert result["class_name"] in [
        "setosa",
        "versicolor",
        "virginica"
    ]

    assert result["prediction"] in [0, 1, 2]


def test_invalid_iris_input(client):
    payload = {
        "sepal_length": -5.1,
        "sepal_width": 3.5,
        "petal_length": 1.6,
        "petal_width": 0.5
    }

    response = client.post("/predict", json=payload)

    # Expects FastAPI/Pydantic to reject negative measurements
    assert response.status_code == 422
