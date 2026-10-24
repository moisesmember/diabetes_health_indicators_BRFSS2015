"""API servindo um modelo real treinado."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from diabetes_risk.domain.schemas import FEATURE_RANGES
from diabetes_risk.models.bundle import ModelBundle
from diabetes_risk.serving.api import create_app, create_app_from_env

pytestmark = pytest.mark.integration

LOW_RISK = {name: low for name, (low, _) in FEATURE_RANGES.items()} | {"BMI": 21.0}
HIGH_RISK = LOW_RISK | {"HighBP": 1.0, "BMI": 45.0, "Age": 13.0}


@pytest.fixture
def client(trained_bundle: ModelBundle) -> TestClient:
    return TestClient(create_app(trained_bundle))


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_model_info(client: TestClient, trained_bundle: ModelBundle) -> None:
    body = client.get("/model").json()
    assert body["model"] == "logistic_regression"
    assert body["threshold"] == trained_bundle.threshold


def test_predict_returns_one_result_per_patient_in_order(client: TestClient) -> None:
    response = client.post("/predict", json={"patients": [LOW_RISK, HIGH_RISK]})

    assert response.status_code == 200
    low, high = response.json()["predictions"]
    assert 0.0 <= low["probability"] < high["probability"] <= 1.0
    assert high["at_risk"] is True


@pytest.mark.parametrize(
    "payload",
    [
        {"patients": []},
        {"patients": [LOW_RISK | {"BMI": 500}]},
        {"patients": [{k: v for k, v in LOW_RISK.items() if k != "Age"}]},
        {"patients": [LOW_RISK | {"unknown": 1}]},
    ],
    ids=["empty", "out-of-range", "missing-field", "extra-field"],
)
def test_invalid_payload_returns_422(client: TestClient, payload: dict[str, object]) -> None:
    assert client.post("/predict", json=payload).status_code == 422


def test_create_app_from_env(
    trained_bundle: ModelBundle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = trained_bundle.save(tmp_path / "model.joblib")
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    client = TestClient(create_app_from_env())

    assert client.get("/health").status_code == 200
