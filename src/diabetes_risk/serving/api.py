"""API REST de inferência (FastAPI).

O app é criado por uma *factory* que recebe o modelo já carregado, o que
torna os testes triviais (basta injetar um bundle) e evita estado global.

Para subir em produção::

    MODEL_PATH=models/model.joblib uvicorn diabetes_risk.serving.api:create_app_from_env --factory
"""

from __future__ import annotations

import os

import pandas as pd
from fastapi import FastAPI

from diabetes_risk.models.bundle import ModelBundle
from diabetes_risk.serving.schemas import (
    ModelInfo,
    Prediction,
    PredictionRequest,
    PredictionResponse,
)

DEFAULT_MODEL_PATH = "models/model.joblib"


def create_app(bundle: ModelBundle) -> FastAPI:
    """Cria a aplicação servindo ``bundle``.

    Endpoints:
        * ``GET /health``: verificação de vida (liveness/readiness);
        * ``GET /model``: metadados do modelo;
        * ``POST /predict``: pontua um lote de pacientes.
    """
    app = FastAPI(
        title="Diabetes Risk API",
        description="Triagem de risco de pré-diabetes/diabetes a partir de indicadores BRFSS.",
        version="0.1.0",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/model", response_model=ModelInfo)
    def model_info() -> ModelInfo:
        return ModelInfo(
            model=bundle.model_name, threshold=bundle.threshold, trained_at=bundle.trained_at
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        features = pd.DataFrame([patient.model_dump() for patient in request.patients])
        probabilities = bundle.predict_proba(features)
        return PredictionResponse(
            model=bundle.model_name,
            threshold=bundle.threshold,
            predictions=[
                Prediction(probability=float(p), at_risk=bool(p >= bundle.threshold))
                for p in probabilities
            ],
        )

    return app


def create_app_from_env() -> FastAPI:
    """Factory para o uvicorn: carrega o modelo de ``$MODEL_PATH``."""
    return create_app(ModelBundle.load(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)))
