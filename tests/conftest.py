"""Fixtures compartilhadas.

Os testes nunca dependem do dataset real: ``make_raw_frame`` gera dados
sintéticos que respeitam o schema do BRFSS e têm sinal real (o risco cresce
com pressão alta, IMC e idade), para que os modelos consigam aprender.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from diabetes_risk.config import PipelineConfig, load_config
from diabetes_risk.domain.schemas import FEATURE_RANGES, TARGET_COLUMN
from diabetes_risk.models.bundle import ModelBundle
from diabetes_risk.models.registry import build_pipeline
from diabetes_risk.models.risk import fit_risk_bands


def make_raw_frame(n_rows: int = 600, seed: int = 0) -> pd.DataFrame:
    """Gera um DataFrame no formato do CSV bruto."""
    rng = np.random.default_rng(seed)
    data = {
        name: rng.integers(low, high + 1, size=n_rows).astype(float)
        for name, (low, high) in FEATURE_RANGES.items()
    }
    data["BMI"] = np.clip(rng.normal(28, 6, size=n_rows).round(), 12, 98)
    logit = -4 + 1.5 * data["HighBP"] + 0.12 * (data["BMI"] - 28) + 0.25 * data["Age"]
    has_risk = rng.random(n_rows) < 1 / (1 + np.exp(-logit))
    # Entre os positivos, ~1/5 são pré-diabetes (1) e o resto diabetes (2).
    data[TARGET_COLUMN] = np.where(has_risk, np.where(rng.random(n_rows) < 0.2, 1.0, 2.0), 0.0)
    return pd.DataFrame(data)[[TARGET_COLUMN, *FEATURE_RANGES]]


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    return make_raw_frame()


@pytest.fixture
def config_dict(tmp_path: Path) -> dict[str, object]:
    """Configuração mínima e rápida, com todos os caminhos dentro de ``tmp_path``."""
    return {
        "data": {
            "raw_path": str(tmp_path / "raw.csv"),
            "processed_dir": str(tmp_path / "processed"),
            "test_size": 0.25,
            "random_state": 7,
        },
        "model": {"name": "logistic_regression", "params": {"class_weight": "balanced"}},
        "training": {"cv_folds": 3, "threshold_beta": 2.0},
        "evaluation": {"min_metrics": {"roc_auc": 0.6}},
        "tracking": {"backend": "local", "uri": str(tmp_path / "runs")},
        "artifacts": {
            "model_path": str(tmp_path / "models" / "model.joblib"),
            "metrics_path": str(tmp_path / "reports" / "metrics.json"),
            "report_dir": str(tmp_path / "reports"),
        },
    }


@pytest.fixture
def config_file(tmp_path: Path, config_dict: dict[str, object]) -> Path:
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")
    return path


@pytest.fixture
def config(config_file: Path) -> PipelineConfig:
    return load_config(config_file)


@pytest.fixture
def mlflow_config(
    config_dict: dict[str, dict[str, object]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> PipelineConfig:
    """Configuração com MLflow em SQLite isolado dentro de ``tmp_path``."""
    pytest.importorskip("mlflow")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.chdir(tmp_path)  # o artifact store padrão (./mlruns) fica no tmp
    config_dict["tracking"] = {
        "backend": "mlflow",
        "uri": f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        "experiment_name": "test-experiment",
        "registered_model_name": "test-model",
    }
    return PipelineConfig.model_validate(config_dict)


@pytest.fixture
def raw_csv(config: PipelineConfig, raw_frame: pd.DataFrame) -> Path:
    """Grava o CSV sintético no caminho esperado pela configuração."""
    raw_frame.to_csv(config.data.raw_path, index=False)
    return config.data.raw_path


@pytest.fixture(scope="session")
def trained_bundle() -> ModelBundle:
    """Bundle pequeno já treinado, reutilizado por testes de API e de bundle."""
    frame = make_raw_frame(n_rows=400, seed=1)
    X = frame.drop(columns=TARGET_COLUMN)
    y = (frame[TARGET_COLUMN] > 0).astype(int)
    pipeline = build_pipeline("logistic_regression").fit(X, y)
    bands = fit_risk_bands(y, pipeline.predict_proba(X)[:, 1], threshold=0.4)
    return ModelBundle(
        pipeline=pipeline, threshold=0.4, model_name="logistic_regression", risk_bands=bands
    )
