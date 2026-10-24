"""Configuração tipada do pipeline.

Todo parâmetro vem de um único arquivo YAML (``params.yaml``) e é validado por
modelos Pydantic. Assim, um erro de configuração falha cedo, com mensagem clara,
em vez de estourar no meio do treino.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class _StrictModel(BaseModel):
    """Base que rejeita chaves desconhecidas (erros de digitação no YAML)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_StrictModel):
    """Origem dos dados e regras de preparação."""

    raw_path: Path
    processed_dir: Path
    positive_classes: list[int] = Field(default=[1, 2], min_length=1)
    drop_duplicates: bool = True
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    random_state: int = 42

    @field_validator("positive_classes")
    @classmethod
    def _classes_are_valid(cls, value: list[int]) -> list[int]:
        invalid = set(value) - {0, 1, 2}
        if invalid:
            raise ValueError(f"Classes inválidas para Diabetes_012: {sorted(invalid)}")
        return value

    @property
    def train_path(self) -> Path:
        """Caminho do conjunto de treino gerado pela etapa ``prepare``."""
        return self.processed_dir / "train.parquet"

    @property
    def test_path(self) -> Path:
        """Caminho do conjunto de teste gerado pela etapa ``prepare``."""
        return self.processed_dir / "test.parquet"


class ModelConfig(_StrictModel):
    """Qual estimador usar e com quais hiperparâmetros."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class TrainingConfig(_StrictModel):
    """Validação cruzada e escolha do limiar de decisão."""

    cv_folds: int = Field(default=5, ge=2)
    threshold_beta: float = Field(default=2.0, gt=0.0)


class EvaluationConfig(_StrictModel):
    """Mínimos exigidos pelo quality gate."""

    min_metrics: dict[str, float] = Field(default_factory=dict)


class TrackingConfig(_StrictModel):
    """Onde registrar parâmetros e métricas dos experimentos."""

    backend: Literal["local", "mlflow"] = "local"
    uri: str = "reports/runs"
    experiment_name: str = "diabetes-risk"


class ArtifactsConfig(_StrictModel):
    """Onde salvar o modelo treinado e o relatório de avaliação."""

    model_path: Path
    metrics_path: Path


class PipelineConfig(_StrictModel):
    """Configuração completa do pipeline."""

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig = TrainingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    tracking: TrackingConfig = TrackingConfig()
    artifacts: ArtifactsConfig


def load_config(path: str | Path) -> PipelineConfig:
    """Lê e valida o arquivo YAML de configuração.

    Args:
        path: Caminho do arquivo YAML (normalmente ``params.yaml``).

    Returns:
        A configuração validada e imutável.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        pydantic.ValidationError: Se algum campo for inválido.
    """
    with Path(path).open(encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    return PipelineConfig.model_validate(raw)
