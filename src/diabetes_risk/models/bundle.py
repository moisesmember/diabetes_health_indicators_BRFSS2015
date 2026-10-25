"""Artefato de modelo versionável.

Um modelo de classificação em produção precisa de mais que o estimador: precisa
do limiar de decisão e das faixas de risco escolhidos no treino, e de metadados
para auditoria. :class:`ModelBundle` empacota tudo isso em um único ``joblib``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from diabetes_risk.domain.schemas import FEATURE_COLUMNS
from diabetes_risk.models.risk import RiskAssessment, RiskBands, assess


@dataclass(frozen=True)
class ModelBundle:
    """Pipeline treinado + limiar de decisão + metadados.

    Attributes:
        pipeline: Pipeline sklearn já ajustado (features, pré-processamento, modelo).
        threshold: Probabilidade mínima para classificar como positivo.
        model_name: Nome do modelo no registry.
        risk_bands: Faixas de risco (baixa/média/alta) aprendidas no treino.
        metadata: Informações livres (parâmetros, métricas de CV etc.).
        trained_at: Data/hora UTC do treino, em ISO 8601.
    """

    pipeline: Pipeline
    threshold: float
    model_name: str
    risk_bands: RiskBands | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    trained_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"Limiar deve estar em [0, 1], recebido {self.threshold}.")
        if self.risk_bands is not None and self.risk_bands.medium != self.threshold:
            raise ValueError("O início da faixa média deve ser igual ao limiar de decisão.")

    @property
    def bands(self) -> RiskBands:
        """Faixas de risco; sem faixas treinadas, usa só o limiar (sem faixa alta)."""
        return self.risk_bands or RiskBands(medium=self.threshold, high=1.0)

    @property
    def run_id(self) -> str | None:
        """Run de tracking que produziu este modelo (``None`` se desconhecido)."""
        run_id = self.metadata.get("run_id")
        return str(run_id) if run_id else None

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Probabilidade de risco (classe positiva) para cada linha."""
        proba: np.ndarray = self.pipeline.predict_proba(features[list(FEATURE_COLUMNS)])[:, 1]
        return proba

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Classe prevista (0/1) aplicando o limiar do bundle."""
        return (self.predict_proba(features) >= self.threshold).astype(int)

    def assess(self, features: pd.DataFrame) -> list[RiskAssessment]:
        """Avaliação completa de cada linha: classificação, nível e indicação."""
        return [assess(p, self.bands) for p in self.predict_proba(features)]

    def assessment_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        """Mesmo que :meth:`assess`, em formato tabular (usado no modelo MLflow)."""
        rows = [
            {
                "probability": a.probability,
                "at_risk": a.at_risk,
                "classification": a.classification.value,
                "risk_level": a.risk_level.value,
                "requires_confirmation": a.requires_confirmation,
            }
            for a in self.assess(features)
        ]
        return pd.DataFrame(rows)

    def save(self, path: str | Path) -> Path:
        """Serializa o bundle, criando diretórios se necessário."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> ModelBundle:
        """Carrega um bundle salvo com :meth:`save`.

        Atenção: ``joblib`` usa pickle; carregue apenas arquivos confiáveis.

        Raises:
            FileNotFoundError: Se o arquivo não existir.
            TypeError: Se o arquivo não contiver um ``ModelBundle``.
        """
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Modelo não encontrado em '{source}'. Rode a etapa 'train'.")
        obj = joblib.load(source)
        if not isinstance(obj, cls):
            raise TypeError(f"'{source}' não contém um ModelBundle (tipo: {type(obj).__name__}).")
        return obj
