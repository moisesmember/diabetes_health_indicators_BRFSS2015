"""Abstrações (portas) das quais o pipeline depende.

O núcleo do pipeline conhece apenas estes contratos, nunca implementações
concretas. Trocar CSV por um banco, ou JSON local por MLflow, é uma questão de
injetar outra classe (princípio da inversão de dependência).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class DataSource(Protocol):
    """Fonte de dados brutos."""

    def load(self) -> pd.DataFrame:
        """Retorna os dados brutos como DataFrame."""
        ...


@runtime_checkable
class ExperimentTracker(Protocol):
    """Registro de parâmetros, métricas e artefatos de um experimento."""

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Registra parâmetros (hiperparâmetros, configuração)."""
        ...

    def log_metrics(self, metrics: Mapping[str, float], prefix: str = "") -> None:
        """Registra métricas numéricas, opcionalmente com um prefixo (ex.: ``"cv_"``)."""
        ...

    def log_artifact(self, path: Path) -> None:
        """Anexa um arquivo ao experimento."""
        ...

    def close(self) -> None:
        """Finaliza o experimento (flush e liberação de recursos)."""
        ...
