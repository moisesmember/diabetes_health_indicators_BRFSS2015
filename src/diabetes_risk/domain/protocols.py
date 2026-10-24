"""Abstrações (portas) das quais o pipeline depende.

O núcleo do pipeline conhece apenas estes contratos, nunca implementações
concretas. Trocar CSV por um banco, ou JSON local por MLflow, é uma questão de
injetar outra classe (princípio da inversão de dependência).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import pandas as pd

if TYPE_CHECKING:
    from diabetes_risk.models.bundle import ModelBundle


@runtime_checkable
class DataSource(Protocol):
    """Fonte de dados brutos."""

    def load(self) -> pd.DataFrame:
        """Retorna os dados brutos como DataFrame."""
        ...


@runtime_checkable
class ExperimentTracker(Protocol):
    """Registro de um experimento (run) e dos modelos que ele produz.

    Um mesmo run atravessa as etapas ``train`` e ``evaluate``: o treino cria o
    run e grava seu ``run_id`` no bundle; a avaliação o retoma. Assim métricas
    de validação cruzada (``cv/*``) e de teste (``test/*``) ficam lado a lado.
    """

    @property
    def run_id(self) -> str:
        """Identificador do run ativo."""
        ...

    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Anexa metadados pesquisáveis (modelo, commit, resultado do gate...)."""
        ...

    def set_description(self, text: str) -> None:
        """Define a descrição (Markdown) do run, exibida para quem o consulta."""
        ...

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Registra parâmetros (hiperparâmetros, configuração)."""
        ...

    def log_metrics(self, metrics: Mapping[str, float], prefix: str = "") -> None:
        """Registra métricas numéricas, opcionalmente com um prefixo (ex.: ``"cv/"``)."""
        ...

    def log_artifact(self, path: Path, artifact_dir: str | None = None) -> None:
        """Anexa um arquivo ao run, opcionalmente dentro de uma subpasta."""
        ...

    def log_dataset(self, df: pd.DataFrame, name: str, context: str, source: str) -> None:
        """Registra a linhagem de um conjunto de dados (nome, origem, tamanho, hash)."""
        ...

    def log_model(self, bundle: ModelBundle, input_example: pd.DataFrame) -> None:
        """Registra o modelo treinado como artefato do run."""
        ...

    def register_model(self, name: str, alias: str | None, tags: Mapping[str, str]) -> str | None:
        """Cria uma versão do modelo no registry; ``alias`` a promove (ex.: champion).

        Returns:
            A versão criada, ou ``None`` se o backend não tiver registry.
        """
        ...

    def close(self) -> None:
        """Finaliza o run (flush e liberação de recursos)."""
        ...
