"""Implementações de :class:`~diabetes_risk.domain.protocols.ExperimentTracker`.

* :class:`LocalJsonTracker`: sem dependências, grava um ``run.json`` por execução;
  ideal para desenvolvimento e CI.
* :class:`MlflowTracker`: integra com MLflow (instale o extra ``[mlflow]``).

Use :func:`create_tracker` para obter a implementação configurada.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from diabetes_risk.config import TrackingConfig
from diabetes_risk.domain.protocols import ExperimentTracker


class LocalJsonTracker:
    """Registra cada execução em ``<root>/<experimento>/<timestamp>/run.json``.

    Os artefatos registrados são copiados para a pasta da execução.

    Args:
        root: Diretório raiz dos experimentos.
        experiment_name: Nome do experimento (subpasta de ``root``).
        run_name: Nome da execução; por padrão, o timestamp UTC atual.
    """

    def __init__(self, root: str | Path, experiment_name: str, run_name: str | None = None) -> None:
        run_name = run_name or datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        self.run_dir = Path(root) / experiment_name / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._record: dict[str, Any] = {"params": {}, "metrics": {}, "artifacts": []}

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Acumula parâmetros (valores não serializáveis viram texto)."""
        self._record["params"].update({key: _jsonable(value) for key, value in params.items()})

    def log_metrics(self, metrics: Mapping[str, float], prefix: str = "") -> None:
        """Acumula métricas, prefixando cada nome com ``prefix``."""
        self._record["metrics"].update({f"{prefix}{k}": float(v) for k, v in metrics.items()})

    def log_artifact(self, path: Path) -> None:
        """Copia o arquivo para a pasta da execução."""
        shutil.copy2(path, self.run_dir / Path(path).name)
        self._record["artifacts"].append(Path(path).name)

    def close(self) -> None:
        """Grava ``run.json``."""
        content = json.dumps(self._record, indent=2, ensure_ascii=False)
        (self.run_dir / "run.json").write_text(content, encoding="utf-8")


class MlflowTracker:
    """Adaptador fino sobre a API de *fluent tracking* do MLflow.

    Args:
        tracking_uri: URI do servidor ou do backend de armazenamento.
        experiment_name: Experimento no qual a execução será criada.
    """

    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        try:
            import mlflow
        except ImportError as error:  # pragma: no cover - depende do ambiente
            raise ImportError("MLflow não está instalado. Use `uv sync --extra mlflow`.") from error
        self._mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run = mlflow.start_run()

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Registra parâmetros na execução ativa."""
        self._mlflow.log_params({key: _jsonable(value) for key, value in params.items()})

    def log_metrics(self, metrics: Mapping[str, float], prefix: str = "") -> None:
        """Registra métricas na execução ativa."""
        self._mlflow.log_metrics({f"{prefix}{k}": float(v) for k, v in metrics.items()})

    def log_artifact(self, path: Path) -> None:
        """Envia o arquivo para o *artifact store* do MLflow."""
        self._mlflow.log_artifact(str(path))

    def close(self) -> None:
        """Encerra a execução."""
        self._mlflow.end_run()


def create_tracker(config: TrackingConfig) -> ExperimentTracker:
    """Factory: devolve o tracker definido em ``config.backend``."""
    if config.backend == "mlflow":
        return MlflowTracker(config.uri, config.experiment_name)
    return LocalJsonTracker(config.uri, config.experiment_name)


def _jsonable(value: Any) -> Any:
    """Mantém tipos JSON nativos e converte o resto (ex.: ``Path``) para ``str``."""
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)
