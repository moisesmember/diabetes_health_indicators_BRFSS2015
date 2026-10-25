"""Implementações de :class:`~diabetes_risk.domain.protocols.ExperimentTracker`.

* :class:`MlflowTracker`: integração completa com MLflow (tracking, datasets,
  modelo pyfunc com assinatura e Model Registry com alias).
* :class:`LocalJsonTracker`: sem dependências, grava um ``run.json`` por run;
  útil em testes ou ambientes sem MLflow.

Use :func:`create_tracker` para obter a implementação configurada.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import warnings
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

import diabetes_risk
from diabetes_risk.config import TrackingConfig
from diabetes_risk.domain.protocols import ExperimentTracker

if TYPE_CHECKING:
    from diabetes_risk.models.bundle import ModelBundle

MODEL_ARTIFACT_NAME = "model"
_MODEL_ID_TAG = "logged_model_id"
_RUNTIME_PACKAGES = ("scikit-learn", "pandas", "numpy", "pandera", "pydantic", "pyyaml", "joblib")


class LocalJsonTracker:
    """Registra cada run em ``<root>/<experimento>/<run_id>/run.json``.

    Os artefatos são copiados para a pasta do run. Passar o ``run_id`` de um
    run existente o retoma, acumulando novos registros.

    Args:
        root: Diretório raiz dos experimentos.
        experiment_name: Nome do experimento (subpasta de ``root``).
        run_id: Run a retomar; por padrão, cria um novo com o timestamp UTC.
    """

    def __init__(self, root: str | Path, experiment_name: str, run_id: str | None = None) -> None:
        self._run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        self.run_dir = Path(root) / experiment_name / self._run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        record_path = self.run_dir / "run.json"
        self._record: dict[str, Any] = (
            json.loads(record_path.read_text(encoding="utf-8"))
            if record_path.exists()
            else {"tags": {}, "params": {}, "metrics": {}, "artifacts": [], "datasets": []}
        )

    @property
    def run_id(self) -> str:
        """Identificador do run (nome da pasta)."""
        return self._run_id

    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Acumula tags."""
        self._record["tags"].update(tags)

    def set_description(self, text: str) -> None:
        """Grava a descrição do run em ``description.md``."""
        (self.run_dir / "description.md").write_text(text, encoding="utf-8")

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Acumula parâmetros (valores não serializáveis viram texto)."""
        self._record["params"].update({key: _jsonable(value) for key, value in params.items()})

    def log_metrics(self, metrics: Mapping[str, float], prefix: str = "") -> None:
        """Acumula métricas, prefixando cada nome com ``prefix``."""
        self._record["metrics"].update({f"{prefix}{k}": float(v) for k, v in metrics.items()})

    def log_artifact(self, path: Path, artifact_dir: str | None = None) -> None:
        """Copia o arquivo para a pasta do run (ou uma subpasta dela)."""
        target_dir = self.run_dir / artifact_dir if artifact_dir else self.run_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target_dir / Path(path).name)
        self._record["artifacts"].append(str(Path(artifact_dir or "") / Path(path).name))

    def log_dataset(self, df: pd.DataFrame, name: str, context: str, source: str) -> None:
        """Registra nome, origem, dimensões e hash do conjunto de dados."""
        self._record["datasets"].append(
            {
                "name": name,
                "context": context,
                "source": source,
                "rows": len(df),
                "columns": df.shape[1],
                "digest": dataframe_digest(df),
            }
        )

    def log_model(self, bundle: ModelBundle, input_example: pd.DataFrame) -> None:
        """Registra os metadados do modelo (o arquivo já é salvo pelo pipeline)."""
        self._record["model"] = {"name": bundle.model_name, "threshold": bundle.threshold}

    def register_model(self, name: str, alias: str | None, tags: Mapping[str, str]) -> str | None:
        """Sem registry no backend local: não faz nada."""
        return None

    def close(self) -> None:
        """Grava ``run.json``."""
        content = json.dumps(self._record, indent=2, ensure_ascii=False)
        (self.run_dir / "run.json").write_text(content, encoding="utf-8")


class MlflowTracker:
    """Adaptador sobre a API do MLflow, seguindo as boas práticas da ferramenta.

    * métricas agrupadas por prefixo (``cv/``, ``test/``), que o MLflow UI
      exibe em seções;
    * datasets registrados com hash (aba *Datasets*), garantindo linhagem;
    * modelo ``pyfunc`` com assinatura, exemplo de entrada, dependências
      fixadas e o limiar de decisão embutido;
    * versões no Model Registry com tags e alias (``@champion``) em vez dos
      *stages*, que foram descontinuados.

    Args:
        tracking_uri: URI do servidor ou do backend (ex.: ``sqlite:///mlflow.db``).
        experiment_name: Experimento no qual o run será criado.
        run_id: Run a retomar (a avaliação retoma o run do treino).
        run_name: Nome legível de um run novo.
    """

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        run_id: str | None = None,
        run_name: str | None = None,
    ) -> None:
        try:
            import mlflow
            from mlflow import MlflowClient
        except ImportError as error:  # pragma: no cover - depende do ambiente
            raise ImportError("MLflow não está instalado. Use `uv sync --extra mlflow`.") from error
        self._mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        self._client = MlflowClient(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run = mlflow.start_run(run_id=run_id, run_name=None if run_id else run_name)

    @property
    def run_id(self) -> str:
        """Identificador do run ativo."""
        return str(self._run.info.run_id)

    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Anexa tags ao run."""
        self._mlflow.set_tags(dict(tags))

    def set_description(self, text: str) -> None:
        """Define a descrição (Markdown) exibida no topo do run no MLflow UI."""
        self._mlflow.set_tag("mlflow.note.content", text)

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Registra parâmetros no run ativo."""
        self._mlflow.log_params({key: _jsonable(value) for key, value in params.items()})

    def log_metrics(self, metrics: Mapping[str, float], prefix: str = "") -> None:
        """Registra métricas no run ativo.

        Se o run já tem um modelo registrado, as métricas também são ligadas a
        ele (aparecem na página do modelo no MLflow 3).
        """
        self._mlflow.log_metrics(
            {f"{prefix}{k}": float(v) for k, v in metrics.items()}, model_id=self._model_id()
        )

    def log_artifact(self, path: Path, artifact_dir: str | None = None) -> None:
        """Envia o arquivo para o *artifact store* do run."""
        self._mlflow.log_artifact(str(path), artifact_path=artifact_dir)

    def log_dataset(self, df: pd.DataFrame, name: str, context: str, source: str) -> None:
        """Registra o dataset (schema, perfil e hash) na aba *Datasets* do run."""
        from mlflow.data.pandas_dataset import from_pandas

        with warnings.catch_warnings():
            # Avisos informativos de inferência de schema (colunas int, fonte local).
            warnings.simplefilter("ignore", UserWarning)
            dataset = from_pandas(df, source=source, name=name, targets="target")
            self._mlflow.log_input(dataset, context=context)

    def log_model(self, bundle: ModelBundle, input_example: pd.DataFrame) -> None:
        """Registra o bundle como modelo pyfunc com assinatura e exemplo de entrada."""
        from mlflow.models import infer_signature

        from diabetes_risk.tracking import mlflow_model

        output_example = bundle.assessment_frame(input_example)
        with tempfile.TemporaryDirectory() as tmp:
            bundle_path = bundle.save(Path(tmp) / "model.joblib")
            model_info = self._mlflow.pyfunc.log_model(
                name=MODEL_ARTIFACT_NAME,
                python_model=mlflow_model.__file__,
                artifacts={mlflow_model.BUNDLE_ARTIFACT_KEY: str(bundle_path)},
                signature=infer_signature(input_example, output_example),
                input_example=input_example,
                code_paths=[str(Path(diabetes_risk.__file__).parent)],
                pip_requirements=[f"{pkg}=={version(pkg)}" for pkg in _RUNTIME_PACKAGES],
                metadata={"threshold": bundle.threshold, "estimator": bundle.model_name},
            )
        # Guardado no run para que a etapa evaluate (outro processo) encontre o modelo.
        self._mlflow.set_tag(_MODEL_ID_TAG, model_info.model_id)

    def register_model(self, name: str, alias: str | None, tags: Mapping[str, str]) -> str | None:
        """Cria uma versão no Model Registry e, se ``alias`` for dado, aponta-o para ela.

        Raises:
            RuntimeError: Se o run não tiver modelo (``log_model`` não foi chamado).
        """
        model_id = self._model_id()
        if model_id is None:
            raise RuntimeError(f"O run {self.run_id} não tem modelo registrado.")
        model_version = self._mlflow.register_model(f"models:/{model_id}", name, tags=dict(tags))
        if alias:
            self._client.set_registered_model_alias(name, alias, model_version.version)
        return str(model_version.version)

    def close(self) -> None:
        """Encerra o run."""
        self._mlflow.end_run()

    def _model_id(self) -> str | None:
        """Id do modelo logado neste run, lido da tag gravada por :meth:`log_model`."""
        tags = self._client.get_run(self.run_id).data.tags
        return tags.get(_MODEL_ID_TAG)


def create_tracker(
    config: TrackingConfig, run_id: str | None = None, run_name: str | None = None
) -> ExperimentTracker:
    """Devolve o tracker definido em ``config.backend``.

    Args:
        config: Configuração de tracking.
        run_id: Run existente a retomar.
        run_name: Nome de um run novo (ignorado no backend local).
    """
    if config.backend == "mlflow":
        return MlflowTracker(config.resolved_uri(), config.experiment_name, run_id, run_name)
    return LocalJsonTracker(config.uri, config.experiment_name, run_id)


def dataframe_digest(df: pd.DataFrame) -> str:
    """Hash curto e determinístico do conteúdo de um DataFrame."""
    row_hashes = pd.util.hash_pandas_object(df, index=False).to_numpy()
    return hashlib.md5(row_hashes.tobytes(), usedforsecurity=False).hexdigest()[:12]


def _jsonable(value: Any) -> Any:
    """Mantém tipos JSON nativos e converte o resto (ex.: ``Path``) para ``str``."""
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)
