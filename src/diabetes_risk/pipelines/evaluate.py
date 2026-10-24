"""Etapa ``evaluate``: métricas no teste hold-out e quality gate."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import pandas as pd

from diabetes_risk.config import PipelineConfig
from diabetes_risk.data.preprocessing import split_features_target
from diabetes_risk.domain.protocols import ExperimentTracker
from diabetes_risk.evaluation.metrics import GateResult, check_quality_gate, compute_metrics
from diabetes_risk.models.bundle import ModelBundle
from diabetes_risk.tracking.trackers import create_tracker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationReport:
    """Métricas no teste e resultado do quality gate."""

    metrics: dict[str, float]
    gate: GateResult


def run_evaluate(
    config: PipelineConfig, tracker: ExperimentTracker | None = None
) -> EvaluationReport:
    """Avalia o modelo salvo no conjunto de teste.

    O relatório é gravado em ``config.artifacts.metrics_path`` (JSON), que o
    DVC usa para comparar experimentos (``dvc metrics diff``).

    Args:
        config: Configuração do pipeline.
        tracker: Destino das métricas; por padrão, o de ``config.tracking``.

    Returns:
        Métricas e resultado do quality gate. A decisão de falhar o processo
        (ex.: no CI) fica com quem chama, como a CLI.
    """
    owns_tracker = tracker is None
    tracker = tracker or create_tracker(config.tracking)
    try:
        bundle = ModelBundle.load(config.artifacts.model_path)
        X, y = split_features_target(pd.read_parquet(config.data.test_path))

        metrics = compute_metrics(
            y, bundle.predict_proba(X), bundle.threshold, beta=config.training.threshold_beta
        )
        gate = check_quality_gate(metrics, config.evaluation.min_metrics)

        metrics_path = config.artifacts.metrics_path
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": bundle.model_name, "gate_passed": gate.passed, **metrics}
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        tracker.log_metrics(metrics, prefix="test_")
        tracker.log_artifact(metrics_path)

        logger.info("Teste: %s", ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        if gate.passed:
            logger.info("Quality gate APROVADO")
        else:
            logger.warning("Quality gate REPROVADO: %s", "; ".join(gate.failures))
        return EvaluationReport(metrics=metrics, gate=gate)
    finally:
        if owns_tracker:
            tracker.close()
