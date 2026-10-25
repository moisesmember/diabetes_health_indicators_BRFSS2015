"""Etapa ``evaluate``: métricas no teste hold-out, relatórios e Model Registry."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from diabetes_risk.config import PipelineConfig
from diabetes_risk.data.preprocessing import split_features_target
from diabetes_risk.domain.protocols import ExperimentTracker
from diabetes_risk.evaluation import plots
from diabetes_risk.evaluation.importance import permutation_feature_importance
from diabetes_risk.evaluation.metrics import (
    GateResult,
    check_quality_gate,
    compute_metrics,
    risk_band_metrics,
)
from diabetes_risk.evaluation.report import build_report
from diabetes_risk.models.bundle import ModelBundle
from diabetes_risk.tracking.lineage import git_commit
from diabetes_risk.tracking.trackers import create_tracker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationReport:
    """Resultado da avaliação.

    Attributes:
        metrics: Métricas no conjunto de teste.
        gate: Resultado do quality gate.
        model_version: Versão criada no Model Registry (``None`` sem registry).
    """

    metrics: dict[str, float]
    gate: GateResult
    model_version: str | None = None


def run_evaluate(
    config: PipelineConfig, tracker: ExperimentTracker | None = None
) -> EvaluationReport:
    """Avalia o modelo salvo no conjunto de teste e publica os resultados.

    Gera em ``config.artifacts.report_dir``:

    * ``metrics.json``: métricas planas (lidas pelo DVC);
    * ``metrics_report.md``: as mesmas métricas explicadas em linguagem natural;
    * ``feature_importance.csv`` e ``figures/*.png``.

    Tudo é registrado no mesmo run do treino (métricas ``test/*``, artefatos em
    ``evaluation/`` e o relatório como descrição do run). Por fim, cria uma
    versão no Model Registry; se o quality gate passar, ela recebe o alias
    ``champion``.

    Args:
        config: Configuração do pipeline.
        tracker: Destino do registro; por padrão, retoma o run do treino.

    Returns:
        Métricas, gate e versão registrada. A decisão de falhar o processo
        (ex.: no CI) fica com quem chama, como a CLI.
    """
    bundle = ModelBundle.load(config.artifacts.model_path)
    test_df = pd.read_parquet(config.data.test_path)
    X, y = split_features_target(test_df)
    beta = config.training.threshold_beta

    owns_tracker = tracker is None
    tracker = tracker or create_tracker(config.tracking, run_id=bundle.run_id)
    try:
        proba = bundle.predict_proba(X)
        metrics = {
            **compute_metrics(y, proba, bundle.threshold, beta=beta),
            **risk_band_metrics(y, proba, bundle.bands),
        }
        gate = check_quality_gate(metrics, config.evaluation.min_metrics)
        report = build_report(metrics, gate, beta=beta, high_risk_threshold=bundle.bands.high)
        importance = permutation_feature_importance(
            bundle.pipeline, X, y, random_state=config.data.random_state
        )

        figures = {
            "confusion_matrix": plots.plot_confusion_matrix(y, proba >= bundle.threshold),
            "roc_curve": plots.plot_roc_curve(y, proba, bundle.threshold),
            "precision_recall_curve": plots.plot_precision_recall_curve(y, proba, bundle.threshold),
            "threshold_tradeoff": plots.plot_threshold_tradeoff(y, proba, bundle.threshold, beta),
            "calibration_curve": plots.plot_calibration_curve(y, proba),
            "feature_importance": plots.plot_feature_importance(importance),
        }
        report_files = _write_reports(config, bundle, metrics, gate, report, importance)
        figure_files = _save_figures(figures, config.artifacts.figures_dir)

        tracker.log_dataset(
            test_df, name="brfss2015-test", context="evaluation", source=str(config.data.test_path)
        )
        tracker.log_metrics(metrics, prefix="test/")
        for path in report_files:
            tracker.log_artifact(path, artifact_dir="evaluation")
        for path in figure_files:
            tracker.log_artifact(path, artifact_dir="evaluation/figures")
        tracker.set_description(report)
        tracker.set_tags({"gate_passed": str(gate.passed).lower()})

        model_version = tracker.register_model(
            config.tracking.registered_model_name,
            alias=config.tracking.champion_alias if gate.passed else None,
            tags={
                "gate_passed": str(gate.passed).lower(),
                "model_name": bundle.model_name,
                "git_commit": git_commit(),
                "threshold": f"{bundle.threshold:.4f}",
                "test_pr_auc": f"{metrics['pr_auc']:.4f}",
                "test_recall": f"{metrics['recall']:.4f}",
            },
        )

        logger.info(
            "Teste: roc_auc=%.4f, pr_auc=%.4f, recall=%.4f, precision=%.4f",
            metrics["roc_auc"],
            metrics["pr_auc"],
            metrics["recall"],
            metrics["precision"],
        )
        if gate.passed:
            logger.info(
                "Quality gate APROVADO (versão %s -> @%s)",
                model_version,
                config.tracking.champion_alias,
            )
        else:
            logger.warning("Quality gate REPROVADO: %s", "; ".join(gate.failures))
        return EvaluationReport(metrics=metrics, gate=gate, model_version=model_version)
    finally:
        if owns_tracker:
            tracker.close()


def _write_reports(
    config: PipelineConfig,
    bundle: ModelBundle,
    metrics: dict[str, float],
    gate: GateResult,
    report: str,
    importance: pd.DataFrame,
) -> list[Path]:
    """Grava métricas (JSON), relatório (Markdown) e importância (CSV)."""
    report_dir = config.artifacts.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.artifacts.metrics_path
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"model": bundle.model_name, "gate_passed": gate.passed, **metrics}
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path = report_dir / "metrics_report.md"
    report_path.write_text(report, encoding="utf-8")
    importance_path = report_dir / "feature_importance.csv"
    importance.to_csv(importance_path, index=False)
    return [metrics_path, report_path, importance_path]


def _save_figures(figures: dict[str, Figure], directory: Path) -> list[Path]:
    """Salva cada figura como ``<nome>.png``."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, figure in figures.items():
        path = directory / f"{name}.png"
        figure.savefig(path, dpi=110)
        paths.append(path)
    return paths
