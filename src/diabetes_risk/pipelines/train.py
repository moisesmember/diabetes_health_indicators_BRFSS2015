"""Etapa ``train``: validação cruzada, escolha do limiar e treino final."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from diabetes_risk.config import PipelineConfig
from diabetes_risk.data.preprocessing import split_features_target
from diabetes_risk.domain.protocols import ExperimentTracker
from diabetes_risk.evaluation.metrics import compute_metrics, find_best_threshold
from diabetes_risk.models.bundle import ModelBundle
from diabetes_risk.models.registry import build_pipeline
from diabetes_risk.tracking.trackers import create_tracker

logger = logging.getLogger(__name__)


def run_train(config: PipelineConfig, tracker: ExperimentTracker | None = None) -> ModelBundle:
    """Treina o modelo e salva o :class:`ModelBundle`.

    1. Gera probabilidades *out-of-fold* com validação cruzada estratificada.
    2. Escolhe o limiar que maximiza a F-beta nessas probabilidades.
    3. Reajusta o pipeline em todo o treino e salva modelo + limiar.

    O conjunto de teste não é tocado aqui; ele fica reservado para ``evaluate``.

    Args:
        config: Configuração do pipeline.
        tracker: Destino de parâmetros/métricas; por padrão, o de ``config.tracking``.

    Returns:
        O bundle treinado (também gravado em ``config.artifacts.model_path``).
    """
    owns_tracker = tracker is None
    tracker = tracker or create_tracker(config.tracking)
    try:
        X, y = split_features_target(pd.read_parquet(config.data.train_path))
        model_cfg, train_cfg = config.model, config.training

        cv = StratifiedKFold(
            n_splits=train_cfg.cv_folds, shuffle=True, random_state=config.data.random_state
        )
        oof_proba = cross_val_predict(
            build_pipeline(model_cfg.name, model_cfg.params), X, y, cv=cv, method="predict_proba"
        )[:, 1]

        threshold = find_best_threshold(y, oof_proba, beta=train_cfg.threshold_beta)
        cv_metrics = compute_metrics(y, oof_proba, threshold, beta=train_cfg.threshold_beta)
        logger.info("CV (%d folds): %s", train_cfg.cv_folds, _fmt(cv_metrics))

        pipeline = build_pipeline(model_cfg.name, model_cfg.params).fit(X, y)
        bundle = ModelBundle(
            pipeline=pipeline,
            threshold=threshold,
            model_name=model_cfg.name,
            metadata={"params": model_cfg.params, "cv_metrics": cv_metrics, "n_train": len(X)},
        )
        model_path = bundle.save(config.artifacts.model_path)
        logger.info("Modelo salvo em %s (limiar=%.4f)", model_path, threshold)

        tracker.log_params({"model": model_cfg.name, **model_cfg.params, **train_cfg.model_dump()})
        tracker.log_metrics(cv_metrics, prefix="cv_")
        tracker.log_artifact(model_path)
        return bundle
    finally:
        if owns_tracker:
            tracker.close()


def _fmt(metrics: dict[str, float]) -> str:
    return ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())
