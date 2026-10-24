"""Etapa ``train``: validação cruzada, escolha do limiar e treino final."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from diabetes_risk.config import PipelineConfig
from diabetes_risk.data.preprocessing import split_features_target
from diabetes_risk.domain.protocols import ExperimentTracker
from diabetes_risk.evaluation.metrics import compute_metrics, find_best_threshold
from diabetes_risk.models.bundle import ModelBundle
from diabetes_risk.models.registry import build_pipeline
from diabetes_risk.tracking.lineage import git_commit
from diabetes_risk.tracking.trackers import create_tracker

logger = logging.getLogger(__name__)


def run_train(config: PipelineConfig, tracker: ExperimentTracker | None = None) -> ModelBundle:
    """Treina o modelo e salva o :class:`ModelBundle`.

    1. Gera probabilidades *out-of-fold* com validação cruzada estratificada.
    2. Escolhe o limiar que maximiza a F-beta nessas probabilidades.
    3. Reajusta o pipeline em todo o treino e salva modelo + limiar.

    No tracker ficam: tags de linhagem, parâmetros, o dataset de treino, as
    métricas ``cv/*`` (incluindo o desvio entre folds) e o modelo pyfunc. O
    ``run_id`` vai para o bundle, para que ``evaluate`` continue o mesmo run.
    O conjunto de teste não é tocado aqui.

    Args:
        config: Configuração do pipeline.
        tracker: Destino do registro; por padrão, o de ``config.tracking``.

    Returns:
        O bundle treinado (também gravado em ``config.artifacts.model_path``).
    """
    model_cfg, train_cfg, data_cfg = config.model, config.training, config.data
    train_df = pd.read_parquet(data_cfg.train_path)
    X, y = split_features_target(train_df)

    owns_tracker = tracker is None
    run_name = f"{model_cfg.name}-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    tracker = tracker or create_tracker(config.tracking, run_name=run_name)
    try:
        tracker.set_tags(
            {
                "model_name": model_cfg.name,
                "git_commit": git_commit(),
                "task": "binary-classification",
                "target": "pré-diabetes ou diabetes (Diabetes_012 ∈ positive_classes)",
            }
        )
        tracker.log_params(
            {
                "model.name": model_cfg.name,
                **{f"model.{key}": value for key, value in model_cfg.params.items()},
                "training.cv_folds": train_cfg.cv_folds,
                "training.threshold_beta": train_cfg.threshold_beta,
                "data.positive_classes": data_cfg.positive_classes,
                "data.drop_duplicates": data_cfg.drop_duplicates,
                "data.test_size": data_cfg.test_size,
                "data.random_state": data_cfg.random_state,
                "data.n_train_rows": len(X),
            }
        )
        tracker.log_dataset(
            train_df, name="brfss2015-train", context="training", source=str(data_cfg.train_path)
        )

        cv = StratifiedKFold(
            n_splits=train_cfg.cv_folds, shuffle=True, random_state=data_cfg.random_state
        )
        folds = list(cv.split(X, y))
        oof_proba = cross_val_predict(
            build_pipeline(model_cfg.name, model_cfg.params), X, y, cv=folds, method="predict_proba"
        )[:, 1]

        threshold = find_best_threshold(y, oof_proba, beta=train_cfg.threshold_beta)
        cv_metrics = {
            **compute_metrics(y, oof_proba, threshold, beta=train_cfg.threshold_beta),
            **_fold_dispersion(y.to_numpy(), oof_proba, folds),
        }
        tracker.log_metrics(cv_metrics, prefix="cv/")
        logger.info("CV (%d folds): %s", train_cfg.cv_folds, _fmt(cv_metrics))

        pipeline = build_pipeline(model_cfg.name, model_cfg.params).fit(X, y)
        bundle = ModelBundle(
            pipeline=pipeline,
            threshold=threshold,
            model_name=model_cfg.name,
            metadata={
                "params": model_cfg.params,
                "cv_metrics": cv_metrics,
                "n_train": len(X),
                "run_id": tracker.run_id,
            },
        )
        model_path = bundle.save(config.artifacts.model_path)
        tracker.log_model(bundle, input_example=X.head(5))
        logger.info(
            "Modelo salvo em %s (limiar=%.4f, run=%s)", model_path, threshold, bundle.run_id
        )
        return bundle
    finally:
        if owns_tracker:
            tracker.close()


def _fold_dispersion(
    y: np.ndarray, proba: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]
) -> dict[str, float]:
    """Desvio padrão de ROC-AUC e PR-AUC entre folds (estabilidade do modelo)."""
    roc = [roc_auc_score(y[val], proba[val]) for _, val in folds]
    pr = [average_precision_score(y[val], proba[val]) for _, val in folds]
    return {"roc_auc_std": float(np.std(roc)), "pr_auc_std": float(np.std(pr))}


def _fmt(metrics: dict[str, float]) -> str:
    keys = ("roc_auc", "pr_auc", "recall", "precision", "threshold")
    return ", ".join(f"{key}={metrics[key]:.4f}" for key in keys)
