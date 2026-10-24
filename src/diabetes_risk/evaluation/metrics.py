"""Métricas de classificação, escolha de limiar e quality gate.

Com ~15% de positivos, acurácia engana (prever "sempre negativo" acerta 85%).
Por isso o foco é em métricas que olham para a classe positiva: PR-AUC, recall
e F-beta, além de ROC-AUC e Brier score (calibração).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: ArrayLike, y_proba: ArrayLike, threshold: float = 0.5, beta: float = 2.0
) -> dict[str, float]:
    """Calcula as métricas de avaliação.

    Args:
        y_true: Alvo real (0/1).
        y_proba: Probabilidade prevista da classe positiva.
        threshold: Limiar que converte probabilidade em classe.
        beta: Peso do recall na F-beta.

    Returns:
        Dicionário ``{nome_da_métrica: valor}``. As métricas de ranking
        (``roc_auc``, ``pr_auc``, ``brier``) independem do limiar.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        f"f{beta:g}": float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0)),
        "threshold": float(threshold),
    }


def find_best_threshold(y_true: ArrayLike, y_proba: ArrayLike, beta: float = 2.0) -> float:
    """Encontra o limiar que maximiza a F-beta.

    Deve ser chamado com predições *out-of-fold* (nunca com dados de teste),
    para que a escolha do limiar não vaze informação para a avaliação final.

    Returns:
        O limiar ótimo, em ``[0, 1]``.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # A curva tem um ponto a mais que `thresholds`; o último (recall=0) é descartado.
    precision, recall = precision[:-1], recall[:-1]
    beta2 = beta**2
    denominator = beta2 * precision + recall
    fbeta = np.divide(
        (1 + beta2) * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(thresholds[int(np.argmax(fbeta))])


@dataclass(frozen=True)
class GateResult:
    """Resultado do quality gate.

    Attributes:
        passed: ``True`` se todas as métricas atingiram o mínimo.
        failures: Mensagens descrevendo cada métrica reprovada.
    """

    passed: bool
    failures: list[str] = field(default_factory=list)


def check_quality_gate(
    metrics: Mapping[str, float], min_metrics: Mapping[str, float]
) -> GateResult:
    """Compara métricas obtidas com os mínimos exigidos.

    Uma métrica exigida que não foi calculada conta como reprovação, para que
    um erro de digitação na configuração não aprove um modelo por omissão.
    """
    failures: list[str] = []
    for name, minimum in min_metrics.items():
        value = metrics.get(name)
        if value is None:
            failures.append(f"{name}: métrica não calculada")
        elif value < minimum:
            failures.append(f"{name}: {value:.4f} < mínimo {minimum:.4f}")
    return GateResult(passed=not failures, failures=failures)
