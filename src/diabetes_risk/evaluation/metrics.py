"""Métricas de classificação, escolha de limiar e quality gate.

Com ~15% de positivos, acurácia engana (prever "sempre negativo" acerta 85%).
Por isso o foco é em métricas que olham para a classe positiva: PR-AUC, recall
e F-beta, além de ROC-AUC e Brier score (calibração).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import NamedTuple

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
        Dicionário ``{nome_da_métrica: valor}`` com três grupos:

        * ranking (independem do limiar): ``roc_auc``, ``pr_auc`` e
          ``pr_auc_lift`` (quantas vezes melhor que o acaso);
        * calibração: ``brier`` e ``brier_skill`` (ganho sobre prever sempre a
          prevalência; negativo indica probabilidades descalibradas);
        * decisão (dependem do limiar): ``precision``, ``recall``, ``f{beta}``
          e as métricas de :func:`confusion_metrics`.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    prevalence = float(y_true.mean())
    pr_auc = float(average_precision_score(y_true, y_proba))
    brier = float(brier_score_loss(y_true, y_proba))
    brier_baseline = prevalence * (1 - prevalence)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": pr_auc,
        "pr_auc_lift": pr_auc / prevalence if prevalence else 0.0,
        "brier": brier,
        "brier_skill": 1 - brier / brier_baseline if brier_baseline else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        f"f{beta:g}": float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0)),
        **confusion_metrics(y_true, y_pred),
        "prevalence": prevalence,
        "threshold": float(threshold),
    }


def confusion_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Métricas da matriz de confusão, incluindo taxas por 1.000 pacientes.

    As taxas por 1.000 traduzem o modelo para a operação: "a cada 1.000
    pessoas triadas, quantos alertas, quantos casos perdidos".

    Returns:
        ``tn``, ``fp``, ``fn``, ``tp`` (contagens), ``specificity``, ``npv``
        (valor preditivo negativo), ``alerts_per_1000``,
        ``missed_cases_per_1000`` e ``false_alarms_per_1000``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    per_1000 = 1000 / max(len(y_true), 1)
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "npv": tn / (tn + fn) if tn + fn else 0.0,
        "alerts_per_1000": (tp + fp) * per_1000,
        "missed_cases_per_1000": fn * per_1000,
        "false_alarms_per_1000": fp * per_1000,
    }


def find_best_threshold(y_true: ArrayLike, y_proba: ArrayLike, beta: float = 2.0) -> float:
    """Encontra o limiar que maximiza a F-beta.

    Deve ser chamado com predições *out-of-fold* (nunca com dados de teste),
    para que a escolha do limiar não vaze informação para a avaliação final.

    Returns:
        O limiar ótimo, em ``[0, 1]``.
    """
    curve = fbeta_curve(y_true, y_proba, beta)
    return float(curve.thresholds[int(np.argmax(curve.fbeta))])


class FBetaCurve(NamedTuple):
    """Precisão, recall e F-beta para cada limiar candidato (arrays alinhados)."""

    thresholds: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    fbeta: np.ndarray


def fbeta_curve(y_true: ArrayLike, y_proba: ArrayLike, beta: float = 2.0) -> FBetaCurve:
    """Calcula precisão, recall e F-beta em todos os limiares possíveis."""
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
    return FBetaCurve(thresholds, precision, recall, fbeta)


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
