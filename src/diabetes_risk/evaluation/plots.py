"""Gráficos de avaliação do classificador.

Cada função recebe dados já calculados e devolve uma ``Figure`` do matplotlib,
sem tocar em estado global (``pyplot``) nem em disco. Quem chama decide onde
salvar ou registrar (MLflow, DVC plots etc.).

Todos os gráficos marcam o **ponto de operação** (limiar escolhido), para que
fique claro onde o modelo está trabalhando na curva.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from diabetes_risk.evaluation.metrics import fbeta_curve


def _mark_operating_point(ax: Axes, x: float, y: float) -> None:
    """Destaca o ponto da curva correspondente ao limiar escolhido."""
    ax.scatter([x], [y], color="crimson", zorder=5, label="ponto de operação")


def plot_confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike) -> Figure:
    """Matriz de confusão com contagens e % por linha (classe real)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    counts = np.array([[np.sum((y_true == r) & (y_pred == c)) for c in (0, 1)] for r in (0, 1)])
    row_pct = counts / counts.sum(axis=1, keepdims=True)

    fig = Figure(figsize=(5, 4.5), layout="constrained")
    ax = fig.subplots()
    ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=1)
    labels = ["sem risco", "com risco"]
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Previsto",
        ylabel="Real",
        title="Matriz de confusão (teste)",
    )
    for r in (0, 1):
        for c in (0, 1):
            color = "white" if row_pct[r, c] > 0.5 else "black"
            ax.text(
                c,
                r,
                f"{counts[r, c]:,}\n({row_pct[r, c]:.1%})",
                ha="center",
                va="center",
                color=color,
            )
    return fig


def plot_roc_curve(y_true: ArrayLike, y_proba: ArrayLike, threshold: float) -> Figure:
    """Curva ROC com a diagonal do acaso e o ponto de operação."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    idx = int(np.argmin(np.abs(thresholds - threshold)))

    fig = Figure(figsize=(5, 4.5), layout="constrained")
    ax = fig.subplots()
    ax.plot(fpr, tpr, label=f"modelo (AUC = {roc_auc_score(y_true, y_proba):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="acaso (AUC = 0,5)")
    _mark_operating_point(ax, fpr[idx], tpr[idx])
    ax.set(
        xlabel="Taxa de falsos positivos (1 − especificidade)",
        ylabel="Taxa de verdadeiros positivos (recall)",
        title="Curva ROC",
    )
    ax.legend(loc="lower right")
    return fig


def plot_precision_recall_curve(y_true: ArrayLike, y_proba: ArrayLike, threshold: float) -> Figure:
    """Curva precisão-recall com a linha de base (prevalência) e o ponto de operação."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    idx = int(np.argmin(np.abs(thresholds - threshold)))
    prevalence = float(np.asarray(y_true).mean())

    fig = Figure(figsize=(5, 4.5), layout="constrained")
    ax = fig.subplots()
    ax.plot(
        recall, precision, label=f"modelo (AP = {average_precision_score(y_true, y_proba):.3f})"
    )
    ax.axhline(prevalence, linestyle="--", color="gray", label=f"acaso ({prevalence:.3f})")
    _mark_operating_point(ax, recall[idx], precision[idx])
    ax.set(xlabel="Recall", ylabel="Precisão", title="Curva precisão-recall", ylim=(0, 1.02))
    ax.legend(loc="upper right")
    return fig


def plot_threshold_tradeoff(
    y_true: ArrayLike, y_proba: ArrayLike, threshold: float, beta: float = 2.0
) -> Figure:
    """Precisão, recall e F-beta em função do limiar, com o limiar escolhido."""
    curve = fbeta_curve(y_true, y_proba, beta)

    fig = Figure(figsize=(6, 4.5), layout="constrained")
    ax = fig.subplots()
    ax.plot(curve.thresholds, curve.precision, label="precisão")
    ax.plot(curve.thresholds, curve.recall, label="recall")
    ax.plot(curve.thresholds, curve.fbeta, label=f"F{beta:g}")
    ax.axvline(threshold, color="crimson", linestyle="--", label=f"limiar = {threshold:.3f}")
    ax.set(
        xlabel="Limiar de decisão",
        ylabel="Valor",
        title="Trade-off por limiar",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)
    return fig


def plot_calibration_curve(y_true: ArrayLike, y_proba: ArrayLike, n_bins: int = 10) -> Figure:
    """Diagrama de confiabilidade: probabilidade prevista vs. frequência observada."""
    observed, predicted = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")

    fig = Figure(figsize=(5, 4.5), layout="constrained")
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="calibração perfeita")
    ax.plot(predicted, observed, marker="o", label="modelo")
    ax.set(
        xlabel="Probabilidade prevista (média do bin)",
        ylabel="Fração real de positivos",
        title="Curva de calibração",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(loc="upper left")
    return fig


def plot_feature_importance(importance: pd.DataFrame, top: int = 15) -> Figure:
    """Barras horizontais da importância por permutação (com desvio padrão).

    Args:
        importance: DataFrame com colunas ``feature``, ``importance_mean`` e
            ``importance_std``.
        top: Quantas features exibir.
    """
    data = importance.nlargest(top, "importance_mean").iloc[::-1]

    fig = Figure(figsize=(6, 0.35 * len(data) + 1.5), layout="constrained")
    ax = fig.subplots()
    ax.barh(
        data["feature"], data["importance_mean"], xerr=data["importance_std"], color="steelblue"
    )
    ax.set(
        xlabel="Queda no PR-AUC ao embaralhar a feature",
        title="Importância das features (permutação)",
    )
    return fig
