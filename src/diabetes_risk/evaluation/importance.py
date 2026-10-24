"""Importância das features por permutação.

Mede quanto o PR-AUC cai quando os valores de uma feature são embaralhados.
Diferente da importância nativa das árvores, funciona para qualquer modelo e é
calculada nas features *originais* (as que o usuário conhece), já que o
pipeline inteiro (engenharia + modelo) é avaliado como uma caixa.
"""

from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


def permutation_feature_importance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    max_rows: int = 10_000,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Calcula a importância por permutação usando PR-AUC como métrica.

    Args:
        pipeline: Pipeline treinado.
        X: Features de avaliação (normalmente o teste).
        y: Alvo correspondente.
        max_rows: Amostra estratificada máxima, para limitar o custo.
        n_repeats: Repetições por feature (gera o desvio padrão).
        random_state: Semente de amostragem e embaralhamento.

    Returns:
        DataFrame com ``feature``, ``importance_mean`` e ``importance_std``,
        ordenado da mais para a menos importante.
    """
    if len(X) > max_rows:
        sample = (
            X.assign(_y=y)
            .groupby("_y", group_keys=False)
            .sample(frac=max_rows / len(X), random_state=random_state)
        )
        X, y = sample.drop(columns="_y"), sample["_y"]

    result = permutation_importance(
        pipeline,
        X,
        y,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=random_state,
    )
    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return importance.sort_values("importance_mean", ascending=False, ignore_index=True)
