"""Detecção de *data drift* com o Population Stability Index (PSI).

O PSI compara a distribuição de uma feature em produção (``current``) com a
do treino (``reference``)::

    PSI = Σ (atual% - referência%) * ln(atual% / referência%)

Regra prática usada no mercado:

* PSI < 0,10: estável;
* 0,10 a 0,25: mudança moderada, vale investigar;
* PSI > 0,25: mudança significativa, considere retreinar.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

import numpy as np
import pandas as pd

DRIFT_THRESHOLD: Final = 0.25
_EPSILON: Final = 1e-6  # evita log(0) e divisão por zero em bins vazios


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Calcula o PSI entre duas amostras de uma mesma feature.

    Features com poucos valores distintos (binárias, ordinais) usam cada valor
    como uma categoria; contínuas usam quantis da referência como bins.

    Args:
        reference: Valores observados no treino.
        current: Valores observados em produção.
        bins: Número máximo de bins para features contínuas.

    Returns:
        O PSI (0 = distribuições idênticas).
    """
    if reference.nunique() <= bins:
        categories = np.union1d(reference.unique(), current.unique())
        ref_dist = reference.value_counts(normalize=True).reindex(categories, fill_value=0)
        cur_dist = current.value_counts(normalize=True).reindex(categories, fill_value=0)
        ref_pct, cur_pct = ref_dist.to_numpy(), cur_dist.to_numpy()
    else:
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        ref_pct = np.histogram(reference, edges)[0] / len(reference)
        cur_pct = np.histogram(current, edges)[0] / len(current)

    ref_pct = np.clip(ref_pct, _EPSILON, None)
    cur_pct = np.clip(cur_pct, _EPSILON, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: Iterable[str],
    threshold: float = DRIFT_THRESHOLD,
) -> pd.DataFrame:
    """Calcula o PSI de cada coluna e sinaliza as que derivaram.

    Returns:
        DataFrame com colunas ``feature``, ``psi`` e ``drift`` (bool),
        ordenado do maior para o menor PSI.
    """
    rows = [
        {"feature": column, "psi": population_stability_index(reference[column], current[column])}
        for column in columns
    ]
    report = pd.DataFrame(rows, columns=["feature", "psi"])
    report["drift"] = report["psi"] > threshold
    return report.sort_values("psi", ascending=False, ignore_index=True)
