"""Contratos de dados do domínio.

Este módulo é a fonte da verdade sobre as colunas do BRFSS 2015 e seus
domínios de valores. Ele define:

* ``RAW_SCHEMA``: contrato Pandera do CSV bruto (validado na ingestão);
* ``FEATURE_RANGES``: faixas válidas, reutilizadas pelo contrato da API
  (:mod:`diabetes_risk.serving.schemas`).
"""

from __future__ import annotations

from typing import Final

import pandera.pandas as pa

TARGET_COLUMN: Final = "Diabetes_012"
"""Coluna alvo original: 0 = sem diabetes, 1 = pré-diabetes, 2 = diabetes."""

BINARY_TARGET: Final = "target"
"""Coluna alvo binária criada na preparação (1 = pré-diabetes ou diabetes)."""

BINARY_FEATURES: Final[tuple[str, ...]] = (
    "HighBP",
    "HighChol",
    "CholCheck",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "DiffWalk",
    "Sex",
)
"""Indicadores 0/1 (ex.: ``HighBP`` = 1 se tem pressão alta)."""

ORDINAL_FEATURES: Final[tuple[str, ...]] = ("GenHlth", "Age", "Education", "Income")
"""Escalas ordenadas (ex.: ``Age`` em 13 faixas de 5 anos)."""

NUMERIC_FEATURES: Final[tuple[str, ...]] = ("BMI", "MentHlth", "PhysHlth")
"""Valores contínuos/contagens (``MentHlth``/``PhysHlth`` = dias ruins nos últimos 30)."""

FEATURE_COLUMNS: Final[tuple[str, ...]] = BINARY_FEATURES + ORDINAL_FEATURES + NUMERIC_FEATURES

FEATURE_RANGES: Final[dict[str, tuple[int, int]]] = {
    **dict.fromkeys(BINARY_FEATURES, (0, 1)),
    "GenHlth": (1, 5),
    "Age": (1, 13),
    "Education": (1, 6),
    "Income": (1, 8),
    "BMI": (12, 98),
    "MentHlth": (0, 30),
    "PhysHlth": (0, 30),
}
"""Faixa válida (inclusiva) de cada feature, conforme o codebook do BRFSS."""


def _range_column(low: int, high: int) -> pa.Column:
    return pa.Column(float, pa.Check.in_range(low, high), nullable=False, coerce=True)


RAW_SCHEMA: Final = pa.DataFrameSchema(
    {
        TARGET_COLUMN: pa.Column(float, pa.Check.isin([0, 1, 2]), nullable=False, coerce=True),
        **{name: _range_column(*bounds) for name, bounds in FEATURE_RANGES.items()},
    },
    strict=True,
    name="brfss2015_raw",
)
"""Contrato do CSV bruto: todas as colunas presentes, sem nulos e dentro da faixa."""
