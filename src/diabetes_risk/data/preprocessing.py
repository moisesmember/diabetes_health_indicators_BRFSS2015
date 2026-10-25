"""Validação, limpeza e divisão dos dados brutos.

Funções puras (DataFrame entra, DataFrame sai): fáceis de testar e de compor
na etapa ``prepare`` do pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import NamedTuple

import pandas as pd
from sklearn.model_selection import train_test_split

from diabetes_risk.domain.schemas import (
    BINARY_TARGET,
    FEATURE_COLUMNS,
    RAW_SCHEMA,
    TARGET_COLUMN,
)


class DataSplit(NamedTuple):
    """Conjuntos de treino e teste."""

    train: pd.DataFrame
    test: pd.DataFrame


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Valida o DataFrame bruto contra ``RAW_SCHEMA``.

    Todos os erros são coletados de uma vez (``lazy=True``), facilitando o
    diagnóstico de arquivos corrompidos.

    Raises:
        pandera.errors.SchemaErrors: Se alguma coluna violar o contrato.
    """
    return RAW_SCHEMA.validate(df, lazy=True)


def build_binary_target(df: pd.DataFrame, positive_classes: Iterable[int]) -> pd.DataFrame:
    """Substitui o alvo de 3 classes por um alvo binário.

    Args:
        df: Dados validados contendo ``Diabetes_012``.
        positive_classes: Classes originais tratadas como positivas (1).

    Returns:
        Novo DataFrame com as features e a coluna ``target`` (int 0/1).
    """
    target = df[TARGET_COLUMN].isin(list(positive_classes)).astype(int)
    return df[list(FEATURE_COLUMNS)].assign(**{BINARY_TARGET: target})


def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas idênticas.

    Linhas repetidas que caíssem em treino *e* teste inflariam artificialmente
    as métricas (vazamento), por isso a deduplicação acontece antes do split.
    """
    return df.drop_duplicates(ignore_index=True)


def split_train_test(df: pd.DataFrame, test_size: float, random_state: int) -> DataSplit:
    """Divide os dados de forma estratificada pelo alvo binário.

    A estratificação mantém a proporção de positivos (~15%) igual em treino e
    teste, o que é essencial com classes desbalanceadas.
    """
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[BINARY_TARGET],
    )
    return DataSplit(train.reset_index(drop=True), test.reset_index(drop=True))


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa a matriz de features (X) do alvo binário (y)."""
    return df[list(FEATURE_COLUMNS)], df[BINARY_TARGET]
