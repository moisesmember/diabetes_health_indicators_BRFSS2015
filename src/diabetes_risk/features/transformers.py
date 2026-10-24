"""Engenharia de features e pré-processamento.

Tudo aqui segue o contrato ``fit``/``transform`` do scikit-learn e entra no
mesmo ``Pipeline`` do modelo. Com isso, a transformação aplicada no treino é
exatamente a aplicada na inferência (sem *training-serving skew*).
"""

from __future__ import annotations

from typing import Final, Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from diabetes_risk.domain.schemas import BINARY_FEATURES, NUMERIC_FEATURES, ORDINAL_FEATURES

COMORBIDITY_COLUMNS: Final = ("HighBP", "HighChol", "Stroke", "HeartDiseaseorAttack", "DiffWalk")
HEALTHY_HABITS: Final = ("PhysActivity", "Fruits", "Veggies")
RISKY_HABITS: Final = ("Smoker", "HvyAlcoholConsump")

BMI_BINS: Final = (-np.inf, 18.5, 25.0, 30.0, np.inf)
"""Limites da OMS: abaixo do peso, normal, sobrepeso, obesidade."""

ENGINEERED_FEATURES: Final = ("ComorbidityCount", "LifestyleScore", "BMICategory")


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Acrescenta features derivadas com significado clínico.

    * ``ComorbidityCount``: nº de condições associadas a diabetes
      (pressão alta, colesterol alto, AVC, doença cardíaca, dificuldade de andar);
    * ``LifestyleScore``: hábitos saudáveis menos hábitos de risco;
    * ``BMICategory``: faixa de IMC da OMS (0 a 3).

    O transformer não aprende nada dos dados (o ``fit`` apenas memoriza os
    nomes das colunas), portanto não há risco de vazamento entre treino e
    validação.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Memoriza os nomes das colunas de entrada (convenção do sklearn)."""
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Retorna uma cópia de ``X`` com as features derivadas."""
        bmi_category = pd.cut(X["BMI"], bins=BMI_BINS, labels=False, right=False)
        return X.assign(
            ComorbidityCount=X[list(COMORBIDITY_COLUMNS)].sum(axis=1),
            LifestyleScore=X[list(HEALTHY_HABITS)].sum(axis=1) - X[list(RISKY_HABITS)].sum(axis=1),
            BMICategory=bmi_category.astype(float),
        )

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        """Nomes das colunas de saída (exigido por ``set_output``)."""
        base = self.feature_names_in_ if input_features is None else input_features
        names = [*base, *ENGINEERED_FEATURES]
        return np.asarray(names, dtype=object)


def build_preprocessor() -> ColumnTransformer:
    """Cria o pré-processador de colunas.

    Variáveis contínuas e ordinais são padronizadas (média 0, desvio 1), o que
    é necessário para modelos lineares e inofensivo para árvores. Indicadores
    binários já estão em 0/1 e passam direto.
    """
    scaled = [*NUMERIC_FEATURES, *ORDINAL_FEATURES, *ENGINEERED_FEATURES]
    return ColumnTransformer(
        transformers=[
            ("scaled", StandardScaler(), scaled),
            ("binary", "passthrough", list(BINARY_FEATURES)),
        ],
        verbose_feature_names_out=False,
    )
