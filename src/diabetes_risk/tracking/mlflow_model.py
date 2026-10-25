"""Modelo ``pyfunc`` do MLflow que embute o limiar de decisão.

Registrar só o estimador sklearn faria ``predict`` usar o limiar padrão de
0,5, divergindo da API. Este wrapper carrega o :class:`ModelBundle` inteiro e
devolve score, classificação (com/sem diabetes), chance (alta/média/baixa) e a
necessidade de confirmação, então qualquer consumidor do registry
(``mlflow.pyfunc.load_model("models:/diabetes-risk@champion")``,
``mlflow models serve``, batch no Spark) obtém exatamente o mesmo resultado.

O arquivo segue o padrão *models from code* do MLflow 3: ele é registrado
como código-fonte (``python_model=<este arquivo>``) em vez de um objeto
serializado com cloudpickle, o que é mais seguro e legível.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from mlflow.models import set_model
from mlflow.pyfunc.model import PythonModel

from diabetes_risk.models.bundle import ModelBundle

BUNDLE_ARTIFACT_KEY = "bundle"


class DiabetesRiskModel(PythonModel):
    """Wrapper pyfunc: ``predict`` retorna score, classificação, nível de risco e confirmação."""

    def load_context(self, context: Any) -> None:
        """Carrega o bundle a partir dos artefatos do modelo."""
        self._bundle = ModelBundle.load(context.artifacts[BUNDLE_ARTIFACT_KEY])

    def predict(self, context: Any, model_input: pd.DataFrame, params: Any = None) -> pd.DataFrame:
        """Pontua um lote de pacientes (DataFrame no formato do BRFSS)."""
        return self._bundle.assessment_frame(model_input)


set_model(DiabetesRiskModel())
