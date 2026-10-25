"""Registro de modelos disponíveis e montagem do pipeline completo.

Para adicionar um modelo novo basta decorar uma *factory* com
:func:`register_model`; nenhum outro código precisa mudar (aberto para
extensão, fechado para modificação). O nome registrado é o usado em
``params.yaml`` (``model.name``).

Exemplo::

    @register_model("svm")
    def _svm(**params):
        return SVC(probability=True, **params)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from diabetes_risk.features.transformers import ClinicalFeatureEngineer, build_preprocessor

ModelFactory = Callable[..., ClassifierMixin]

_REGISTRY: dict[str, ModelFactory] = {}


def register_model(name: str) -> Callable[[ModelFactory], ModelFactory]:
    """Decorador que registra uma factory de classificador sob ``name``.

    Raises:
        ValueError: Se ``name`` já estiver registrado.
    """

    def decorator(factory: ModelFactory) -> ModelFactory:
        if name in _REGISTRY:
            raise ValueError(f"Modelo '{name}' já está registrado.")
        _REGISTRY[name] = factory
        return factory

    return decorator


def available_models() -> list[str]:
    """Nomes de todos os modelos registrados, em ordem alfabética."""
    return sorted(_REGISTRY)


def create_classifier(name: str, params: dict[str, Any] | None = None) -> ClassifierMixin:
    """Instancia o classificador registrado como ``name``.

    Raises:
        KeyError: Se ``name`` não estiver registrado (a mensagem lista as opções).
    """
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise KeyError(f"Modelo '{name}' desconhecido. Opções: {available_models()}") from None
    return factory(**(params or {}))


def build_pipeline(name: str, params: dict[str, Any] | None = None) -> Pipeline:
    """Monta o pipeline completo: features -> pré-processamento -> classificador.

    O objeto retornado é o artefato que vai para produção, carregando consigo
    toda a transformação dos dados.
    """
    pipeline = Pipeline(
        steps=[
            ("features", ClinicalFeatureEngineer()),
            ("preprocess", build_preprocessor()),
            ("classifier", create_classifier(name, params)),
        ]
    )
    return pipeline.set_output(transform="pandas")


@register_model("logistic_regression")
def _logistic_regression(**params: Any) -> ClassifierMixin:
    return LogisticRegression(**{"max_iter": 1000, **params})


@register_model("hist_gradient_boosting")
def _hist_gradient_boosting(**params: Any) -> ClassifierMixin:
    return HistGradientBoostingClassifier(**params)


@register_model("random_forest")
def _random_forest(**params: Any) -> ClassifierMixin:
    return RandomForestClassifier(**{"n_jobs": -1, **params})
