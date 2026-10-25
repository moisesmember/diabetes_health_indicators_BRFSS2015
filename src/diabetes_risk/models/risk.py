"""Tradução do score do modelo em uma avaliação de risco compreensível.

O score do modelo (``probability``) ordena bem as pessoas, mas não está
calibrado: 0,49 não significa "49% de chance". Por isso a resposta ao usuário
é baseada em **faixas** com significado empírico:

* **baixa**: score abaixo do limiar de decisão → ``sem diabetes``;
* **média**: score acima do limiar, mas abaixo do corte de alto risco → ``com diabetes``;
* **alta**: faixa em que, nos dados de treino, ao menos ``high_precision``
  (padrão 50%) das pessoas tinham diabetes ou pré-diabetes → ``com diabetes``.

Os cortes e a prevalência observada em cada faixa são aprendidos no treino com
predições *out-of-fold*, nunca com o conjunto de teste.

Toda avaliação positiva vem com a indicação de **confirmação por exame
laboratorial**: o modelo faz triagem, não diagnóstico.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

import numpy as np
from numpy.typing import ArrayLike


class RiskLevel(StrEnum):
    """Nível de chance de diabetes ou pré-diabetes."""

    LOW = "baixa"
    MEDIUM = "média"
    HIGH = "alta"


class Classification(StrEnum):
    """Classificação binária exibida ao usuário (indicativo de triagem)."""

    POSITIVE = "com diabetes"
    NEGATIVE = "sem diabetes"


CONFIRMATORY_TESTS: Final = (
    "Glicemia de jejum",
    "Hemoglobina glicada (HbA1c)",
    "Teste oral de tolerância à glicose (TOTG), se indicado pelo médico",
)
"""Exames usados para confirmar ou descartar diabetes e pré-diabetes."""

DISCLAIMER: Final = (
    "Resultado de triagem estatística baseado em respostas autodeclaradas. "
    "Não é diagnóstico: apenas exames laboratoriais avaliados por um profissional "
    "de saúde confirmam diabetes ou pré-diabetes."
)

_RECOMMENDATIONS: Final = {
    RiskLevel.HIGH: (
        "Chance alta de diabetes ou pré-diabetes. Procure um médico em breve para "
        "confirmar com exames laboratoriais."
    ),
    RiskLevel.MEDIUM: (
        "Chance moderada de diabetes ou pré-diabetes. Recomenda-se realizar exames "
        "laboratoriais para confirmar ou descartar."
    ),
    RiskLevel.LOW: (
        "Chance baixa de diabetes ou pré-diabetes. Mantenha hábitos saudáveis e o "
        "acompanhamento de rotina; procure um médico se surgirem sintomas como sede "
        "excessiva, urina frequente ou perda de peso sem causa."
    ),
}


@dataclass(frozen=True)
class RiskBands:
    """Cortes das faixas de risco e prevalência observada em cada uma.

    Attributes:
        medium: Score mínimo da faixa média (igual ao limiar de decisão).
        high: Score mínimo da faixa alta (``>= medium``).
        prevalence: Fração de pessoas com diabetes/pré-diabetes em cada faixa,
            medida no treino (``None`` se a faixa ficou vazia).
    """

    medium: float
    high: float
    prevalence: dict[RiskLevel, float | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.medium <= self.high <= 1.0:
            raise ValueError(
                f"Cortes inválidos: exige 0 <= medium <= high <= 1 "
                f"(medium={self.medium}, high={self.high})."
            )

    def level(self, probability: float) -> RiskLevel:
        """Faixa correspondente a um score."""
        if probability >= self.high:
            return RiskLevel.HIGH
        if probability >= self.medium:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def levels(self, probabilities: ArrayLike) -> np.ndarray:
        """Faixa de cada score (vetorizado)."""
        p = np.asarray(probabilities)
        return np.where(
            p >= self.high,
            RiskLevel.HIGH.value,
            np.where(p >= self.medium, RiskLevel.MEDIUM.value, RiskLevel.LOW.value),
        )


def fit_risk_bands(
    y_true: ArrayLike, y_proba: ArrayLike, threshold: float, high_precision: float = 0.5
) -> RiskBands:
    """Aprende os cortes das faixas a partir de predições *out-of-fold*.

    O corte de alto risco é o **menor** score a partir do qual a prevalência
    acumulada (precisão) ainda é ``>= high_precision``. Com o padrão de 0,5,
    "alta" significa: entre as pessoas com esse score ou mais, pelo menos metade
    tem diabetes ou pré-diabetes.

    Args:
        y_true: Alvo real (0/1).
        y_proba: Scores *out-of-fold* do modelo.
        threshold: Limiar de decisão (início da faixa média).
        high_precision: Prevalência mínima exigida na faixa alta.

    Returns:
        Os cortes e a prevalência observada em cada faixa. Se nenhum corte
        atingir ``high_precision``, a faixa alta fica vazia (``high = 1``).
    """
    y = np.asarray(y_true)
    p = np.asarray(y_proba)

    order = np.argsort(-p, kind="stable")
    cumulative_precision = np.cumsum(y[order]) / np.arange(1, len(y) + 1)
    reaches = np.flatnonzero(cumulative_precision >= high_precision)
    high = float(p[order][int(reaches[-1])]) if reaches.size else 1.0
    high = min(max(high, threshold), 1.0)

    provisional = RiskBands(medium=threshold, high=high)
    levels = provisional.levels(p)
    prevalence: dict[RiskLevel, float | None] = {}
    for level in RiskLevel:
        mask = levels == level.value
        prevalence[level] = float(y[mask].mean()) if mask.any() else None
    return RiskBands(medium=threshold, high=high, prevalence=prevalence)


@dataclass(frozen=True)
class RiskAssessment:
    """Resultado completo da triagem de uma pessoa.

    Attributes:
        probability: Score do modelo (ordena o risco; não é calibrado).
        at_risk: ``True`` se o score atinge o limiar de decisão.
        classification: ``com diabetes`` ou ``sem diabetes`` (indicativo de triagem).
        risk_level: Chance ``alta``, ``média`` ou ``baixa``.
        band_prevalence: Fração de pessoas com diabetes/pré-diabetes na mesma
            faixa, nos dados de treino (``None`` se desconhecida).
        recommendation: Indicação do que fazer a seguir.
        requires_confirmation: ``True`` quando é preciso confirmar com exames.
        confirmatory_tests: Exames indicados para a confirmação.
    """

    probability: float
    at_risk: bool
    classification: Classification
    risk_level: RiskLevel
    band_prevalence: float | None
    recommendation: str
    requires_confirmation: bool
    confirmatory_tests: tuple[str, ...]


def assess(probability: float, bands: RiskBands) -> RiskAssessment:
    """Converte um score na avaliação completa (classificação, nível e indicação)."""
    level = bands.level(probability)
    at_risk = level is not RiskLevel.LOW
    return RiskAssessment(
        probability=float(probability),
        at_risk=at_risk,
        classification=Classification.POSITIVE if at_risk else Classification.NEGATIVE,
        risk_level=level,
        band_prevalence=bands.prevalence.get(level),
        recommendation=_RECOMMENDATIONS[level],
        requires_confirmation=at_risk,
        confirmatory_tests=CONFIRMATORY_TESTS if at_risk else (),
    )
