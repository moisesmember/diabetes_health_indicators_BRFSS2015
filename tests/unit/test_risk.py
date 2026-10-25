import numpy as np
import pytest

from diabetes_risk.models.risk import (
    CONFIRMATORY_TESTS,
    Classification,
    RiskBands,
    RiskLevel,
    assess,
    fit_risk_bands,
)

BANDS = RiskBands(
    medium=0.4,
    high=0.7,
    prevalence={RiskLevel.LOW: 0.05, RiskLevel.MEDIUM: 0.25, RiskLevel.HIGH: 0.55},
)


class TestRiskBands:
    @pytest.mark.parametrize(
        ("probability", "level"),
        [
            (0.0, RiskLevel.LOW),
            (0.399, RiskLevel.LOW),
            (0.4, RiskLevel.MEDIUM),
            (0.699, RiskLevel.MEDIUM),
            (0.7, RiskLevel.HIGH),
            (1.0, RiskLevel.HIGH),
        ],
    )
    def test_level_boundaries_are_inclusive(self, probability: float, level: RiskLevel) -> None:
        assert BANDS.level(probability) is level

    def test_vectorized_levels_match_scalar(self) -> None:
        scores = np.array([0.1, 0.5, 0.9])
        assert BANDS.levels(scores).tolist() == [BANDS.level(s).value for s in scores]

    @pytest.mark.parametrize(("medium", "high"), [(0.7, 0.4), (-0.1, 0.5), (0.4, 1.1)])
    def test_rejects_invalid_cuts(self, medium: float, high: float) -> None:
        with pytest.raises(ValueError, match="Cortes inválidos"):
            RiskBands(medium=medium, high=high)


class TestFitRiskBands:
    def test_high_band_reaches_target_prevalence(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 2_000)
        proba = np.clip(0.4 * y + 0.6 * rng.random(2_000), 0, 1)

        bands = fit_risk_bands(y, proba, threshold=0.3, high_precision=0.8)

        high = proba >= bands.high
        assert y[high].mean() >= 0.8
        assert bands.medium == 0.3
        assert bands.high > bands.medium
        assert bands.prevalence[RiskLevel.HIGH] == pytest.approx(y[high].mean())

    def test_prevalence_increases_with_level(self) -> None:
        rng = np.random.default_rng(1)
        y = (rng.random(5_000) < 0.2).astype(int)  # prevalência realista (~20%)
        proba = np.clip(0.3 * y + 0.7 * rng.random(5_000), 0, 1)

        prev = fit_risk_bands(y, proba, threshold=0.4).prevalence

        assert prev[RiskLevel.LOW] < prev[RiskLevel.MEDIUM] < prev[RiskLevel.HIGH]

    def test_unreachable_precision_leaves_high_band_empty(self) -> None:
        y = np.array([0, 1, 0, 1])
        proba = np.array([0.9, 0.8, 0.7, 0.1])  # o maior score é negativo

        bands = fit_risk_bands(y, proba, threshold=0.5, high_precision=0.99)

        assert bands.high == 1.0
        assert bands.prevalence[RiskLevel.HIGH] is None

    def test_high_cut_never_below_threshold(self) -> None:
        y = np.array([1, 1, 1, 0])
        proba = np.array([0.9, 0.3, 0.2, 0.1])  # precisão alta mesmo em scores baixos

        assert fit_risk_bands(y, proba, threshold=0.6, high_precision=0.5).high >= 0.6


class TestAssess:
    def test_high_risk_requires_confirmation(self) -> None:
        result = assess(0.85, BANDS)

        assert result.classification is Classification.POSITIVE
        assert result.classification == "com diabetes"
        assert result.risk_level == "alta"
        assert result.at_risk
        assert result.requires_confirmation
        assert result.confirmatory_tests == CONFIRMATORY_TESTS
        assert result.band_prevalence == 0.55
        assert "confirmar" in result.recommendation

    def test_medium_risk_is_positive_and_requires_confirmation(self) -> None:
        result = assess(0.5, BANDS)
        assert result.classification == "com diabetes"
        assert result.risk_level == "média"
        assert result.requires_confirmation

    def test_low_risk_is_negative_without_confirmation(self) -> None:
        result = assess(0.1, BANDS)

        assert result.classification == "sem diabetes"
        assert result.risk_level == "baixa"
        assert not result.at_risk
        assert not result.requires_confirmation
        assert result.confirmatory_tests == ()

    def test_unknown_band_prevalence_is_none(self) -> None:
        assert assess(0.5, RiskBands(medium=0.4, high=0.7)).band_prevalence is None
