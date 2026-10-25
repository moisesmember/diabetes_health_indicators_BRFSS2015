import numpy as np
import pytest

from diabetes_risk.evaluation.metrics import (
    check_quality_gate,
    compute_metrics,
    confusion_metrics,
    fbeta_curve,
    find_best_threshold,
    risk_band_metrics,
)
from diabetes_risk.models.risk import RiskBands


class TestComputeMetrics:
    def test_perfect_classifier(self) -> None:
        metrics = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], threshold=0.5)

        assert metrics["roc_auc"] == 1.0
        assert metrics["pr_auc"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["f2"] == 1.0
        assert metrics["threshold"] == 0.5

    def test_threshold_changes_classification_metrics_only(self) -> None:
        y, proba = [0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9]

        low = compute_metrics(y, proba, threshold=0.3)
        high = compute_metrics(y, proba, threshold=0.7)

        assert low["recall"] == 1.0
        assert high["recall"] == 0.5
        assert low["roc_auc"] == high["roc_auc"]

    def test_no_positive_predictions_does_not_raise(self) -> None:
        metrics = compute_metrics([0, 1], [0.1, 0.2], threshold=0.9)
        assert metrics["precision"] == 0.0

    def test_metric_name_reflects_beta(self) -> None:
        assert "f1" in compute_metrics([0, 1], [0.2, 0.8], beta=1.0)


class TestFindBestThreshold:
    def test_separable_data_picks_threshold_between_classes(self) -> None:
        threshold = find_best_threshold([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.8])
        assert 0.2 < threshold <= 0.7

    def test_higher_beta_never_raises_threshold(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 500)
        proba = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)

        assert find_best_threshold(y, proba, beta=3.0) <= find_best_threshold(y, proba, beta=0.5)


class TestQualityGate:
    def test_passes_when_all_minimums_met(self) -> None:
        result = check_quality_gate({"roc_auc": 0.8, "recall": 0.7}, {"roc_auc": 0.75})
        assert result.passed
        assert result.failures == []

    def test_fails_below_minimum(self) -> None:
        result = check_quality_gate({"roc_auc": 0.7}, {"roc_auc": 0.75})
        assert not result.passed
        assert "roc_auc" in result.failures[0]

    def test_missing_metric_fails(self) -> None:
        result = check_quality_gate({}, {"roc_auc": 0.75})
        assert result.failures == ["roc_auc: métrica não calculada"]

    @pytest.mark.parametrize("value", [0.75, 0.9])
    def test_minimum_is_inclusive(self, value: float) -> None:
        assert check_quality_gate({"roc_auc": value}, {"roc_auc": 0.75}).passed


class TestInterpretableMetrics:
    def test_confusion_metrics_and_rates_per_1000(self) -> None:
        y_true = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        y_pred = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]

        metrics = confusion_metrics(y_true, y_pred)

        assert (metrics["tp"], metrics["fn"], metrics["fp"], metrics["tn"]) == (1, 1, 1, 7)
        assert metrics["specificity"] == pytest.approx(7 / 8)
        assert metrics["npv"] == pytest.approx(7 / 8)
        assert metrics["alerts_per_1000"] == pytest.approx(200)
        assert metrics["missed_cases_per_1000"] == pytest.approx(100)
        assert metrics["false_alarms_per_1000"] == pytest.approx(100)

    def test_pr_auc_lift_is_relative_to_prevalence(self) -> None:
        metrics = compute_metrics([0, 0, 0, 1], [0.1, 0.2, 0.3, 0.9])
        assert metrics["prevalence"] == 0.25
        assert metrics["pr_auc_lift"] == pytest.approx(metrics["pr_auc"] / 0.25)

    def test_brier_skill_compares_with_prevalence_baseline(self) -> None:
        y = np.array([0, 0, 0, 1])
        constant = np.full(4, 0.25)  # prever sempre a prevalência
        assert compute_metrics(y, constant)["brier_skill"] == pytest.approx(0.0)
        assert compute_metrics(y, [0.0, 0.0, 0.0, 1.0])["brier_skill"] == pytest.approx(1.0)

    def test_fbeta_curve_arrays_are_aligned(self) -> None:
        curve = fbeta_curve([0, 1, 0, 1], [0.2, 0.8, 0.4, 0.6])
        lengths = {len(curve.thresholds), len(curve.precision), len(curve.recall), len(curve.fbeta)}
        assert len(lengths) == 1


def test_risk_band_metrics_share_and_prevalence() -> None:
    bands = RiskBands(medium=0.4, high=0.7)
    y = [0, 0, 1, 0, 1, 1]
    proba = [0.1, 0.2, 0.3, 0.5, 0.6, 0.9]  # baixa: 3, média: 2, alta: 1

    metrics = risk_band_metrics(y, proba, bands)

    assert metrics["band_share_low"] == pytest.approx(3 / 6)
    assert metrics["band_share_medium"] == pytest.approx(2 / 6)
    assert metrics["band_share_high"] == pytest.approx(1 / 6)
    assert metrics["band_prevalence_low"] == pytest.approx(1 / 3)
    assert metrics["band_prevalence_medium"] == pytest.approx(1 / 2)
    assert metrics["band_prevalence_high"] == 1.0


def test_risk_band_metrics_skips_prevalence_of_empty_band() -> None:
    metrics = risk_band_metrics([0, 1], [0.1, 0.2], RiskBands(medium=0.4, high=0.7))
    assert metrics["band_share_high"] == 0.0
    assert "band_prevalence_high" not in metrics
