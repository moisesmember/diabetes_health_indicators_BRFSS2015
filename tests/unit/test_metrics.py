import numpy as np
import pytest

from diabetes_risk.evaluation.metrics import (
    check_quality_gate,
    compute_metrics,
    find_best_threshold,
)


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
