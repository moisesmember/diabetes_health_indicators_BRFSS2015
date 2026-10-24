import numpy as np
import pandas as pd
import pytest

from diabetes_risk.monitoring.drift import drift_report, population_stability_index


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_identical_distributions_have_zero_psi(rng: np.random.Generator) -> None:
    sample = pd.Series(rng.normal(size=1000))
    assert population_stability_index(sample, sample) == pytest.approx(0.0)


def test_shifted_continuous_distribution_drifts(rng: np.random.Generator) -> None:
    reference = pd.Series(rng.normal(28, 5, size=2000))
    current = pd.Series(rng.normal(35, 5, size=2000))
    assert population_stability_index(reference, current) > 0.25


def test_same_distribution_new_sample_is_stable(rng: np.random.Generator) -> None:
    reference = pd.Series(rng.normal(size=5000))
    current = pd.Series(rng.normal(size=5000))
    assert population_stability_index(reference, current) < 0.1


def test_categorical_feature_with_unseen_category(rng: np.random.Generator) -> None:
    reference = pd.Series(rng.integers(0, 2, size=1000))
    current = pd.Series(rng.integers(0, 3, size=1000))
    assert population_stability_index(reference, current) > 0.25


def test_drift_report_flags_and_sorts(rng: np.random.Generator) -> None:
    reference = pd.DataFrame({"stable": rng.normal(size=2000), "shifted": rng.normal(size=2000)})
    current = pd.DataFrame(
        {"stable": rng.normal(size=2000), "shifted": rng.normal(3, 1, size=2000)}
    )

    report = drift_report(reference, current, ["stable", "shifted"])

    assert report["feature"].tolist() == ["shifted", "stable"]
    assert report["drift"].tolist() == [True, False]
