import pandas as pd
import pytest

from diabetes_risk.domain.schemas import FEATURE_COLUMNS, TARGET_COLUMN
from diabetes_risk.features.transformers import (
    ENGINEERED_FEATURES,
    ClinicalFeatureEngineer,
    build_preprocessor,
)


def _patient(**overrides: float) -> dict[str, float]:
    base = dict.fromkeys(FEATURE_COLUMNS, 0.0)
    base.update({"GenHlth": 1.0, "Age": 1.0, "Education": 1.0, "Income": 1.0, "BMI": 22.0})
    return base | overrides


class TestClinicalFeatureEngineer:
    def test_counts_comorbidities(self) -> None:
        X = pd.DataFrame([_patient(HighBP=1, HighChol=1, Stroke=1, Smoker=1)])
        assert ClinicalFeatureEngineer().fit_transform(X)["ComorbidityCount"].item() == 3

    def test_lifestyle_score_subtracts_risky_habits(self) -> None:
        X = pd.DataFrame([_patient(PhysActivity=1, Fruits=1, Veggies=1, Smoker=1)])
        assert ClinicalFeatureEngineer().fit_transform(X)["LifestyleScore"].item() == 2

    @pytest.mark.parametrize(
        ("bmi", "category"),
        [(17.0, 0), (18.5, 1), (24.9, 1), (25.0, 2), (29.9, 2), (30.0, 3), (60.0, 3)],
    )
    def test_bmi_category_follows_who_limits(self, bmi: float, category: int) -> None:
        X = pd.DataFrame([_patient(BMI=bmi)])
        assert ClinicalFeatureEngineer().fit_transform(X)["BMICategory"].item() == category

    def test_does_not_mutate_input(self) -> None:
        X = pd.DataFrame([_patient()])
        ClinicalFeatureEngineer().fit_transform(X)
        assert list(X.columns) == list(FEATURE_COLUMNS)

    def test_feature_names_out(self) -> None:
        X = pd.DataFrame([_patient()])
        names = ClinicalFeatureEngineer().fit(X).get_feature_names_out()
        assert list(names) == [*FEATURE_COLUMNS, *ENGINEERED_FEATURES]


def test_preprocessor_scales_continuous_and_keeps_binary(raw_frame: pd.DataFrame) -> None:
    engineered = ClinicalFeatureEngineer().fit_transform(raw_frame.drop(columns=TARGET_COLUMN))

    output = build_preprocessor().set_output(transform="pandas").fit_transform(engineered)

    assert output["BMI"].mean() == pytest.approx(0.0, abs=1e-9)
    assert output["BMI"].std(ddof=0) == pytest.approx(1.0)
    assert set(output["HighBP"].unique()) <= {0.0, 1.0}
    assert len(output.columns) == len(FEATURE_COLUMNS) + len(ENGINEERED_FEATURES)
