from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from diabetes_risk.domain.schemas import TARGET_COLUMN
from diabetes_risk.models import registry
from diabetes_risk.models.bundle import ModelBundle
from diabetes_risk.models.registry import (
    available_models,
    build_pipeline,
    create_classifier,
    register_model,
)
from diabetes_risk.models.risk import RiskBands, RiskLevel


class TestRegistry:
    def test_builtin_models_are_registered(self) -> None:
        assert {"logistic_regression", "hist_gradient_boosting", "random_forest"} <= set(
            available_models()
        )

    def test_create_classifier_forwards_params(self) -> None:
        model = create_classifier("random_forest", {"n_estimators": 7})
        assert model.get_params()["n_estimators"] == 7

    def test_unknown_model_lists_options(self) -> None:
        with pytest.raises(KeyError, match="logistic_regression"):
            create_classifier("does_not_exist")

    def test_register_new_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(registry, "_REGISTRY", dict(registry._REGISTRY))

        @register_model("dummy")
        def _dummy(**params: object) -> DummyClassifier:
            return DummyClassifier(**params)  # type: ignore[arg-type]

        assert isinstance(create_classifier("dummy"), DummyClassifier)

    def test_duplicate_registration_fails(self) -> None:
        with pytest.raises(ValueError, match="já está registrado"):
            register_model("logistic_regression")(lambda: None)  # type: ignore[arg-type,return-value]


@pytest.mark.parametrize("name", available_models())
def test_every_registered_model_trains(name: str, raw_frame: pd.DataFrame) -> None:
    X = raw_frame.drop(columns=TARGET_COLUMN)
    y = (raw_frame[TARGET_COLUMN] > 0).astype(int)

    proba = build_pipeline(name).fit(X, y).predict_proba(X)[:, 1]

    assert proba.shape == (len(X),)
    assert np.all((proba >= 0) & (proba <= 1))


class TestModelBundle:
    def test_predict_applies_threshold(
        self, trained_bundle: ModelBundle, raw_frame: pd.DataFrame
    ) -> None:
        proba = trained_bundle.predict_proba(raw_frame)
        expected = (proba >= trained_bundle.threshold).astype(int)
        np.testing.assert_array_equal(trained_bundle.predict(raw_frame), expected)

    def test_ignores_extra_columns(
        self, trained_bundle: ModelBundle, raw_frame: pd.DataFrame
    ) -> None:
        with_extra = raw_frame.assign(extra=1.0)
        np.testing.assert_allclose(
            trained_bundle.predict_proba(with_extra), trained_bundle.predict_proba(raw_frame)
        )

    def test_save_and_load_roundtrip(
        self, trained_bundle: ModelBundle, raw_frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        path = trained_bundle.save(tmp_path / "nested" / "model.joblib")

        loaded = ModelBundle.load(path)

        assert loaded.threshold == trained_bundle.threshold
        np.testing.assert_allclose(
            loaded.predict_proba(raw_frame), trained_bundle.predict_proba(raw_frame)
        )

    @pytest.mark.parametrize("threshold", [-0.1, 1.1])
    def test_rejects_invalid_threshold(self, trained_bundle: ModelBundle, threshold: float) -> None:
        with pytest.raises(ValueError, match="Limiar"):
            ModelBundle(pipeline=trained_bundle.pipeline, threshold=threshold, model_name="x")

    def test_assess_is_consistent_with_threshold(
        self, trained_bundle: ModelBundle, raw_frame: pd.DataFrame
    ) -> None:
        results = trained_bundle.assess(raw_frame)

        expected_at_risk = trained_bundle.predict(raw_frame).astype(bool).tolist()
        assert [r.at_risk for r in results] == expected_at_risk
        for result in results:
            expected = "com diabetes" if result.at_risk else "sem diabetes"
            assert result.classification == expected
            assert result.requires_confirmation == result.at_risk

    def test_without_trained_bands_there_is_no_high_level(
        self, trained_bundle: ModelBundle, raw_frame: pd.DataFrame
    ) -> None:
        bundle = ModelBundle(pipeline=trained_bundle.pipeline, threshold=0.0, model_name="x")
        assert {r.risk_level for r in bundle.assess(raw_frame)} == {RiskLevel.MEDIUM}

    def test_assessment_frame_columns(
        self, trained_bundle: ModelBundle, raw_frame: pd.DataFrame
    ) -> None:
        frame = trained_bundle.assessment_frame(raw_frame.head(3))
        assert list(frame.columns) == [
            "probability",
            "at_risk",
            "classification",
            "risk_level",
            "requires_confirmation",
        ]
        assert len(frame) == 3

    def test_rejects_bands_not_aligned_with_threshold(self, trained_bundle: ModelBundle) -> None:
        with pytest.raises(ValueError, match="faixa média"):
            ModelBundle(
                pipeline=trained_bundle.pipeline,
                threshold=0.4,
                model_name="x",
                risk_bands=RiskBands(medium=0.3, high=0.7),
            )

    def test_load_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="train"):
            ModelBundle.load(tmp_path / "missing.joblib")

    def test_load_wrong_object_type(self, tmp_path: Path) -> None:
        path = tmp_path / "not_a_bundle.joblib"
        joblib.dump({"a": 1}, path)
        with pytest.raises(TypeError, match="ModelBundle"):
            ModelBundle.load(path)
