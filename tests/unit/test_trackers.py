import json
from pathlib import Path

import pandas as pd
import pytest

from diabetes_risk.config import PipelineConfig, TrackingConfig
from diabetes_risk.domain.protocols import ExperimentTracker
from diabetes_risk.models.bundle import ModelBundle
from diabetes_risk.tracking.trackers import (
    LocalJsonTracker,
    MlflowTracker,
    create_tracker,
    dataframe_digest,
)


class TestLocalJsonTracker:
    def test_writes_run_file(self, tmp_path: Path, trained_bundle: ModelBundle) -> None:
        artifact = tmp_path / "model.bin"
        artifact.write_bytes(b"content")
        tracker = LocalJsonTracker(tmp_path / "runs", "exp", run_id="run1")

        tracker.set_tags({"model_name": "lr"})
        tracker.log_params({"alpha": 0.1, "path": Path("a/b")})
        tracker.log_metrics({"auc": 0.8}, prefix="cv/")
        tracker.log_artifact(artifact, artifact_dir="evaluation")
        tracker.log_dataset(pd.DataFrame({"a": [1, 2]}), "train", "training", "x.parquet")
        tracker.log_model(trained_bundle, input_example=pd.DataFrame())
        tracker.set_description("# Relatório")
        tracker.close()

        run_dir = tmp_path / "runs" / "exp" / "run1"
        record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert record["tags"] == {"model_name": "lr"}
        assert record["params"] == {"alpha": 0.1, "path": str(Path("a/b"))}
        assert record["metrics"] == {"cv/auc": 0.8}
        assert record["datasets"][0]["rows"] == 2
        assert record["model"]["threshold"] == trained_bundle.threshold
        assert (run_dir / "evaluation" / "model.bin").read_bytes() == b"content"
        assert (run_dir / "description.md").read_text(encoding="utf-8") == "# Relatório"

    def test_resumes_existing_run(self, tmp_path: Path) -> None:
        first = LocalJsonTracker(tmp_path, "exp")
        first.log_metrics({"auc": 0.8}, prefix="cv/")
        first.close()

        resumed = LocalJsonTracker(tmp_path, "exp", run_id=first.run_id)
        resumed.log_metrics({"auc": 0.7}, prefix="test/")
        resumed.close()

        record = json.loads((resumed.run_dir / "run.json").read_text(encoding="utf-8"))
        assert record["metrics"] == {"cv/auc": 0.8, "test/auc": 0.7}

    def test_has_no_registry(self, tmp_path: Path) -> None:
        assert LocalJsonTracker(tmp_path, "exp").register_model("m", "champion", {}) is None

    def test_satisfies_protocol(self, tmp_path: Path) -> None:
        assert isinstance(LocalJsonTracker(tmp_path, "exp"), ExperimentTracker)


def test_factory_returns_local_tracker(tmp_path: Path) -> None:
    tracker = create_tracker(TrackingConfig(backend="local", uri=str(tmp_path)))
    assert isinstance(tracker, LocalJsonTracker)


def test_dataframe_digest_is_deterministic_and_content_sensitive() -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    assert dataframe_digest(frame) == dataframe_digest(frame.copy())
    assert dataframe_digest(frame) != dataframe_digest(frame.assign(a=[1, 3]))


class TestMlflowTracker:
    def test_full_lifecycle(
        self, mlflow_config: PipelineConfig, trained_bundle: ModelBundle, raw_frame: pd.DataFrame
    ) -> None:
        import mlflow

        tracking = mlflow_config.tracking
        features = raw_frame.drop(columns="Diabetes_012")
        tracker = create_tracker(tracking, run_name="unit")
        assert isinstance(tracker, MlflowTracker)
        tracker.set_tags({"model_name": "lr"})
        tracker.log_params({"alpha": 0.1})
        tracker.log_metrics({"auc": 0.8}, prefix="cv/")
        tracker.log_dataset(features.assign(target=0), "train", "training", "train.parquet")
        tracker.log_model(trained_bundle, input_example=features.head(3))
        tracker.close()

        resumed = create_tracker(tracking, run_id=tracker.run_id)
        resumed.log_metrics({"auc": 0.75}, prefix="test/")
        resumed.set_description("# Relatório")
        version = resumed.register_model("test-model", "champion", {"gate_passed": "true"})
        resumed.close()

        run = mlflow.get_run(tracker.run_id)
        assert run.data.metrics == {"cv/auc": 0.8, "test/auc": 0.75}
        assert run.data.tags["mlflow.note.content"] == "# Relatório"
        assert run.inputs.dataset_inputs[0].dataset.name == "train"

        champion = mlflow.MlflowClient().get_model_version_by_alias("test-model", "champion")
        assert str(champion.version) == version
        assert champion.tags["gate_passed"] == "true"

        model = mlflow.pyfunc.load_model("models:/test-model@champion")
        predictions = model.predict(features.head(5))
        assert list(predictions.columns) == ["probability", "at_risk"]
        expected = trained_bundle.predict(features.head(5)).astype(bool)
        assert predictions["at_risk"].tolist() == expected.tolist()

    def test_register_without_model_fails(self, mlflow_config: PipelineConfig) -> None:
        tracker = create_tracker(mlflow_config.tracking)
        try:
            with pytest.raises(RuntimeError, match="não tem modelo"):
                tracker.register_model("test-model", None, {})
        finally:
            tracker.close()

    def test_env_var_overrides_uri(
        self, mlflow_config: PipelineConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        assert mlflow_config.tracking.resolved_uri() == "http://mlflow:5000"
