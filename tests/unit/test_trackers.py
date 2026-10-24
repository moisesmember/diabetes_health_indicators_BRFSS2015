import json
from pathlib import Path

import pytest

from diabetes_risk.config import TrackingConfig
from diabetes_risk.domain.protocols import ExperimentTracker
from diabetes_risk.tracking.trackers import LocalJsonTracker, MlflowTracker, create_tracker


def test_local_tracker_writes_run_file(tmp_path: Path) -> None:
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"content")
    tracker = LocalJsonTracker(tmp_path / "runs", "exp", run_name="run1")

    tracker.log_params({"alpha": 0.1, "path": Path("a/b")})
    tracker.log_metrics({"auc": 0.8}, prefix="cv_")
    tracker.log_artifact(artifact)
    tracker.close()

    run_dir = tmp_path / "runs" / "exp" / "run1"
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["params"] == {"alpha": 0.1, "path": str(Path("a/b"))}
    assert record["metrics"] == {"cv_auc": 0.8}
    assert record["artifacts"] == ["model.bin"]
    assert (run_dir / "model.bin").read_bytes() == b"content"


def test_local_tracker_satisfies_protocol(tmp_path: Path) -> None:
    assert isinstance(LocalJsonTracker(tmp_path, "exp"), ExperimentTracker)


def test_factory_returns_local_tracker(tmp_path: Path) -> None:
    tracker = create_tracker(TrackingConfig(backend="local", uri=str(tmp_path)))
    assert isinstance(tracker, LocalJsonTracker)


def test_mlflow_tracker_logs_run(tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    artifact = tmp_path / "metrics.json"
    artifact.write_text("{}", encoding="utf-8")

    tracker = create_tracker(TrackingConfig(backend="mlflow", uri=uri, experiment_name="test"))
    assert isinstance(tracker, MlflowTracker)
    tracker.log_params({"alpha": 0.1})
    tracker.log_metrics({"auc": 0.8}, prefix="test_")
    tracker.close()

    runs = mlflow.search_runs(experiment_names=["test"])
    assert runs.loc[0, "metrics.test_auc"] == 0.8
    assert runs.loc[0, "params.alpha"] == "0.1"
