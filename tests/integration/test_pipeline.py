"""Pipeline ponta a ponta: prepare -> train -> evaluate em dados sintéticos."""

import json
from pathlib import Path

import pandas as pd
import pytest

from diabetes_risk.config import PipelineConfig
from diabetes_risk.domain.schemas import BINARY_TARGET, FEATURE_COLUMNS
from diabetes_risk.models.bundle import ModelBundle
from diabetes_risk.pipelines.evaluate import run_evaluate
from diabetes_risk.pipelines.prepare import run_prepare
from diabetes_risk.pipelines.train import run_train

pytestmark = pytest.mark.integration


class _InMemorySource:
    """Fonte alternativa: prova que o pipeline depende só da abstração DataSource."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def load(self) -> pd.DataFrame:
        return self._frame


def test_prepare_writes_validated_splits(config: PipelineConfig, raw_csv: Path) -> None:
    split = run_prepare(config)

    train = pd.read_parquet(config.data.train_path)
    test = pd.read_parquet(config.data.test_path)
    assert list(train.columns) == [*FEATURE_COLUMNS, BINARY_TARGET]
    assert len(train) == len(split.train)
    assert len(test) / (len(train) + len(test)) == pytest.approx(0.25, abs=0.01)


def test_prepare_accepts_injected_source(config: PipelineConfig, raw_frame: pd.DataFrame) -> None:
    duplicated = pd.concat([raw_frame, raw_frame.head(50)], ignore_index=True)

    split = run_prepare(config, source=_InMemorySource(duplicated))

    assert len(split.train) + len(split.test) == len(raw_frame.drop_duplicates())


def test_full_pipeline(config: PipelineConfig, raw_csv: Path) -> None:
    run_prepare(config)
    bundle = run_train(config)
    report = run_evaluate(config)

    assert ModelBundle.load(config.artifacts.model_path).threshold == bundle.threshold
    assert report.gate.passed, report.gate.failures
    assert report.metrics["roc_auc"] > 0.6

    saved = json.loads(config.artifacts.metrics_path.read_text(encoding="utf-8"))
    assert saved["gate_passed"] is True
    assert saved["roc_auc"] == pytest.approx(report.metrics["roc_auc"])

    runs = sorted(Path(config.tracking.uri, config.tracking.experiment_name).glob("*/run.json"))
    assert len(runs) == 2  # uma execução para train, outra para evaluate
    train_run = json.loads(runs[0].read_text(encoding="utf-8"))
    assert "cv_roc_auc" in train_run["metrics"]


def test_gate_failure_is_reported(config_dict: dict[str, dict[str, object]], raw_csv: Path) -> None:
    config_dict["evaluation"]["min_metrics"] = {"roc_auc": 0.999}
    config = PipelineConfig.model_validate(config_dict)

    run_prepare(config)
    run_train(config)
    report = run_evaluate(config)

    assert not report.gate.passed
    assert "roc_auc" in report.gate.failures[0]
