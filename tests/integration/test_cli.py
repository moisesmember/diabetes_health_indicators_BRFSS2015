"""CLI: as mesmas etapas que o DVC executa."""

from pathlib import Path

import pytest
import yaml

from diabetes_risk.cli import EXIT_GATE_FAILED, main

pytestmark = pytest.mark.integration


def test_cli_runs_all_stages(config_file: Path, raw_csv: Path, tmp_path: Path) -> None:
    for stage in ("prepare", "train", "evaluate"):
        assert main(["--config", str(config_file), stage]) == 0

    assert (tmp_path / "models" / "model.joblib").exists()
    assert (tmp_path / "reports" / "metrics.json").exists()


def test_cli_run_executes_all_stages(config_file: Path, raw_csv: Path, tmp_path: Path) -> None:
    assert main(["--config", str(config_file), "run"]) == 0
    assert (tmp_path / "reports" / "metrics_report.md").exists()


def test_cli_evaluate_exits_with_error_when_gate_fails(
    config_file: Path, config_dict: dict[str, dict[str, object]], raw_csv: Path
) -> None:
    config_dict["evaluation"]["min_metrics"] = {"recall": 1.01}
    config_file.write_text(yaml.safe_dump(config_dict), encoding="utf-8")

    main(["--config", str(config_file), "prepare"])
    main(["--config", str(config_file), "train"])

    assert main(["--config", str(config_file), "evaluate"]) == EXIT_GATE_FAILED


def test_cli_requires_stage() -> None:
    with pytest.raises(SystemExit):
        main([])
