from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from diabetes_risk.config import PipelineConfig, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_loads_valid_file(config: PipelineConfig, tmp_path: Path) -> None:
    assert config.model.name == "logistic_regression"
    assert config.data.train_path == tmp_path / "processed" / "train.parquet"
    assert config.data.test_path == tmp_path / "processed" / "test.parquet"


def test_project_params_file_is_valid() -> None:
    config = load_config(PROJECT_ROOT / "params.yaml")
    assert config.data.positive_classes == [1, 2]


def test_rejects_unknown_keys(config_dict: dict[str, object]) -> None:
    config_dict["unexpected"] = 1
    with pytest.raises(ValidationError, match="unexpected"):
        PipelineConfig.model_validate(config_dict)


@pytest.mark.parametrize("test_size", [0.0, 1.0, 1.5])
def test_rejects_invalid_test_size(
    config_dict: dict[str, dict[str, object]], test_size: float
) -> None:
    config_dict["data"]["test_size"] = test_size
    with pytest.raises(ValidationError, match="test_size"):
        PipelineConfig.model_validate(config_dict)


def test_rejects_invalid_positive_class(config_dict: dict[str, dict[str, object]]) -> None:
    config_dict["data"]["positive_classes"] = [1, 5]
    with pytest.raises(ValidationError, match="Classes inválidas"):
        PipelineConfig.model_validate(config_dict)


def test_optional_sections_have_defaults(config_dict: dict[str, object], tmp_path: Path) -> None:
    for section in ("training", "evaluation", "tracking"):
        config_dict.pop(section)
    path = tmp_path / "minimal.yaml"
    path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")

    config = load_config(path)

    assert config.training.cv_folds == 5
    assert config.tracking.backend == "local"
    assert config.evaluation.min_metrics == {}
