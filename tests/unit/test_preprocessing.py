from pathlib import Path

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from diabetes_risk.data.preprocessing import (
    build_binary_target,
    drop_duplicate_rows,
    split_features_target,
    split_train_test,
    validate_raw,
)
from diabetes_risk.data.sources import CsvDataSource
from diabetes_risk.domain.schemas import BINARY_TARGET, FEATURE_COLUMNS, TARGET_COLUMN


class TestValidateRaw:
    def test_accepts_valid_frame(self, raw_frame: pd.DataFrame) -> None:
        assert len(validate_raw(raw_frame)) == len(raw_frame)

    def test_rejects_out_of_range_values(self, raw_frame: pd.DataFrame) -> None:
        raw_frame.loc[0, "Age"] = 99
        with pytest.raises(SchemaErrors, match="Age"):
            validate_raw(raw_frame)

    def test_rejects_missing_values(self, raw_frame: pd.DataFrame) -> None:
        raw_frame.loc[0, "BMI"] = None
        with pytest.raises(SchemaErrors, match="BMI"):
            validate_raw(raw_frame)

    def test_rejects_missing_column(self, raw_frame: pd.DataFrame) -> None:
        with pytest.raises(SchemaErrors, match="Income"):
            validate_raw(raw_frame.drop(columns="Income"))

    def test_rejects_unexpected_column(self, raw_frame: pd.DataFrame) -> None:
        with pytest.raises(SchemaErrors, match="extra"):
            validate_raw(raw_frame.assign(extra=1.0))


@pytest.mark.parametrize(
    ("positive_classes", "expected"),
    [([1, 2], [0, 1, 1]), ([2], [0, 0, 1])],
)
def test_build_binary_target(positive_classes: list[int], expected: list[int]) -> None:
    raw = pd.DataFrame(
        {TARGET_COLUMN: [0.0, 1.0, 2.0], **{name: [1.0, 1.0, 1.0] for name in FEATURE_COLUMNS}}
    )

    result = build_binary_target(raw, positive_classes)

    assert result[BINARY_TARGET].tolist() == expected
    assert TARGET_COLUMN not in result.columns


def test_drop_duplicate_rows() -> None:
    frame = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
    assert drop_duplicate_rows(frame).to_dict("list") == {"a": [1, 2], "b": [3, 4]}


def test_split_is_stratified_and_disjoint(raw_frame: pd.DataFrame) -> None:
    dataset = build_binary_target(raw_frame, [1, 2]).reset_index(names="row_id")

    train, test = split_train_test(dataset, test_size=0.25, random_state=0)

    assert len(train) + len(test) == len(dataset)
    assert set(train["row_id"]).isdisjoint(test["row_id"])
    assert train[BINARY_TARGET].mean() == pytest.approx(test[BINARY_TARGET].mean(), abs=0.02)


def test_split_features_target(raw_frame: pd.DataFrame) -> None:
    X, y = split_features_target(build_binary_target(raw_frame, [1, 2]))
    assert list(X.columns) == list(FEATURE_COLUMNS)
    assert y.name == BINARY_TARGET


class TestCsvDataSource:
    def test_loads_csv(self, tmp_path: Path, raw_frame: pd.DataFrame) -> None:
        path = tmp_path / "data.csv"
        raw_frame.to_csv(path, index=False)
        pd.testing.assert_frame_equal(CsvDataSource(path).load(), raw_frame)

    def test_missing_file_has_helpful_message(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Kaggle"):
            CsvDataSource(tmp_path / "missing.csv").load()
