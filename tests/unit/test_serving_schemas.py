import pytest
from pydantic import ValidationError

from diabetes_risk.domain.schemas import FEATURE_RANGES
from diabetes_risk.serving.schemas import PatientFeatures, PredictionRequest


def _bounds(field_name: str) -> tuple[float, float]:
    metadata = PatientFeatures.model_fields[field_name].metadata
    low = next(m.ge for m in metadata if hasattr(m, "ge"))
    high = next(m.le for m in metadata if hasattr(m, "le"))
    return low, high


def test_api_contract_matches_domain_ranges() -> None:
    assert set(PatientFeatures.model_fields) == set(FEATURE_RANGES)
    for name, expected in FEATURE_RANGES.items():
        assert _bounds(name) == expected, name


def test_rejects_out_of_range_value() -> None:
    payload = {name: low for name, (low, _) in FEATURE_RANGES.items()} | {"BMI": 5}
    with pytest.raises(ValidationError, match="BMI"):
        PatientFeatures.model_validate(payload)


def test_rejects_empty_batch() -> None:
    with pytest.raises(ValidationError, match="patients"):
        PredictionRequest(patients=[])
