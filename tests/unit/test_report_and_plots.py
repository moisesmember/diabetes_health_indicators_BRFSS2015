import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from diabetes_risk.domain.schemas import TARGET_COLUMN
from diabetes_risk.evaluation import plots
from diabetes_risk.evaluation.importance import permutation_feature_importance
from diabetes_risk.evaluation.metrics import GateResult, compute_metrics, risk_band_metrics
from diabetes_risk.evaluation.report import build_report
from diabetes_risk.models.bundle import ModelBundle
from diabetes_risk.models.risk import RiskBands


@pytest.fixture
def predictions() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    proba = np.clip(0.35 * y + 0.65 * rng.random(400), 0, 1)
    return y, proba


class TestReport:
    def test_translates_metrics_into_sentences(
        self, predictions: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y, proba = predictions
        metrics = compute_metrics(y, proba, threshold=0.5)

        report = build_report(metrics, GateResult(passed=True))

        assert "✅ APROVADO" in report
        recall_per_100 = round(100 * metrics["recall"])
        assert f"o modelo sinaliza **{recall_per_100}**" in report
        assert "Alertas emitidos" in report

    def test_lists_gate_failures(self, predictions: tuple[np.ndarray, np.ndarray]) -> None:
        metrics = compute_metrics(*predictions)
        gate = GateResult(passed=False, failures=["roc_auc: 0.5 < mínimo 0.8"])

        report = build_report(metrics, gate)

        assert "❌ REPROVADO" in report
        assert "- roc_auc: 0.5 < mínimo 0.8" in report

    def test_includes_risk_band_section_when_available(
        self, predictions: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y, proba = predictions
        bands = RiskBands(medium=0.5, high=0.8)
        metrics = {**compute_metrics(y, proba, 0.5), **risk_band_metrics(y, proba, bands)}

        report = build_report(metrics, GateResult(passed=True), high_risk_threshold=0.8)

        assert "## Faixas de chance" in report
        assert "| Alta | com diabetes | ≥ 0.800 |" in report
        assert "| Baixa | sem diabetes | < 0.500 |" in report

    def test_omits_risk_band_section_without_band_metrics(
        self, predictions: tuple[np.ndarray, np.ndarray]
    ) -> None:
        report = build_report(compute_metrics(*predictions), GateResult(passed=True))
        assert "Faixas de chance" not in report

    def test_warns_when_probabilities_are_uncalibrated(self) -> None:
        y = np.array([0] * 90 + [1] * 10)
        overconfident = np.where(y == 1, 0.9, 0.6)  # ordena bem, mas superestima o risco

        report = build_report(compute_metrics(y, overconfident), GateResult(passed=True))

        assert "não estão calibradas" in report


class TestPlots:
    @pytest.mark.parametrize(
        "make_figure",
        [
            lambda y, p: plots.plot_confusion_matrix(y, p >= 0.5),
            lambda y, p: plots.plot_roc_curve(y, p, 0.5),
            lambda y, p: plots.plot_precision_recall_curve(y, p, 0.5),
            lambda y, p: plots.plot_threshold_tradeoff(y, p, 0.5),
            lambda y, p: plots.plot_calibration_curve(y, p),
        ],
        ids=["confusion", "roc", "pr", "tradeoff", "calibration"],
    )
    def test_curve_plots_render(self, make_figure, predictions, tmp_path) -> None:  # type: ignore[no-untyped-def]
        figure = make_figure(*predictions)

        assert isinstance(figure, Figure)
        assert figure.axes[0].get_title()
        figure.savefig(tmp_path / "plot.png")
        assert (tmp_path / "plot.png").stat().st_size > 0

    def test_feature_importance_plot_shows_top_features(self) -> None:
        importance = pd.DataFrame(
            {"feature": list("abcde"), "importance_mean": [5, 4, 3, 2, 1], "importance_std": 0.1}
        )
        figure = plots.plot_feature_importance(importance, top=3)
        labels = [tick.get_text() for tick in figure.axes[0].get_yticklabels()]
        assert labels == ["c", "b", "a"]  # maior no topo


def test_permutation_importance_ranks_informative_feature_first(
    trained_bundle: ModelBundle, raw_frame: pd.DataFrame
) -> None:
    X = raw_frame.drop(columns=TARGET_COLUMN)
    y = (raw_frame[TARGET_COLUMN] > 0).astype(int)

    importance = permutation_feature_importance(
        trained_bundle.pipeline, X, y, max_rows=300, n_repeats=2
    )

    assert list(importance.columns) == ["feature", "importance_mean", "importance_std"]
    assert set(importance["feature"]) == set(X.columns)
    # O gerador sintético usa HighBP, BMI e Age para definir o risco.
    assert set(importance["feature"].head(3)) & {"HighBP", "BMI", "Age"}
