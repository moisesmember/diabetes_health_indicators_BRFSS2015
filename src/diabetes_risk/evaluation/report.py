"""Relatório de avaliação em linguagem natural.

Métricas soltas ("recall = 0,855") são difíceis de comunicar. Este módulo
traduz cada uma para uma frase acionável ("de cada 100 pessoas com
diabetes/pré-diabetes, o modelo sinaliza 86"). O texto vira o artefato
``metrics_report.md`` e a descrição do run no MLflow.
"""

from __future__ import annotations

from collections.abc import Mapping

from diabetes_risk.evaluation.metrics import GateResult


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _per_100(value: float) -> str:
    return f"{round(100 * value)}"


def build_report(metrics: Mapping[str, float], gate: GateResult, beta: float = 2.0) -> str:
    """Gera o relatório Markdown a partir das métricas de :func:`compute_metrics`.

    Args:
        metrics: Métricas calculadas no conjunto de teste.
        gate: Resultado do quality gate.
        beta: Beta usado na F-score (para localizar a métrica ``f{beta}``).

    Returns:
        Texto Markdown com veredito, tabelas interpretadas e alertas.
    """
    m = metrics
    fbeta_name = f"f{beta:g}"
    status = "✅ APROVADO" if gate.passed else "❌ REPROVADO"

    lines = [
        "# Relatório de avaliação — risco de diabetes",
        "",
        f"**Quality gate:** {status}",
        *[f"- {failure}" for failure in gate.failures],
        "",
        f"Prevalência no teste: **{_pct(m['prevalence'])}** "
        f"(pré-diabetes + diabetes). Limiar de decisão: **{m['threshold']:.3f}**.",
        "",
        "## Capacidade de ordenar risco (independe do limiar)",
        "",
        "| Métrica | Valor | Como ler |",
        "|---|---|---|",
        f"| ROC-AUC | {m['roc_auc']:.3f} | Sorteando uma pessoa com e uma sem a doença, o modelo "
        f"dá risco maior à primeira em {_pct(m['roc_auc'])} das vezes (50% = acaso). |",
        f"| PR-AUC | {m['pr_auc']:.3f} | Precisão média ao longo dos limiares; o acaso daria "
        f"{m['prevalence']:.3f}, então o modelo é **{m['pr_auc_lift']:.1f}×** melhor "
        "que o acaso. |",
        "",
        "## Qualidade da decisão (no limiar escolhido)",
        "",
        "| Métrica | Valor | Como ler |",
        "|---|---|---|",
        f"| Recall (sensibilidade) | {m['recall']:.3f} | De cada 100 pessoas com diabetes ou "
        f"pré-diabetes, o modelo sinaliza **{_per_100(m['recall'])}**. |",
        f"| Precisão (VPP) | {m['precision']:.3f} | De cada 100 alertas, "
        f"**{_per_100(m['precision'])}** são casos reais. |",
        f"| Especificidade | {m['specificity']:.3f} | De cada 100 pessoas sem a doença, "
        f"**{_per_100(m['specificity'])}** não recebem alerta. |",
        f'| VPN | {m["npv"]:.3f} | Quando o modelo diz "sem risco", acerta '
        f"**{_pct(m['npv'])}** das vezes. |",
        f"| F{beta:g} | {m[fbeta_name]:.3f} | Equilíbrio precisão/recall com recall "
        f"{beta:g}× mais importante (critério de escolha do limiar). |",
        "",
        "## Impacto operacional (a cada 1.000 pessoas triadas)",
        "",
        "| Indicador | Valor |",
        "|---|---|",
        f"| Alertas emitidos | {m['alerts_per_1000']:.0f} |",
        f"| Casos perdidos (falsos negativos) | {m['missed_cases_per_1000']:.0f} |",
        f"| Falsos alarmes (falsos positivos) | {m['false_alarms_per_1000']:.0f} |",
        "",
        "## Calibração das probabilidades",
        "",
        f"Brier score: **{m['brier']:.3f}** · Brier skill: **{m['brier_skill']:+.3f}** "
        "(ganho sobre prever sempre a prevalência; 0 = igual, 1 = perfeito).",
        "",
    ]
    if m["brier_skill"] < 0:
        lines += [
            "> ⚠️ Brier skill negativo: as probabilidades **não estão calibradas** "
            "(efeito esperado de `class_weight=balanced`). Use o score para **ordenar e "
            "triar**, não como risco absoluto. Veja `figures/calibration_curve.png`.",
            "",
        ]
    lines += [
        "## Gráficos",
        "",
        "`confusion_matrix.png`, `roc_curve.png`, `precision_recall_curve.png`, "
        "`threshold_tradeoff.png`, `calibration_curve.png`, `feature_importance.png`.",
        "",
    ]
    return "\n".join(lines)
