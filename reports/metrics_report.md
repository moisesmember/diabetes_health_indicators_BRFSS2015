# Relatório de avaliação — risco de diabetes

**Quality gate:** ✅ APROVADO

Prevalência no teste: **17.3%** (pré-diabetes + diabetes). Limiar de decisão: **0.396**.

## Capacidade de ordenar risco (independe do limiar)

| Métrica | Valor | Como ler |
|---|---|---|
| ROC-AUC | 0.810 | Sorteando uma pessoa com e uma sem a doença, o modelo dá risco maior à primeira em 81.0% das vezes (50% = acaso). |
| PR-AUC | 0.466 | Precisão média ao longo dos limiares; o acaso daria 0.173, então o modelo é **2.7×** melhor que o acaso. |

## Qualidade da decisão (no limiar escolhido)

| Métrica | Valor | Como ler |
|---|---|---|
| Recall (sensibilidade) | 0.855 | De cada 100 pessoas com diabetes ou pré-diabetes, o modelo sinaliza **86**. |
| Precisão (VPP) | 0.308 | De cada 100 alertas, **31** são casos reais. |
| Especificidade | 0.600 | De cada 100 pessoas sem a doença, **60** não recebem alerta. |
| VPN | 0.952 | Quando o modelo diz "sem risco", acerta **95.2%** das vezes. |
| F2 | 0.631 | Equilíbrio precisão/recall com recall 2× mais importante (critério de escolha do limiar). |

## Impacto operacional (a cada 1.000 pessoas triadas)

| Indicador | Valor |
|---|---|
| Alertas emitidos | 479 |
| Casos perdidos (falsos negativos) | 25 |
| Falsos alarmes (falsos positivos) | 331 |

## Calibração das probabilidades

Brier score: **0.180** · Brier skill: **-0.262** (ganho sobre prever sempre a prevalência; 0 = igual, 1 = perfeito).

## Faixas de chance (o que o usuário recebe)

| Chance | Classificação | Score | % das pessoas | Casos reais na faixa | Indicação |
|---|---|---|---|---|---|
| Alta | com diabetes | ≥ 0.747 | 13.0% | **51 em 100** | exame confirmatório em breve |
| Média | com diabetes | 0.396 a 0.747 | 34.9% | **23 em 100** | exame confirmatório |
| Baixa | sem diabetes | < 0.396 | 52.1% | **5 em 100** | acompanhamento de rotina |

> ⚠️ Brier skill negativo: as probabilidades **não estão calibradas** (efeito esperado de `class_weight=balanced`). Use o score para **ordenar e triar**, não como risco absoluto. Veja `figures/calibration_curve.png`.

## Gráficos

`confusion_matrix.png`, `roc_curve.png`, `precision_recall_curve.png`, `threshold_tradeoff.png`, `calibration_curve.png`, `feature_importance.png`.
