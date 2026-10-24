# Diabetes Risk — pipeline de ML (BRFSS 2015)

Pipeline reproduzível de triagem de **risco de pré-diabetes/diabetes** a partir
dos indicadores de saúde da pesquisa
[BRFSS 2015 (CDC)](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset).
Cobre do CSV bruto até uma API REST, com validação de dados, rastreamento de
experimentos, quality gate e CI.

## Resultados (teste hold-out, 20%)

| Métrica | Valor | Leitura |
|---|---|---|
| ROC-AUC | 0,810 | boa separação entre classes |
| PR-AUC | 0,466 | ~3× a taxa base de positivos (~15%) |
| Recall | 0,855 | 85% dos casos de risco são detectados |
| Precisão | 0,308 | ~1 em cada 3 alertas é positivo |
| F2 | 0,631 | métrica de escolha do limiar (prioriza recall) |

O limiar de decisão (≈0,40) é escolhido no treino, via validação cruzada,
para maximizar a F2: em triagem, deixar passar um caso (falso negativo) é
mais caro do que um alerta extra. As métricas completas ficam em
[reports/metrics.json](reports/metrics.json).

## Arquitetura

```
CSV bruto ─► prepare ─► train ─► evaluate ─► ModelBundle ─► API (FastAPI)
             │          │        │                │
             Pandera    CV +     quality gate      Pydantic
             dedup      limiar   (exit 1 se falhar)
             split      ótimo
```

| Camada | Módulo | Responsabilidade |
|---|---|---|
| Domínio | [domain/](src/diabetes_risk/domain/) | colunas, faixas válidas, schema Pandera, *Protocols* |
| Dados | [data/](src/diabetes_risk/data/) | fontes de dados, validação, alvo binário, deduplicação, split |
| Features | [features/](src/diabetes_risk/features/) | transformer clínico + pré-processamento (sklearn) |
| Modelos | [models/](src/diabetes_risk/models/) | registry de modelos e `ModelBundle` (pipeline + limiar) |
| Avaliação | [evaluation/](src/diabetes_risk/evaluation/) | métricas, limiar ótimo, quality gate |
| Tracking | [tracking/](src/diabetes_risk/tracking/) | tracker local (JSON) ou MLflow |
| Monitoramento | [monitoring/](src/diabetes_risk/monitoring/) | drift por PSI |
| Etapas | [pipelines/](src/diabetes_risk/pipelines/) | orquestra as camadas: `prepare`, `train`, `evaluate` |
| Serving | [serving/](src/diabetes_risk/serving/) | API REST e contratos de entrada/saída |

### Decisões de design

- **Uma fonte de configuração**: [params.yaml](params.yaml), validado por
  Pydantic ([config.py](src/diabetes_risk/config.py)) e lido pelo DVC. Chaves
  desconhecidas são rejeitadas.
- **Sem *training-serving skew***: a engenharia de features e o pré-processamento
  vivem dentro do `Pipeline` sklearn salvo no `ModelBundle`.
- **Sem vazamento**: deduplicação antes do split; limiar escolhido com
  predições *out-of-fold*; o teste só é usado em `evaluate`.
- **SOLID**:
  - *SRP*: um módulo por responsabilidade;
  - *OCP*: novo modelo = uma função com `@register_model`;
  - *LSP*: transformers seguem o contrato sklearn;
  - *ISP/DIP*: as etapas dependem dos *Protocols* `DataSource` e
    `ExperimentTracker`, injetáveis.

## Como usar

Pré-requisitos: Python ≥ 3.11 e [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras              # dependências (+ MLflow) e dev tools
```

Baixe `diabetes_012_health_indicators_BRFSS2015.csv` do
[Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
para `data/raw/`.

### Rodar o pipeline

```bash
uv tool install dvc               # uma vez
dvc repro                         # roda só as etapas que mudaram
dvc metrics show                  # métricas atuais
dvc params diff                   # o que mudou desde o último commit
```

Ou etapa por etapa, sem DVC:

```bash
uv run diabetes-risk prepare
uv run diabetes-risk train
uv run diabetes-risk evaluate     # exit code 1 se o quality gate reprovar
```

Para trocar de modelo, edite `model.name` em `params.yaml` (`logistic_regression`,
`hist_gradient_boosting` ou `random_forest`) e rode `dvc repro`.
Para usar MLflow, defina `tracking.backend: mlflow` e `tracking.uri`
(ex.: `sqlite:///mlflow.db`); depois `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`.

### Servir o modelo

```bash
uv run diabetes-risk serve        # http://127.0.0.1:8000/docs
```

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "patients": [{"HighBP":1,"HighChol":1,"CholCheck":1,"BMI":38,"Smoker":1,"Stroke":0,
    "HeartDiseaseorAttack":1,"PhysActivity":0,"Fruits":0,"Veggies":1,"HvyAlcoholConsump":0,
    "AnyHealthcare":1,"NoDocbcCost":0,"GenHlth":4,"MentHlth":5,"PhysHlth":10,"DiffWalk":1,
    "Sex":1,"Age":10,"Education":4,"Income":3}]}'
# {"model":"hist_gradient_boosting","threshold":0.396,"predictions":[{"probability":0.90,"at_risk":true}]}
```

Com Docker (o modelo é montado como volume, não embutido na imagem):

```bash
docker build -t diabetes-risk .
docker run -p 8000:8000 -v ./models:/app/models:ro diabetes-risk
```

### Monitorar drift

```python
from diabetes_risk.monitoring.drift import drift_report
drift_report(train_df, production_df, columns=FEATURE_COLUMNS)  # PSI > 0,25 = drift
```

## Desenvolvimento

```bash
uv run pytest                     # unitários + integração
uv run pytest -m "not integration"   # só unitários
uv run pytest --cov               # com cobertura
uv run ruff check src tests && uv run ruff format src tests
uv run mypy src
uv run pre-commit install         # checagens automáticas a cada commit
```

Os testes usam dados **sintéticos** gerados em
[tests/conftest.py](tests/conftest.py); não dependem do dataset real.
O CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) roda lint, mypy,
testes em Python 3.11 e 3.13 (cobertura mínima de 90%) e o build da imagem Docker.

## Limitações e próximos passos

- **Calibração**: `class_weight=balanced` desloca as probabilidades para cima
  (Brier ≈ 0,18). Se a probabilidade for usada como risco absoluto, adicione
  `CalibratedClassifierCV`.
- **Tuning**: os hiperparâmetros são fixos; Optuna pode ser plugado na etapa `train`.
- **Remote do DVC**: configure um (`dvc remote add`) para compartilhar dados e modelos.
- **Uso clínico**: o modelo é uma ferramenta de triagem a partir de
  autorrelato, não um diagnóstico.
