# Tutorial: executando o projeto com Docker Compose

Este guia sobe a stack completa de MLOps na sua máquina: servidor MLflow com
banco PostgreSQL, treino em container e API de inferência.

```
               ┌────────────┐       ┌───────────┐
 docker compose│  trainer   │──────►│  mlflow   │──► postgres (metadados)
   run trainer │ (pipeline) │ HTTP  │ :5000     │──► volume mlflow-artifacts
               └─────┬──────┘       └───────────┘    (modelos, gráficos)
                     │ grava models/model.joblib
                     ▼
               ┌────────────┐
               │    api     │ :8000  (lê ./models, somente leitura)
               └────────────┘
```

| Serviço | Imagem | Quando roda | Endereço |
|---|---|---|---|
| `postgres` | `postgres:16.4-alpine` | sempre (dependência do MLflow) | interno |
| `mlflow` | `diabetes-risk/mlflow:3.17.2` | `docker compose up` | <http://localhost:5000> |
| `trainer` | `diabetes-risk/trainer` | sob demanda (`docker compose run`) | — |
| `api` | `diabetes-risk/api` | `docker compose up` | <http://localhost:8000/docs> |

---

## 1. Pré-requisitos

- Docker Desktop (Windows/macOS) ou Docker Engine + plugin Compose v2 (Linux).
- O dataset `diabetes_012_health_indicators_BRFSS2015.csv`, baixado do
  [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
  e salvo em `data/raw/`.

```bash
docker compose version      # deve mostrar v2.x
ls data/raw/                # diabetes_012_health_indicators_BRFSS2015.csv
```

## 2. Configuração (opcional)

Todos os valores têm padrão. Para mudar portas ou senhas:

```bash
cp .env.example .env        # depois edite o .env
```

> Em qualquer ambiente compartilhado, troque `POSTGRES_PASSWORD`. O `.env`
> está no `.gitignore` e nunca deve ser commitado.

## 3. Primeira execução

**3.1 Construir as imagens** gravando o commit do código (aparece nos runs do MLflow):

```bash
# bash / zsh
export GIT_COMMIT=$(git rev-parse --short HEAD)
# PowerShell
$env:GIT_COMMIT = git rev-parse --short HEAD

docker compose --profile train build
```

**3.2 Subir o MLflow** (o Postgres sobe junto; `--wait` espera ficar saudável):

```bash
docker compose up -d --wait mlflow
```

Abra <http://localhost:5000>: o experimento ainda estará vazio.

**3.3 Treinar** (prepare → train → evaluate, registrando tudo no MLflow):

```bash
docker compose run --rm trainer
```

O container termina sozinho. Exit code `1` significa que o modelo foi
**reprovado no quality gate** (veja o motivo no log ou em `reports/metrics_report.md`).

**3.4 Subir a API:**

```bash
docker compose up -d --wait api
curl http://localhost:8000/model
```

Documentação interativa em <http://localhost:8000/docs>.

## 4. Lendo os resultados no MLflow

No experimento **diabetes-risk**, cada linha é um run (um modelo treinado).
Treino e avaliação escrevem no **mesmo run**:

| Onde | O que tem |
|---|---|
| **Overview → Description** | relatório em linguagem natural ("de cada 100 pessoas com diabetes, o modelo sinaliza 86") |
| **Model metrics** `cv/*` | validação cruzada no treino, incluindo `cv/roc_auc_std` (estabilidade entre folds) |
| **Model metrics** `test/*` | teste hold-out: as métricas que valem para decidir |
| **Parameters** | `model.*`, `training.*` e `data.*` vindos do `params.yaml` |
| **Tags** | `model_name`, `git_commit`, `gate_passed` |
| **Datasets** | `brfss2015-train` e `brfss2015-test`, com hash e schema (linhagem) |
| **Artifacts → evaluation/** | `metrics_report.md`, `feature_importance.csv`, `figures/*.png` |
| **Artifacts → model/** | modelo pyfunc com assinatura, exemplo de entrada e limiar embutido |

Métricas para olhar primeiro:

| Métrica | Pergunta que responde | Bom sinal |
|---|---|---|
| `test/recall` | Quantos doentes o modelo pega? | alto (triagem) |
| `test/missed_cases_per_1000` | Quantos casos escapam a cada 1.000 triados? | baixo |
| `test/alerts_per_1000` | Qual a carga de trabalho gerada? | compatível com a operação |
| `test/pr_auc_lift` | Quantas vezes melhor que o acaso? | > 1, quanto maior melhor |
| `test/brier_skill` | As probabilidades são confiáveis como risco absoluto? | > 0 |

Para **comparar modelos**, selecione vários runs e clique em **Compare**.

### Model Registry

Em **Models → diabetes-risk**, cada avaliação cria uma versão, com tags
`gate_passed`, `test_pr_auc`, `test_recall` e `git_commit`. A versão aprovada
mais recente recebe o alias **`@champion`**. Para consumi-la de qualquer lugar:

```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
model = mlflow.pyfunc.load_model("models:/diabetes-risk@champion")
model.predict(df)   # -> colunas probability e at_risk (limiar já aplicado)
```

## 5. Operações do dia a dia

### Retreinar com novos parâmetros

O `params.yaml` é montado no container, então não precisa rebuildar a imagem:

```bash
# edite params.yaml (ex.: model.name: random_forest)
docker compose run --rm trainer
docker compose restart api        # a API carrega o novo models/model.joblib
```

### Atualizar as imagens após mudar o código

```bash
export GIT_COMMIT=$(git rev-parse --short HEAD)
docker compose --profile train build         # rebuilda trainer e api (cache por camada)
docker compose up -d --wait api              # recria a API com a nova imagem
docker compose run --rm trainer              # (opcional) retreina com o novo código
```

Só as camadas afetadas são refeitas: mudar `src/` não reinstala dependências;
mudar o `uv.lock` sim.

### Versionar imagens por commit

```bash
export IMAGE_TAG=$(git rev-parse --short HEAD) GIT_COMMIT=$(git rev-parse --short HEAD)
docker compose --profile train build         # gera diabetes-risk/api:<commit>
docker compose up -d api                     # sobe exatamente essa versão
docker image ls diabetes-risk/api            # versões disponíveis para rollback
```

**Rollback:** `IMAGE_TAG=<commit-anterior> docker compose up -d api`.

### Atualizar a versão do MLflow

1. Atualize o cliente: `uv lock --upgrade-package mlflow && uv sync --all-extras`.
2. Use a mesma versão em `docker/mlflow/Dockerfile` (`ARG MLFLOW_VERSION`) e na
   tag `image:` do serviço `mlflow` no `compose.yaml`.
3. Rebuilde e suba: `docker compose build mlflow && docker compose up -d --wait mlflow`.
   O MLflow migra o schema do Postgres automaticamente ao iniciar.

> Faça backup do banco antes de trocar a versão principal (ex.: 3.x → 4.x).

### Logs, status e shell

```bash
docker compose ps                      # status e healthchecks
docker compose logs -f mlflow api      # acompanhar logs
docker compose exec api sh             # shell no container da API
```

### Parar e limpar

```bash
docker compose stop                    # para, mantendo tudo
docker compose down                    # remove containers (volumes preservados)
docker compose down -v                 # ⚠️ apaga também o histórico do MLflow e os modelos registrados
```

### Backup e restauração do MLflow

```bash
# metadados (experimentos, runs, registry)
docker compose exec postgres pg_dump -U mlflow mlflow > mlflow-backup.sql
# artefatos (modelos, gráficos)
docker run --rm -v diabetes-risk_mlflow-artifacts:/data -v "$PWD":/backup alpine \
    tar czf /backup/mlflow-artifacts.tgz -C /data .

# restauração dos metadados
docker compose exec -T postgres psql -U mlflow mlflow < mlflow-backup.sql
```

## 6. Solução de problemas

| Sintoma | Causa provável | Solução |
|---|---|---|
| `Dataset não encontrado em 'data/raw/...'` | CSV não baixado | veja a seção 1 |
| `api` reiniciando em loop | ainda não há `models/model.joblib` | rode `docker compose run --rm trainer` |
| `port is already allocated` | porta 5000/8000 ocupada (no macOS, o AirPlay usa a 5000) | defina `MLFLOW_PORT`/`API_PORT` no `.env` |
| `403 Invalid Host header` ao acessar o MLflow | host fora do `--allowed-hosts` | acrescente o host na lista do serviço `mlflow` no `compose.yaml` |
| `git_commit = unknown` nos runs | imagem construída sem `GIT_COMMIT` | rebuild conforme o passo 3.1 |
| trainer termina com exit code 1 | quality gate reprovado | leia `reports/metrics_report.md` |
| `mlflow` unhealthy | Postgres indisponível ou credenciais trocadas após criar o volume | `docker compose logs mlflow`; se mudou a senha, recrie o volume (`down -v`) |

## 7. Indo para produção

Esta stack é para desenvolvimento local. Em produção:

- troque o volume de artefatos por um bucket (`--artifacts-destination s3://...`);
- use um Postgres gerenciado e segredos fora do `.env` (Docker secrets, Vault, Secret Manager);
- coloque o MLflow atrás de autenticação (`mlflow server --app-name basic-auth` ou proxy com SSO);
- publique as imagens num registry (GHCR/ECR) a partir do CI, com a tag do commit.
