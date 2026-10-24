# syntax=docker/dockerfile:1
#
# Imagens do projeto (multi-stage, multi-target):
#
#   trainer  executa o pipeline (prepare -> train -> evaluate) e registra no MLflow
#   api      serve o modelo via FastAPI (alvo padrão; sem MLflow, imagem enxuta)
#
#   docker build --target trainer -t diabetes-risk/trainer .
#   docker build -t diabetes-risk/api .
#
# Dados, modelos e relatórios NÃO são embutidos: entram por volume, então a
# mesma imagem serve qualquer versão de dados/modelo.

ARG PYTHON_VERSION=3.13

# ---------------------------------------------------------------- builders ---
FROM python:${PYTHON_VERSION}-slim AS builder-base
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./

FROM builder-base AS api-builder
# Dependências primeiro: esta camada só é refeita quando o uv.lock muda.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

FROM builder-base AS trainer-builder
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra mlflow --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra mlflow --no-editable

# ----------------------------------------------------------------- runtime ---
FROM python:${PYTHON_VERSION}-slim AS runtime-base
ARG GIT_COMMIT=unknown
LABEL org.opencontainers.image.source="https://github.com/moisesmember/diabetes_health_indicators_BRFSS2015" \
      org.opencontainers.image.revision="${GIT_COMMIT}"
RUN useradd --create-home --uid 1000 app
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    GIT_COMMIT=${GIT_COMMIT}

FROM runtime-base AS trainer
COPY --from=trainer-builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app params.yaml ./
RUN mkdir -p data models reports && chown -R app:app /app
ENV MPLCONFIGDIR=/tmp/matplotlib \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    GIT_PYTHON_REFRESH=quiet
USER app
CMD ["diabetes-risk", "run"]

FROM runtime-base AS api
COPY --from=api-builder --chown=app:app /app/.venv /app/.venv
ENV MODEL_PATH=/app/models/model.joblib
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "diabetes_risk.serving.api:create_app_from_env", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]
