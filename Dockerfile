# Imagem de inferência. O modelo NÃO é embutido: monte-o em /app/models
# (ex.: `docker run -v ./models:/app/models ...`), assim a mesma imagem serve
# qualquer versão do modelo.

FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

# Dependências primeiro (camada cacheada enquanto o lockfile não muda).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev


FROM python:3.13-slim
WORKDIR /app
RUN useradd --create-home --uid 1000 app
COPY --from=builder --chown=app:app /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    MODEL_PATH=/app/models/model.joblib \
    PYTHONUNBUFFERED=1
USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "diabetes_risk.serving.api:create_app_from_env", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]
