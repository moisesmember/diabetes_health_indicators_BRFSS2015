# Servidor de tracking do MLflow com backend PostgreSQL.
# A versão acompanha a do uv.lock, para que cliente e servidor sejam compatíveis.

FROM python:3.13-slim

ARG MLFLOW_VERSION=3.17.2
ARG PSYCOPG_VERSION=3.2.3
RUN pip install --no-cache-dir \
        "mlflow==${MLFLOW_VERSION}" \
        "psycopg[binary]==${PSYCOPG_VERSION}"

# Usuário sem privilégios; o volume de artefatos herda a posse desta pasta.
RUN useradd --create-home --uid 1000 mlflow \
    && mkdir -p /mlartifacts \
    && chown mlflow:mlflow /mlartifacts
USER mlflow
ENV MLFLOW_DISABLE_AGENT_HINT=1

EXPOSE 5000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=6 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"
