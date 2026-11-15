# Imagem única do pipeline de sustentação (treino, validação, monitoramento e MLflow UI)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 CREDIT_RISK_ROOT=/app \
    MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY pyproject.toml .
COPY src ./src
RUN pip install -e . && useradd -m appuser \n    && mkdir -p /app/mlstore /app/data /app/reports /app/logs /app/models && chown -R appuser /app
USER appuser

CMD ["credit-risk", "run-all"]
