"""Configuração central do projeto (caminhos, sementes e limiares de alerta)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.getenv("CREDIT_RISK_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REFERENCE_DIR = DATA_DIR / "reference"
PRODUCTION_DIR = DATA_DIR / "production"
RESTRICTED_DIR = DATA_DIR / "restricted"   # atributos sensíveis segregados (só auditoria)
QUARANTINE_DIR = DATA_DIR / "quarantine"   # lotes bloqueados pelo contrato
STATE_FILE = DATA_DIR / "monitoring_state.json"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"
LOG_FILE = LOGS_DIR / "pipeline.jsonl"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
MLFLOW_EXPERIMENT = "credit-risk-observability"
MODEL_NAME = "credit-scoring-xgb"

# Dataset público: "Default of Credit Card Clients" (UCI / OpenML id 42477, 30.000 linhas)
OPENML_DATA_ID = 42477
RAW_FILE = RAW_DIR / "credit_default.csv"

RANDOM_STATE = 42
REFERENCE_SHARE = 0.6          # 60% referência (treino) / 40% produção
N_PRODUCTION_WINDOWS = 6       # meses simulados de produção
WINDOW_START = "2026-01"       # primeiro mês simulado

TARGET = "default"
ID_COL = "customer_id"
SENSITIVE_COLS = ["sex", "marriage"]   # nunca entram no modelo (privacy by default)

PAY_STATUS_COLS = [f"pay_{i}" for i in range(1, 7)]
BILL_COLS = [f"bill_amt_{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"pay_amt_{i}" for i in range(1, 7)]
FEATURES = ["limit_bal", "education", "age", *PAY_STATUS_COLS, *BILL_COLS, *PAY_AMT_COLS]
KEY_FEATURES = ["limit_bal", "age", "pay_1", "bill_amt_1", "pay_amt_1"]  # destaque nos painéis

# Limiares de alerta (documentados no README)
PSI_ALERT = 0.20               # PSI > 0,2 = mudança significativa
PSI_WARN = 0.10
KS_PVALUE_ALERT = 0.05
DRIFT_SHARE_ALERT = 0.30       # >30% das features com drift => dataset drift
AUC_DROP_ALERT = 0.05          # queda absoluta de AUC vs. holdout de referência
DEFAULT_RATE_DELTA_ALERT = 0.05

for _d in (RAW_DIR, REFERENCE_DIR, PRODUCTION_DIR, RESTRICTED_DIR, QUARANTINE_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
