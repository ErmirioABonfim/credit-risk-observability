"""Data Contract (Pandera) — "lixo não entra no modelo".

Regras rígidas aplicadas em TODO lote antes de treino ou inferência. Se o lote viola o
contrato, ele é enviado para quarentena e a ingestão é bloqueada (DataContractViolation).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

from . import config
from .logging_utils import log_event, stage

CONTRACT_VERSION = "1.0.0"


class DataContractViolation(Exception):
    """Lote rejeitado pelo contrato de dados — ingestão bloqueada."""

    def __init__(self, batch: str, failures: pd.DataFrame):
        self.batch = batch
        self.failures = failures
        super().__init__(f"Lote '{batch}' bloqueado: {len(failures)} violação(ões) do contrato v{CONTRACT_VERSION}")


def _money(lo: float, hi: float, **kw) -> Column:
    return Column(float, Check.in_range(lo, hi), nullable=False, coerce=True, **kw)


CREDIT_CONTRACT = DataFrameSchema(
    name=f"credit_scoring_contract_v{CONTRACT_VERSION}",
    columns={
        # R1 — identificador pseudonimizado, obrigatório e único (sem duplicatas de cliente)
        config.ID_COL: Column(str, Check.str_matches(r"^[0-9a-f]{20}$"), nullable=False, unique=True),
        # R2 — limite de crédito (proxy de renda): não nulo e dentro de faixa plausível
        "limit_bal": _money(1_000, 2_000_000),
        # R3 — maioridade civil: só clientes com 18+ anos (CC art. 5º) e idade plausível
        "age": Column(int, [Check.ge(18), Check.le(100)], nullable=False, coerce=True),
        # R4 — domínio fechado de escolaridade
        "education": Column(int, Check.isin([1, 2, 3, 4]), nullable=False, coerce=True),
        # R5 — status de pagamento no domínio documentado (-2 .. 8 meses de atraso)
        **{c: Column(int, Check.in_range(-2, 8), nullable=False, coerce=True) for c in config.PAY_STATUS_COLS},
        # R6 — fatura pode ser negativa (crédito a favor), mas dentro de limites
        **{c: _money(-1_000_000, 5_000_000) for c in config.BILL_COLS},
        # R7 — pagamentos nunca negativos
        **{c: _money(0, 5_000_000) for c in config.PAY_AMT_COLS},
        # R8 — rótulo binário (opcional: em inferência o rótulo chega com atraso)
        config.TARGET: Column(int, Check.isin([0, 1]), nullable=False, coerce=True, required=False),
    },
    checks=[
        # R9 — sem linhas duplicadas no lote
        Check(lambda df: ~df.duplicated(), element_wise=False, name="no_duplicate_rows",
              error="linhas duplicadas no lote"),
        # R10 — lote não pode estar vazio
        Check(lambda df: len(df) > 0, name="non_empty_batch", error="lote vazio"),
    ],
    # R11 — strict: colunas não previstas (ex.: CPF, nome) são rejeitadas => protege contra vazamento de PII
    strict=True,
    ordered=False,
)


def validate_batch(df: pd.DataFrame, batch: str, block: bool = True) -> pd.DataFrame:
    """Valida o lote; em falha grava quarentena + relatório e (se block) bloqueia a ingestão."""
    with stage("validate_contract", batch=batch, rows_in=len(df)) as ctx:
        try:
            validated = CREDIT_CONTRACT.validate(df, lazy=True)
        except pa.errors.SchemaErrors as err:
            failures = err.failure_cases.copy()
            failures["failure_case"] = failures["failure_case"].astype(str)
            _quarantine(df, batch, failures)
            ctx.update(status_contract="rejected", n_violations=len(failures),
                       rules_violated=sorted(failures["check"].astype(str).unique().tolist()))
            if block:
                log_event("ingestion_blocked", level="ERROR", batch=batch, n_violations=len(failures))
                raise DataContractViolation(batch, failures) from None
            return df.iloc[0:0]
        ctx.update(status_contract="accepted", rows_valid=len(validated))
    return validated


def _quarantine(df: pd.DataFrame, batch: str, failures: pd.DataFrame) -> None:
    df.to_csv(config.QUARANTINE_DIR / f"{batch}_rejected_batch.csv", index=False)
    failures.to_csv(config.QUARANTINE_DIR / f"{batch}_violations.csv", index=False)
    f = failures.copy()
    # checks de nível de lote (ex.: duplicatas) são reportados uma vez por linha, não por coluna
    is_df_check = (f["schema_context"] == "DataFrameSchema") & (f["check"] != "column_in_schema")
    f.loc[is_df_check, "column"] = "<lote>"
    f.loc[f["check"] == "column_in_schema", "column"] = f.loc[f["check"] == "column_in_schema", "failure_case"]
    f["row"] = f["index"].astype(str)
    summary = (f.groupby(["column", "check"], dropna=False)["row"].nunique()
               .reset_index(name="occurrences").sort_values("column"))
    (config.QUARANTINE_DIR / f"{batch}_summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), ensure_ascii=False, indent=2), encoding="utf-8")


def make_corrupted_batch(clean: pd.DataFrame, n: int = 300, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Lote propositalmente sujo para demonstrar o contrato bloqueando a ingestão."""
    rng = np.random.default_rng(seed)
    bad = clean.sample(n, random_state=seed).reset_index(drop=True).astype({"limit_bal": float})
    bad.loc[rng.choice(n, 8, replace=False), "limit_bal"] = np.nan        # renda/limite nulo
    bad.loc[rng.choice(n, 5, replace=False), "age"] = 15                   # menor de idade
    bad.loc[rng.choice(n, 2, replace=False), "age"] = 140                  # idade impossível
    bad.loc[rng.choice(n, 4, replace=False), "education"] = 9              # fora do domínio
    bad.loc[rng.choice(n, 4, replace=False), "pay_amt_1"] = -500.0         # pagamento negativo
    bad.loc[rng.choice(n, 3, replace=False), "limit_bal"] = 50_000_000.0   # limite absurdo
    bad = pd.concat([bad, bad.iloc[:5]], ignore_index=True)                # duplicatas
    bad["cpf"] = "000.000.000-00"                                          # PII não autorizada
    return bad
