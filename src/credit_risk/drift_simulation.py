"""Simulação da passagem do tempo em produção com Data Drift e Concept Drift.

Cenário macroeconômico (6 meses, intensidade crescente k = 0 → 1):

| Mecanismo (componente)  | O que muda                                         | Tipo           |
|-------------------------|----------------------------------------------------|----------------|
| inflation_bills         | faturas (bill_amt_*) sobem até +45%                | Data drift     |
| new_young_low_limit     | até 50% dos clientes novos: 18-28 anos, limite 30-70% menor | Data drift |
| lower_payments          | pagamentos (pay_amt_*) caem até -40%               | Data drift     |
| payment_delays          | +1 mês de atraso em pay_1..pay_3 (prob. até 40%)   | Data drift     |
| concept_default         | clientes com utilização > 60% passam a dar default | Concept drift  |
|                         | (até 35% dos bons pagadores viram default)         |                |

O mês 1 (k=0) é o controle sem drift. Cada componente pode ser ligado/desligado para a
análise causal (contrafactual) em `causal_analysis.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data import window_months
from .logging_utils import stage

ALL_COMPONENTS = ("inflation_bills", "new_young_low_limit", "lower_payments", "payment_delays", "concept_default")
DATA_DRIFT_COMPONENTS = ALL_COMPONENTS[:-1]


def intensity(month_idx: int) -> float:
    return month_idx / (config.N_PRODUCTION_WINDOWS - 1)


def simulate(df: pd.DataFrame, k: float, seed: int, components=ALL_COMPONENTS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)       # mesmo seed => mesmos sorteios => contrafactuais comparáveis
    out = df.copy()
    n = len(out)
    u_profile, u_limit, u_age = rng.random(n), rng.uniform(0.3, 0.7, n), rng.integers(18, 29, n)
    u_delay = rng.random((n, 3))
    u_concept = rng.random(n)

    if "inflation_bills" in components:
        out[config.BILL_COLS] = (out[config.BILL_COLS] * (1 + 0.45 * k)).round(0)
    if "new_young_low_limit" in components:
        new = u_profile < 0.5 * k
        out.loc[new, "age"] = u_age[new]
        out.loc[new, "limit_bal"] = np.maximum(10_000, (out.loc[new, "limit_bal"] * u_limit[new]).round(-3))
    if "lower_payments" in components:
        out[config.PAY_AMT_COLS] = (out[config.PAY_AMT_COLS] * (1 - 0.40 * k)).round(0)
    if "payment_delays" in components:
        for j, col in enumerate(["pay_1", "pay_2", "pay_3"]):
            hit = u_delay[:, j] < 0.40 * k
            out.loc[hit, col] = np.clip(np.maximum(out.loc[hit, col], 0) + 1, -2, 8)
    if "concept_default" in components:
        # mesma feature, outro desfecho: utilização alta agora leva a default (P(y|X) mudou)
        util = out["bill_amt_1"] / out["limit_bal"]
        flip = (out[config.TARGET] == 0) & (util > 0.6) & (u_concept < 0.35 * k)
        out.loc[flip, config.TARGET] = 1

    int_cols = ["age", "education", *config.PAY_STATUS_COLS, config.TARGET]
    out[int_cols] = out[int_cols].astype(int)
    return out


def load_clean_window(month: str) -> pd.DataFrame:
    return pd.read_parquet(config.PRODUCTION_DIR / f"clean_{month}.parquet")


def build_production_windows() -> list[str]:
    """Gera `production/window_<mês>.parquet` com drift progressivo."""
    paths = []
    with stage("simulate_drift") as ctx:
        for i, month in enumerate(window_months()):
            drifted = simulate(load_clean_window(month), intensity(i), seed=config.RANDOM_STATE + i)
            path = config.PRODUCTION_DIR / f"window_{month}.parquet"
            drifted.to_parquet(path, index=False)
            paths.append(str(path))
            ctx[f"default_rate_{month}"] = round(float(drifted[config.TARGET].mean()), 4)
    return paths
