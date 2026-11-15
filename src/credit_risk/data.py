"""Ingestão, pseudonimização, minimização e particionamento Referência x Produção."""
from __future__ import annotations

import hashlib
import hmac
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config
from .logging_utils import log_event, stage

RENAME = {
    "x1": "limit_bal", "x2": "sex", "x3": "education", "x4": "marriage", "x5": "age",
    **{f"x{6 + i}": f"pay_{i + 1}" for i in range(6)},
    **{f"x{12 + i}": f"bill_amt_{i + 1}" for i in range(6)},
    **{f"x{18 + i}": f"pay_amt_{i + 1}" for i in range(6)},
    "y": config.TARGET,
}


def download_raw() -> pd.DataFrame:
    """Baixa o dataset público (OpenML 42477) se ainda não existir localmente."""
    if config.RAW_FILE.exists():
        return pd.read_csv(config.RAW_FILE)
    from sklearn.datasets import fetch_openml

    frame = fetch_openml(data_id=config.OPENML_DATA_ID, as_frame=True,
                         data_home=str(config.RAW_DIR / "openml")).frame
    frame.to_csv(config.RAW_FILE, index=False)
    return frame


def pseudonymize(values: pd.Series) -> pd.Series:
    """HMAC-SHA256 com segredo fora do código (LGPD art. 13 §4 — pseudonimização).

    Em produção o identificador real (CPF) seria substituído na borda de ingestão; o
    segredo fica em cofre (ex.: Vault/Secrets Manager) e não é versionado.
    """
    salt = os.getenv("PSEUDONYMIZATION_KEY", "dev-only-key-change-me").encode()
    return values.astype(str).map(lambda v: hmac.new(salt, v.encode(), hashlib.sha256).hexdigest()[:20])


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=RENAME).copy()
    df[config.TARGET] = df[config.TARGET].astype(int)
    # Códigos não documentados de escolaridade (0, 5, 6) => 4 ("outros"), conforme dicionário UCI
    df["education"] = df["education"].replace({0: 4, 5: 4, 6: 4})
    df["marriage"] = df["marriage"].replace({0: 3})
    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    df.insert(0, config.ID_COL, pseudonymize(pd.Series([f"client-{i}" for i in range(len(df))])))
    log_event("raw_cleaned", rows=len(df), duplicates_removed=n_dup)
    return df


def segregate_sensitive(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Minimização: atributos sensíveis saem do dataset de modelagem e vão para área restrita."""
    sensitive = df[[config.ID_COL, *config.SENSITIVE_COLS]].copy()
    model_df = df.drop(columns=config.SENSITIVE_COLS)
    return model_df, sensitive


def prepare() -> dict:
    with stage("prepare_data") as ctx:
        df = clean(download_raw())
        model_df, sensitive = segregate_sensitive(df)
        ref, prod = train_test_split(model_df, train_size=config.REFERENCE_SHARE,
                                     stratify=model_df[config.TARGET], random_state=config.RANDOM_STATE)
        ref.reset_index(drop=True).to_parquet(config.REFERENCE_DIR / "reference.parquet", index=False)
        sensitive.to_parquet(config.RESTRICTED_DIR / "sensitive_attributes.parquet", index=False)

        # Produção "sem drift" (baseline): fatiada em janelas mensais; o drift é injetado depois
        prod = prod.sample(frac=1.0, random_state=config.RANDOM_STATE).reset_index(drop=True)
        windows = np.array_split(np.arange(len(prod)), config.N_PRODUCTION_WINDOWS)
        months = pd.period_range(config.WINDOW_START, periods=config.N_PRODUCTION_WINDOWS, freq="M")
        for month, idx in zip(months, windows):
            prod.iloc[idx].to_parquet(config.PRODUCTION_DIR / f"clean_{month}.parquet", index=False)

        ctx.update(reference_rows=len(ref), production_rows=len(prod), windows=len(months))
    return ctx


def load_reference() -> pd.DataFrame:
    return pd.read_parquet(config.REFERENCE_DIR / "reference.parquet")


def window_months() -> list[str]:
    return [str(p) for p in pd.period_range(config.WINDOW_START, periods=config.N_PRODUCTION_WINDOWS, freq="M")]
