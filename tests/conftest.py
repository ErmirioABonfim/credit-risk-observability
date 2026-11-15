import os
import tempfile

import numpy as np
import pandas as pd
import pytest

# isola os artefatos dos testes do workspace real
os.environ.setdefault("CREDIT_RISK_ROOT", tempfile.mkdtemp(prefix="credit-risk-tests-"))

from credit_risk import config  # noqa: E402
from credit_risk.data import pseudonymize  # noqa: E402


@pytest.fixture
def clean_batch() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({
        config.ID_COL: pseudonymize(pd.Series([f"c{i}" for i in range(n)])),
        "limit_bal": rng.integers(10, 500, n) * 1000.0,
        "education": rng.integers(1, 5, n),
        "age": rng.integers(21, 70, n),
        **{c: rng.integers(-2, 3, n) for c in config.PAY_STATUS_COLS},
        **{c: rng.normal(40_000, 20_000, n).round() for c in config.BILL_COLS},
        **{c: rng.exponential(4_000, n).round() for c in config.PAY_AMT_COLS},
        config.TARGET: rng.binomial(1, 0.22, n),
    })
    return df
