import numpy as np
import pandas as pd

from credit_risk import config
from credit_risk.contracts import validate_batch
from credit_risk.drift_detection import psi, statistical_drift_table
from credit_risk.drift_simulation import ALL_COMPONENTS, simulate


def test_psi_is_zero_for_identical_and_high_for_shift():
    rng = np.random.default_rng(1)
    a = pd.Series(rng.normal(0, 1, 5000))
    assert psi(a, a) < 1e-6
    assert psi(a, pd.Series(rng.normal(0, 1, 5000))) < 0.05
    assert psi(a, pd.Series(rng.normal(1.0, 1, 5000))) > 0.2


def test_zero_intensity_means_no_drift(clean_batch):
    out = simulate(clean_batch, k=0.0, seed=1)
    pd.testing.assert_frame_equal(out, clean_batch.astype(out.dtypes.to_dict()))


def test_full_drift_is_detected_and_still_respects_contract(clean_batch):
    drifted = simulate(clean_batch, k=1.0, seed=1)
    # drift é dado VÁLIDO porém diferente: o contrato aceita, o monitor de drift detecta
    validate_batch(drifted, batch="t_drifted")
    tbl = statistical_drift_table(clean_batch, drifted)
    flagged = set(tbl.loc[tbl["drift_detected"] | tbl["ks_drift"], "feature"])
    assert {"age", "pay_1"} <= flagged
    assert drifted[config.TARGET].mean() > clean_batch[config.TARGET].mean()


def test_components_can_be_toggled(clean_batch):
    only_concept = simulate(clean_batch, 1.0, 1, components=("concept_default",))
    pd.testing.assert_frame_equal(only_concept[config.FEATURES], clean_batch[config.FEATURES].astype(only_concept[config.FEATURES].dtypes.to_dict()))
    assert set(ALL_COMPONENTS) >= {"concept_default", "payment_delays"}
