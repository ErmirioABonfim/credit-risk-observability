import numpy as np
import pytest

from credit_risk import config
from credit_risk.contracts import CREDIT_CONTRACT, DataContractViolation, make_corrupted_batch, validate_batch


def test_clean_batch_passes(clean_batch):
    out = validate_batch(clean_batch, batch="t_clean")
    assert len(out) == len(clean_batch)


@pytest.mark.parametrize("mutate, rule", [
    (lambda d: d.assign(limit_bal=d["limit_bal"].where(d.index != 0, np.nan)), "not_nullable"),
    (lambda d: d.assign(age=d["age"].where(d.index != 0, 16)), "greater_than_or_equal_to(18)"),
    (lambda d: d.assign(limit_bal=d["limit_bal"].where(d.index != 0, 9e7)), "in_range(1000, 2000000)"),
    (lambda d: d.assign(education=d["education"].where(d.index != 0, 9)), "isin([1, 2, 3, 4])"),
    (lambda d: d.assign(pay_amt_1=d["pay_amt_1"].where(d.index != 0, -1.0)), "in_range(0, 5000000)"),
    (lambda d: d.assign(cpf="123"), "column_in_schema"),
])
def test_each_rule_blocks_ingestion(clean_batch, mutate, rule):
    with pytest.raises(DataContractViolation) as exc:
        validate_batch(mutate(clean_batch.copy()), batch="t_rule")
    assert rule in set(exc.value.failures["check"].astype(str))


def test_duplicates_are_blocked(clean_batch):
    dup = clean_batch.iloc[[0]]
    with pytest.raises(DataContractViolation):
        validate_batch(clean_batch._append(dup, ignore_index=True), batch="t_dup")


def test_corrupted_demo_batch_is_quarantined(clean_batch):
    bad = make_corrupted_batch(clean_batch, n=200)
    with pytest.raises(DataContractViolation):
        validate_batch(bad, batch="t_demo")
    assert (config.QUARANTINE_DIR / "t_demo_summary.json").exists()


def test_target_is_optional_for_inference(clean_batch):
    CREDIT_CONTRACT.validate(clean_batch.drop(columns=[config.TARGET]))
