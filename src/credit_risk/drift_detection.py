"""Detecção estatística de drift: PSI + Kolmogorov-Smirnov (próprios) e relatórios Evidently AI."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from . import config

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

CATEGORICAL = ["education", *config.PAY_STATUS_COLS]
NUMERICAL = [c for c in config.FEATURES if c not in CATEGORICAL]
DRIFT_DIR = config.REPORTS_DIR / "drift"
DRIFT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------- PSI / KS
def psi(reference: pd.Series, current: pd.Series, bins: int = 10, categorical: bool = False) -> float:
    """Population Stability Index. <0,1 estável | 0,1–0,2 atenção | >0,2 mudança significativa."""
    eps = 1e-4
    if categorical:
        cats = sorted(set(reference.unique()) | set(current.unique()))
        ref_p = reference.value_counts(normalize=True).reindex(cats, fill_value=0).values
        cur_p = current.value_counts(normalize=True).reindex(cats, fill_value=0).values
    else:
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        ref_p = np.histogram(reference, edges)[0] / len(reference)
        cur_p = np.histogram(current, edges)[0] / len(current)
    ref_p, cur_p = np.clip(ref_p, eps, None), np.clip(cur_p, eps, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def statistical_drift_table(reference: pd.DataFrame, current: pd.DataFrame, features=config.FEATURES) -> pd.DataFrame:
    rows = []
    for col in features:
        cat = col in CATEGORICAL
        p = psi(reference[col], current[col], categorical=cat)
        ks = ks_2samp(reference[col], current[col])
        rows.append({
            "feature": col, "type": "categorical" if cat else "numerical",
            "psi": round(p, 4), "ks_statistic": round(float(ks.statistic), 4), "ks_pvalue": float(ks.pvalue),
            "ref_mean": round(float(reference[col].mean()), 2), "cur_mean": round(float(current[col].mean()), 2),
            "psi_level": "ALERTA" if p > config.PSI_ALERT else ("atenção" if p > config.PSI_WARN else "estável"),
            # decisão: PSI é a métrica primária (robusta a n grande); KS confirma
            "drift_detected": bool(p > config.PSI_ALERT),
            "ks_drift": bool(ks.pvalue < config.KS_PVALUE_ALERT),
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------- Evidently
def _column_mapping(with_prediction: bool):
    from evidently import ColumnMapping

    return ColumnMapping(
        target=config.TARGET if with_prediction else None,
        prediction="prediction_proba" if with_prediction else None,
        numerical_features=NUMERICAL, categorical_features=CATEGORICAL, id=config.ID_COL,
    )


def evidently_data_drift(reference: pd.DataFrame, current: pd.DataFrame, name: str) -> dict:
    """Relatório HTML de data drift (KS para numéricas, PSI para categóricas)."""
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.report import Report

    cols = [config.ID_COL, *config.FEATURES]
    report = Report(metrics=[
        DataDriftPreset(num_stattest="ks", cat_stattest="psi", cat_stattest_threshold=config.PSI_ALERT,
                        drift_share=config.DRIFT_SHARE_ALERT),
        DataQualityPreset(),
    ])
    report.run(reference_data=reference[cols], current_data=current[cols], column_mapping=_column_mapping(False))
    path = DRIFT_DIR / f"data_drift_{name}.html"
    report.save_html(str(path))
    result = _find(report.as_dict(), "drift_by_columns")
    dataset = _find(report.as_dict(), "share_of_drifted_columns", parent=True)
    return {
        "html": str(path),
        "share_of_drifted_columns": float(dataset["share_of_drifted_columns"]),
        "dataset_drift": bool(dataset["dataset_drift"]),
        "drifted_features": sorted(c for c, v in result.items() if v["drift_detected"]),
        "stattests": {c: v["stattest_name"] for c, v in result.items()},
    }


def evidently_model_report(ref_scored: pd.DataFrame, cur_scored: pd.DataFrame, threshold: float, name: str) -> str:
    """Concept drift: qualidade de classificação + drift de target e de predição."""
    from evidently.metric_preset import ClassificationPreset, TargetDriftPreset
    from evidently.report import Report

    cols = [config.ID_COL, *config.FEATURES, config.TARGET, "prediction_proba"]
    report = Report(metrics=[ClassificationPreset(probas_threshold=threshold), TargetDriftPreset()])
    report.run(reference_data=ref_scored[cols], current_data=cur_scored[cols], column_mapping=_column_mapping(True))
    path = DRIFT_DIR / f"model_performance_{name}.html"
    report.save_html(str(path))
    return str(path)


def evidently_test_suite(reference: pd.DataFrame, ref_scored: pd.DataFrame, cur_scored: pd.DataFrame,
                         baseline_auc: float, name: str) -> dict:
    """Suite de testes pass/fail => base dos alertas automáticos."""
    from evidently.test_suite import TestSuite
    from evidently.tests import TestColumnDrift, TestRocAuc, TestShareOfDriftedColumns

    cols = [config.ID_COL, *config.FEATURES, config.TARGET, "prediction_proba"]
    ref = ref_scored[cols]
    suite = TestSuite(tests=[
        TestShareOfDriftedColumns(lt=config.DRIFT_SHARE_ALERT, columns=config.FEATURES),
        *[TestColumnDrift(column_name=c) for c in config.KEY_FEATURES],
        TestColumnDrift(column_name="prediction_proba"),
        TestRocAuc(gte=round(baseline_auc - config.AUC_DROP_ALERT, 4)),
    ])
    suite.run(reference_data=ref, current_data=cur_scored[cols], column_mapping=_column_mapping(True))
    path = DRIFT_DIR / f"test_suite_{name}.html"
    suite.save_html(str(path))
    tests = suite.as_dict()["tests"]
    return {"html": str(path),
            "tests": [{"name": t["name"], "status": t["status"], "description": t["description"]} for t in tests],
            "failed": sum(t["status"] == "FAIL" for t in tests)}


def _find(obj, key, parent=False):
    if isinstance(obj, dict):
        if key in obj:
            return obj if parent else obj[key]
        for v in obj.values():
            r = _find(v, key, parent)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find(v, key, parent)
            if r is not None:
                return r
    return None
