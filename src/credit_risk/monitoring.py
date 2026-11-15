"""Job de monitoramento por janela de produção (mês).

Para cada janela: ingestão -> contrato -> inferência -> drift (PSI/KS + Evidently) ->
performance (concept drift) -> alertas -> MLflow + Prometheus + histórico do dashboard.
Pensado para rodar agendado (1 janela por execução) via `credit-risk monitor --window next`.
"""
from __future__ import annotations

import json
import time

import pandas as pd

from . import config
from .contracts import DataContractViolation, validate_batch
from .data import load_reference, window_months
from .drift_detection import (evidently_data_drift, evidently_model_report, evidently_test_suite,
                              statistical_drift_table)
from .logging_utils import RUN_ID, log_event, stage
from .train import HOLDOUT_FILE, classification_metrics, load_model

HISTORY_FILE = config.REPORTS_DIR / "monitoring_history.json"
ALERTS_FILE = config.REPORTS_DIR / "alerts.jsonl"
PROM_FILE = config.REPORTS_DIR / "metrics" / "credit_risk.prom"


# ----------------------------------------------------------------------------- estado / agenda
def _state() -> dict:
    return json.loads(config.STATE_FILE.read_text()) if config.STATE_FILE.exists() else {"processed": []}


def next_window() -> str | None:
    done = set(_state()["processed"])
    return next((m for m in window_months() if m not in done), None)


def _mark_processed(month: str) -> None:
    st = _state()
    st["processed"] = sorted(set(st["processed"]) | {month})
    config.STATE_FILE.write_text(json.dumps(st, indent=2))


def reset_state() -> None:
    for f in (config.STATE_FILE, HISTORY_FILE, ALERTS_FILE):
        f.unlink(missing_ok=True)


# ----------------------------------------------------------------------------- alertas
def build_alerts(month: str, drift_tbl: pd.DataFrame, evid: dict, perf: dict, base: dict) -> list[dict]:
    alerts = []
    for _, r in drift_tbl[drift_tbl["drift_detected"]].iterrows():
        alerts.append({"severity": "warning", "type": "feature_drift", "feature": r["feature"],
                       "message": f"PSI {r['psi']:.3f} > {config.PSI_ALERT} em '{r['feature']}' "
                                  f"(média {r['ref_mean']} → {r['cur_mean']}; KS={r['ks_statistic']:.3f})"})
    if evid["dataset_drift"]:
        alerts.append({"severity": "critical", "type": "dataset_drift",
                       "message": f"Evidently: {evid['share_of_drifted_columns']:.0%} das features com drift "
                                  f"(limite {config.DRIFT_SHARE_ALERT:.0%})"})
    drop = base["auc"] - perf["auc"]
    if drop > config.AUC_DROP_ALERT:
        alerts.append({"severity": "critical", "type": "concept_drift",
                       "message": f"AUC caiu {drop:.3f} ({base['auc']:.3f} → {perf['auc']:.3f}) — degradação do modelo"})
    d_rate = perf["observed_default_rate"] - base["observed_default_rate"]
    if abs(d_rate) > config.DEFAULT_RATE_DELTA_ALERT:
        alerts.append({"severity": "critical", "type": "target_drift",
                       "message": f"Taxa de default observada {base['observed_default_rate']:.1%} → "
                                  f"{perf['observed_default_rate']:.1%}"})
    for a in alerts:
        a.update(window=month, run_id=RUN_ID)
        log_event("alert_raised", level="WARNING" if a["severity"] == "warning" else "ERROR", **a)
        with open(ALERTS_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(a, ensure_ascii=False) + "\n")
    return alerts


# ----------------------------------------------------------------------------- job
def run_window(month: str) -> dict:
    import mlflow

    t0 = time.perf_counter()
    record: dict = {"window": month, "run_id": RUN_ID}
    stages: dict[str, dict] = {}
    model, meta = load_model()
    base = meta["holdout_metrics"]

    try:
        with stage("ingest", window=month) as ctx:
            raw = pd.read_parquet(config.PRODUCTION_DIR / f"window_{month}.parquet")
            ctx["rows"] = len(raw)
        stages["ingest"] = ctx

        try:
            batch = validate_batch(raw, batch=f"window_{month}")
            stages["validate_contract"] = {"status": "success", "rows_valid": len(batch)}
        except DataContractViolation as exc:
            stages["validate_contract"] = {"status": "failed", "violations": len(exc.failures)}
            record.update(status="blocked", rows_in=len(raw), rows_rejected=len(raw))
            raise

        with stage("inference", window=month, rows=len(batch)) as ctx:
            t = time.perf_counter()
            proba = model.predict_proba(batch[config.FEATURES])[:, 1]
            ctx["latency_ms_per_1k"] = round((time.perf_counter() - t) * 1000 / len(batch) * 1000, 2)
            scored = batch.assign(prediction_proba=proba, prediction=(proba >= meta["threshold"]).astype(int))
            scored.to_parquet(config.PRODUCTION_DIR / f"scored_{month}.parquet", index=False)
        stages["inference"] = ctx

        with stage("drift_detection", window=month) as ctx:
            reference = load_reference()
            ref_scored = pd.read_parquet(HOLDOUT_FILE)
            drift_tbl = statistical_drift_table(reference, scored)
            drift_tbl.to_csv(config.REPORTS_DIR / "drift" / f"psi_ks_{month}.csv", index=False)
            evid = evidently_data_drift(reference, scored, month)
            model_html = evidently_model_report(ref_scored, scored, meta["threshold"], month)
            suite = evidently_test_suite(reference, ref_scored, scored, base["auc"], month)
            ctx.update(n_drifted_psi=int(drift_tbl["drift_detected"].sum()),
                       evidently_share=round(evid["share_of_drifted_columns"], 3), tests_failed=suite["failed"])
        stages["drift_detection"] = ctx

        with stage("performance_monitoring", window=month) as ctx:
            perf = classification_metrics(scored[config.TARGET].values, proba, meta["threshold"])
            ctx.update(auc=perf["auc"], recall=perf["recall"])
        stages["performance_monitoring"] = ctx

        alerts = build_alerts(month, drift_tbl, evid, perf, base)
        record.update(
            status="success", rows_in=len(raw), rows_rejected=0,
            mean_score=round(float(proba.mean()), 4),
            psi=dict(zip(drift_tbl["feature"], drift_tbl["psi"])),
            ks=dict(zip(drift_tbl["feature"], drift_tbl["ks_statistic"])),
            drifted_features_psi=drift_tbl.loc[drift_tbl["drift_detected"], "feature"].tolist(),
            drifted_features_evidently=evid["drifted_features"],
            share_drifted_evidently=round(evid["share_of_drifted_columns"], 4),
            dataset_drift=evid["dataset_drift"], performance=perf,
            tests=suite["tests"], tests_failed=suite["failed"],
            alerts=alerts, n_alerts=len(alerts),
            reports={"data_drift": evid["html"], "model": model_html, "tests": suite["html"]},
        )
    finally:
        record["stages"] = stages
        record["total_latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        record.setdefault("status", "failed")
        _persist(record)
        _log_mlflow(mlflow, record, base)
        _export_prometheus(record)
        log_event("window_processed", window=month, status=record["status"],
                  n_alerts=record.get("n_alerts", 0), total_latency_ms=record["total_latency_ms"])
    _mark_processed(month)
    return record


def _persist(record: dict) -> None:
    hist = json.loads(HISTORY_FILE.read_text(encoding="utf-8")) if HISTORY_FILE.exists() else []
    hist = [h for h in hist if h["window"] != record["window"]] + [record]
    hist.sort(key=lambda h: h["window"])
    HISTORY_FILE.write_text(json.dumps(hist, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _log_mlflow(mlflow, rec: dict, base: dict) -> None:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    step = window_months().index(rec["window"])
    with mlflow.start_run(run_name=f"monitor-{rec['window']}"):
        mlflow.set_tags({"stage": "monitoring", "window": rec["window"], "status": rec["status"],
                         "pipeline_run_id": rec["run_id"]})
        m = {"rows_in": rec.get("rows_in", 0), "rows_rejected": rec.get("rows_rejected", 0),
             "pipeline_success": int(rec["status"] == "success"), "total_latency_ms": rec["total_latency_ms"]}
        if rec["status"] == "success":
            m.update({"share_drifted_evidently": rec["share_drifted_evidently"],
                      "n_drifted_features_psi": len(rec["drifted_features_psi"]),
                      "n_alerts": rec["n_alerts"], "tests_failed": rec["tests_failed"],
                      "mean_score": rec["mean_score"], "auc_drop": round(base["auc"] - rec["performance"]["auc"], 4),
                      **{f"perf_{k}": v for k, v in rec["performance"].items()},
                      **{f"psi_{k}": v for k, v in rec["psi"].items()}})
            for name, path in rec["reports"].items():
                mlflow.log_artifact(path, artifact_path="evidently")
        for s, v in rec["stages"].items():
            if "duration_ms" in v:
                m[f"latency_{s}_ms"] = v["duration_ms"]
        mlflow.log_metrics(m, step=step)
        mlflow.log_dict(rec, "window_record.json")


def _export_prometheus(rec: dict) -> None:
    """Formato textfile do node_exporter / Pushgateway (integração opcional com a Fase 3)."""
    from prometheus_client import CollectorRegistry, Gauge, write_to_textfile

    PROM_FILE.parent.mkdir(parents=True, exist_ok=True)
    reg = CollectorRegistry()
    lbl = ["window"]
    Gauge("credit_pipeline_success", "1 se a janela foi processada com sucesso", lbl, registry=reg).labels(rec["window"]).set(int(rec["status"] == "success"))
    Gauge("credit_pipeline_rows_processed", "linhas ingeridas", lbl, registry=reg).labels(rec["window"]).set(rec.get("rows_in", 0))
    Gauge("credit_pipeline_latency_ms", "latência total do job", lbl, registry=reg).labels(rec["window"]).set(rec["total_latency_ms"])
    if rec["status"] == "success":
        Gauge("credit_model_auc", "AUC na janela", lbl, registry=reg).labels(rec["window"]).set(rec["performance"]["auc"])
        Gauge("credit_drift_share", "fração de features com drift (Evidently)", lbl, registry=reg).labels(rec["window"]).set(rec["share_drifted_evidently"])
        Gauge("credit_alerts_total", "alertas gerados", lbl, registry=reg).labels(rec["window"]).set(rec["n_alerts"])
        g = Gauge("credit_feature_psi", "PSI por feature", ["window", "feature"], registry=reg)
        for f, v in rec["psi"].items():
            g.labels(rec["window"], f).set(v)
    write_to_textfile(str(PROM_FILE), reg)
