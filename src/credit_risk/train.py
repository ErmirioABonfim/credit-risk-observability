"""Treino do modelo baseline (XGBoost) sobre o Dataset de Referência validado."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from . import config
from .contracts import validate_batch
from .data import load_reference
from .logging_utils import log_event, stage

MODEL_FILE = config.MODELS_DIR / "model.joblib"
META_FILE = config.MODELS_DIR / "model_metadata.json"
HOLDOUT_FILE = config.REFERENCE_DIR / "holdout_scored.parquet"


def classification_metrics(y_true, proba, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    pred = (proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else float("nan")
    ks = ks_2samp(proba[y_true == 1], proba[y_true == 0]).statistic if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "auc": round(float(auc), 4),
        "gini": round(float(2 * auc - 1), 4),
        "ks_separation": round(float(ks), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "approval_rate": round(float(1 - pred.mean()), 4),
        "observed_default_rate": round(float(y_true.mean()), 4),
        "predicted_default_rate": round(float(pred.mean()), 4),
    }


def fairness_audit(scored: pd.DataFrame) -> dict:
    """Auditoria de viés com atributos sensíveis segregados (nunca usados como feature)."""
    sens = pd.read_parquet(config.RESTRICTED_DIR / "sensitive_attributes.parquet")
    df = scored.merge(sens, on=config.ID_COL, how="left")
    df["approved"] = 1 - df["prediction"]
    out = {}
    for col, labels in {"sex": {1: "masculino", 2: "feminino"}, "marriage": {1: "casado", 2: "solteiro", 3: "outros"}}.items():
        rates = df.groupby(col)["approved"].mean()
        out[col] = {
            "approval_rate": {labels.get(k, str(k)): round(float(v), 4) for k, v in rates.items()},
            "disparate_impact_ratio": round(float(rates.min() / rates.max()), 4),  # regra dos 4/5: >= 0,8
        }
    age_band = pd.cut(df["age"], [17, 25, 35, 50, 120], labels=["18-25", "26-35", "36-50", "51+"])
    rates = df.groupby(age_band, observed=True)["approved"].mean()
    out["age_band"] = {"approval_rate": {str(k): round(float(v), 4) for k, v in rates.items()},
                       "disparate_impact_ratio": round(float(rates.min() / rates.max()), 4)}
    return out


def train() -> dict:
    import mlflow

    with stage("train_model") as ctx:
        ref = validate_batch(load_reference(), batch="reference")
        train_df, holdout = train_test_split(ref, test_size=0.2, stratify=ref[config.TARGET],
                                             random_state=config.RANDOM_STATE)
        fit_df, val_df = train_test_split(train_df, test_size=0.15, stratify=train_df[config.TARGET],
                                          random_state=config.RANDOM_STATE)
        params = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
                      eval_metric="auc", early_stopping_rounds=30, random_state=config.RANDOM_STATE)
        model = XGBClassifier(**params)
        model.fit(fit_df[config.FEATURES], fit_df[config.TARGET],
                  eval_set=[(val_df[config.FEATURES], val_df[config.TARGET])], verbose=False)

        # limiar que maximiza F1 na validação (nunca no holdout)
        val_proba = model.predict_proba(val_df[config.FEATURES])[:, 1]
        grid = np.linspace(0.1, 0.9, 81)
        threshold = float(grid[np.argmax([f1_score(val_df[config.TARGET], val_proba >= t) for t in grid])])

        proba = model.predict_proba(holdout[config.FEATURES])[:, 1]
        metrics = classification_metrics(holdout[config.TARGET].values, proba, threshold)
        scored = holdout.assign(prediction_proba=proba, prediction=(proba >= threshold).astype(int))
        scored.to_parquet(HOLDOUT_FILE, index=False)
        fairness = fairness_audit(scored)

        importance = pd.Series(model.feature_importances_, index=config.FEATURES).sort_values(ascending=False)
        joblib.dump(model, MODEL_FILE)
        meta = {"model_name": config.MODEL_NAME, "features": config.FEATURES, "threshold": threshold,
                "best_iteration": int(model.best_iteration), "holdout_metrics": metrics,
                "fairness": fairness, "feature_importance": importance.round(4).to_dict(),
                "train_rows": len(fit_df), "valid_rows": len(val_df), "holdout_rows": len(holdout)}
        META_FILE.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name="baseline-training") as run:
            mlflow.set_tags({"stage": "training", "dataset": "reference", "contract": "passed"})
            mlflow.log_params({k: v for k, v in params.items() if k != "eval_metric"} | {"threshold": threshold})
            mlflow.log_metrics({f"holdout_{k}": v for k, v in metrics.items()})
            mlflow.log_metrics({"fairness_dir_sex": fairness["sex"]["disparate_impact_ratio"],
                                "fairness_dir_age_band": fairness["age_band"]["disparate_impact_ratio"]})
            mlflow.log_artifact(str(META_FILE))
            mlflow.xgboost.log_model(model, name="model", registered_model_name=config.MODEL_NAME,
                                     input_example=holdout[config.FEATURES].head(3))
            ctx["mlflow_run_id"] = run.info.run_id

        ctx.update(threshold=round(threshold, 3), **{f"holdout_{k}": v for k, v in metrics.items() if k in ("auc", "ks_separation", "f1")})
    log_event("model_trained", **meta["holdout_metrics"])
    return meta


def load_model():
    return joblib.load(MODEL_FILE), json.loads(META_FILE.read_text(encoding="utf-8"))
