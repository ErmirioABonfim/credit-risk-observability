"""Análise causal da degradação (intervenções contrafactuais sobre o mecanismo de drift).

Pergunta: *qual mudança no mundo causou a queda de performance do modelo?*

Como controlamos o processo gerador (simulação), podemos fazer intervenções do tipo
do(componente = desligado) mantendo TODO o resto igual (mesmos clientes, mesmos sorteios).
Comparamos o modelo em cada mundo contrafactual:

* add-one  : mundo sem drift + apenas 1 mecanismo  -> efeito isolado do mecanismo
* drop-one : mundo com todos os mecanismos - 1     -> quanto do dano some se removermos a causa
* covariate vs concept: separa "o cliente mudou" (P(X)) de "a relação mudou" (P(y|X)).
"""
from __future__ import annotations

import json

import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config
from .data import window_months
from .drift_detection import psi
from .drift_simulation import ALL_COMPONENTS, DATA_DRIFT_COMPONENTS, intensity, load_clean_window, simulate
from .logging_utils import stage
from .train import load_model

OUT_JSON = config.REPORTS_DIR / "causal_analysis.json"
OUT_MD = config.REPORTS_DIR / "causal_analysis.md"


def _evaluate(model, threshold: float, df: pd.DataFrame, reference: pd.DataFrame) -> dict:
    proba = model.predict_proba(df[config.FEATURES])[:, 1]
    return {
        "auc": round(float(roc_auc_score(df[config.TARGET], proba)), 4),
        "observed_default_rate": round(float(df[config.TARGET].mean()), 4),
        "predicted_default_rate": round(float((proba >= threshold).mean()), 4),
        "mean_score": round(float(proba.mean()), 4),
        "score_psi": round(psi(pd.Series(model.predict_proba(reference[config.FEATURES])[:, 1]), pd.Series(proba)), 4),
    }


def run() -> dict:
    model, meta = load_model()
    thr = meta["threshold"]
    months = window_months()
    i = len(months) - 1                                   # janela de drift máximo
    month = months[i]
    clean = load_clean_window(month)
    seed, k = config.RANDOM_STATE + i, intensity(i)
    reference = pd.read_parquet(config.REFERENCE_DIR / "holdout_scored.parquet")

    with stage("causal_analysis", window=month) as ctx:
        scen = {"sem_drift (controle)": (), "todos os mecanismos (observado)": ALL_COMPONENTS,
                "apenas covariate shift (P(X))": DATA_DRIFT_COMPONENTS, "apenas concept drift (P(y|X))": ("concept_default",)}
        scen |= {f"+ {c}": (c,) for c in ALL_COMPONENTS}
        scen |= {f"todos - {c}": tuple(x for x in ALL_COMPONENTS if x != c) for c in ALL_COMPONENTS}
        results = {name: _evaluate(model, thr, simulate(clean, k, seed, comps), reference) for name, comps in scen.items()}

        base, full = results["sem_drift (controle)"], results["todos os mecanismos (observado)"]
        total_drop = base["auc"] - full["auc"]
        attribution = []
        for c in ALL_COMPONENTS:
            alone = base["auc"] - results[f"+ {c}"]["auc"]
            removed = results[f"todos - {c}"]["auc"] - full["auc"]   # AUC recuperada ao remover a causa
            attribution.append({"mecanismo": c, "tipo": "concept" if c == "concept_default" else "covariate",
                                "queda_auc_isolada": round(alone, 4), "auc_recuperada_se_removido": round(removed, 4),
                                "delta_taxa_default_prevista": round(results[f"+ {c}"]["predicted_default_rate"] - base["predicted_default_rate"], 4),
                                "delta_taxa_default_real": round(results[f"+ {c}"]["observed_default_rate"] - base["observed_default_rate"], 4)})
        attribution.sort(key=lambda r: r["auc_recuperada_se_removido"], reverse=True)
        out = {"window": month, "intensity": k, "baseline_auc": base["auc"], "observed_auc": full["auc"],
               "total_auc_drop": round(total_drop, 4), "scenarios": results, "attribution": attribution,
               "root_cause": attribution[0]["mecanismo"]}
        OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        OUT_MD.write_text(_markdown(out), encoding="utf-8")
        ctx.update(total_auc_drop=out["total_auc_drop"], root_cause=out["root_cause"])
    return out


def _markdown(o: dict) -> str:
    lines = [f"# Análise causal — janela {o['window']}", "",
             f"AUC controle (sem drift): **{o['baseline_auc']}** → AUC observada: **{o['observed_auc']}** "
             f"(queda {o['total_auc_drop']}). Causa raiz principal: **{o['root_cause']}**.", "",
             "| Cenário | AUC | Default real | Default previsto | PSI do score |", "|---|---|---|---|---|"]
    lines += [f"| {n} | {r['auc']} | {r['observed_default_rate']:.1%} | {r['predicted_default_rate']:.1%} | {r['score_psi']} |"
              for n, r in o["scenarios"].items()]
    lines += ["", "| Mecanismo | Tipo | Queda AUC isolada | AUC recuperada se removido | Δ default previsto | Δ default real |",
              "|---|---|---|---|---|---|"]
    lines += [f"| {a['mecanismo']} | {a['tipo']} | {a['queda_auc_isolada']} | {a['auc_recuperada_se_removido']} | "
              f"{a['delta_taxa_default_prevista']:+.1%} | {a['delta_taxa_default_real']:+.1%} |" for a in o["attribution"]]
    return "\n".join(lines) + "\n"
