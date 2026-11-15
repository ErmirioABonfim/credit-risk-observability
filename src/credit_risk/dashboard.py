"""Dashboard HTML estático de observabilidade (pipeline + infraestrutura + modelo + alertas)."""
from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import config
from .logging_utils import read_events
from .monitoring import HISTORY_FILE

OUT = config.REPORTS_DIR / "monitoring_dashboard.html"
PALETTE = {"ok": "#1b9e77", "warn": "#d95f02", "crit": "#c0392b", "blue": "#2c7fb8", "grey": "#7f8c8d"}


def _fig_html(fig: go.Figure, height: int = 360) -> str:
    fig.update_layout(template="plotly_white", margin=dict(l=40, r=20, t=50, b=40), height=height,
                      font=dict(family="Inter, Segoe UI, sans-serif", size=12))
    return fig.to_html(full_html=False, include_plotlyjs=False)


def build() -> Path:
    hist = json.loads(HISTORY_FILE.read_text(encoding="utf-8")) if HISTORY_FILE.exists() else []
    meta = json.loads((config.MODELS_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    base = meta["holdout_metrics"]
    events = read_events()
    ok = [h for h in hist if h["status"] == "success"]
    months = [h["window"] for h in ok]
    sections = []

    # ---------- KPIs
    last = ok[-1] if ok else None
    contract_runs = [e for e in events if e["event"] in ("stage_succeeded", "stage_failed") and e.get("stage") == "validate_contract"]
    rejected = sum(e["status"] == "failed" for e in contract_runs)
    stage_runs = [e for e in events if e["event"] in ("stage_succeeded", "stage_failed")]
    success_rate = sum(e["status"] == "success" for e in stage_runs) / max(len(stage_runs), 1)
    kpis = [
        ("Janelas processadas", f"{len(ok)}/{config.N_PRODUCTION_WINDOWS}", "ok"),
        ("Taxa de sucesso das tarefas", f"{success_rate:.0%}", "ok" if success_rate > .95 else "warn"),
        ("Lotes bloqueados pelo contrato", str(rejected), "crit" if rejected else "ok"),
        ("AUC baseline → atual", f"{base['auc']:.3f} → {last['performance']['auc']:.3f}" if last else "-",
         "crit" if last and base["auc"] - last["performance"]["auc"] > config.AUC_DROP_ALERT else "ok"),
        ("Features com drift (Evidently)", f"{last['share_drifted_evidently']:.0%}" if last else "-",
         "crit" if last and last["dataset_drift"] else "ok"),
        ("Alertas (total)", str(sum(h.get("n_alerts", 0) for h in hist)), "crit" if any(h.get("n_alerts") for h in hist) else "ok"),
    ]
    sections.append('<div class="kpis">' + "".join(
        f'<div class="kpi {c}"><span>{html.escape(t)}</span><b>{html.escape(v)}</b></div>' for t, v, c in kpis) + "</div>")

    # ---------- 1. Saúde do pipeline
    if hist:
        stage_names = ["ingest", "validate_contract", "inference", "drift_detection", "performance_monitoring"]
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Latência por tarefa (ms)", "Volume processado x rejeitado"))
        for s in stage_names:
            fig.add_bar(name=s, x=[h["window"] for h in hist],
                        y=[h["stages"].get(s, {}).get("duration_ms", 0) for h in hist], row=1, col=1)
        fig.add_bar(name="linhas aceitas", x=[h["window"] for h in hist],
                    y=[h.get("rows_in", 0) - h.get("rows_rejected", 0) for h in hist], marker_color=PALETTE["ok"], row=1, col=2)
        fig.add_bar(name="linhas rejeitadas", x=[h["window"] for h in hist],
                    y=[h.get("rows_rejected", 0) for h in hist], marker_color=PALETTE["crit"], row=1, col=2)
        fig.update_layout(barmode="stack")
        status_rows = "".join(
            f"<tr><td>{h['window']}</td>" + "".join(
                f"<td class='{'ok' if h['stages'].get(s, {}).get('status', 'success' if s in h['stages'] else '') == 'success' else 'crit'}'>"
                f"{'✔' if s in h['stages'] and h['stages'][s].get('status', 'success') == 'success' else '✖'}</td>" for s in stage_names)
            + f"<td>{h['total_latency_ms']:.0f} ms</td></tr>" for h in hist)
        sections.append("<h2>1. Saúde do pipeline de dados</h2>" + _fig_html(fig) +
                        "<table><tr><th>Janela</th>" + "".join(f"<th>{s}</th>" for s in stage_names) +
                        "<th>Latência total</th></tr>" + status_rows + "</table>")

    # ---------- 2. Contrato de dados
    summaries = sorted(config.QUARANTINE_DIR.glob("*_summary.json"), key=lambda p: p.stat().st_mtime)
    cnt = Counter(e.get("status_contract") for e in events if e.get("stage") == "validate_contract" and e.get("status_contract"))
    block = f"<p>Validações: <b>{cnt.get('accepted', 0)}</b> lotes aceitos, <b>{cnt.get('rejected', 0)}</b> rejeitados (quarentena em <code>data/quarantine/</code>).</p>"
    for s in summaries[-1:]:
        rows = json.loads(s.read_text(encoding="utf-8"))
        block += f"<h3>Lote bloqueado: {html.escape(s.stem.replace('_summary', ''))}</h3><table><tr><th>Coluna</th><th>Regra violada</th><th>Linhas</th></tr>" + "".join(
            f"<tr><td>{html.escape(str(r['column']))}</td><td>{html.escape(str(r['check']))}</td><td>{r['occurrences']}</td></tr>" for r in rows) + "</table>"
    sections.append("<h2>2. Data Contract (Pandera)</h2>" + block)

    if ok:
        # ---------- 3. Drift
        feats = config.FEATURES
        z = [[h["psi"][f] for h in ok] for f in feats]
        heat = go.Figure(go.Heatmap(z=z, x=months, y=feats, colorscale="RdYlGn_r", zmin=0, zmax=0.5,
                                    colorbar=dict(title="PSI"), text=[[f"{v:.2f}" for v in r] for r in z], texttemplate="%{text}"))
        heat.update_layout(title="PSI por feature e janela (>0,2 = drift)")
        heat_html = _fig_html(heat, height=620)
        share = go.Figure()
        share.add_scatter(x=months, y=[h["share_drifted_evidently"] for h in ok], mode="lines+markers", name="Evidently (KS/PSI)")
        share.add_scatter(x=months, y=[len(h["drifted_features_psi"]) / len(feats) for h in ok], mode="lines+markers", name="PSI > 0,2")
        share.add_hline(y=config.DRIFT_SHARE_ALERT, line_dash="dash", line_color=PALETTE["crit"], annotation_text="limite dataset drift")
        share.update_layout(title="Fração de features com drift", yaxis_tickformat=".0%")
        sections.append("<h2>3. Data Drift</h2>" + _fig_html(share) + heat_html)

        # ---------- 4. Modelo
        perf = make_subplots(rows=1, cols=2, subplot_titles=("Qualidade do modelo (rótulos chegam com atraso)", "Taxa de default e aprovação"))
        for k, c in [("auc", PALETTE["blue"]), ("ks_separation", PALETTE["grey"]), ("recall", PALETTE["warn"])]:
            perf.add_scatter(x=months, y=[h["performance"][k] for h in ok], name=k, mode="lines+markers", line_color=c, row=1, col=1)
        perf.add_hline(y=base["auc"] - config.AUC_DROP_ALERT, line_dash="dash", line_color=PALETTE["crit"], row=1, col=1,
                       annotation_text="limite AUC")
        for k, c in [("observed_default_rate", PALETTE["crit"]), ("predicted_default_rate", PALETTE["warn"]), ("approval_rate", PALETTE["ok"])]:
            perf.add_scatter(x=months, y=[h["performance"][k] for h in ok], name=k, mode="lines+markers", line_color=c, row=1, col=2)
        perf.update_yaxes(tickformat=".0%", row=1, col=2)
        sections.append("<h2>4. Saúde do modelo (Concept Drift)</h2>" + _fig_html(perf))

    # ---------- 5. Causal
    causal = config.REPORTS_DIR / "causal_analysis.json"
    if causal.exists():
        c = json.loads(causal.read_text(encoding="utf-8"))
        att = pd.DataFrame(c["attribution"])
        fig = go.Figure(go.Bar(x=att["auc_recuperada_se_removido"], y=att["mecanismo"], orientation="h",
                               marker_color=[PALETTE["crit"] if t == "concept" else PALETTE["blue"] for t in att["tipo"]]))
        fig.update_layout(title=f"AUC recuperada ao remover cada causa (queda total {c['total_auc_drop']:.3f})", yaxis_autorange="reversed")
        sections.append("<h2>5. Análise causal da degradação</h2>" + _fig_html(fig) +
                        f"<p>Causa raiz: <b>{c['root_cause']}</b>. Detalhes em <code>reports/causal_analysis.md</code>.</p>")

    # ---------- 6. Alertas
    alerts = [a for h in hist for a in h.get("alerts", [])]
    sections.append("<h2>6. Alertas gerados</h2><table><tr><th>Janela</th><th>Severidade</th><th>Tipo</th><th>Mensagem</th></tr>" + "".join(
        f"<tr><td>{a['window']}</td><td class='{'crit' if a['severity'] == 'critical' else 'warn'}'>{a['severity']}</td>"
        f"<td>{a['type']}</td><td>{html.escape(a['message'])}</td></tr>" for a in alerts) + "</table>")

    # ---------- 7. Relatórios
    links = "".join(
        f"<li><b>{h['window']}</b>: " + " · ".join(
            f"<a href='{Path(p).relative_to(config.REPORTS_DIR).as_posix()}'>{n}</a>" for n, p in h.get("reports", {}).items()) + "</li>" for h in ok)
    sections.append(f"<h2>7. Relatórios Evidently</h2><ul>{links}</ul>")

    page = f"""<!doctype html><html lang="pt-br"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Credit Risk Monitoring</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body{{font-family:Inter,'Segoe UI',sans-serif;margin:0;background:#f5f6f8;color:#1f2933}}
main{{max-width:1200px;margin:auto;padding:16px}} h1{{margin:8px 0}} h2{{margin-top:32px;border-bottom:2px solid #dde1e6;padding-bottom:4px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}}
.kpi{{background:#fff;border-radius:8px;padding:12px;border-left:6px solid {PALETTE['ok']};box-shadow:0 1px 2px #0001}}
.kpi span{{display:block;font-size:12px;color:#52606d}} .kpi b{{font-size:20px}}
.kpi.warn{{border-color:{PALETTE['warn']}}} .kpi.crit{{border-color:{PALETTE['crit']}}}
table{{border-collapse:collapse;width:100%;background:#fff;font-size:13px;margin:8px 0}} td,th{{border:1px solid #e4e7eb;padding:6px;text-align:left}}
td.ok{{color:{PALETTE['ok']};font-weight:600}} td.crit{{color:{PALETTE['crit']};font-weight:600}} td.warn{{color:{PALETTE['warn']};font-weight:600}}
.sub{{color:#52606d}} .plotly-graph-div{{background:#fff}}
</style></head><body><main>
<h1>Credit Scoring — Observabilidade em Produção</h1>
<p class="sub">Modelo <b>{config.MODEL_NAME}</b> · limiar {meta['threshold']:.2f} · baseline AUC {base['auc']:.3f} ·
gerado a partir de <code>logs/pipeline.jsonl</code>, <code>reports/monitoring_history.json</code> e MLflow.</p>
{''.join(sections)}
</main></body></html>"""
    OUT.write_text(page, encoding="utf-8")
    return OUT
