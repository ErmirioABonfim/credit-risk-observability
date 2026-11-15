"""CLI do pipeline: `credit-risk <comando>` (ou `python -m credit_risk.cli <comando>`).

Sequência de execução (mesma ordem do deploy):
  1. prepare      -> baixa dataset público, limpa, pseudonimiza, separa Referência x Produção
  2. train        -> valida Referência no contrato e treina o baseline (MLflow + registry)
  3. validate     -> demonstra o contrato bloqueando um lote corrompido
  4. simulate     -> gera as 6 janelas mensais de produção com drift progressivo
  5. monitor      -> processa janela(s) de produção (agendável: --window next)
  6. causal       -> análise causal contrafactual da degradação
  7. dashboard    -> gera reports/monitoring_dashboard.html
  run-all         -> 1..7 em sequência (demo completa)
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
warnings.filterwarnings("ignore")


def cmd_prepare(_):
    from .data import prepare
    prepare()


def cmd_train(_):
    from .train import train
    meta = train()
    print(f"Holdout: {meta['holdout_metrics']}")


def cmd_validate(args):
    import pandas as pd

    from . import config
    from .contracts import DataContractViolation, make_corrupted_batch, validate_batch

    if args.file:
        df = pd.read_parquet(args.file) if args.file.endswith(".parquet") else pd.read_csv(args.file)
        name = os.path.splitext(os.path.basename(args.file))[0]
    else:
        df = make_corrupted_batch(pd.read_parquet(config.PRODUCTION_DIR / "clean_2026-01.parquet"))
        name = "demo_corrupted_batch"
    try:
        validate_batch(df, batch=name)
        print(f"[OK] Lote '{name}' aceito pelo contrato.")
    except DataContractViolation as exc:
        print(f"[BLOQUEADO] {exc}")
        print(exc.failures.groupby(["column", "check"], dropna=False).size().to_string())
        if args.strict:
            sys.exit(1)


def cmd_simulate(_):
    from .drift_simulation import build_production_windows
    build_production_windows()


def cmd_monitor(args):
    from .data import window_months
    from .monitoring import next_window, reset_state, run_window

    if args.reset:
        reset_state()
    if args.window == "all":
        months = window_months()
    elif args.window == "next":
        nxt = next_window()
        if nxt is None:
            print("Todas as janelas já foram processadas (use --reset para recomeçar).")
            return
        months = [nxt]
    else:
        months = [args.window]
    for m in months:
        rec = run_window(m)
        print(f"[{m}] status={rec['status']} AUC={rec.get('performance', {}).get('auc')} "
              f"drift={rec.get('share_drifted_evidently')} alertas={rec.get('n_alerts')}")
    if args.dashboard:
        cmd_dashboard(args)


def cmd_causal(_):
    from .causal_analysis import run
    out = run()
    print(f"Queda AUC {out['total_auc_drop']} | causa raiz: {out['root_cause']}")


def cmd_dashboard(_):
    from .dashboard import build
    print(f"Dashboard: {build()}")


def cmd_run_all(args):
    args.window, args.reset, args.dashboard, args.file, args.strict = "all", True, False, None, False
    for step in (cmd_prepare, cmd_train, cmd_validate, cmd_simulate, cmd_monitor, cmd_causal, cmd_dashboard):
        step(args)


def main(argv=None):
    p = argparse.ArgumentParser(prog="credit-risk", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare").set_defaults(func=cmd_prepare)
    sub.add_parser("train").set_defaults(func=cmd_train)
    v = sub.add_parser("validate", help="valida um arquivo ou o lote corrompido de demonstração")
    v.add_argument("--file")
    v.add_argument("--strict", action="store_true", help="exit 1 se o lote for bloqueado (CI)")
    v.set_defaults(func=cmd_validate)
    sub.add_parser("simulate").set_defaults(func=cmd_simulate)
    m = sub.add_parser("monitor")
    m.add_argument("--window", default="next", help="'next' (agendado), 'all' ou AAAA-MM")
    m.add_argument("--reset", action="store_true")
    m.add_argument("--dashboard", action="store_true", help="regera o dashboard ao final")
    m.set_defaults(func=cmd_monitor)
    sub.add_parser("causal").set_defaults(func=cmd_causal)
    sub.add_parser("dashboard").set_defaults(func=cmd_dashboard)
    sub.add_parser("run-all").set_defaults(func=cmd_run_all)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
