# Execução completa na ordem do deploy (ver README > "Sequência de execução / deploy")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
if (-not (Test-Path ".venv")) { python -m venv .venv }
.\.venv\Scripts\python -m pip install -q -r requirements.txt
.\.venv\Scripts\python -m pip install -q -e .
.\.venv\Scripts\credit-risk prepare      # 1. dados de referência x produção
.\.venv\Scripts\credit-risk train        # 2. contrato na referência + baseline
.\.venv\Scripts\credit-risk validate     # 3. demo: lote corrompido bloqueado
.\.venv\Scripts\credit-risk simulate     # 4. 6 janelas mensais com drift
.\.venv\Scripts\credit-risk monitor --window all --reset   # 5. monitoramento
.\.venv\Scripts\credit-risk causal       # 6. análise causal
.\.venv\Scripts\credit-risk dashboard    # 7. dashboard
