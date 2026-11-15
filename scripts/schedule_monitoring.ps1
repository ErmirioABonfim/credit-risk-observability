<#
.SYNOPSIS
  Registra (ou remove) a tarefa agendada que executa o monitoramento de drift.

.DESCRIPTION
  Cada execução processa a PRÓXIMA janela mensal de produção ainda não monitorada
  (`credit-risk monitor --window next --dashboard`), reproduzindo a passagem do tempo:
  1 execução = 1 mês de produção. Em produção real o intervalo seria mensal/semanal;
  para a demonstração usamos um intervalo curto (padrão: 10 minutos).

.EXAMPLE
  ./scripts/schedule_monitoring.ps1 -IntervalMinutes 10      # registra
  ./scripts/schedule_monitoring.ps1 -Unregister              # remove
  Start-ScheduledTask -TaskName CreditRisk-DriftMonitor      # dispara agora
#>
param(
    [int]$IntervalMinutes = 10,
    [string]$TaskName = "CreditRisk-DriftMonitor",
    [switch]$Unregister
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Tarefa '$TaskName' removida."
    return
}

$exe = Join-Path $root ".venv\Scripts\credit-risk.exe"
if (-not (Test-Path $exe)) { throw "Ambiente não encontrado. Rode antes: python -m venv .venv; .venv\Scripts\pip install -e ." }

$action  = New-ScheduledTaskAction -Execute $exe -Argument "monitor --window next --dashboard" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
           -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes) -RepetitionDuration (New-TimeSpan -Days 1)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
           -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Description "Tech Challenge Fase 4 - monitoramento de drift do modelo de credito (1 janela por execucao)" -Force | Out-Null
Write-Host "Tarefa '$TaskName' registrada: a cada $IntervalMinutes min processa a próxima janela de produção."
Write-Host "Acompanhe: Get-Content $root\logs\pipeline.jsonl -Wait -Tail 5"
