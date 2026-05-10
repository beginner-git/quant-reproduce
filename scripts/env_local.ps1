# Set up GPTQ conda env on the local Windows machine.
# Run from repo root in PowerShell (with conda available on PATH).
#
# Usage: .\scripts\env_local.ps1 GPTQ
#
# Reuses HF_HOME (~/.cache/huggingface by default) so model weights aren't
# re-downloaded across envs.

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("GPTQ", "AWQ", "BiLLM", "KIVI")]
    [string]$Method
)

$ErrorActionPreference = "Stop"

$EnvName = "quant-$($Method.ToLower())"
$YmlPath = Join-Path $PSScriptRoot ".." $Method "env.yml"

if (-not (Test-Path $YmlPath)) {
    Write-Error "No env.yml at $YmlPath"
    exit 1
}

Write-Host "Creating conda env $EnvName from $YmlPath" -ForegroundColor Cyan
conda env create -f $YmlPath

Write-Host "Activate then install common:" -ForegroundColor Cyan
Write-Host "  conda activate $EnvName" -ForegroundColor Yellow
Write-Host "  pip install -e ." -ForegroundColor Yellow

if (-not $env:HF_HOME) {
    $defaultHfHome = Join-Path $HOME ".cache" "huggingface"
    Write-Host "Tip: set `$env:HF_HOME = `"$defaultHfHome`" to share model cache across envs."
}
