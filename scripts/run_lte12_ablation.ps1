param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Device = "cuda",
    [ValidateSet("random-v1", "unseen-sentence-v1", "signerindep-v1")]
    [string]$Split = "random-v1",
    [switch]$SkipBertScore
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

& $Python scripts\train_sample.py --config configs\full_tconv_lte12.yaml --device $Device --split $Split
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$RunDir = Get-ChildItem data_full\runs -Directory |
    Where-Object {
        $ConfigPath = Join-Path $_.FullName "config.json"
        if (-not (Test-Path $ConfigPath)) { return $false }
        try {
            $Config = Get-Content $ConfigPath -Raw | ConvertFrom-Json
            return (
                $Config.trial.name -eq "data_full_tconv_lte12" -and
                $Config.split.kind -eq $Split -and
                $Config.data.max_target_tokens -eq 12
            )
        } catch {
            return $false
        }
    } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $RunDir) {
    throw "Could not find the completed data_full_tconv_lte12 run directory."
}

$Pred = Join-Path $RunDir.FullName "predictions.csv"
$ScoreArgs = @("scripts\score_predictions.py", "--pred", $Pred, "--per-example")
if (-not $SkipBertScore) {
    $ScoreArgs += "--bertscore"
}
& $Python @ScoreArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python scripts\prediction_diagnostics.py `
    --pred $Pred `
    --out-md "docs\logs\DIAG_tconv_lte12_$Split.md" `
    --out-json "docs\logs\DIAG_tconv_lte12_$Split.json"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python scripts\trivial_baselines.py --out-dir data_full --config configs\full_tconv_lte12.yaml --split $Split
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "LTE12 ablation complete."
Write-Host "Split: $Split"
Write-Host "Run directory: $($RunDir.FullName)"
Write-Host "Diagnostics: docs\logs\DIAG_tconv_lte12_$Split.md"
