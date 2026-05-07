param(
    [int]$SmokeEpochs = 1,
    [switch]$WithMlflow
)

$ErrorActionPreference = "Stop"

Write-Host "== pytest =="
python -m pytest

Write-Host "== predict smoke =="
python predict.py

Write-Host "== train smoke =="
$trainArgs = @(
    "train.py",
    "--epochs", "$SmokeEpochs",
    "--patience", "1",
    "--batch-size", "16",
    "--seed", "42"
)

if ($WithMlflow) {
    $trainArgs += @(
        "--enable-mlflow",
        "--mlflow-tracking-uri", "sqlite:///mlflow.db",
        "--mlflow-run-name", "local-ci-smoke"
    )
}

python @trainArgs

Write-Host "Local CI passed."
