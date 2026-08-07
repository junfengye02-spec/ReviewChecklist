[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

Write-Host "[A7 local] protocol and admission tests"
python -m pytest -q -p no:cacheprovider tests/test_a7_load_admission.py
if ($LASTEXITCODE -ne 0) {
    throw "A7 protocol tests failed (exit $LASTEXITCODE)"
}

$TemporaryRoot = Join-Path ([System.IO.Path]::GetTempPath()) (
    "tender-review-a7-" + [guid]::NewGuid().ToString("N")
)
[System.IO.Directory]::CreateDirectory($TemporaryRoot) | Out-Null
try {
    $PlanPath = Join-Path $TemporaryRoot "not-run-plan.json"
    Write-Host "[A7 local] render explicit NOT_RUN plan"
    python -m tender_review.performance plan --output $PlanPath
    if ($LASTEXITCODE -ne 0) {
        throw "A7 NOT_RUN plan failed (exit $LASTEXITCODE)"
    }
    $Plan = Get-Content -LiteralPath $PlanPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (
        $Plan.status -ne "NOT_RUN" -or
        $Plan.matrix.Count -ne 60 -or
        $Plan.report.status -ne "NOT_RUN" -or
        $Plan.report.claims_allowed -ne $false -or
        $Plan.report.queue_decision -ne "NO_DECISION" -or
        $Plan.report.operational_action -ne "KEEP_MYSQL_QUEUE" -or
        $Plan.report.scenario_metrics.Count -ne 0
    ) {
        throw "A7 local plan crossed the real-evidence boundary"
    }
}
finally {
    $ResolvedTemporaryRoot = [System.IO.Path]::GetFullPath($TemporaryRoot)
    $ResolvedSystemTemp = [System.IO.Path]::GetFullPath(
        [System.IO.Path]::GetTempPath()
    )
    if ($ResolvedTemporaryRoot.StartsWith($ResolvedSystemTemp)) {
        Remove-Item -LiteralPath $ResolvedTemporaryRoot -Recurse -Force
    }
}

[ordered]@{
    scope = "a7_protocol_only"
    status = "NOT_RUN"
    evidence = "local_contract_tests_only"
    matrix_cells = 60
    claims_allowed = $false
    queue_decision = "NO_DECISION"
    operational_action = "KEEP_MYSQL_QUEUE"
    redis_decision = "NO_DECISION"
    automatic_stack_change_allowed = $false
    real_environment_gates = [ordered]@{
        mysql = "not_run"
        minio = "not_run"
        model = "not_run"
        independent_workers = "not_run"
        real_pdf_end_to_end = "not_run"
        load_matrix = "not_run"
    }
} | ConvertTo-Json -Depth 3

