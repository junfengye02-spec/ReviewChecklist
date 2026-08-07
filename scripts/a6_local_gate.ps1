[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

function Invoke-A6GateStep {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Action
    )

    Write-Host "[A6 local] $Name"
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "A6 local gate step failed: $Name (exit $LASTEXITCODE)"
    }
}

$FocusedTests = @(
    "tests/test_a6_reliability_observability.py",
    "tests/test_a2_review_job_handler.py::ReviewJobHandlerIntegrationTests::test_single_process_fake_fault_drills_resume_without_duplicate_side_effects",
    "tests/test_phase3_integration.py::DocumentParsingIntegrationTests::test_single_process_fake_parse_fault_resumes_after_durable_checkpoint",
    "tests/test_phase2_worker_reliability.py::ReliableWorkerTests::test_single_process_sqlite_expired_and_reclaimed_leases_fence_old_writes",
    "tests/test_phase3_storage_lifecycle.py::LifecycleTests::test_database_failure_and_orphan_cleanup_share_a_safe_decision_trace",
    "tests/test_a2_openai_compatible.py::OpenAICompatibleLlmProviderTests::test_permanent_rejection_emits_a_safe_correlated_attempt",
    "tests/test_a2_openai_compatible.py::OpenAICompatibleEmbeddingProviderTests::test_permanent_rejection_emits_a_safe_correlated_attempt",
    "tests/test_openapi_contract.py"
)

Invoke-A6GateStep "single-process SQLite/Fake reliability contracts" {
    python -m pytest -q -p no:cacheprovider @FocusedTests
}

Invoke-A6GateStep "Docker Compose configuration parse only" {
    docker compose config --quiet
}

Invoke-A6GateStep "Alembic unique head" {
    $Heads = @(alembic heads)
    if ($LASTEXITCODE -ne 0) {
        return
    }
    $HeadText = $Heads -join "`n"
    if ($HeadText -notmatch "(?m)^d5b0f6a8c214 \(head\)$") {
        throw "Unexpected Alembic head: $HeadText"
    }
    $Heads | Write-Host
}

$OfflineDirectory = Join-Path ([System.IO.Path]::GetTempPath()) (
    "tender-review-a6-" + [guid]::NewGuid().ToString("N")
)
[System.IO.Directory]::CreateDirectory($OfflineDirectory) | Out-Null
$PreviousDatabaseUrl = $env:DATABASE_URL
try {
    $env:DATABASE_URL = (
        "mysql+pymysql://a6_offline:a6_offline@127.0.0.1:3306/" +
        "a6_offline?charset=utf8mb4"
    )
    $UpgradeSql = Join-Path $OfflineDirectory "upgrade.sql"
    $DowngradeSql = Join-Path $OfflineDirectory "downgrade.sql"

    Invoke-A6GateStep "MySQL dialect offline upgrade SQL" {
        alembic upgrade base:head --sql | Out-File -FilePath $UpgradeSql -Encoding utf8
    }
    Invoke-A6GateStep "MySQL dialect offline downgrade SQL" {
        alembic downgrade head:base --sql | Out-File -FilePath $DowngradeSql -Encoding utf8
    }
    if (
        (Get-Item -LiteralPath $UpgradeSql).Length -eq 0 -or
        (Get-Item -LiteralPath $DowngradeSql).Length -eq 0
    ) {
        throw "Alembic offline SQL output was empty"
    }
}
finally {
    if ($null -eq $PreviousDatabaseUrl) {
        Remove-Item Env:DATABASE_URL -ErrorAction SilentlyContinue
    }
    else {
        $env:DATABASE_URL = $PreviousDatabaseUrl
    }
    $ResolvedTemporaryRoot = [System.IO.Path]::GetFullPath($OfflineDirectory)
    $ResolvedSystemTemp = [System.IO.Path]::GetFullPath(
        [System.IO.Path]::GetTempPath()
    )
    if ($ResolvedTemporaryRoot.StartsWith($ResolvedSystemTemp)) {
        Remove-Item -LiteralPath $ResolvedTemporaryRoot -Recurse -Force
    }
}

[ordered]@{
    scope = "local_contract_only"
    evidence = "single_process_sqlite_fake_and_offline_configuration"
    alembic_head = "d5b0f6a8c214"
    human_annotations = "0/361"
    claims_allowed = $false
    real_environment_gates = [ordered]@{
        mysql = "not_run"
        minio = "not_run"
        api = "not_run"
        two_workers = "not_run"
        real_model = "not_run"
        real_pdf_end_to_end = "not_run"
        takeover_and_fencing = "not_run"
    }
} | ConvertTo-Json -Depth 3
