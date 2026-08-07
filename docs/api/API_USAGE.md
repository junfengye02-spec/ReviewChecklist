# Tender Review API v1 usage

> Stage 9R validation: this reference targets the independent
> `tender_review_backend/workbench` application. The superseded Stage 9 output
> is retained only as an invalid audit record. Current engineering evidence is
> under `docs/stage9_tender_final_validation/`.

The frozen OpenAPI document is
[`contracts/openapi-v1.json`](../../contracts/openapi-v1.json). Regenerate it
only after an intentional contract change:

```powershell
python -m tender_review.openapi
python -m tender_review.openapi --check
```

The Stage 9R check confirms that the generated FastAPI schema is byte-identical
to the frozen document. Examples below use local Fake/demo data unless stated
otherwise. Demo responses remain `provisional` or `external-platform` with
`claims_allowed=false`; they are not production or human-verified results.

## Local API

```powershell
$env:TENDER_REVIEW_ADAPTER_MODE = "fake"
$env:TENDER_REVIEW_ENVIRONMENT = "local"
$env:TENDER_REVIEW_WORKBENCH_DEMO_ENABLED = "true"
python -m uvicorn tender_review.api.main:app --host 127.0.0.1 --port 8000
```

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/live
Invoke-RestMethod http://127.0.0.1:8000/health/ready
Invoke-RestMethod http://127.0.0.1:8000/api/v1/workbench
```

Representative workbench response:

```json
{
  "schema_version": 1,
  "demo_mode": true,
  "environment": "local",
  "source_type": "external-platform",
  "status": "provisional",
  "claims_allowed": false,
  "human_annotation_cases": 0,
  "required_human_cases": 4,
  "review_job_ids": ["demo-review-job-1"],
  "evaluation_run_ids": ["phase4-provisional-navigation-20260728"]
}
```

## Documents and jobs

Upload a PDF. `source_system + source_document_id + content` is idempotent; a
different immutable content hash for the same source is rejected.

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/documents `
  -F "source_system=local-demo" `
  -F "source_document_id=synthetic-pdf-001" `
  -F "file=@local-data/tender-documents/synthetic-tender.pdf;type=application/pdf"
```

Create a review job with immutable input hashes. Replace IDs and hashes with
the values registered in the selected environment.

```powershell
$body = @{
  schema_version = 1
  document_snapshot_id = "document-snapshot-id"
  document_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  rule_version_id = "rule-version-id"
  rule_version_hash = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
  model_config_id = "model-config-id"
  model_config_hash = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
  max_attempts = 3
} | ConvertTo-Json

Invoke-RestMethod http://127.0.0.1:8000/api/v1/review-jobs `
  -Method Post `
  -ContentType application/json `
  -Headers @{ "Idempotency-Key" = "demo-review-001"; "X-Caller-ID" = "stage9-demo" } `
  -Body $body
```

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/review-jobs/demo-review-job-1
Invoke-RestMethod http://127.0.0.1:8000/api/v1/review-jobs/demo-review-job-1/checkpoints
Invoke-RestMethod http://127.0.0.1:8000/api/v1/review-jobs/demo-review-job-1/findings
```

Job responses expose lifecycle, processing stage, attempt count, lease token,
failure stage, retryability, and timestamps. They do not expose internal graph
nodes.

## Findings and human decisions

Read a Finding and its immutable decision history:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/findings/demo-finding-1
Invoke-RestMethod http://127.0.0.1:8000/api/v1/findings/demo-finding-1/decisions
```

A local demo may explicitly record `REJECT` or `INSUFFICIENT_EVIDENCE`. The
server rejects `APPROVE` when the Finding is provisional. The local demo script
never sends `APPROVE` or `PUBLISH`.

Representative Finding boundary returned by the demo API:

```json
{
  "finding_id": "demo-finding-1",
  "status": "PENDING_DECISION",
  "provenance": {
    "source_kind": "provisional_retrieval",
    "status": "provisional",
    "claims_allowed": false
  }
}
```

```powershell
$decision = @{
  reviewer_kind = "human"
  reviewer_id = "demo-human-reviewer"
  decision = "INSUFFICIENT_EVIDENCE"
  reason = "Local demo only; evidence is not independently verified."
} | ConvertTo-Json

Invoke-RestMethod http://127.0.0.1:8000/api/v1/findings/demo-finding-1/decisions `
  -Method Post -ContentType application/json -Body $decision
```

## Rules and optimization

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/rule-sets/demo-success-set/versions
Invoke-RestMethod "http://127.0.0.1:8000/api/v1/rule-versions/demo-success-rule-2/diff?against_version_id=demo-success-rule-1"
Invoke-RestMethod http://127.0.0.1:8000/api/v1/optimization-jobs/demo-success-optimization-1
Invoke-RestMethod http://127.0.0.1:8000/api/v1/optimization-jobs/demo-success-optimization-1/attempts
Invoke-RestMethod http://127.0.0.1:8000/api/v1/optimization-jobs/demo-failure-optimization-1
```

The successful demo trace ends at a non-claimable `DRAFT` candidate and
`WAITING_APPROVAL`. The failure trace ends at `OPTIMIZATION_FAILED` after the
configured limit. Neither trace creates a trusted completed evaluation gate,
human approval, or published rule.

## Reports and audit

```powershell
$runId = "phase4-provisional-navigation-20260728"
Invoke-RestMethod "http://127.0.0.1:8000/api/v1/evaluation-runs/$runId"
Invoke-RestMethod "http://127.0.0.1:8000/api/v1/evaluation-runs/$runId/report"
Invoke-RestMethod "http://127.0.0.1:8000/api/v1/audit-events?limit=30"
```

Representative metric boundary:

```json
{
  "metric_id": "recall-at-10",
  "label": "Recall@10",
  "value": null,
  "source_type": "provisional",
  "status": "unknown",
  "claims_allowed": false,
  "collected": false,
  "interpretation": "No independently reviewed human relevance labels are available."
}
```

## Public route inventory

| Area | Methods and paths |
| --- | --- |
| Service | `GET /api/v1`, `GET /health/live`, `GET /health/ready` |
| Documents | `POST /api/v1/documents` |
| Jobs | `POST /api/v1/review-jobs`, `GET /api/v1/review-jobs/{job_id}`, `POST .../cancel`, `POST .../rerun`, `GET .../checkpoints`, `GET .../findings` |
| Findings | `GET /api/v1/findings/{finding_id}`, `GET/POST .../decisions` |
| Rules | `GET/POST /api/v1/rule-sets/{rule_set_id}/versions`, `GET /api/v1/rule-versions/{version_id}`, `GET .../diff`, `POST .../evaluate`, `POST .../optimize`, `POST .../publish`, `POST /api/v1/rule-sets/{rule_set_id}/rollback` |
| Optimization | `GET /api/v1/optimization-jobs/{id}`, `GET .../attempts`, `POST .../cancel` |
| Evaluation | `GET /api/v1/evaluation-runs/{run_id}`, `GET .../report` |
| Workbench/audit | `GET /api/v1/workbench`, `GET /api/v1/audit-events` |
