# Tender Review

Python backend and independent Vue workbench for reviewing tender documents.
The repository contains application code, migrations, automated tests, API
contracts, and synthetic examples only.

## Data privacy

Real tender documents, bidder information, approval opinions, generated review
reports, evaluation baselines, credentials, and runtime output are not included.
Keep private inputs under `local-data/`; that directory and common document
formats are ignored by Git.

Synthetic examples live under `tests/fixtures/templates/`. Do not replace them
with production data.

## Backend setup

Python 3.12 is supported.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --requirement requirements-dev.txt
```

Run the API with offline fake adapters:

```powershell
$env:TENDER_REVIEW_ADAPTER_MODE = "fake"
$env:TENDER_REVIEW_ENVIRONMENT = "local"
uvicorn tender_review.api.main:app --host 127.0.0.1 --port 8000
```

Run one Worker polling cycle:

```powershell
python -m tender_review.worker --once
```

The API is available under `/api/v1`. Liveness and readiness endpoints are
`/health/live` and `/health/ready`.

## Local infrastructure

Copy `.env.example` to `.env` and replace every local placeholder before using
production adapters:

```powershell
docker compose up --build
```

Production mode requires MySQL, MinIO, and explicit model-provider settings. It
never falls back to fake adapters.

## Private inputs

Legacy command-line workflows require explicit private paths. Use
`examples/legacy-config.example.json` as a structure-only template and provide
paths through command arguments or these environment variables:

- `TENDER_REVIEW_LEGACY_CONFIG`
- `TENDER_REVIEW_EXCEL_PATH`
- `TENDER_REVIEW_PDF_PATH`

Suggested local layout:

```text
local-data/
  tender-documents/
  review-rules.xlsx
  baseline/
```

Never commit files from `local-data/`.

## Workbench

```powershell
cd workbench
npm install
npm run dev
```

The development server uses `http://127.0.0.1:5178` and proxies `/api/v1` to
the backend URL configured in `workbench/.env.example`.

## Verification

```powershell
python -m unittest discover -s tests -v
python -m ruff check tender_review tests alembic
python -m compileall -q tender_review tests alembic
cd workbench
npm run build
```

Tests that validate private historical baselines are skipped unless their
explicit `TENDER_REVIEW_PRIVATE_*` environment variables are set. All default
tests and demo data are synthetic.

See `docs/api/API_USAGE.md` for local API examples.
