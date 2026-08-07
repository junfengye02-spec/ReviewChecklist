from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .batch import IndividualTaskBatchRunner, discover_documents
from .config import PROJECT_DIR, Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch tender PDF review")
    parser.add_argument("--legacy-config", type=Path)
    parser.add_argument(
        "--excel",
        type=Path,
        default=PROJECT_DIR / "local-data" / "review-rules.xlsx",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=PROJECT_DIR / "local-data" / "tender-documents",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "runs" / "review-opinions",
    )
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_DIR / "runs")
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument(
        "--sequential-recovery",
        action="store_true",
        help="Harvest completed reports and retry unresolved PDFs one at a time",
    )
    parser.add_argument(
        "--isolated-recovery",
        action="store_true",
        help="Recover unresolved PDFs with a fresh product per file",
    )
    parser.add_argument(
        "--write-partial",
        action="store_true",
        help="Write valid saved reports and mark unresolved PDFs as missing",
    )
    parser.add_argument("--timeout-seconds", type=float, default=14400)
    return parser


def main() -> int:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = build_parser().parse_args()
    documents = discover_documents(args.source_dir)
    settings = Settings.load(
        legacy_config=args.legacy_config,
        excel_path=args.excel,
        pdf_path=documents[0].path,
        runs_dir=args.runs_dir,
    )
    runner = IndividualTaskBatchRunner(
        settings=settings,
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        timeout_seconds=args.timeout_seconds,
        run_dir=args.resume_run,
    )
    try:
        if args.write_partial:
            if not args.resume_run:
                raise ValueError("--write-partial requires --resume-run")
            result = runner.write_partial_outputs()
        elif args.isolated_recovery:
            if not args.resume_run:
                raise ValueError("--isolated-recovery requires --resume-run")
            result = runner.recover_with_isolated_products()
        elif args.sequential_recovery:
            if not args.resume_run:
                raise ValueError("--sequential-recovery requires --resume-run")
            result = runner.recover_sequentially()
        else:
            result = runner.resume() if args.resume_run else runner.run()
    except Exception as exc:
        print(f"[BATCH] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"[BATCH] Artifacts: {runner.run_dir}", file=sys.stderr)
        return 1

    print(json.dumps(result["totals"], ensure_ascii=False, indent=2))
    print(f"[BATCH] Opinions: {args.output_dir.resolve()}")
    print(f"[BATCH] Artifacts: {runner.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
