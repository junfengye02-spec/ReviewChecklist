from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .approval_optimizer import ApprovalOptimizer
from .config import PROJECT_DIR, Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize review points from structured approval opinions"
    )
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
        "--opinion-dir",
        type=Path,
        default=PROJECT_DIR / "runs" / "review-opinions",
    )
    parser.add_argument("--batch-run", type=Path)
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_DIR / "runs")
    parser.add_argument(
        "--resume-run",
        type=Path,
        help="Continue an existing approval optimization run",
    )
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument(
        "--stability-runs",
        type=int,
        default=1,
        help="Consecutive passing workflow tests required for every case",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate approval inputs and workflow metadata without network calls",
    )
    return parser


def find_latest_batch_run(runs_dir: Path) -> Path:
    candidates = sorted(
        (
            path
            for path in runs_dir.glob("batch_*")
            if path.is_dir() and (path / "03_任务索引.json").is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No batch run found under: {runs_dir.resolve()}")
    return candidates[0].resolve()


def main() -> int:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = build_parser().parse_args()
    source_dir = args.source_dir.resolve()
    pdfs = sorted(path for path in source_dir.rglob("*.pdf") if path.is_file())
    if not pdfs:
        print(f"[APPROVAL] FAILED: no PDF files under {source_dir}", file=sys.stderr)
        return 1
    batch_run = (
        args.batch_run.resolve()
        if args.batch_run
        else find_latest_batch_run(args.runs_dir.resolve())
    )
    settings = Settings.load(
        legacy_config=args.legacy_config,
        excel_path=args.excel,
        pdf_path=pdfs[0],
        runs_dir=args.runs_dir,
    )
    optimizer = ApprovalOptimizer(
        settings=settings,
        opinion_dir=args.opinion_dir,
        source_dir=source_dir,
        batch_run=batch_run,
        max_iterations=args.max_iterations,
        stability_runs=args.stability_runs,
        run_dir=args.resume_run,
    )
    try:
        result = optimizer.validate() if args.dry_run else optimizer.run()
    except Exception as exc:
        failure = optimizer.run_dir / "99_运行失败.json"
        failure.write_text(
            json.dumps(
                {"type": type(exc).__name__, "message": str(exc)},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[APPROVAL] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"[APPROVAL] Artifacts: {optimizer.run_dir}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print(f"[APPROVAL] Artifacts: {optimizer.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
