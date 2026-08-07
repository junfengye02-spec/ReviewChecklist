from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings
from .pipeline import MvpPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Foshan tender review backend MVP")
    parser.add_argument("--legacy-config", type=Path)
    parser.add_argument("--excel", type=Path)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument(
        "--resume-run",
        type=Path,
        help="Resume evaluation/optimization from an existing run directory",
    )
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument(
        "--stability-runs",
        type=int,
        default=3,
        help="Consecutive workflow-test hits required before a review point passes",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and build artifacts without calling remote APIs",
    )
    return parser


def main() -> int:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = build_parser().parse_args()
    settings = Settings.load(
        legacy_config=args.legacy_config,
        excel_path=args.excel,
        pdf_path=args.pdf,
        runs_dir=args.runs_dir,
    )
    pipeline = MvpPipeline(
        settings,
        max_iterations=args.max_iterations,
        stability_runs=args.stability_runs,
        run_dir=args.resume_run,
    )
    try:
        if args.resume_run:
            result = pipeline.resume()
        else:
            result = pipeline.prepare() if args.dry_run else pipeline.run()
    except Exception as exc:
        pipeline.artifacts.write_json(
            "99_运行失败.json", {"type": type(exc).__name__, "message": str(exc)}
        )
        print(f"[MVP] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"[MVP] Artifacts: {pipeline.artifacts.run_dir}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print(f"[MVP] Artifacts: {pipeline.artifacts.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
