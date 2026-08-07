from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from tender_review.bootstrap.phase4_annotation import rebuild_chunk_document
from tender_review.config import PROJECT_DIR

from .annotation import (
    ImportedAnnotationBundle,
    annotation_gaps,
    build_annotation_work_package,
    freeze_dataset,
    import_annotations,
    load_work_package,
    render_json,
    validate_work_package_directory,
    work_package_outputs,
)
from .provisional import (
    build_provisional_input,
    default_provisional_config,
    render_provisional_artifacts,
    run_provisional_comparison,
    validate_provisional_artifacts,
    write_provisional_artifacts,
)
from .retrieval import (
    RetrievalDataset,
    RetrievalVariantRun,
    build_evaluation_report,
    evaluate_variant_run,
)
from .runs import EvaluationReport as A4EvaluationReport
from .runs import EvaluationRun as A4EvaluationRun


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 4 retrieval annotation and evaluation pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser(
        "export",
        help="Rebuild the real-PDF candidate annotation work package.",
    )
    export.add_argument(
        "--baseline-manifest",
        type=Path,
        default=PROJECT_DIR / "local-data" / "baseline" / "dataset_manifest.json",
    )
    export.add_argument(
        "--phase3-audit",
        type=Path,
        default=(
            PROJECT_DIR
            / "local-data"
            / "reports"
            / "real_pdf_audit.json"
        ),
    )
    export.add_argument(
        "--pdf-root",
        type=Path,
        default=PROJECT_DIR / "local-data" / "tender-documents",
    )
    export.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "local-data" / "baseline" / "phase4_retrieval",
    )
    export.add_argument(
        "--check",
        action="store_true",
        help="Freshly rebuild in memory and compare with the existing package.",
    )

    validate = subparsers.add_parser(
        "validate-work-package",
        help="Strictly validate package hashes, chunk references, and checksums.",
    )
    validate.add_argument("--package-dir", required=True, type=Path)

    import_parser = subparsers.add_parser(
        "import-annotations",
        help="Import and canonicalize completed human annotation/review decisions.",
    )
    import_parser.add_argument("--package-dir", required=True, type=Path)
    import_parser.add_argument("--input", required=True, type=Path)
    import_parser.add_argument("--output", required=True, type=Path)

    freeze = subparsers.add_parser(
        "freeze",
        help="Freeze an immutable DatasetVersion after the human gate passes.",
    )
    freeze.add_argument("--package-dir", required=True, type=Path)
    freeze.add_argument("--annotations", required=True, type=Path)
    freeze.add_argument("--dataset-version-id", required=True)
    freeze.add_argument("--output", required=True, type=Path)
    freeze.add_argument(
        "--allow-subset",
        action="store_true",
        help="Freeze only approved cases; the default requires all package cases.",
    )

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate Stage 4A ranked-results artifacts against a dataset.",
    )
    evaluate.add_argument("--dataset", required=True, type=Path)
    evaluate.add_argument("--results", required=True, nargs="+", type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument(
        "--allow-provisional",
        action="store_true",
        help="Write an explicitly non-claimable provisional report if the gate fails.",
    )

    provisional = subparsers.add_parser(
        "provisional",
        help="Compare BM25, Vector-only and Hybrid/RRF using navigation hints only.",
    )
    provisional.add_argument(
        "--package-dir",
        type=Path,
        default=PROJECT_DIR / "local-data" / "baseline" / "phase4_retrieval",
    )
    provisional.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_DIR
            / "local-data"
            / "baseline"
            / "phase4_retrieval"
            / "provisional"
        ),
    )
    provisional.add_argument(
        "--check",
        action="store_true",
        help="Validate existing provisional artifacts and their shared hashes.",
    )
    a4_verify = subparsers.add_parser(
        "a4-verify",
        help="Verify immutable A4 run/report hashes and dataset binding offline.",
    )
    a4_verify.add_argument("--run", required=True, type=Path)
    a4_verify.add_argument("--report", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    options = build_parser().parse_args(argv)
    try:
        if options.command == "export":
            return _export(options)
        if options.command == "validate-work-package":
            gaps = validate_work_package_directory(options.package_dir)
            print(render_json(gaps), end="")
            return 0
        if options.command == "import-annotations":
            package, chunks = load_work_package(options.package_dir)
            imported = import_annotations(
                package=package,
                chunks=chunks,
                raw_bundle=_read_json(options.input),
            )
            options.output.parent.mkdir(parents=True, exist_ok=True)
            options.output.write_text(
                render_json(imported), encoding="utf-8", newline="\n"
            )
            print(render_json(annotation_gaps(package, imported)), end="")
            return 0
        if options.command == "freeze":
            package, _ = load_work_package(options.package_dir)
            imported = ImportedAnnotationBundle.model_validate(
                _read_json(options.annotations)
            )
            dataset = freeze_dataset(
                package=package,
                bundle=imported,
                dataset_version_id=options.dataset_version_id,
                require_complete_package=not options.allow_subset,
            )
            options.output.parent.mkdir(parents=True, exist_ok=True)
            options.output.write_text(
                render_json(dataset), encoding="utf-8", newline="\n"
            )
            print(
                json.dumps(
                    {
                        "dataset_version_id": dataset.dataset_version_id,
                        "dataset_sha256": dataset.dataset_sha256,
                        "cases": len(dataset.labels),
                        "status": dataset.status,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        if options.command == "evaluate":
            dataset = RetrievalDataset.model_validate(_read_json(options.dataset))
            runs = [
                RetrievalVariantRun.model_validate(_read_json(path))
                for path in options.results
            ]
            metrics = tuple(
                evaluate_variant_run(dataset=dataset, run=run) for run in runs
            )
            report = build_evaluation_report(
                dataset=dataset,
                variants=metrics,
                require_real_baseline=not options.allow_provisional,
            )
            options.output.parent.mkdir(parents=True, exist_ok=True)
            options.output.write_text(
                render_json(report), encoding="utf-8", newline="\n"
            )
            print(render_json(report), end="")
            return 0
        if options.command == "provisional":
            return _provisional(options)
        if options.command == "a4-verify":
            run = A4EvaluationRun.model_validate(_read_json(options.run))
            report = A4EvaluationReport.model_validate(_read_json(options.report))
            if run.run_id != report.run_id:
                raise ValueError("A4 run and report IDs differ")
            if run.report_sha256 != report.report_sha256:
                raise ValueError("A4 run and report hashes differ")
            if run.binding != report.binding or run.dataset != report.dataset:
                raise ValueError("A4 run and report provenance differs")
            print(json.dumps({
                "run_id": run.run_id,
                "status": run.status.value,
                "source_type": report.source_type.value,
                "provenance_status": report.status,
                "claims_allowed": report.claims_allowed,
                "release_gate_passed": report.release_gate.passed,
                "binding_sha256": run.binding.binding_sha256,
                "result_sha256": run.result_sha256,
                "report_sha256": report.report_sha256,
            }, ensure_ascii=False, sort_keys=True))
            return 0
    except (ValueError, ValidationError, FileNotFoundError, KeyError) as exc:
        if isinstance(exc, ValidationError):
            details = "; ".join(
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors()[:3]
            )
            if len(exc.errors()) > 3:
                details += f"; ... ({len(exc.errors())} total errors)"
            message = f"{len(exc.errors())} validation errors: {details}"
        else:
            message = str(exc)
        print(f"Stage 4 evaluation command failed: {message}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command: {options.command}")


def _export(options: argparse.Namespace) -> int:
    package, chunks, template = build_annotation_work_package(
        baseline_manifest_path=options.baseline_manifest.resolve(),
        phase3_audit_path=options.phase3_audit.resolve(),
        pdf_root=options.pdf_root.resolve(),
        project_root=PROJECT_DIR,
        rebuild_document=rebuild_chunk_document,
    )
    outputs = work_package_outputs(package, chunks, template)
    if options.check:
        mismatches = [
            name
            for name, content in outputs.items()
            if not (options.output_dir / name).is_file()
            or (options.output_dir / name).read_text(encoding="utf-8") != content
        ]
        if mismatches:
            raise ValueError(
                "work package rebuild differs: " + ", ".join(mismatches)
            )
        print(f"Stage 4 work package matches: {options.output_dir}")
        return 0
    options.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        (options.output_dir / name).write_text(
            content, encoding="utf-8", newline="\n"
        )
    print(
        json.dumps(
            {
                "work_package_sha256": package.work_package_sha256,
                "documents": len(package.documents),
                "candidate_cases": len(package.cases),
                "chunks": len(chunks),
                "approved_human_cases": 0,
                "real_dataset_ready": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _provisional(options: argparse.Namespace) -> int:
    if options.check:
        validate_provisional_artifacts(options.output_dir)
        print(f"Stage 4C provisional artifacts valid: {options.output_dir}")
        return 0
    package, chunks = load_work_package(options.package_dir)
    input_contract = build_provisional_input(package, chunks)
    config = default_provisional_config()
    runs, report = run_provisional_comparison(
        input_contract=input_contract,
        chunks=chunks,
        config=config,
    )
    artifacts = render_provisional_artifacts(
        input_contract=input_contract,
        runs=runs,
        report=report,
        config=config,
    )
    write_provisional_artifacts(options.output_dir, artifacts)
    print(
        json.dumps(
            {
                "output_dir": str(options.output_dir),
                "input_sha256": input_contract.input_sha256,
                "claims_allowed": False,
                "human_annotation_cases": 0,
                "required_human_cases": len(input_contract.cases),
                "variants": [item.variant for item in runs],
                "default_strategy_candidate": report.default_strategy_candidate,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
