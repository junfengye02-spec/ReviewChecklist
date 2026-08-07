from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tender_review.config import PROJECT_DIR
from tender_review.evaluation.public import (
    CreateDatasetVersion,
    DatasetProvenance,
    DatasetSampleInput,
    DatasetSourceType,
    DatasetSplit,
    DatasetVersionService,
    InMemoryDatasetVersionRepository,
)
from tender_review.review.public import FakeLlmProvider
from tender_review.rule_management.public import (
    CreateRuleVersion,
    InMemoryRuleVersionRepository,
    RuleProvenance,
    RuleVersionService,
)
from tender_review.shared.clock import FixedClock
from tender_review.shared.ids import SequentialIdGenerator

from .application import OptimizationService, RootCauseAnalyzer
from .fakes import (
    FakeCandidateGenerator,
    FakeRegressionEvaluator,
    InMemoryOptimizationRepository,
)
from .migration import (
    ApprovalOptimizerMigration,
    HistoricalOptimizationGroup,
    migrate_approval_optimizer_baseline,
)
from .models import (
    CreateOptimizationJob,
    FailureSignals,
    OptimizationProvenance,
    OptimizationSample,
    SampleRole,
    SourceArtifact,
    SourceType,
    stable_sha256,
)
from .rule_candidates import RuleVersionCandidateStager


DEFAULT_SOURCE = PROJECT_DIR / "baseline" / "platform_optimization_baseline.json"
DEFAULT_OUTPUT = PROJECT_DIR / "baseline" / "phase7_optimization"
DEMO_TIME = datetime(2026, 7, 28, 8, 0, tzinfo=timezone.utc)


def render_demo_artifacts(
    source_path: Path = DEFAULT_SOURCE,
) -> dict[str, bytes]:
    """Return the frozen pre-A5 demonstration without executing optimization.

    A5 rejects this provisional source as NOT_READY. The historical artifact remains
    readable and byte-verifiable, but it must not be regenerated as a new trajectory.
    """
    migration = migrate_approval_optimizer_baseline(source_path)
    frozen_names = (
        "migration.json",
        "success_trace.json",
        "failure_trace.json",
        "README.md",
        "manifest.json",
        "checksums.json",
    )
    frozen = {name: (DEFAULT_OUTPUT / name).read_bytes() for name in frozen_names}
    manifest = json.loads(frozen["manifest.json"])
    if manifest.get("source_input_sha256") != migration.source_input_sha256:
        raise ValueError("frozen Phase 7 demo targets another source input")
    return frozen


def write_demo_artifacts(
    output_dir: Path = DEFAULT_OUTPUT,
    source_path: Path = DEFAULT_SOURCE,
) -> tuple[Path, ...]:
    artifacts = render_demo_artifacts(source_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, content in artifacts.items():
        path = output_dir / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
        written.append(path)
    return tuple(written)


def validate_demo_artifacts(
    output_dir: Path = DEFAULT_OUTPUT,
    source_path: Path = DEFAULT_SOURCE,
) -> list[str]:
    expected = render_demo_artifacts(source_path)
    errors: list[str] = []
    for name, content in expected.items():
        path = output_dir / name
        if not path.is_file():
            errors.append(f"missing: {name}")
        elif path.read_bytes() != content:
            errors.append(f"changed: {name}")
    extras = sorted(
        path.name for path in output_dir.glob("*") if path.name not in expected
    ) if output_dir.is_dir() else []
    errors.extend(f"unexpected: {name}" for name in extras)
    return errors


def _run_scenario(
    migration: ApprovalOptimizerMigration,
    group: HistoricalOptimizationGroup,
    *,
    scenario: str,
    plans: tuple[tuple[bool, bool, bool], ...],
    max_rounds: int,
    candidates_per_round: int,
) -> dict[str, Any]:
    clock = FixedClock(DEMO_TIME)
    rules = InMemoryRuleVersionRepository()
    rule_service = RuleVersionService(
        rules, SequentialIdGenerator(prefix=f"demo-{scenario}-rule"), clock
    )
    base = rule_service.create_version(
        CreateRuleVersion(
            rule_set_id=f"demo-{scenario}-set",
            rule_key=f"historical-review-item-{group.review_item}",
            rule_set_name=f"Historical review item {group.review_item}",
            content_json=json.dumps(
                {"rule_text": f"recorded base for item {group.review_item}"},
                sort_keys=True,
                separators=(",", ":"),
            ),
            change_summary="immutable demonstration base",
            provenance=RuleProvenance(
                source_type="provisional",
                status="provisional",
                claims_allowed=False,
            ),
        )
    )
    datasets = InMemoryDatasetVersionRepository()
    dataset_service = DatasetVersionService(
        datasets,
        SequentialIdGenerator(prefix=f"demo-{scenario}-dataset"),
        clock,
    )
    target_id = f"platform-group-{group.review_item}-target-aggregate"
    protection_id = f"platform-group-{group.review_item}-protection-aggregate"
    target_document = f"external-platform-group-{group.review_item}-target"
    protection_document = f"external-platform-group-{group.review_item}-protection"
    dataset = dataset_service.create_version(
        CreateDatasetVersion(
            dataset_name=f"phase7-{scenario}-provisional",
            requested_status="PROVISIONAL",
            change_summary=(
                "Migrated aggregate external-platform observations; not chunk labels."
            ),
            provenance=DatasetProvenance(
                status="provisional",
                claims_allowed=False,
                source_description=migration.source_claim_boundary,
                source_manifest_sha256=migration.migration_sha256,
            ),
            samples=(
                _dataset_sample(target_id, target_document, DatasetSplit.OPTIMIZATION),
                _dataset_sample(
                    protection_id, protection_document, DatasetSplit.VALIDATION
                ),
            ),
        )
    )
    repository = InMemoryOptimizationRepository()
    generator = FakeCandidateGenerator(
        base_content_json=base.content_json,
        base_execution_config_json=base.execution_config_json,
    )
    evaluator = FakeRegressionEvaluator(plans)
    service = OptimizationService(
        repository=repository,
        rule_versions=rules,
        datasets=datasets,
        ids=SequentialIdGenerator(prefix=f"demo-{scenario}-optimization"),
        clock=clock,
        root_causes=RootCauseAnalyzer(
            FakeLlmProvider(
                (
                    '{"rationale":"Historical workflow output indicates a semantic rule '
                    'coverage candidate; this diagnosis remains provisional.",'
                    '"root_cause":"RULE_GAP"}',
                )
                * max_rounds
            )
        ),
        candidates=generator,
        evaluator=evaluator,
        stager=RuleVersionCandidateStager(rule_service, rules),
    )
    source_reference = (
        f"baseline/platform_optimization_baseline.json#"
        f"approval_optimizer_baseline.groups.{group.review_item}"
    )
    command = CreateOptimizationJob(
        base_rule_version_id=base.rule_version_id,
        dataset_version_id=dataset.dataset_version_id,
        max_rounds=max_rounds,
        candidates_per_round=candidates_per_round,
        required_stability_runs=2,
        model_sha256=stable_sha256("historical-configured-model"),
        prompt_sha256=stable_sha256("phase7-root-cause-and-candidate-prompts-v1"),
        retriever_sha256=stable_sha256("historical-platform-retriever-unknown"),
        tool_sha256=stable_sha256("historical-platform-tools-unknown"),
        samples=(
            _optimization_sample(
                target_id,
                SampleRole.TARGET,
                target_document,
                source_reference,
                FailureSignals(
                    failure_summary=(
                        f"Recorded external-platform group status={group.platform_status}; "
                        f"target_count={group.target_count}. Internal stage telemetry is absent."
                    )
                ),
            ),
            _optimization_sample(
                protection_id,
                SampleRole.PROTECTION,
                protection_document,
                source_reference,
                None,
            ),
        ),
        provenance=OptimizationProvenance(
            source_type=SourceType.EXTERNAL_PLATFORM,
            status="provisional",
            claims_allowed=False,
            source_description=(
                "Real tender materials evaluated by a historical external platform. "
                "Platform status is not a human-approved or chunk-level label."
            ),
            source_artifacts=tuple(
                SourceArtifact(
                    path=item.path,
                    sha256=item.sha256,
                    kind=item.kind,
                )
                for item in migration.source_artifacts
            ),
            human_annotation_cases=0,
            required_human_cases=migration.required_human_cases,
        ),
    )
    created = service.create(command)
    completed = service.run(created.optimization_job_id)
    attempts = service.list_attempts(created.optimization_job_id)
    candidate = (
        rules.get_version(completed.candidate_rule_version_id)
        if completed.candidate_rule_version_id
        else None
    )
    return {
        "schema_version": 1,
        "scenario": scenario,
        "source_group": group.model_dump(mode="json"),
        "boundary": {
            "source_type": "EXTERNAL_PLATFORM",
            "status": "provisional",
            "claims_allowed": False,
            "human_annotation_cases": 0,
            "required_human_cases": migration.required_human_cases,
            "historical_platform_status_is_chunk_label": False,
            "accuracy_improvement_claimed": False,
            "complete_evaluation_gate_created": False,
            "human_approval_recorded": False,
            "auto_published": False,
        },
        "job": completed.model_dump(mode="json"),
        "attempts": [item.model_dump(mode="json") for item in attempts],
        "candidate_rule_version": (
            candidate.model_dump(mode="json") if candidate is not None else None
        ),
    }


def _dataset_sample(
    sample_id: str, document_id: str, split: DatasetSplit
) -> DatasetSampleInput:
    return DatasetSampleInput(
        sample_id=sample_id,
        document_id=document_id,
        document_sha256=stable_sha256(document_id),
        split=split,
        source_type=DatasetSourceType.EXTERNAL_PLATFORM,
        provenance_status="provisional",
        label_version="external-platform-observation-v1",
        label_json='{"label":"aggregate-platform-observation-not-human-truth"}',
        review_input_sha256=stable_sha256(f"review-input:{sample_id}"),
        evidence_sha256=stable_sha256(f"evidence-reference:{sample_id}"),
    )


def _optimization_sample(
    sample_id: str,
    role: SampleRole,
    document_id: str,
    source_reference: str,
    signals: FailureSignals | None,
) -> OptimizationSample:
    return OptimizationSample(
        sample_id=sample_id,
        role=role,
        document_id=document_id,
        document_sha256=stable_sha256(document_id),
        source_type=SourceType.EXTERNAL_PLATFORM,
        provenance_status="provisional",
        claims_allowed=False,
        source_reference=source_reference,
        review_input_sha256=stable_sha256(f"review-input:{sample_id}"),
        evidence_sha256=stable_sha256(f"evidence-reference:{sample_id}"),
        signals=signals,
    )


def _group(
    migration: ApprovalOptimizerMigration, review_item: str
) -> HistoricalOptimizationGroup:
    return next(item for item in migration.groups if item.review_item == review_item)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _readme() -> str:
    return """# Phase 7 provisional optimization demonstration

Rebuild or check these artifacts with:

```text
python -m tender_review.optimization.demo
python -m tender_review.optimization.demo --check
```

The success trace uses historical external-platform review item 47. The bounded
failure trace uses item 12. These are aggregate observations over real tender
materials, not human-approved labels and not chunk relevance truth. Every trace
keeps `source_type=EXTERNAL_PLATFORM`, `status=provisional`, and
`claims_allowed=false`. A passing three-gate result creates only an immutable
draft rule candidate at the human approval boundary; it does not create a
release `CompleteEvaluationGate`, record human approval, or publish anything.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify or copy the frozen pre-A5 provisional optimization artifacts"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        errors = validate_demo_artifacts(args.output_dir, args.source)
        if errors:
            for error in errors:
                print(error)
            return 1
        print(
            "Frozen pre-A5 artifacts verified; no optimization was executed: "
            f"{args.output_dir}"
        )
        return 0
    written = write_demo_artifacts(args.output_dir, args.source)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
