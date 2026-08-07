from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from tender_review.config import PROJECT_DIR
from tender_review.shared.config import AppSettings


DEFAULT_BASELINE = PROJECT_DIR / "contracts" / "openapi-v1.json"
_POST_FREEZE_PATH_PREFIXES = (
    "/api/v1/annotation-datasets",
    "/api/v1/a4/",
    "/api/v1/a5/",
    "/api/v1/a7/",
)
_POST_FREEZE_COMPONENTS = {
    "AnnotationDatasetProvenance",
    "AnnotationDatasetVersion",
    "AnnotationEvidenceChunk",
    "AnnotationSampleInput",
    "AnnotationSampleStatus",
    "ChunkRelevanceLabel",
    "CreateAnnotationDatasetRequest",
    "DatasetAnnotationSample",
    "DatasetSplit",
    "DatasetStatus",
    "HumanLabelRecord",
    "SubmitAnnotationLabelRequest",
    "CreateEvaluationRunRequest",
    "EngineeringMetrics",
    "EvaluationDatasetSnapshot",
    "EvaluationDifferenceSource",
    "EvaluationFailureSample",
    "EvaluationMetrics",
    "EvaluationPurpose",
    "EvaluationResult",
    "EvaluationRunBinding",
    "EvaluationRunStatus",
    "EvaluationSourceType",
    "GateAssessmentStatus",
    "MetricDifference",
    "ReleaseGateAssessment",
    "RetrievalMetrics",
    "ReviewMetrics",
    "StabilityMetrics",
    "A5OptimizeRuleVersionRequest",
    "OptimizationReadiness",
    "OptimizationReadinessStatus",
    "OptimizationTraceEvent",
    "OptimizationTraceOutcome",
    "A7AdmissionReport",
    "A7Authenticity",
    "A7ExecutionBinding",
    "A7RunStatus",
    "A7SourceType",
    "MetricId",
    "QueueDecision",
    "RedisDecision",
    "ScenarioMetrics",
    "ThresholdAssessment",
}


def render_openapi() -> str:
    from tender_review.bootstrap import create_api_app

    app = create_api_app(
        AppSettings(environment="contract", adapter_mode="fake", log_json=False)
    )
    schema = app.openapi()
    schema["paths"] = {
        path: value
        for path, value in schema["paths"].items()
        if path != "/api/v1/" and not path.startswith(_POST_FREEZE_PATH_PREFIXES)
    }
    components = schema.get("components", {}).get("schemas", {})
    for name in _POST_FREEZE_COMPONENTS:
        components.pop(name, None)
    optimization_job = components.get("OptimizationJob", {})
    properties = optimization_job.get("properties", {})
    properties.pop("readiness", None)
    properties.pop("graph_trace", None)
    required = optimization_job.get("required", [])
    optimization_job["required"] = [
        name for name in required if name not in {"readiness", "graph_trace"}
    ]
    optimization_status = components.get("OptimizationStatus", {})
    optimization_status["enum"] = [
        value
        for value in optimization_status.get("enum", [])
        if value not in {"NOT_READY", "BLOCKED"}
    ]
    review_job_response = components.get("ReviewJobResponse", {})
    review_job_properties = review_job_response.get("properties", {})
    a6_review_job_fields = {
        "recovery_count",
        "recovery_metric_source",
        "safe_failure_code",
        "safe_failure_category",
        "safe_failure_retryable",
    }
    for name in a6_review_job_fields:
        review_job_properties.pop(name, None)
    review_job_response["required"] = [
        name
        for name in review_job_response.get("required", [])
        if name not in a6_review_job_fields
    ]
    audit_event = components.get("AuditEvent", {})
    audit_event_properties = audit_event.get("properties", {})
    for name in {
        "job_id",
        "thread_id",
        "checkpoint_id",
        "rule_version",
        "dataset_version",
        "model_config",
    }:
        audit_event_properties.pop(name, None)
    for name in tuple(components):
        if name.startswith("tender_review__evaluation__runs__"):
            components.pop(name, None)
    for model_name in ("EvaluationReport", "EvaluationRun"):
        qualified = f"tender_review__stage8__models__{model_name}"
        if qualified in components:
            components[model_name] = components.pop(qualified)
    schema = json.loads(
        json.dumps(schema)
        .replace(
            "#/components/schemas/tender_review__stage8__models__EvaluationReport",
            "#/components/schemas/EvaluationReport",
        )
        .replace(
            "#/components/schemas/tender_review__stage8__models__EvaluationRun",
            "#/components/schemas/EvaluationRun",
        )
    )
    return json.dumps(
        schema, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"


def check_openapi(path: Path = DEFAULT_BASELINE) -> list[str]:
    if not path.is_file():
        return [f"Missing OpenAPI baseline: {path}"]
    if path.read_text(encoding="utf-8") != render_openapi():
        return [f"OpenAPI baseline differs: {path}"]
    return []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze the v1 OpenAPI contract")
    parser.add_argument("--output", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.resolve()
    if args.check:
        mismatches = check_openapi(output)
        if mismatches:
            for mismatch in mismatches:
                print(mismatch)
            return 1
        print(f"OpenAPI baseline matches: {output}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_openapi(), encoding="utf-8")
    print(f"Wrote OpenAPI baseline: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
