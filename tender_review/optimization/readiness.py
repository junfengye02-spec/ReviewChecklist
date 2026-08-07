from __future__ import annotations

from tender_review.evaluation.public import (
    AnnotationDatasetRepository,
    AnnotationSampleStatus,
    DatasetSplit,
    DatasetStatus,
    EvaluationPurpose,
    EvaluationRunRepository,
    EvaluationRunStatus,
    EvaluationSourceType,
)
from tender_review.shared.clock import Clock
from tender_review.shared.errors import ServiceError

from .models import (
    CreateOptimizationJob,
    OptimizationReadiness,
    OptimizationReadinessStatus,
    SampleRole,
    SourceType,
)


class UnavailableOptimizationReadinessVerifier:
    """Fail closed when A3/A4 evidence repositories are not configured."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock

    def assess(self, command: CreateOptimizationJob) -> OptimizationReadiness:
        return OptimizationReadiness(
            status=OptimizationReadinessStatus.NOT_READY,
            claims_allowed=False,
            blockers=(
                "A3 verified/FROZEN dataset verification is not configured",
                "A4 claimable failure-sample report verification is not configured",
            ),
            a4_evaluation_run_id=command.a4_evaluation_run_id,
            a4_report_sha256=command.a4_report_sha256,
            assessed_at=self._clock.now(),
        )


class A4OptimizationReadinessVerifier:
    """Bind optimization inputs to one frozen A3 dataset and one A4 report."""

    def __init__(
        self,
        annotations: AnnotationDatasetRepository,
        evaluations: EvaluationRunRepository,
        clock: Clock,
    ) -> None:
        self._annotations = annotations
        self._evaluations = evaluations
        self._clock = clock

    def assess(self, command: CreateOptimizationJob) -> OptimizationReadiness:
        not_ready: list[str] = []
        integrity_blockers: list[str] = []
        dataset = None
        run = None
        report = None

        try:
            dataset = self._annotations.get_version(command.dataset_version_id)
        except ServiceError:
            not_ready.append("A3 annotation dataset is unavailable")

        if command.a4_evaluation_run_id is None:
            not_ready.append("A4 evaluation run id is missing")
        else:
            try:
                run = self._evaluations.get(command.a4_evaluation_run_id)
                report = self._evaluations.get_report(command.a4_evaluation_run_id)
            except ServiceError:
                not_ready.append("A4 evaluation run or report is unavailable")
        if command.a4_report_sha256 is None:
            not_ready.append("A4 report SHA-256 is missing")

        dataset_sample_ids: tuple[str, ...] = ()
        if dataset is not None:
            dataset_sample_ids = tuple(sorted(item.sample_id for item in dataset.samples))
            if dataset.status is not DatasetStatus.FROZEN:
                not_ready.append("A3 dataset status is not FROZEN")
            if (
                dataset.provenance.status != "verified"
                or not dataset.provenance.claims_allowed
                or dataset.required_human_cases != len(dataset.samples)
            ):
                not_ready.append("A3 dataset is not fully verified real data")
            sample_map = {item.sample_id: item for item in dataset.samples}
            for supplied in command.samples:
                source = sample_map.get(supplied.sample_id)
                if source is None:
                    integrity_blockers.append(
                        f"sample {supplied.sample_id} is outside the A3 dataset"
                    )
                    continue
                expected_split = (
                    DatasetSplit.OPTIMIZATION
                    if supplied.role is SampleRole.TARGET
                    else DatasetSplit.VALIDATION
                )
                if source.split is not expected_split:
                    integrity_blockers.append(
                        f"sample {supplied.sample_id} has an invalid optimization role/split"
                    )
                if source.status is not AnnotationSampleStatus.FROZEN:
                    not_ready.append(
                        f"sample {supplied.sample_id} is not independently verified and FROZEN"
                    )
                decision_ids = {
                    item.human_decision_id
                    for item in (source.annotation, source.review, source.adjudication)
                    if item is not None
                }
                identity_matches = (
                    supplied.source_type is SourceType.REAL
                    and supplied.provenance_status == "verified"
                    and supplied.claims_allowed
                    and supplied.document_id == source.document_snapshot_id
                    and supplied.document_sha256 == source.document_sha256
                    and supplied.source_reference == source.source_pdf_reference
                    and supplied.review_input_sha256 == source.source_case_sha256
                    and supplied.evidence_sha256 == source.evidence_catalog_sha256
                    and supplied.finding_id == source.finding_id
                    and supplied.human_decision_id in decision_ids
                )
                if not identity_matches:
                    integrity_blockers.append(
                        f"sample {supplied.sample_id} provenance differs from A3"
                    )

        verified_failure_ids: tuple[str, ...] = ()
        if run is not None and report is not None:
            if command.a4_report_sha256 != report.report_sha256:
                integrity_blockers.append("A4 report SHA-256 does not match persisted report")
            if run.report_sha256 != report.report_sha256:
                integrity_blockers.append("A4 run/report identity does not match")
            if (
                report.run_id != run.run_id
                or report.purpose is not run.purpose
                or report.binding != run.binding
                or report.result_sha256 != run.result_sha256
                or report.dataset != run.dataset
            ):
                integrity_blockers.append("A4 run/report binding or result identity differs")
            if dataset is not None and (
                run.dataset.dataset_version_id != dataset.dataset_version_id
                or report.dataset.dataset_version_id != dataset.dataset_version_id
                or run.dataset.manifest_sha256 != dataset.manifest_sha256
                or report.dataset.manifest_sha256 != dataset.manifest_sha256
                or run.binding.dataset_manifest_sha256 != dataset.manifest_sha256
            ):
                integrity_blockers.append("A4 evidence targets another dataset manifest")
            if (
                run.status is not EvaluationRunStatus.COMPLETED
                or run.purpose is not EvaluationPurpose.CANDIDATE_DIAGNOSTIC
                or run.binding.dataset_split is not DatasetSplit.OPTIMIZATION
                or run.source_type is not EvaluationSourceType.REAL
                or run.provenance_status != "verified"
                or not run.claims_allowed
                or not report.claims_allowed
                or report.status != "verified"
                or report.source_type is not EvaluationSourceType.REAL
                or report.result_sha256 is None
            ):
                not_ready.append(
                    "A4 candidate-diagnostic report is not completed, real, verified, and claimable"
                )
            failure_map = {item.sample_id: item for item in report.failure_samples}
            verified_failure_ids = tuple(sorted(failure_map))
            for target in (
                item for item in command.samples if item.role is SampleRole.TARGET
            ):
                failure = failure_map.get(target.sample_id)
                if failure is None:
                    integrity_blockers.append(
                        f"target {target.sample_id} is not an A4 failure sample"
                    )
                elif target.evidence_sha256 not in failure.evidence_sha256s:
                    integrity_blockers.append(
                        f"target {target.sample_id} A4 evidence hash does not match A3"
                    )

        claimable_input = (
            command.provenance.source_type is SourceType.REAL
            and command.provenance.status == "verified"
            and command.provenance.claims_allowed
            and command.provenance.human_annotation_cases
            == command.provenance.required_human_cases
            and all(
                item.source_type is SourceType.REAL
                and item.provenance_status == "verified"
                and item.claims_allowed
                for item in command.samples
            )
        )
        if not claimable_input:
            not_ready.append("optimization inputs are not real verified claimable data")

        blockers = tuple(dict.fromkeys((*integrity_blockers, *not_ready)))
        status = (
            OptimizationReadinessStatus.BLOCKED
            if integrity_blockers
            else OptimizationReadinessStatus.NOT_READY
            if blockers
            else OptimizationReadinessStatus.READY
        )
        return OptimizationReadiness(
            status=status,
            claims_allowed=status is OptimizationReadinessStatus.READY,
            blockers=blockers,
            dataset_manifest_sha256=(dataset.manifest_sha256 if dataset else None),
            a4_evaluation_run_id=(run.run_id if run else command.a4_evaluation_run_id),
            a4_run_sha256=(run.run_sha256 if run else None),
            a4_report_sha256=(report.report_sha256 if report else command.a4_report_sha256),
            a4_binding_sha256=(run.binding.binding_sha256 if run else None),
            a4_result_sha256=(report.result_sha256 if report else None),
            verified_failure_sample_ids=verified_failure_ids,
            dataset_sample_ids=dataset_sample_ids,
            assessed_at=self._clock.now(),
        )
