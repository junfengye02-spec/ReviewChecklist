from __future__ import annotations

from tender_review.rule_management.public import (
    CreateRuleVersion,
    RuleProvenance,
    RuleVersionRepository,
    RuleVersionService,
)

from .models import (
    OptimizationAttempt,
    OptimizationCandidate,
    OptimizationJob,
    SourceType,
)


class RuleVersionCandidateStager:
    """Create one immutable draft version per accepted optimization candidate."""

    def __init__(
        self,
        service: RuleVersionService,
        repository: RuleVersionRepository,
    ) -> None:
        self._service = service
        self._repository = repository

    def stage_candidate(
        self,
        job: OptimizationJob,
        attempt: OptimizationAttempt,
        candidate: OptimizationCandidate,
    ) -> str:
        base = self._repository.get_version(job.base_rule_version_id)
        rule_set = self._repository.get_rule_set(base.rule_set_id)
        marker = (
            f"[optimization:{job.optimization_job_id}:"
            f"{attempt.attempt_number}:{candidate.candidate_id}]"
        )
        for version in self._repository.list_versions(base.rule_set_id):
            if marker in version.change_summary:
                return version.rule_version_id

        real_verified = (
            job.provenance.source_type is SourceType.REAL
            and job.provenance.status == "verified"
            and job.provenance.claims_allowed
        )
        findings = tuple(
            sorted({item.finding_id for item in job.samples if item.finding_id})
        )
        decisions = tuple(
            sorted(
                {
                    item.human_decision_id
                    for item in job.samples
                    if item.human_decision_id
                }
            )
        )
        provenance = RuleProvenance(
            source_type="optimization" if real_verified else "provisional",
            status="verified" if real_verified else "provisional",
            claims_allowed=real_verified,
            source_finding_ids=findings,
            source_decision_ids=decisions,
            review_input_sha256s=tuple(
                sorted({item.review_input_sha256 for item in job.samples})
            ),
            evidence_sha256s=tuple(
                sorted({item.evidence_sha256 for item in job.samples})
            ),
            dataset_version_id=job.dataset_version_id,
        )
        created = self._service.create_version(
            CreateRuleVersion(
                rule_set_id=base.rule_set_id,
                rule_key=rule_set.rule_key,
                rule_set_name=rule_set.name,
                rule_set_description=rule_set.description,
                parent_version_id=base.rule_version_id,
                content_json=candidate.content_json,
                execution_config_json=candidate.execution_config_json,
                change_summary=f"{marker} {candidate.rationale}",
                provenance=provenance,
            )
        )
        return created.rule_version_id
