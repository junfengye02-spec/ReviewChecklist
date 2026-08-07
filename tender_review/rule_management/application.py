from __future__ import annotations

import json
from typing import Any

from tender_review.shared.clock import Clock
from tender_review.shared.errors import ConflictError, NotFoundError, PermanentError
from tender_review.shared.ids import IdGenerator

from .models import (
    CompleteEvaluationGate,
    CreateRuleVersion,
    EvaluationGate,
    EvaluationGateStatus,
    PublishRuleVersion,
    RollbackRuleSet,
    RuleDiffChange,
    RuleSet,
    RuleVersion,
    RuleVersionDiff,
    RuleVersionStatus,
    canonical_json,
    require_named_human,
    stable_sha256,
)
from .ports import ReleaseGateVerifier, RuleVersionRepository


class _UnavailableReleaseGateVerifier:
    def assert_dataset_release_ready(self, dataset_version_id: str) -> None:
        del dataset_version_id
        raise PermanentError(
            "A4 release-gate verification is not configured",
            code="release_gate_verifier_unavailable",
        )

    def assert_release_eligible(self, **identity: str) -> None:
        del identity
        raise PermanentError(
            "A4 release-gate verification is not configured",
            code="release_gate_verifier_unavailable",
        )


class RuleVersionService:
    def __init__(
        self,
        repository: RuleVersionRepository,
        ids: IdGenerator,
        clock: Clock,
        release_gates: ReleaseGateVerifier | None = None,
    ) -> None:
        self._repository = repository
        self._ids = ids
        self._clock = clock
        self._release_gates = release_gates or _UnavailableReleaseGateVerifier()

    def list_versions(self, rule_set_id: str) -> tuple[RuleVersion, ...]:
        return self._repository.list_versions(rule_set_id)

    def create_version(self, command: CreateRuleVersion) -> RuleVersion:
        rule_set = self._get_or_create_set(command)
        versions = self._repository.list_versions(rule_set.rule_set_id)
        version_number = len(versions) + 1
        expected_parent = versions[-1].rule_version_id if versions else None
        if command.parent_version_id != expected_parent:
            raise ConflictError(
                "parent_version_id must reference the latest immutable version",
                code="rule_parent_not_latest",
                details={"expected_parent_version_id": expected_parent},
            )
        content = json.loads(command.content_json)
        execution = json.loads(command.execution_config_json)
        version = RuleVersion(
            rule_version_id=self._ids.new(),
            rule_set_id=rule_set.rule_set_id,
            version_number=version_number,
            parent_version_id=command.parent_version_id,
            status=RuleVersionStatus.DRAFT,
            content_json=command.content_json,
            execution_config_json=command.execution_config_json,
            content_sha256=stable_sha256({"content": content, "execution_config": execution}),
            change_summary=command.change_summary,
            provenance=command.provenance,
            created_at=self._clock.now(),
        )
        return self._repository.add_version(version)

    def diff(self, from_version_id: str, to_version_id: str) -> RuleVersionDiff:
        before = self._repository.get_version(from_version_id)
        after = self._repository.get_version(to_version_id)
        if before.rule_set_id != after.rule_set_id:
            raise PermanentError("cannot diff versions from different rule sets", code="rule_diff_set_mismatch")
        changes = _diff_values(json.loads(before.content_json), json.loads(after.content_json))
        return RuleVersionDiff(
            from_version_id=from_version_id,
            to_version_id=to_version_id,
            changes=tuple(changes),
        )

    def request_evaluation(self, rule_version_id: str, dataset_version_id: str) -> RuleVersion:
        version = self._repository.get_version(rule_version_id)
        if version.status not in {RuleVersionStatus.DRAFT, RuleVersionStatus.OPTIMIZING}:
            raise ConflictError("rule version cannot enter evaluation", code="rule_evaluation_state_invalid")
        self._release_gates.assert_dataset_release_ready(dataset_version_id)
        payload = {
            "schema_version": 1,
            "gate_id": self._ids.new(),
            "rule_version_id": version.rule_version_id,
            "dataset_version_id": dataset_version_id,
            "status": EvaluationGateStatus.PENDING,
            "provisional": version.provenance.status == "provisional",
            "claims_allowed": False,
            "evaluation_run_id": None,
            "report_sha256": None,
            "requested_at": self._clock.now(),
            "completed_at": None,
        }
        gate = EvaluationGate(**payload, gate_sha256=stable_sha256(payload))
        evaluating = version.model_copy(update={"status": RuleVersionStatus.EVALUATING})
        return self._repository.save_gate(evaluating, gate)

    def complete_evaluation(self, command: CompleteEvaluationGate) -> RuleVersion:
        version = self._repository.get_version(command.rule_version_id)
        gate = version.evaluation_gate
        if version.status is not RuleVersionStatus.EVALUATING or gate is None:
            raise ConflictError("rule version is not awaiting evaluation", code="rule_evaluation_not_pending")
        if gate.gate_id != command.gate_id:
            raise ConflictError("evaluation gate identity differs", code="rule_gate_mismatch")
        if command.status == "PASSED" and (
            command.provisional or not command.claims_allowed or version.provenance.status == "provisional"
        ):
            raise PermanentError("provisional-only evaluation cannot pass the release gate", code="provisional_gate_cannot_pass")
        if command.status == "PASSED":
            self._release_gates.assert_release_eligible(
                rule_version_id=version.rule_version_id,
                dataset_version_id=gate.dataset_version_id,
                evaluation_run_id=command.evaluation_run_id,
                report_sha256=command.report_sha256,
            )
        status = EvaluationGateStatus(command.status)
        payload = {
            "schema_version": 1,
            "gate_id": gate.gate_id,
            "rule_version_id": version.rule_version_id,
            "dataset_version_id": gate.dataset_version_id,
            "status": status,
            "provisional": command.provisional,
            "claims_allowed": command.claims_allowed,
            "evaluation_run_id": command.evaluation_run_id,
            "report_sha256": command.report_sha256,
            "requested_at": gate.requested_at,
            "completed_at": self._clock.now(),
        }
        completed = EvaluationGate(**payload, gate_sha256=stable_sha256(payload))
        next_status = (
            RuleVersionStatus.WAITING_APPROVAL
            if status is EvaluationGateStatus.PASSED
            else RuleVersionStatus.REJECTED
        )
        return self._repository.save_gate(
            version.model_copy(update={"status": next_status}), completed
        )

    def publish(self, command: PublishRuleVersion) -> RuleVersion:
        approver = require_named_human(command.approver_id)
        version = self._repository.get_version(command.rule_version_id)
        gate = version.evaluation_gate
        if version.status is not RuleVersionStatus.WAITING_APPROVAL:
            raise ConflictError("rule version is not waiting for approval", code="rule_publish_state_invalid")
        if gate is None or gate.status is not EvaluationGateStatus.PASSED:
            raise PermanentError("rule version has not passed its evaluation gate", code="rule_gate_not_passed")
        if gate.provisional or not gate.claims_allowed or version.provenance.status == "provisional":
            raise PermanentError("provisional-only rule version cannot be published", code="provisional_rule_publish_forbidden")
        self._release_gates.assert_release_eligible(
            rule_version_id=version.rule_version_id,
            dataset_version_id=gate.dataset_version_id,
            evaluation_run_id=gate.evaluation_run_id or "",
            report_sha256=gate.report_sha256 or "",
        )
        rule_set = self._repository.get_rule_set(version.rule_set_id)
        published = version.model_copy(update={
            "status": RuleVersionStatus.PUBLISHED,
            "published_at": self._clock.now(),
            "published_by": approver,
        })
        return self._repository.publish(rule_set, published)

    def rollback(self, command: RollbackRuleSet) -> RuleVersion:
        require_named_human(command.approver_id)
        rule_set = self._repository.get_rule_set(command.rule_set_id)
        if rule_set.current_version_id is None:
            raise ConflictError("rule set has no published version", code="rule_set_not_published")
        current = self._repository.get_version(rule_set.current_version_id)
        target = self._repository.get_version(command.target_version_id)
        if target.rule_set_id != rule_set.rule_set_id or target.rule_version_id == current.rule_version_id:
            raise PermanentError("rollback target must be another version in this rule set", code="rule_rollback_target_invalid")
        if target.published_at is None:
            raise PermanentError("rollback target was never published", code="rule_rollback_target_unpublished")
        rolled_current = current.model_copy(update={"status": RuleVersionStatus.ROLLED_BACK})
        restored = target.model_copy(update={"status": RuleVersionStatus.PUBLISHED})
        return self._repository.rollback(rule_set, rolled_current, restored)

    def _get_or_create_set(self, command: CreateRuleVersion) -> RuleSet:
        try:
            existing = self._repository.get_rule_set(command.rule_set_id)
        except NotFoundError:
            return self._repository.create_rule_set(RuleSet(
                rule_set_id=command.rule_set_id,
                rule_key=command.rule_key,
                name=command.rule_set_name,
                description=command.rule_set_description,
                created_at=self._clock.now(),
            ))
        if existing.rule_key != command.rule_key or existing.name != command.rule_set_name:
            raise ConflictError("rule set metadata differs", code="rule_set_conflict")
        return existing


def _diff_values(before: Any, after: Any, path: str = "$") -> list[RuleDiffChange]:
    if isinstance(before, dict) and isinstance(after, dict):
        changes: list[RuleDiffChange] = []
        for key in sorted(set(before) | set(after)):
            child = f"{path}.{key}"
            if key not in before:
                changes.append(RuleDiffChange(path=child, operation="add", after_json=canonical_json(after[key])))
            elif key not in after:
                changes.append(RuleDiffChange(path=child, operation="remove", before_json=canonical_json(before[key])))
            else:
                changes.extend(_diff_values(before[key], after[key], child))
        return changes
    if isinstance(before, list) and isinstance(after, list):
        changes = []
        for index in range(max(len(before), len(after))):
            child = f"{path}[{index}]"
            if index >= len(before):
                changes.append(RuleDiffChange(path=child, operation="add", after_json=canonical_json(after[index])))
            elif index >= len(after):
                changes.append(RuleDiffChange(path=child, operation="remove", before_json=canonical_json(before[index])))
            else:
                changes.extend(_diff_values(before[index], after[index], child))
        return changes
    if before == after:
        return []
    return [RuleDiffChange(
        path=path,
        operation="replace",
        before_json=canonical_json(before),
        after_json=canonical_json(after),
    )]
