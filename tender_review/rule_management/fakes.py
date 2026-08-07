from __future__ import annotations

from threading import RLock

from tender_review.shared.errors import ConflictError, NotFoundError, PermanentError

from .models import EvaluationGate, RuleSet, RuleVersion
from .ports import ReleaseGateVerifier


class InMemoryRuleVersionRepository:
    def __init__(self, release_gates: ReleaseGateVerifier | None = None) -> None:
        self._sets: dict[str, RuleSet] = {}
        self._versions: dict[str, RuleVersion] = {}
        self._lock = RLock()
        self._release_gates = release_gates

    def set_release_gate_verifier(self, verifier: ReleaseGateVerifier) -> None:
        self._release_gates = verifier

    def get_rule_set(self, rule_set_id: str) -> RuleSet:
        with self._lock:
            try:
                return self._sets[rule_set_id]
            except KeyError as exc:
                raise NotFoundError("rule set does not exist", code="rule_set_not_found") from exc

    def create_rule_set(self, rule_set: RuleSet) -> RuleSet:
        with self._lock:
            existing = self._sets.get(rule_set.rule_set_id)
            if existing is not None:
                if existing.rule_key != rule_set.rule_key or existing.name != rule_set.name:
                    raise ConflictError("rule set identity differs", code="rule_set_conflict")
                return existing
            if any(item.rule_key == rule_set.rule_key for item in self._sets.values()):
                raise ConflictError("rule key already exists", code="rule_key_conflict")
            self._sets[rule_set.rule_set_id] = rule_set
            return rule_set

    def get_version(self, rule_version_id: str) -> RuleVersion:
        with self._lock:
            try:
                return self._versions[rule_version_id]
            except KeyError as exc:
                raise NotFoundError("rule version does not exist", code="rule_version_not_found") from exc

    def list_versions(self, rule_set_id: str) -> tuple[RuleVersion, ...]:
        with self._lock:
            self.get_rule_set(rule_set_id)
            return tuple(sorted(
                (item for item in self._versions.values() if item.rule_set_id == rule_set_id),
                key=lambda item: item.version_number,
            ))

    def add_version(self, version: RuleVersion) -> RuleVersion:
        with self._lock:
            if version.rule_version_id in self._versions:
                raise ConflictError("rule version already exists", code="rule_version_conflict")
            if any(
                item.rule_set_id == version.rule_set_id
                and (item.version_number == version.version_number or item.content_sha256 == version.content_sha256)
                for item in self._versions.values()
            ):
                raise ConflictError("duplicate rule version", code="rule_version_duplicate")
            self._versions[version.rule_version_id] = version
            return version

    def save_gate(self, version: RuleVersion, gate: EvaluationGate) -> RuleVersion:
        with self._lock:
            current = self.get_version(version.rule_version_id)
            if current.content_sha256 != version.content_sha256:
                raise ConflictError("rule content is immutable", code="rule_content_changed")
            updated = version.model_copy(update={"evaluation_gate": gate})
            self._versions[version.rule_version_id] = updated
            return updated

    def publish(self, rule_set: RuleSet, version: RuleVersion) -> RuleVersion:
        with self._lock:
            gate = self.get_version(version.rule_version_id).evaluation_gate
            if gate is None or gate.evaluation_run_id is None or gate.report_sha256 is None:
                raise PermanentError("persisted release gate is incomplete", code="release_gate_persistence_invalid")
            if self._release_gates is None:
                raise PermanentError("release-gate verifier is unavailable", code="release_gate_verifier_unavailable")
            self._release_gates.assert_release_eligible(
                rule_version_id=version.rule_version_id,
                dataset_version_id=gate.dataset_version_id,
                evaluation_run_id=gate.evaluation_run_id,
                report_sha256=gate.report_sha256,
            )
            current_set = self.get_rule_set(rule_set.rule_set_id)
            if current_set.current_version_id and current_set.current_version_id != version.rule_version_id:
                previous = self.get_version(current_set.current_version_id)
                self._versions[previous.rule_version_id] = previous.model_copy(
                    update={"status": "ROLLED_BACK"}
                )
            self._sets[rule_set.rule_set_id] = rule_set.model_copy(
                update={"current_version_id": version.rule_version_id}
            )
            self._versions[version.rule_version_id] = version
            return version

    def rollback(
        self, rule_set: RuleSet, current: RuleVersion, target: RuleVersion
    ) -> RuleVersion:
        with self._lock:
            self.get_rule_set(rule_set.rule_set_id)
            self._versions[current.rule_version_id] = current
            self._versions[target.rule_version_id] = target
            self._sets[rule_set.rule_set_id] = rule_set.model_copy(
                update={"current_version_id": target.rule_version_id}
            )
            return target
