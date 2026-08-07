from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import EvaluationGate, RuleSet, RuleVersion


@runtime_checkable
class RuleVersionRepository(Protocol):
    def get_rule_set(self, rule_set_id: str) -> RuleSet: ...

    def create_rule_set(self, rule_set: RuleSet) -> RuleSet: ...

    def get_version(self, rule_version_id: str) -> RuleVersion: ...

    def list_versions(self, rule_set_id: str) -> tuple[RuleVersion, ...]: ...

    def add_version(self, version: RuleVersion) -> RuleVersion: ...

    def save_gate(self, version: RuleVersion, gate: EvaluationGate) -> RuleVersion: ...

    def publish(self, rule_set: RuleSet, version: RuleVersion) -> RuleVersion: ...

    def rollback(
        self, rule_set: RuleSet, current: RuleVersion, target: RuleVersion
    ) -> RuleVersion: ...


@runtime_checkable
class ReleaseGateVerifier(Protocol):
    def assert_dataset_release_ready(self, dataset_version_id: str) -> None: ...

    def assert_release_eligible(
        self,
        *,
        rule_version_id: str,
        dataset_version_id: str,
        evaluation_run_id: str,
        report_sha256: str,
    ) -> object: ...
