from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from tender_review.shared.contracts import ContractModel

from .models import SourceArtifact, SourceType, stable_sha256


class HistoricalOptimizationGroup(ContractModel):
    review_item: str = Field(min_length=1, max_length=128)
    platform_status: Literal["optimized", "failed", "already_covered", "no_target"]
    target_count: int = Field(ge=0)
    protection_count: int = Field(ge=0)
    accepted_iteration: int | None = Field(default=None, ge=0)
    iterations_run: int | None = Field(default=None, ge=0)
    final_summary_json: str | None = None
    reason: str | None = None
    source_type: Literal[SourceType.EXTERNAL_PLATFORM] = SourceType.EXTERNAL_PLATFORM
    status: Literal["provisional"] = "provisional"
    claims_allowed: Literal[False] = False
    sample_semantics: Literal["aggregate_external_platform_observation"] = (
        "aggregate_external_platform_observation"
    )

    @model_validator(mode="after")
    def status_is_consistent(self):
        if self.platform_status == "optimized" and self.accepted_iteration is None:
            raise ValueError("optimized historical group needs its recorded iteration")
        return self


class ApprovalOptimizerMigration(ContractModel):
    source_run_id: str = Field(min_length=1, max_length=256)
    source_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_artifacts: tuple[SourceArtifact, ...] = Field(min_length=1)
    source_claim_boundary: str = Field(min_length=1, max_length=8000)
    source_type: Literal[SourceType.EXTERNAL_PLATFORM] = SourceType.EXTERNAL_PLATFORM
    status: Literal["provisional"] = "provisional"
    claims_allowed: Literal[False] = False
    human_annotation_cases: Literal[0] = 0
    required_human_cases: int = Field(ge=1)
    groups: tuple[HistoricalOptimizationGroup, ...] = Field(min_length=1)
    migration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def migration_hash_matches(self):
        payload = self.model_dump(mode="json", exclude={"migration_sha256"})
        if self.migration_sha256 != stable_sha256(payload):
            raise ValueError("migration_sha256 does not match migrated content")
        return self


def migrate_approval_optimizer_baseline(
    baseline_path: Path,
) -> ApprovalOptimizerMigration:
    resolved = baseline_path.resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    baseline = _object(payload.get("approval_optimizer_baseline"), "approval baseline")
    input_counts = _object(baseline.get("input_counts"), "approval input counts")
    required_human_cases = int(input_counts.get("usable_cases") or 0)
    if required_human_cases < 1:
        raise ValueError("approval baseline must contain at least one usable case")
    groups_payload = _object(baseline.get("groups"), "approval groups")
    groups: list[HistoricalOptimizationGroup] = []
    for review_item, raw_group in sorted(
        groups_payload.items(), key=lambda item: _review_item_key(item[0])
    ):
        group = _object(raw_group, f"approval group {review_item}")
        final = group.get("final")
        groups.append(
            HistoricalOptimizationGroup(
                review_item=str(review_item),
                platform_status=str(group["status"]),
                target_count=int(group.get("target_count") or 0),
                protection_count=int(group.get("protection_count") or 0),
                accepted_iteration=group.get("accepted_iteration"),
                iterations_run=group.get("iterations_run"),
                final_summary_json=(
                    json.dumps(
                        final,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    if isinstance(final, dict)
                    else None
                ),
                reason=group.get("reason"),
            )
        )
    source_artifacts = tuple(
        SourceArtifact(
            path=str(item["path"]),
            sha256=str(item["sha256"]),
            kind=(
                "approval_opinion"
                if str(item["path"]).startswith("审批意见/")
                else "platform_run"
            ),
        )
        for raw in payload.get("source_artifacts") or []
        for item in [_object(raw, "source artifact")]
    )
    source_artifacts = (
        SourceArtifact(
            path=(
                f"{resolved.parent.name}/{resolved.name}"
                if resolved.parent.name
                else resolved.name
            ),
            sha256=_file_sha256(resolved),
            kind="manifest",
        ),
        *source_artifacts,
    )
    migrated = {
        "schema_version": 1,
        "source_run_id": baseline["run_id"],
        "source_input_sha256": payload["dataset_sha256"],
        "source_artifacts": [item.model_dump(mode="json") for item in source_artifacts],
        "source_claim_boundary": baseline["claim_boundary"],
        "source_type": SourceType.EXTERNAL_PLATFORM,
        "status": "provisional",
        "claims_allowed": False,
        "human_annotation_cases": 0,
        "required_human_cases": required_human_cases,
        "groups": [item.model_dump(mode="json") for item in groups],
    }
    return ApprovalOptimizerMigration(
        **migrated,
        migration_sha256=stable_sha256(migrated),
    )


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _review_item_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
