from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from tender_review.findings.public import (
    EvidenceReference as EvidenceDto,
    Finding as FindingDto,
    FindingProvenance,
    HumanDecision as DecisionDto,
)
from tender_review.shared.errors import ConflictError, NotFoundError, RetryableError

from .models import EvidenceReference, Finding, HumanDecision


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


class SqlAlchemyFindingRepository:
    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

    def add_finding(self, finding: FindingDto) -> FindingDto:
        try:
            with self._sessions.begin() as session:
                existing = session.get(Finding, finding.finding_id)
                if existing is not None:
                    current = self._to_finding(session, existing)
                    if current == finding:
                        return current
                    raise ConflictError("finding already exists", code="finding_conflict")
                row = Finding(
                    id=finding.finding_id,
                    review_job_id=finding.review_job_id,
                    rule_version_id=finding.rule_version_id,
                    review_item=finding.review_item_id,
                    status=finding.status.value,
                    workflow_state=finding.workflow_state.value,
                    compliant=(finding.conclusion == "compliant") if finding.conclusion else None,
                    title=finding.message[:512],
                    description=finding.message,
                    explanation_json={},
                    review_input_sha256=finding.provenance.review_input_sha256,
                    finding_content_sha256=finding.finding_content_sha256,
                    provenance_json=finding.provenance.model_dump(mode="json"),
                    documents_json=[item.model_dump(mode="json") for item in finding.documents],
                    created_at=finding.created_at,
                    updated_at=finding.created_at,
                    schema_version=str(finding.schema_version),
                )
                session.add(row)
                for index, evidence in enumerate(finding.evidence, start=1):
                    session.add(EvidenceReference(
                        id=str(uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"tender-review:{finding.finding_id}:evidence:{index}",
                        )),
                        finding_id=finding.finding_id,
                        document_snapshot_id=evidence.document_id,
                        chunk_id=evidence.chunk_id,
                        page_start=evidence.page_number,
                        page_end=evidence.page_end or evidence.page_number,
                        section_path=json.dumps(evidence.section_path, ensure_ascii=False),
                        excerpt=evidence.excerpt,
                        text_sha256=evidence.text_sha256,
                        schema_version=str(evidence.schema_version),
                    ))
                session.flush()
                return finding
        except IntegrityError as exc:
            raise ConflictError("finding persistence conflict", code="finding_conflict") from exc
        except SQLAlchemyError as exc:
            raise RetryableError("finding storage is unavailable", code="finding_storage_unavailable") from exc

    def get_finding(self, finding_id: str) -> FindingDto:
        try:
            with self._sessions() as session:
                row = session.get(Finding, finding_id)
                if row is None:
                    raise NotFoundError("finding does not exist", code="finding_not_found")
                return self._to_finding(session, row)
        except SQLAlchemyError as exc:
            raise RetryableError("finding storage is unavailable", code="finding_storage_unavailable") from exc

    def list_findings(self, review_job_id: str) -> tuple[FindingDto, ...]:
        try:
            with self._sessions() as session:
                rows = session.scalars(
                    select(Finding)
                    .where(Finding.review_job_id == review_job_id)
                    .order_by(Finding.created_at, Finding.id)
                )
                return tuple(self._to_finding(session, row) for row in rows)
        except SQLAlchemyError as exc:
            raise RetryableError(
                "finding storage is unavailable", code="finding_storage_unavailable"
            ) from exc

    def list_decisions(self, finding_id: str) -> tuple[DecisionDto, ...]:
        try:
            with self._sessions() as session:
                if session.get(Finding, finding_id) is None:
                    raise NotFoundError("finding does not exist", code="finding_not_found")
                rows = session.scalars(
                    select(HumanDecision)
                    .where(HumanDecision.finding_id == finding_id)
                    .order_by(HumanDecision.created_at, HumanDecision.id)
                )
                return tuple(self._to_decision(row) for row in rows)
        except SQLAlchemyError as exc:
            raise RetryableError("decision storage is unavailable", code="decision_storage_unavailable") from exc

    def append_decision(self, *, finding: FindingDto, decision: DecisionDto) -> FindingDto:
        try:
            with self._sessions.begin() as session:
                row = session.scalar(
                    select(Finding).where(Finding.id == finding.finding_id).with_for_update()
                )
                if row is None:
                    raise NotFoundError("finding does not exist", code="finding_not_found")
                if row.finding_content_sha256 != finding.finding_content_sha256:
                    raise ConflictError("finding content is immutable", code="finding_content_changed")
                latest = session.scalar(
                    select(HumanDecision)
                    .where(HumanDecision.finding_id == finding.finding_id)
                    .order_by(HumanDecision.created_at.desc(), HumanDecision.id.desc())
                    .limit(1)
                    .with_for_update()
                )
                if latest is None and decision.supersedes_decision_id is not None:
                    raise ConflictError("there is no decision to supersede", code="decision_supersedes_missing")
                if latest is not None and decision.supersedes_decision_id != latest.id:
                    raise ConflictError("latest decision must be superseded", code="decision_supersedes_latest_required")
                session.add(HumanDecision(
                    id=decision.decision_id,
                    finding_id=decision.finding_id,
                    supersedes_decision_id=decision.supersedes_decision_id,
                    reviewer_id=decision.reviewer_id,
                    reviewer_kind=decision.reviewer_kind,
                    decision=decision.decision.value,
                    comment=decision.reason,
                    modified_finding_json=(decision.revision.model_dump(mode="json") if decision.revision else None),
                    review_input_sha256=decision.review_input_sha256,
                    finding_content_sha256=decision.finding_content_sha256,
                    evidence_sha256=decision.evidence_sha256,
                    decision_sha256=decision.decision_sha256,
                    created_at=decision.decided_at,
                    schema_version=str(decision.schema_version),
                ))
                row.status = finding.status.value
                session.flush()
                return finding
        except IntegrityError as exc:
            raise ConflictError("decision already exists", code="decision_duplicate") from exc
        except SQLAlchemyError as exc:
            raise RetryableError("decision storage is unavailable", code="decision_storage_unavailable") from exc

    @staticmethod
    def _to_finding(session: Session, row: Finding) -> FindingDto:
        evidence_rows = session.scalars(
            select(EvidenceReference)
            .where(EvidenceReference.finding_id == row.id)
            .order_by(EvidenceReference.created_at, EvidenceReference.id)
        )
        evidence = tuple(EvidenceDto(
            document_id=item.document_snapshot_id,
            chunk_id=item.chunk_id,
            page_number=item.page_start,
            page_end=item.page_end,
            section_path=tuple(json.loads(item.section_path or "[]")),
            excerpt=item.excerpt,
            text_sha256=item.text_sha256,
        ) for item in evidence_rows)
        conclusion = None if row.compliant is None else ("compliant" if row.compliant else "noncompliant")
        return FindingDto(
            finding_id=row.id,
            review_job_id=row.review_job_id,
            rule_version_id=row.rule_version_id,
            review_item_id=row.review_item,
            workflow_state=row.workflow_state,
            status=row.status,
            conclusion=conclusion,
            message=row.description,
            evidence=evidence,
            documents=tuple(row.documents_json or ()),
            provenance=FindingProvenance.model_validate(row.provenance_json),
            created_at=_aware(row.created_at),
            finding_content_sha256=row.finding_content_sha256,
        )

    @staticmethod
    def _to_decision(row: HumanDecision) -> DecisionDto:
        return DecisionDto(
            decision_id=row.id,
            finding_id=row.finding_id,
            reviewer_kind=row.reviewer_kind,
            reviewer_id=row.reviewer_id,
            decision=row.decision,
            reason=row.comment,
            revision=row.modified_finding_json,
            supersedes_decision_id=row.supersedes_decision_id,
            decided_at=_aware(row.created_at),
            review_input_sha256=row.review_input_sha256,
            finding_content_sha256=row.finding_content_sha256,
            evidence_sha256=row.evidence_sha256,
            decision_sha256=row.decision_sha256,
        )
