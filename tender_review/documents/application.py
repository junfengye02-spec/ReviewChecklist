from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Protocol

from tender_review.documents.lifecycle import (
    ArtifactSubmission,
    ArtifactType,
    DocumentLifecycleRepository,
    DocumentLifecycleService,
    SnapshotSaveResult,
    SourceDocument,
)
from tender_review.documents.parsing.application import DocumentParsingService
from tender_review.documents.parsing.models import (
    ChunkSet,
    ParseArtifact,
    ParseRequest,
)
from tender_review.documents.parsing.ports import ChunkingStrategy
from tender_review.jobs.public import JobMessage, JobResult
from tender_review.shared.contracts import CallContext
from tender_review.shared.faults import DisabledFaultInjector, FaultInjector
from tender_review.shared.observability import (
    CorrelationContext,
    log_event,
    record_metric,
)


FAULT_AFTER_PARSE_CHECKPOINT = "parsing.after_checkpoint"


@dataclass(frozen=True, slots=True)
class DocumentParseOutcome:
    snapshot_id: str
    parsed_artifact_id: str
    quality_report_artifact_id: str
    parse_artifact_sha256: str
    chunk_set_sha256: str
    page_count: int
    chunk_count: int
    ocr_candidate_count: int
    ocr_failure_count: int


class ParsingExecutionContext(Protocol):
    def latest_checkpoint(self) -> object | None: ...

    def save_checkpoint(
        self,
        *,
        node_name: str,
        stage: str,
        state_json: dict[str, object],
        output_artifact_id: str | None = None,
    ) -> object: ...


class DocumentService:
    """Public document use cases over immutable storage and structured parsing."""

    def __init__(
        self,
        *,
        lifecycle: DocumentLifecycleService,
        repository: DocumentLifecycleRepository,
        parser: DocumentParsingService,
        chunker: ChunkingStrategy,
    ) -> None:
        self._lifecycle = lifecycle
        self._repository = repository
        self._parser = parser
        self._chunker = chunker

    def upload(
        self,
        *,
        source_system: str,
        source_document_id: str,
        file_name: str,
        content: bytes,
        media_type: str = "application/pdf",
    ) -> SnapshotSaveResult:
        return self._lifecycle.upload_document(
            SourceDocument(
                source_system=source_system,
                source_document_id=source_document_id,
                file_name=file_name,
                content=content,
                media_type=media_type,
            )
        )

    def get_snapshot(self, snapshot_id: str):
        return self._repository.get_snapshot(snapshot_id)

    def parse(self, snapshot_id: str, *, call: CallContext) -> DocumentParseOutcome:
        snapshot = self._repository.set_parse_status(snapshot_id, "PARSING")
        try:
            content = self._lifecycle.read_snapshot(snapshot_id)
            parsed = self._parser.parse(
                ParseRequest(
                    document_id=snapshot_id,
                    pdf_bytes=content,
                    document_sha256=snapshot.object.sha256,
                    call=call,
                )
            )
            chunks = self._chunker.chunk(parsed)
            parsed_record = self._write_parse_artifact(snapshot_id, parsed, chunks)
            report_record = self._write_quality_report(snapshot_id, parsed)
            descriptor = parsed.document.parser
            self._repository.set_parse_status(
                snapshot_id,
                "PARSED",
                parser_name=descriptor.name,
                parser_version=descriptor.version,
            )
            stats = parsed.quality_report.statistics
            return DocumentParseOutcome(
                snapshot_id=snapshot_id,
                parsed_artifact_id=parsed_record.artifact.id,
                quality_report_artifact_id=report_record.artifact.id,
                parse_artifact_sha256=parsed.artifact_sha256,
                chunk_set_sha256=chunks.chunk_set_sha256,
                page_count=parsed.document.page_count,
                chunk_count=len(chunks.chunks),
                ocr_candidate_count=stats.ocr_candidate_count,
                ocr_failure_count=stats.ocr_failure_count,
            )
        except Exception:
            self._repository.set_parse_status(snapshot_id, "FAILED")
            raise

    def _write_parse_artifact(
        self, snapshot_id: str, parsed: ParseArtifact, chunks: ChunkSet
    ):
        payload = {
            "schema_version": "1",
            "parse_artifact": parsed.model_dump(mode="json"),
            "chunk_set": chunks.model_dump(mode="json"),
        }
        return self._lifecycle.write_artifact(
            ArtifactSubmission(
                document_snapshot_id=snapshot_id,
                artifact_type=ArtifactType.PARSED_JSON,
                content=_json_bytes(payload),
                media_type="application/json",
                schema_version="1",
                metadata={
                    "parse_artifact_sha256": parsed.artifact_sha256,
                    "chunk_set_sha256": chunks.chunk_set_sha256,
                },
            )
        )

    def _write_quality_report(self, snapshot_id: str, parsed: ParseArtifact):
        payload = {
            "schema_version": "1",
            "quality_report": parsed.quality_report.model_dump(mode="json"),
        }
        return self._lifecycle.write_artifact(
            ArtifactSubmission(
                document_snapshot_id=snapshot_id,
                artifact_type=ArtifactType.REPORT,
                content=_json_bytes(payload),
                media_type="application/json",
                schema_version="1",
                metadata={
                    "report_sha256": parsed.quality_report.report_sha256,
                },
            )
        )


class DocumentParsingJobHandler:
    """Resumable parsing-only Worker handler; it never claims review completion."""

    job_type = "document_parse"

    def __init__(
        self,
        documents: DocumentService,
        *,
        fault_injector: FaultInjector | None = None,
    ) -> None:
        self._documents = documents
        self._faults = fault_injector or DisabledFaultInjector()
        self._logger = logging.getLogger("tender_review.worker.parsing")

    def __call__(
        self, job: JobMessage, context: ParsingExecutionContext
    ) -> JobResult:
        correlation = CorrelationContext(
            job_id=job.job_id,
            thread_id=job.job_id,
            call_id=f"document-parse:{job.job_id}",
        )
        completed_artifact = _completed_parse_artifact(context.latest_checkpoint())
        if completed_artifact is not None:
            log_event(
                self._logger,
                logging.INFO,
                event="parsing.checkpoint_recovered",
                message="Parsing resumed from a completed checkpoint",
                context=correlation,
                recovery_count=max(0, job.attempt - 1),
                output_artifact_id=completed_artifact,
            )
            return JobResult(
                output_reference=completed_artifact,
                summary="Parsing checkpoint already completed",
            )
        started = time.perf_counter()
        outcome = self._documents.parse(
            job.input_reference,
            call=CallContext(
                call_id=f"document-parse:{job.job_id}",
                timeout_seconds=3600,
            ),
        )
        context.save_checkpoint(
            node_name="parse",
            stage="PARSING",
            state_json={
                "schema_version": 1,
                "values": [
                    {
                        "schema_version": 1,
                        "key": "parsed_artifact_id",
                        "value": outcome.parsed_artifact_id,
                    },
                    {
                        "schema_version": 1,
                        "key": "quality_report_artifact_id",
                        "value": outcome.quality_report_artifact_id,
                    },
                    {
                        "schema_version": 1,
                        "key": "call_id",
                        "value": f"document-parse:{job.job_id}",
                    },
                    {
                        "schema_version": 1,
                        "key": "metrics_source",
                        "value": "process_monotonic",
                    },
                ],
            },
            output_artifact_id=outcome.parsed_artifact_id,
        )
        duration_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        log_event(
            self._logger,
            logging.INFO,
            event="parsing.checkpoint_saved",
            message="Parsing checkpoint saved",
            context=correlation,
            page_count=outcome.page_count,
            chunk_count=outcome.chunk_count,
        )
        record_metric(
            self._logger,
            name="parsing_node_duration",
            value=duration_ms,
            unit="ms",
            source="process_monotonic",
            context=correlation,
            page_count=outcome.page_count,
            chunk_count=outcome.chunk_count,
        )
        self._faults.trip(FAULT_AFTER_PARSE_CHECKPOINT)
        return JobResult(
            output_reference=outcome.parsed_artifact_id,
            summary=(
                f"Parsed {outcome.page_count} pages into {outcome.chunk_count} chunks"
            ),
        )


def _completed_parse_artifact(checkpoint: object | None) -> str | None:
    if checkpoint is None or getattr(checkpoint, "node_name", None) != "parse":
        return None
    output = getattr(checkpoint, "output_artifact_id", None)
    if output:
        return str(output)
    state = getattr(checkpoint, "state", None)
    values = getattr(state, "values", ())
    for item in values:
        if getattr(item, "key", None) == "parsed_artifact_id":
            return str(getattr(item, "value"))
    return None


def _json_bytes(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
