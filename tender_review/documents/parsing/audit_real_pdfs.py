from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from tender_review.shared.contracts import CallContext

from .adapters.pymupdf import PyMuPDFStructuredParser
from .application import DocumentParsingService, EvidenceValidator
from .chunking import StructuralChunker
from .fakes import UnavailableOcrProvider
from .models import ChunkSet, ParseArtifact, ParseRequest, sha256_bytes, stable_sha256


def audit_pdf_directory(pdf_root: Path) -> dict[str, object]:
    """Parse local PDFs without pretending that unavailable OCR succeeded."""
    parser = PyMuPDFStructuredParser()
    unavailable_ocr = UnavailableOcrProvider()
    service = DocumentParsingService(
        parser=parser,
        renderer=parser,
        ocr_provider=unavailable_ocr,
    )
    chunker = StructuralChunker()
    documents: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    totals: Counter[str] = Counter()

    paths = sorted(pdf_root.rglob("*.pdf"), key=lambda path: str(path).casefold())
    for index, path in enumerate(paths, start=1):
        relative_path = path.relative_to(pdf_root).as_posix()
        content = path.read_bytes()
        document_sha256 = sha256_bytes(content)
        try:
            artifact = service.parse(
                ParseRequest(
                    document_id=relative_path,
                    pdf_bytes=content,
                    document_sha256=document_sha256,
                    call=CallContext(
                        call_id=f"phase3-real-pdf-audit-{index}",
                        timeout_seconds=3600,
                    ),
                )
            )
            chunks = chunker.chunk(artifact)
        except Exception as exc:
            failures.append(
                {
                    "relative_path": relative_path,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            continue

        status_counts = Counter(page.status.value for page in artifact.document.pages)
        candidate_reasons = Counter(
            quality.candidate_reason
            for quality in artifact.quality_report.pages
            if quality.candidate_reason is not None
        )
        block_count = sum(len(page.blocks) for page in artifact.document.pages)
        heading_count = sum(
            block.title_candidate
            for page in artifact.document.pages
            for block in page.blocks
        )
        table_count = sum(len(page.tables) for page in artifact.document.pages)
        table_cell_count = sum(
            len(table.cells)
            for page in artifact.document.pages
            for table in page.tables
        )
        checked_references, invalid_references = _audit_chunk_references(
            artifact,
            chunks,
        )
        max_chunk_characters = max(
            (len(chunk.raw_text) for chunk in chunks.chunks), default=0
        )
        page_failures = [
            quality.model_dump(mode="json")
            for quality in artifact.quality_report.pages
            if quality.status.value in {"ocr_failed", "extraction_failed"}
        ]
        documents.append(
            {
                "relative_path": relative_path,
                "document_sha256": document_sha256,
                "page_count": artifact.document.page_count,
                "parse_artifact_sha256": artifact.artifact_sha256,
                "chunk_set_sha256": chunks.chunk_set_sha256,
                "chunk_count": len(chunks.chunks),
                "block_count": block_count,
                "heading_count": heading_count,
                "table_count": table_count,
                "table_cell_count": table_cell_count,
                "chunk_reference_count": checked_references,
                "invalid_chunk_references": invalid_references,
                "max_chunk_characters": max_chunk_characters,
                "status_counts": dict(sorted(status_counts.items())),
                "ocr_candidate_reasons": dict(sorted(candidate_reasons.items())),
                "ocr_candidate_pages": [
                    quality.page_number
                    for quality in artifact.quality_report.pages
                    if quality.ocr_candidate
                ],
                "failed_pages": page_failures,
                "table_warning_pages": [
                    quality.page_number
                    for quality in artifact.quality_report.pages
                    if quality.table_warnings
                ],
            }
        )
        totals["documents"] += 1
        totals["pages"] += artifact.document.page_count
        totals["chunks"] += len(chunks.chunks)
        totals["blocks"] += block_count
        totals["headings"] += heading_count
        totals["tables"] += table_count
        totals["table_cells"] += table_cell_count
        totals["chunk_references_checked"] += checked_references
        totals["invalid_chunk_references"] += len(invalid_references)
        totals["max_chunk_characters"] = max(
            totals["max_chunk_characters"], max_chunk_characters
        )
        totals["ocr_candidates"] += artifact.quality_report.statistics.ocr_candidate_count
        totals["ocr_called"] += artifact.quality_report.statistics.ocr_called_count
        totals["ocr_success"] += artifact.quality_report.statistics.ocr_success_count
        totals["ocr_failures"] += artifact.quality_report.statistics.ocr_failure_count
        totals["table_warnings"] += artifact.quality_report.statistics.table_warning_count
        for status, count in status_counts.items():
            totals[f"page_status:{status}"] += count
        for reason, count in candidate_reasons.items():
            totals[f"ocr_candidate_reason:{reason}"] += count
        print(
            f"[{index}/{len(paths)}] {relative_path}: "
            f"{artifact.document.page_count} pages, {len(chunks.chunks)} chunks",
            flush=True,
        )

    payload: dict[str, object] = {
        "schema_version": 1,
        "audit_name": "phase3_pdf_parsing_offline_audit",
        "pdf_root": str(pdf_root),
        "parser": parser.descriptor.model_dump(mode="json"),
        "chunker": chunker.descriptor.model_dump(mode="json"),
        "ocr": {
            "provider": unavailable_ocr.name,
            "version": unavailable_ocr.version,
            "mode": "explicitly_unavailable",
            "interpretation": (
                "OCR candidates are invoked, but unavailable OCR is recorded as "
                "a failure and never reported as OCR success."
            ),
        },
        "documents": documents,
        "document_failures": failures,
        "totals": dict(sorted(totals.items())),
    }
    payload["report_sha256"] = stable_sha256(payload)
    return payload


def _audit_chunk_references(
    artifact: ParseArtifact,
    chunks: ChunkSet,
) -> tuple[int, list[dict[str, str]]]:
    blocks = {
        block.block_id: block
        for page in artifact.document.pages
        for block in page.blocks
    }
    checked = 0
    invalid: list[dict[str, str]] = []
    validator = EvidenceValidator(artifact=artifact, chunks=chunks)
    for chunk in chunks.chunks:
        for reference in chunk.source_blocks:
            checked += 1
            block = blocks.get(reference.block_id)
            if block is None:
                invalid.append(
                    {"chunk_id": chunk.chunk_id, "reason": "source block is missing"}
                )
                continue
            if block.page_number != reference.page_number:
                invalid.append(
                    {"chunk_id": chunk.chunk_id, "reason": "page number differs"}
                )
                continue
            if block.bbox != reference.bbox:
                invalid.append(
                    {"chunk_id": chunk.chunk_id, "reason": "bounding box differs"}
                )
                continue
            if (
                reference.text_start < 0
                or reference.text_end > len(block.text)
                or reference.text_start >= reference.text_end
            ):
                invalid.append(
                    {"chunk_id": chunk.chunk_id, "reason": "source range is invalid"}
                )
                continue
            excerpt = block.text[reference.text_start : reference.text_end][:32]
            try:
                validator.create(
                    chunk_id=chunk.chunk_id,
                    source_block_id=reference.block_id,
                    excerpt=excerpt,
                )
            except ValueError as exc:
                invalid.append({"chunk_id": chunk.chunk_id, "reason": str(exc)})
    return checked, invalid


def render_report(report: dict[str, object]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Phase 3 structured-PDF parsing audit over local PDFs."
    )
    parser.add_argument("--pdf-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the frozen report exactly matches a fresh offline audit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    options = build_parser().parse_args(argv)
    if options.check and not options.output.is_file():
        print(f"Missing Phase 3 PDF audit baseline: {options.output}")
        return 1
    report = audit_pdf_directory(options.pdf_root)
    rendered = render_report(report)
    if options.check:
        if options.output.read_text(encoding="utf-8") != rendered:
            print(f"Phase 3 PDF audit baseline differs: {options.output}")
            return 1
        print(f"Phase 3 PDF audit baseline matches: {options.output}")
        return 0
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(rendered, encoding="utf-8")
    print(json.dumps(report["totals"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
