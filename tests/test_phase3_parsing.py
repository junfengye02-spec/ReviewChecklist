from __future__ import annotations

import unittest

import fitz

from tender_review.documents.parsing.adapters.pymupdf import PyMuPDFStructuredParser
from tender_review.documents.parsing.application import (
    DocumentParsingService,
    create_evidence_reference,
    validate_evidence_reference,
)
from tender_review.documents.parsing.chunking import StructuralChunker
from tender_review.documents.parsing.fakes import FakeOcrProvider
from tender_review.documents.parsing.models import (
    BlockSource,
    PageParseStatus,
    ParseRequest,
    sha256_bytes,
    sha256_text,
)
from tender_review.shared.contracts import CallContext


def build_request(pdf_bytes: bytes, *, document_id: str = "synthetic.pdf") -> ParseRequest:
    return ParseRequest(
        document_id=document_id,
        pdf_bytes=pdf_bytes,
        document_sha256=sha256_bytes(pdf_bytes),
        call=CallContext(call_id=f"test:{document_id}"),
    )


def new_pdf() -> fitz.Document:
    return fitz.open()


class StructuredPdfParsingTests(unittest.TestCase):
    def test_ocr_is_called_only_for_empty_or_low_text_pages_and_failures_are_recorded(self):
        document = new_pdf()
        normal_page = document.new_page()
        normal_page.insert_text(
            (72, 72),
            "This is an ordinary PDF text layer with enough content to avoid OCR.",
        )
        document.new_page()
        low_text_page = document.new_page()
        low_text_page.insert_text((72, 72), "x")
        pdf_bytes = document.tobytes()
        document.close()

        parser = PyMuPDFStructuredParser(min_text_characters=20)
        ocr = FakeOcrProvider(
            {
                2: "Recognized OCR text for the empty page.",
                3: RuntimeError("OCR endpoint is unavailable"),
            }
        )
        artifact = DocumentParsingService(
            parser=parser,
            renderer=parser,
            ocr_provider=ocr,
        ).parse(build_request(pdf_bytes))

        self.assertEqual([call.page_number for call in ocr.calls], [2, 3])
        self.assertEqual(
            [page.status for page in artifact.document.pages],
            [
                PageParseStatus.OK,
                PageParseStatus.OCR_APPLIED,
                PageParseStatus.OCR_FAILED,
            ],
        )
        self.assertTrue(
            any(
                block.source is BlockSource.OCR
                for block in artifact.document.pages[1].blocks
            )
        )
        quality = artifact.quality_report
        self.assertEqual(quality.statistics.page_count, 3)
        self.assertEqual(quality.statistics.ocr_candidate_count, 2)
        self.assertEqual(quality.statistics.ocr_called_count, 2)
        self.assertEqual(quality.statistics.ocr_success_count, 1)
        self.assertEqual(quality.statistics.ocr_failure_count, 1)
        self.assertEqual(quality.pages[1].ocr_attempt.confidence_lower, 0.8)
        self.assertEqual(quality.pages[1].ocr_attempt.confidence_upper, 0.9)
        self.assertEqual(quality.pages[2].ocr_attempt.failure.error_type, "RuntimeError")
        self.assertEqual(artifact.schema_version, 1)
        self.assertEqual(artifact.document.parser.schema_version, 1)
        self.assertEqual(quality.schema_version, 1)
        tampered = artifact.model_dump(mode="json")
        tampered["artifact_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "artifact_sha256"):
            type(artifact).model_validate(tampered)

    def test_render_failure_is_not_counted_as_an_ocr_provider_call(self):
        document = new_pdf()
        document.new_page()
        pdf_bytes = document.tobytes()
        document.close()
        parser = PyMuPDFStructuredParser()
        ocr = FakeOcrProvider({1: "must not be used"})

        class FailingRenderer:
            descriptor = parser.descriptor

            def render_page(self, request):
                raise RuntimeError("render failed")

        artifact = DocumentParsingService(
            parser=parser,
            renderer=FailingRenderer(),
            ocr_provider=ocr,
        ).parse(build_request(pdf_bytes))

        attempt = artifact.quality_report.pages[0].ocr_attempt
        self.assertEqual(ocr.calls, [])
        self.assertFalse(attempt.provider_called)
        self.assertEqual(artifact.quality_report.statistics.ocr_called_count, 0)
        self.assertEqual(artifact.quality_report.statistics.ocr_failure_count, 1)

    def test_real_table_detection_produces_cells_source_blocks_and_bounded_table_chunk(self):
        document = new_pdf()
        page = document.new_page()
        page.draw_rect(fitz.Rect(72, 72, 272, 132))
        page.draw_line((172, 72), (172, 132))
        page.draw_line((72, 102), (272, 102))
        page.insert_text((80, 90), "Header A")
        page.insert_text((180, 90), "Header B")
        page.insert_text((80, 120), "Value A")
        page.insert_text((180, 120), "Value B")
        pdf_bytes = document.tobytes()
        document.close()

        artifact = PyMuPDFStructuredParser().parse(build_request(pdf_bytes))
        page = artifact.document.pages[0]
        self.assertTrue(page.table_detection_attempted)
        self.assertEqual(page.table_warnings, ())
        self.assertEqual(len(page.tables), 1)
        self.assertEqual(len(page.tables[0].cells), 4)
        self.assertTrue(page.tables[0].source_block_ids)

        chunks = StructuralChunker(max_characters=100).chunk(artifact)
        table_chunks = [chunk for chunk in chunks.chunks if chunk.table_ids]
        self.assertEqual(len(table_chunks), 1)
        self.assertEqual(table_chunks[0].table_ids, (page.tables[0].table_id,))
        self.assertLessEqual(len(table_chunks[0].raw_text), 100)
        self.assertTrue(table_chunks[0].source_blocks)

    def test_long_source_blocks_are_split_without_exceeding_the_configured_limit(self):
        document = new_pdf()
        page = document.new_page()
        page.insert_textbox(
            fitz.Rect(72, 72, 520, 760),
            "evidence " * 120,
        )
        pdf_bytes = document.tobytes()
        document.close()

        artifact = PyMuPDFStructuredParser().parse(build_request(pdf_bytes))
        chunks = StructuralChunker(max_characters=100).chunk(artifact)

        self.assertGreater(len(chunks.chunks), 1)
        self.assertTrue(
            all(len(chunk.raw_text) <= 100 for chunk in chunks.chunks)
        )
        self.assertTrue(all(chunk.source_blocks for chunk in chunks.chunks))

    def test_evidence_reference_must_match_its_chunk_block_page_text_and_hash(self):
        document = new_pdf()
        page = document.new_page()
        page.insert_text((72, 72), "1. Scope")
        page.insert_text(
            (72, 100),
            "The bidder must provide the signed certificate before submission.",
        )
        pdf_bytes = document.tobytes()
        document.close()

        artifact = PyMuPDFStructuredParser().parse(build_request(pdf_bytes))
        chunks = StructuralChunker(max_characters=200).chunk(artifact)
        source_chunk = next(chunk for chunk in chunks.chunks if chunk.source_blocks)
        source_block_id = source_chunk.source_blocks[0].block_id
        source_block = next(
            block
            for parsed_page in artifact.document.pages
            for block in parsed_page.blocks
            if block.block_id == source_block_id
        )
        excerpt = source_block.text[:12]
        reference = create_evidence_reference(
            artifact=artifact,
            chunks=chunks,
            chunk_id=source_chunk.chunk_id,
            source_block_id=source_block_id,
            excerpt=excerpt,
        )
        validate_evidence_reference(reference, artifact=artifact, chunks=chunks)

        with self.assertRaisesRegex(ValueError, "page number"):
            validate_evidence_reference(
                reference.model_copy(update={"page_number": 99}),
                artifact=artifact,
                chunks=chunks,
            )
        with self.assertRaisesRegex(ValueError, "source block hash"):
            validate_evidence_reference(
                reference.model_copy(update={"source_block_text_sha256": "0" * 64}),
                artifact=artifact,
                chunks=chunks,
            )
        with self.assertRaisesRegex(ValueError, "outside the source block range"):
            validate_evidence_reference(
                reference.model_copy(
                    update={
                        "excerpt": "not in source",
                        "excerpt_sha256": sha256_text("not in source"),
                    }
                ),
                artifact=artifact,
                chunks=chunks,
            )


if __name__ == "__main__":
    unittest.main()
