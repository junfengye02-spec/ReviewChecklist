from __future__ import annotations

import unittest
from types import SimpleNamespace

import fitz
from fastapi.testclient import TestClient

from tender_review.api import create_app
from tender_review.bootstrap import build_container
from tender_review.documents.application import DocumentParsingJobHandler
from tender_review.jobs.models import JobMessage
from tender_review.shared.config import AppSettings
from tender_review.shared.contracts import CallContext
from tender_review.shared.faults import InjectedFault, OneShotFaultInjector
from tender_review.documents.application import FAULT_AFTER_PARSE_CHECKPOINT


def one_page_pdf(text: str = "Tender Review Evidence Heading") -> bytes:
    document = fitz.open()
    try:
        page = document.new_page()
        page.insert_text((72, 72), text)
        return document.tobytes()
    finally:
        document.close()


class DocumentUploadApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.container = build_container(
            AppSettings(
                environment="test",
                adapter_mode="fake",
                log_json=False,
                document_max_upload_bytes=4096,
            )
        )
        self.client = TestClient(create_app(self.container))

    def tearDown(self) -> None:
        self.container.close()

    def upload(self, content: bytes, *, source_id: str = "source-1"):
        return self.client.post(
            "/api/v1/documents",
            files={"file": ("tender.pdf", content, "application/pdf")},
            data={"source_system": "test", "source_document_id": source_id},
        )

    def test_upload_is_content_addressed_and_source_idempotent(self) -> None:
        content = one_page_pdf()
        created = self.upload(content)
        replay = self.upload(content)

        self.assertEqual(created.status_code, 201)
        self.assertEqual(replay.status_code, 200)
        self.assertTrue(created.json()["created"])
        self.assertFalse(replay.json()["created"])
        self.assertEqual(created.json()["id"], replay.json()["id"])
        self.assertEqual(created.json()["sha256"], replay.json()["sha256"])
        self.assertEqual(created.json()["parse_status"], "UPLOADED")

        conflict = self.upload(one_page_pdf("Changed immutable content"))
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(
            conflict.json()["error"]["code"], "source_document_hash_conflict"
        )

    def test_upload_rejects_non_pdf_empty_and_oversized_content(self) -> None:
        non_pdf = self.upload(b"not a pdf", source_id="bad")
        empty = self.upload(b"", source_id="empty")
        oversized = self.upload(b"%PDF-" + b"x" * 5000, source_id="large")
        wrong_media = self.client.post(
            "/api/v1/documents",
            files={"file": ("tender.txt", b"%PDF-1.7", "text/plain")},
            data={"source_system": "test", "source_document_id": "wrong-media"},
        )

        self.assertEqual(non_pdf.json()["error"]["code"], "document_pdf_invalid")
        self.assertEqual(empty.json()["error"]["code"], "document_empty")
        self.assertEqual(oversized.json()["error"]["code"], "document_too_large")
        self.assertEqual(
            wrong_media.json()["error"]["code"],
            "document_media_type_unsupported",
        )


class DocumentParsingIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.container = build_container(
            AppSettings(environment="test", adapter_mode="fake", log_json=False)
        )
        self.snapshot = self.container.documents.upload(
            source_system="test",
            source_document_id="parse-source",
            file_name="parse.pdf",
            content=one_page_pdf("1 Scope\nEvidence text for a deterministic chunk."),
        ).snapshot

    def tearDown(self) -> None:
        self.container.close()

    def test_parse_persists_versioned_artifacts_and_quality_status(self) -> None:
        outcome = self.container.documents.parse(
            self.snapshot.id,
            call=CallContext(call_id="phase3-integration", timeout_seconds=30),
        )

        stored = self.container.documents.get_snapshot(self.snapshot.id)
        artifacts = self.container.document_repository.list_artifacts(self.snapshot.id)
        self.assertEqual(stored.parse_status, "PARSED")
        self.assertEqual(stored.parser_name, "pymupdf-structured-parser")
        self.assertEqual(outcome.page_count, 1)
        self.assertGreaterEqual(outcome.chunk_count, 1)
        self.assertEqual(len(artifacts), 2)
        self.assertEqual(
            {artifact.artifact_type.value for artifact in artifacts},
            {"parsed_json", "report"},
        )
        for artifact in artifacts:
            payload = self.container.document_lifecycle.read_artifact(artifact.id)
            self.assertIn(b'"schema_version":"1"', payload)

    def test_worker_handler_resumes_completed_parse_checkpoint(self) -> None:
        handler = self.container.worker_handlers[DocumentParsingJobHandler.job_type]
        saved: list[dict[str, object]] = []

        class Context:
            checkpoint = None

            def latest_checkpoint(self):
                return self.checkpoint

            def save_checkpoint(self, **values):
                saved.append(values)
                self.checkpoint = SimpleNamespace(
                    node_name=values["node_name"],
                    output_artifact_id=values["output_artifact_id"],
                )
                return self.checkpoint

        context = Context()
        job = JobMessage(
            job_id="parse-job",
            job_type="document_parse",
            input_reference=self.snapshot.id,
        )
        first = handler(job, context)
        artifact_count = len(
            self.container.document_repository.list_artifacts(self.snapshot.id)
        )
        second = handler(job, context)

        self.assertEqual(len(saved), 1)
        self.assertEqual(first.output_reference, second.output_reference)
        self.assertEqual(
            len(self.container.document_repository.list_artifacts(self.snapshot.id)),
            artifact_count,
        )
        self.assertIn("already completed", second.summary)

    def test_single_process_fake_parse_fault_resumes_after_durable_checkpoint(
        self,
    ) -> None:
        handler = DocumentParsingJobHandler(
            self.container.documents,
            fault_injector=OneShotFaultInjector(FAULT_AFTER_PARSE_CHECKPOINT),
        )
        saved: list[dict[str, object]] = []

        class Context:
            checkpoint = None

            def latest_checkpoint(self):
                return self.checkpoint

            def save_checkpoint(self, **values):
                saved.append(values)
                self.checkpoint = SimpleNamespace(
                    node_name=values["node_name"],
                    output_artifact_id=values["output_artifact_id"],
                )
                return self.checkpoint

        context = Context()
        job = JobMessage(
            job_id="parse-fault-job",
            job_type="document_parse",
            input_reference=self.snapshot.id,
        )

        with self.assertRaises(InjectedFault):
            handler(job, context)
        artifact_count = len(
            self.container.document_repository.list_artifacts(self.snapshot.id)
        )
        recovered = handler(job.model_copy(update={"attempt": 2}), context)

        self.assertEqual(len(saved), 1)
        self.assertEqual(artifact_count, 2)
        self.assertEqual(
            len(self.container.document_repository.list_artifacts(self.snapshot.id)),
            artifact_count,
        )
        self.assertEqual(recovered.output_reference, saved[0]["output_artifact_id"])
        self.assertIn("already completed", recovered.summary)


if __name__ == "__main__":
    unittest.main()
