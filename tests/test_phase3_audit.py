from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tender_review.documents.parsing.audit_real_pdfs import main, render_report


class Phase3AuditCliTests(unittest.TestCase):
    def test_check_requires_an_existing_baseline_before_running_the_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "missing.json"
            with patch(
                "tender_review.documents.parsing.audit_real_pdfs.audit_pdf_directory",
                side_effect=AssertionError("audit should not run"),
            ):
                exit_code = main(
                    ["--pdf-root", temporary, "--output", str(output), "--check"]
                )

        self.assertEqual(exit_code, 1)

    def test_check_compares_the_complete_deterministic_report(self) -> None:
        report = {
            "schema_version": 1,
            "audit_name": "phase3_pdf_parsing_offline_audit",
            "totals": {"documents": 1},
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "audit.json"
            output.write_text(render_report(report), encoding="utf-8")
            with patch(
                "tender_review.documents.parsing.audit_real_pdfs.audit_pdf_directory",
                return_value=report,
            ):
                matching = main(
                    ["--pdf-root", temporary, "--output", str(output), "--check"]
                )
                output.write_text("{}\n", encoding="utf-8")
                different = main(
                    ["--pdf-root", temporary, "--output", str(output), "--check"]
                )

        self.assertEqual(matching, 0)
        self.assertEqual(different, 1)


if __name__ == "__main__":
    unittest.main()
