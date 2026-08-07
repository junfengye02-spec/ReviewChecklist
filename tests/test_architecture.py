from __future__ import annotations

import unittest

from architecture_rules import (
    cross_module_internal_violations,
    dependency_cycles,
    domain_dependency_violations,
    load_python_sources,
    port_annotation_violations,
)
from tender_review.config import PROJECT_DIR


class ArchitectureGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = load_python_sources(PROJECT_DIR / "tender_review")

    def test_repository_obeys_dependency_guards(self):
        self.assertEqual(domain_dependency_violations(self.sources), [])
        self.assertEqual(cross_module_internal_violations(self.sources), [])
        self.assertEqual(dependency_cycles(self.sources), [])
        self.assertEqual(port_annotation_violations(self.sources), [])

    def test_domain_guard_rejects_framework_and_infrastructure_imports(self):
        sources = {
            "tender_review.documents.models": (
                "import sqlalchemy\n"
                "from tender_review.infrastructure.database import Base\n"
            )
        }
        self.assertEqual(
            domain_dependency_violations(sources),
            [
                "tender_review.documents.models -> sqlalchemy",
                "tender_review.documents.models -> tender_review.infrastructure.database",
            ],
        )

    def test_cross_module_guard_rejects_internal_import(self):
        sources = {
            "tender_review.review.service": (
                "from tender_review.documents.models import ParsedPage\n"
            )
        }
        self.assertEqual(
            cross_module_internal_violations(sources),
            ["tender_review.review.service -> tender_review.documents.models"],
        )

    def test_cycle_guard_rejects_a_real_import_cycle(self):
        sources = {
            "tender_review.documents.service": (
                "from tender_review.review.public import ReviewSummary\n"
            ),
            "tender_review.review.public": (
                "from tender_review.documents.service import DocumentService\n"
            ),
        }
        self.assertEqual(
            dependency_cycles(sources),
            [
                (
                    "tender_review.documents.service",
                    "tender_review.review.public",
                )
            ],
        )

    def test_port_guard_rejects_sdk_session_and_raw_dict_types(self):
        sources = {
            "tender_review.documents.ports": (
                "class BadPort:\n"
                "    def save(self, session: Session, value: dict[str, object]) -> Any: ...\n"
            )
        }
        self.assertEqual(
            port_annotation_violations(sources),
            [
                "tender_review.documents.ports.save: Any",
                "tender_review.documents.ports.save: Session",
                "tender_review.documents.ports.save: dict[str, object]",
            ],
        )


if __name__ == "__main__":
    unittest.main()
