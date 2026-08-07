from __future__ import annotations

import tempfile
import os
import unittest
from pathlib import Path

from tender_review.optimization.demo import (
    render_demo_artifacts,
    validate_demo_artifacts,
    write_demo_artifacts,
)
from tender_review.optimization.migration import migrate_approval_optimizer_baseline


PRIVATE_OPTIMIZATION_BASELINE_ENV = "TENDER_REVIEW_PRIVATE_OPTIMIZATION_BASELINE"


@unittest.skipUnless(
    os.environ.get(PRIVATE_OPTIMIZATION_BASELINE_ENV),
    f"requires private fixture via {PRIVATE_OPTIMIZATION_BASELINE_ENV}",
)
class Phase7MigrationDemoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = Path(os.environ[PRIVATE_OPTIMIZATION_BASELINE_ENV]).resolve()

    def test_migration_preserves_external_platform_boundary_and_both_outcomes(self):
        migrated = migrate_approval_optimizer_baseline(self.source)
        counts: dict[str, int] = {}
        for group in migrated.groups:
            counts[group.platform_status] = counts.get(group.platform_status, 0) + 1
            self.assertEqual(group.source_type.value, "EXTERNAL_PLATFORM")
            self.assertEqual(group.status, "provisional")
            self.assertFalse(group.claims_allowed)
            self.assertEqual(
                group.sample_semantics, "aggregate_external_platform_observation"
            )
        self.assertTrue(counts)
        self.assertTrue(set(counts).issubset({"failed", "no_target", "already_covered", "optimized"}))
        self.assertEqual(migrated.human_annotation_cases, 0)
        self.assertGreater(migrated.required_human_cases, 0)

    def test_demo_contains_success_failure_and_no_release_claim(self):
        artifacts = render_demo_artifacts(self.source)
        success = artifacts["success_trace.json"].decode("utf-8")
        failure = artifacts["failure_trace.json"].decode("utf-8")
        self.assertIn('"status": "WAITING_APPROVAL"', success)
        self.assertIn('"status": "DRAFT"', success)
        self.assertIn('"complete_evaluation_gate_created": false', success)
        self.assertIn('"status": "OPTIMIZATION_FAILED"', failure)
        self.assertIn('"auto_published": false', failure)

    def test_written_demo_is_byte_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_demo_artifacts(output, self.source)
            self.assertEqual(validate_demo_artifacts(output, self.source), [])


if __name__ == "__main__":
    unittest.main()
