from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from tender_review.baseline import check_rebuild, sha256_value


PRIVATE_BASELINE_ENV = "TENDER_REVIEW_PRIVATE_BASELINE_DIR"


@unittest.skipUnless(
    os.environ.get(PRIVATE_BASELINE_ENV),
    f"requires private fixtures via {PRIVATE_BASELINE_ENV}",
)
class PrivateBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.baseline_dir = Path(os.environ[PRIVATE_BASELINE_ENV]).resolve()
        cls.config_path = cls.baseline_dir / "stage0_config.json"
        cls.manifest = json.loads(
            (cls.baseline_dir / "dataset_manifest.json").read_text(encoding="utf-8")
        )

    def test_case_ids_and_manifest_hash_are_consistent(self) -> None:
        cases = self.manifest["cases"]
        case_ids = [row["case_id"] for row in cases]
        self.assertEqual(len(case_ids), len(set(case_ids)))
        for case in cases:
            stored_hash = case["case_sha256"]
            unhashed = {key: value for key, value in case.items() if key != "case_sha256"}
            self.assertEqual(stored_hash, sha256_value(unhashed))

    def test_private_artifacts_match_a_clean_rebuild(self) -> None:
        self.assertEqual(check_rebuild(self.config_path, self.baseline_dir), [])


if __name__ == "__main__":
    unittest.main()
