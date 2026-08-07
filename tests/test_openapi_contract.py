from __future__ import annotations

import unittest

from tender_review.openapi import DEFAULT_BASELINE, check_openapi


class OpenApiContractTests(unittest.TestCase):
    def test_frozen_v1_schema_matches_application(self):
        self.assertEqual(check_openapi(DEFAULT_BASELINE), [])


if __name__ == "__main__":
    unittest.main()
