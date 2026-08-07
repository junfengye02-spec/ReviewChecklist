import unittest

from tender_review.results import (
    deterministic_issue_evaluation,
    repair_mojibake,
)


EXPECTED = "招标文件不同章节中的提交要求不一致"


class ResultEvaluationTests(unittest.TestCase):
    def test_repairs_gbk_text_decoded_as_latin1(self):
        self.assertEqual(repair_mojibake("ÖÐÎÄ"), "中文")

    def test_explicit_issue_is_a_hit(self):
        result = deterministic_issue_evaluation(
            EXPECTED,
            {
                "errorReason": "章节 A 要求纸质提交，章节 B 要求电子提交，两处要求不一致。"
            },
        )
        self.assertTrue(result["hit"])

    def test_compliant_ai_result_overrides_document_keywords(self):
        result = deterministic_issue_evaluation(
            EXPECTED,
            {
                "ai_result": {"result": "true", "reason": "企业材料完全符合审评要点。"},
                "content_result": {
                    "text": "章节 A 与章节 B 的提交要求不一致。"
                },
            },
        )
        self.assertFalse(result["hit"])


if __name__ == "__main__":
    unittest.main()
