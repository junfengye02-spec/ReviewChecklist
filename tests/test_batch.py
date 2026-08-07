import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from tender_review.batch import (
    build_batch_zip,
    discover_documents,
    write_structured_outputs,
)


class BatchReviewTests(unittest.TestCase):
    def test_archive_uses_neutral_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            (source / "合成规则-A").mkdir(parents=True)
            (source / "合成规则-B").mkdir(parents=True)
            (source / "合成规则-A" / "a.pdf").write_bytes(b"a")
            (source / "合成规则-B" / "b.pdf").write_bytes(b"b")

            documents = discover_documents(source)
            target = build_batch_zip(documents, root / "batch.zip")

            with zipfile.ZipFile(target) as archive:
                names = archive.namelist()
            self.assertEqual(len(names), 2)
            self.assertEqual({Path(name).name for name in names}, {"a.pdf", "b.pdf"})
            self.assertTrue(all(name.startswith("待审项目_") for name in names))
            self.assertNotIn("合成规则-A", "".join(names))
            self.assertNotIn("合成规则-B", "".join(names))

    def test_structured_outputs_correlate_files_and_issues(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            (source / "合成规则-A").mkdir(parents=True)
            (source / "合成规则-B").mkdir(parents=True)
            (source / "合成规则-A" / "a.pdf").write_bytes(b"a")
            (source / "合成规则-B" / "b.pdf").write_bytes(b"b")
            documents = discover_documents(source)
            report = {
                "data": [
                    {
                        "fileName": "a.pdf",
                        "reviewItem": "1",
                        "reviewStatus": "PENDING_CONFIRMATION",
                        "errorReason": "存在问题",
                        "filePageNumber": 3,
                    },
                    {
                        "fileName": "b.pdf",
                        "reviewItem": "1",
                        "reviewStatus": "SUCCESS",
                        "errorReason": "符合要求",
                    },
                ]
            }
            raw = root / "raw.json"
            raw.write_text("{}", encoding="utf-8")

            summary = write_structured_outputs(
                report=report,
                documents=documents,
                output_dir=root / "审批意见",
                run_id="batch_test",
                product_id="product",
                task_id="task",
                review_item_ids=["1"],
                raw_report_path=raw,
            )

            self.assertEqual(summary["totals"]["completed_files"], 2)
            self.assertEqual(summary["totals"]["issue_count"], 1)
            result = json.loads(
                (root / "审批意见" / "合成规则-A" / "a.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(result["source"]["expected_issue"], "合成规则-A")
            self.assertFalse(result["review"]["opinions"][0]["compliant"])


if __name__ == "__main__":
    unittest.main()
