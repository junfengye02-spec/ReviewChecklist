from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from .artifacts import RunArtifacts
from .config import Settings
from .llm import LlmClient
from .platform import PlatformClient
from .results import (
    deterministic_issue_evaluation,
    find_review_point_meta,
    parse_workflow_output,
    report_item_output,
    select_review_items,
)
from .rules import load_review_rule, write_optimized_excel


class MvpPipeline:
    def __init__(
        self,
        settings: Settings,
        max_iterations: int = 3,
        stability_runs: int = 3,
        run_dir=None,
    ) -> None:
        self.settings = settings
        self.max_iterations = max(0, max_iterations)
        self.stability_runs = max(1, stability_runs)
        self.artifacts = (
            RunArtifacts.open_existing(run_dir)
            if run_dir is not None
            else RunArtifacts(settings.runs_dir)
        )
        self.platform = PlatformClient(settings.base_url, settings.username)
        self.llm = LlmClient(settings.llm_url, settings.llm_api_key, settings.llm_model)

    def prepare(self) -> dict[str, Any]:
        rule = load_review_rule(self.settings.excel_path, self.settings.review_item)
        zip_path = self.artifacts.build_zip(self.settings.pdf_path)
        excel_copy = self.artifacts.copy_excel(self.settings.excel_path)
        manifest = {
            "run_id": self.artifacts.run_id,
            "excel_path": str(self.settings.excel_path),
            "pdf_path": str(self.settings.pdf_path),
            "zip_path": str(zip_path),
            "excel_copy": str(excel_copy),
            "review_item": self.settings.review_item,
            "expected_issue": self.settings.expected_issue,
            "rule": asdict(rule),
        }
        self.artifacts.write_json("00_manifest.json", manifest)
        return manifest

    def run(self) -> dict[str, Any]:
        manifest = self.prepare()
        rule = load_review_rule(self.settings.excel_path, self.settings.review_item)
        zip_path = self.artifacts.run_dir / "待审核材料_MVP.zip"

        self._log("生成临时平台 token")
        self.platform.refresh_token()

        product_name = f"招标文件审核_MVP_{time.strftime('%Y%m%d%H%M%S')}"
        self._log(f"创建隔离产品：{product_name}")
        product = self.platform.create_product(product_name, self.settings.product_property)
        product_id = str((product["response"].get("data") or {})["id"])
        self.artifacts.write_json("01_创建产品.json", product)

        self._log("导入完整审核要点 Excel")
        imported = self.platform.import_review_points(product_id, self.settings.excel_path)
        self.artifacts.write_json("02_导入审核要点_响应.json", imported)

        self._log("上传单样本 ZIP")
        upload = self.platform.upload_zip(product_id, zip_path)
        self.artifacts.write_json("03_上传ZIP_响应.json", upload)
        matching_items = sorted(
            {
                str(item.get("reviewItem"))
                for item in self.platform._matches(upload)
                if item.get("reviewItem") is not None
            }
        )
        if self.settings.review_item not in matching_items:
            raise RuntimeError(
                f"ZIP was parsed but review item {self.settings.review_item} was not matched; "
                f"matched items={matching_items}"
            )

        self._log("创建并启动 AI 审核任务")
        task = self.platform.create_task(
            product_id=product_id,
            product_name=product_name,
            applicant_name=self.settings.applicant_name,
            product_property=self.settings.product_property,
            clinical_evaluation=self.settings.clinical_evaluation,
            upload=upload,
        )
        task_id = task["task_id"]
        self.artifacts.write_json("04_创建任务.json", task)
        started = self.platform.start_task(task_id)
        self.artifacts.write_json("05_启动任务_响应.json", started)

        self._log(f"等待任务 {task_id} 完成")
        progress = self.platform.wait_for_task(
            task_id,
            self.settings.poll_interval_seconds,
            self.settings.poll_timeout_seconds,
            on_progress=self._progress,
        )
        self.artifacts.write_json("06_最终进度.json", progress)

        report = self.platform.get_results(task_id)
        self.artifacts.write_json("07_AI审核报告.json", report)
        return self._finalize(
            manifest=manifest,
            product_id=product_id,
            product_name=product_name,
            task_id=task_id,
            upload=upload,
            report=report,
            rule=rule,
        )

    def resume(self) -> dict[str, Any]:
        manifest = self.artifacts.read_json("00_manifest.json")
        product = self.artifacts.read_json("01_创建产品.json")
        task = self.artifacts.read_json("04_创建任务.json")
        upload = self.artifacts.read_json("03_上传ZIP_响应.json")
        report = self.artifacts.read_json("07_AI审核报告.json")
        product_id = str((product["response"].get("data") or {})["id"])
        product_name = str(product["request"]["productName"])
        task_id = str(task["task_id"])
        rule = load_review_rule(self.settings.excel_path, self.settings.review_item)
        self._log(f"从运行 {self.artifacts.run_id} 的平台报告继续")
        self.platform.refresh_token()
        return self._finalize(
            manifest=manifest,
            product_id=product_id,
            product_name=product_name,
            task_id=task_id,
            upload=upload,
            report=report,
            rule=rule,
        )

    def _finalize(
        self,
        manifest: dict[str, Any],
        product_id: str,
        product_name: str,
        task_id: str,
        upload: dict[str, Any],
        report: dict[str, Any],
        rule,
    ) -> dict[str, Any]:
        matching_items = sorted(
            {
                str(item.get("reviewItem"))
                for item in self.platform._matches(upload)
                if item.get("reviewItem") is not None
            }
        )
        item_rows = select_review_items(report, self.settings.review_item)
        if not item_rows:
            raise RuntimeError(
                f"AI report has no result for review item {self.settings.review_item}"
            )
        initial_outputs = [report_item_output(item) for item in item_rows]
        initial_eval = self._strict_evaluate(
            initial_outputs,
            "\n\n".join(str(item.get("reasoning") or "") for item in item_rows),
        )
        self.artifacts.write_json(
            "08_初始命中判定.json",
            {"outputs": initial_outputs, "evaluation": initial_eval},
        )

        meta = find_review_point_meta(upload, self.settings.review_item)
        current_point = rule.point
        self._log(
            f"使用原始审核要点执行 {self.stability_runs} 次快速重审稳定性检查"
        )
        baseline_check, baseline_output = self._workflow_stability_check(
            meta=meta,
            rule=rule,
            point=current_point,
            task_id=task_id,
            label="基线",
        )
        baseline_eval = baseline_check["evaluation"]
        self.artifacts.write_json(
            "08b_基线工作流测试.json",
            baseline_check,
        )

        current_output: Any = baseline_output
        current_eval = baseline_eval
        attempts: list[dict[str, Any]] = []

        for iteration in range(1, self.max_iterations + 1):
            if current_eval.get("hit") is True:
                break
            self._log(f"第 {iteration} 轮：最小调整项目 {self.settings.review_item} 审核要点")
            optimized = self._optimize_point(
                rule=rule,
                current_point=current_point,
                current_output=current_output,
                current_eval=current_eval,
                attempts=attempts,
                iteration=iteration,
            )
            candidate = str(optimized.get("optimized_point") or "").strip()
            if not candidate:
                raise RuntimeError("Optimizer returned an empty review point")

            payload = self._workflow_payload(
                meta=meta, rule=rule, point=candidate, task_id=task_id
            )

            stability_check, parsed_output = self._workflow_stability_check(
                meta=meta,
                rule=rule,
                point=candidate,
                task_id=task_id,
                label=f"优化{iteration}",
            )
            evaluation = stability_check["evaluation"]
            attempt = {
                "iteration": iteration,
                "change_note": optimized.get("change_note"),
                "optimizer_method": optimized.get("optimizer_method", "llm"),
                "point": candidate,
                "workflow_request": payload,
                "workflow_output": parsed_output,
                "workflow_checks": stability_check["checks"],
                "evaluation": evaluation,
            }
            attempts.append(attempt)
            self.artifacts.write_json(f"09_循环_{iteration}.json", attempt)
            current_point = candidate
            current_output = parsed_output
            current_eval = evaluation

        optimized_excel = self.artifacts.run_dir / "审核要点_优化结果.xlsx"
        write_optimized_excel(
            self.settings.excel_path,
            optimized_excel,
            self.settings.review_item,
            current_point,
        )
        self.artifacts.write_text("审核要点_项目41_最终版.md", current_point + "\n")

        summary = {
            **manifest,
            "product_id": product_id,
            "product_name": product_name,
            "task_id": task_id,
            "matched_review_items": matching_items,
            "initial_batch_evaluation": initial_eval,
            "baseline_workflow_evaluation": baseline_eval,
            "stability_runs_required": self.stability_runs,
            "iterations_run": len(attempts),
            "final_evaluation": current_eval,
            "final_hit": current_eval.get("hit") is True,
            "optimized_excel": str(optimized_excel),
        }
        self.artifacts.write_json("10_运行总结.json", summary)
        return summary

    def _workflow_stability_check(
        self,
        meta,
        rule,
        point: str,
        task_id: str,
        label: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        payload = self._workflow_payload(
            meta=meta, rule=rule, point=point, task_id=task_id
        )
        checks: list[dict[str, Any]] = []
        representative: dict[str, Any] = {}
        for run_number in range(1, self.stability_runs + 1):
            self._log(f"{label}快速重审 {run_number}/{self.stability_runs}")
            raw_output = self.platform.workflow_test(payload)
            parsed_output = parse_workflow_output(raw_output)
            evaluation = self._strict_evaluate(
                parsed_output, str(parsed_output.get("reasoning") or "")
            )
            checks.append(
                {
                    "run": run_number,
                    "output": parsed_output,
                    "evaluation": evaluation,
                }
            )
            representative = parsed_output
            if evaluation.get("hit") is not True:
                break

        passed = sum(1 for check in checks if check["evaluation"].get("hit") is True)
        stable = passed == self.stability_runs and len(checks) == self.stability_runs
        evaluation = {
            "hit": stable,
            "method": "repeated_workflow_test",
            "reason": (
                f"连续快速重审命中 {passed}/{self.stability_runs}；"
                + ("达到稳定性门槛。" if stable else "未达到全量命中门槛。")
            ),
            "detected_issue": self.settings.expected_issue if stable else "",
        }
        return {
            "workflow_request": payload,
            "checks": checks,
            "evaluation": evaluation,
        }, representative

    def _optimize_point(
        self,
        rule,
        current_point: str,
        current_output: Any,
        current_eval: dict[str, Any],
        attempts: list[dict[str, Any]],
        iteration: int,
    ) -> dict[str, Any]:
        optimized = self.llm.optimize_review_point(
            original_point=rule.point,
            current_point=current_point,
            expected_issue=self.settings.expected_issue,
            platform_output=current_output,
            evaluation_reason=str(current_eval.get("reason") or ""),
            previous_attempts=attempts,
        )
        optimized["optimizer_method"] = f"llm:{self.settings.llm_model}"
        return optimized

    @staticmethod
    def _workflow_payload(meta, rule, point: str, task_id: str) -> dict[str, Any]:
        payload = {
            "id": str(meta.get("id") or ""),
            "workflowId": str(meta.get("workFlowId") or rule.workflow_id),
            "content": meta.get("content") or "",
            "reviewItemName": meta.get("file") or rule.file_name,
            "point": point,
            "taskId": task_id,
        }
        missing = [key for key, value in payload.items() if value in (None, "")]
        if missing:
            raise RuntimeError(f"Workflow-test payload is missing: {', '.join(missing)}")
        return payload

    def _strict_evaluate(self, output: Any, reasoning: str) -> dict[str, Any]:
        deterministic = deterministic_issue_evaluation(
            self.settings.expected_issue, output
        )
        if deterministic is not None:
            return deterministic
        return self.llm.strict_evaluate(
            self.settings.expected_issue, output, reasoning
        )

    def _log(self, message: str) -> None:
        print(f"[MVP] {message}", flush=True)

    @staticmethod
    def _progress(snapshot: dict[str, Any]) -> None:
        print(
            "[MVP] 进度: "
            f"{snapshot.get('percentage', 0)}% "
            f"status={snapshot.get('status')} "
            f"detail={snapshot.get('progressStatus', '')}",
            flush=True,
        )
