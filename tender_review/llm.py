from __future__ import annotations

import json
import re
import time
from typing import Any

import requests


class LlmClient:
    def __init__(self, url: str, api_key: str, model: str) -> None:
        self.url = url
        self.api_key = api_key
        self.model = model
        self.session = requests.Session()

    def strict_evaluate(
        self, expected_issue: str, platform_output: Any, reasoning: str = ""
    ) -> dict[str, Any]:
        system = (
            "你是招标文件审核结果验收员。必须严格判断平台是否明确发现了指定问题。"
            "只有结果明确指出该问题确实存在，才算命中。仅提到相关章节、提出一般检查建议、"
            "表示无法判断，或表示文件符合要求，均算未命中。只返回合法JSON。"
        )
        user = f"""【必须检出的人工已知问题】
{expected_issue}

【平台审核输出】
{self._to_text(platform_output)}

【平台推理】
{reasoning}

返回：
{{"hit": true或false, "reason": "严格判定理由", "detected_issue": "平台实际发现的问题，未发现则为空"}}
"""
        return self._json_call(system, user)

    def optimize_review_point(
        self,
        original_point: str,
        current_point: str,
        expected_issue: str,
        platform_output: Any,
        evaluation_reason: str,
        previous_attempts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        system = (
            "你是公共资源交易招标文件审核规则专家。请对审核要点做最小、通用、可复用的增补，"
            "使审核模型能够识别给定的目标问题。不得写入项目名称、文件名、页码或样本答案，"
            "不得删除原有要求，"
            "不得只针对单个样本过拟合。保持原Markdown结构，只返回合法JSON。"
        )
        attempts_text = json.dumps(previous_attempts, ensure_ascii=False, indent=2)
        user = f"""【原始审核要点】
{original_point}

【当前审核要点】
{current_point}

【人工已知问题】
{expected_issue}

【本轮平台输出】
{self._to_text(platform_output)}

【未命中原因】
{evaluation_reason}

【此前失败尝试】
{attempts_text}

请对目标问题涉及的章节、条款和要求逐项比较，明确指出任何增删、替换、范围或口径差异。

返回：
{{"optimized_point": "优化后的完整审核要点", "change_note": "本轮最小修改说明"}}
"""
        return self._json_call(system, user)

    def evaluate_approval_case(
        self,
        *,
        expected_compliant: bool,
        expected_opinions: list[str],
        platform_output: Any,
    ) -> dict[str, Any]:
        system = (
            "你是招标文件审核结果验收员。请根据人工审批意见，严格判断平台本次审核输出是否达标。"
            "人工意见只用于验收语义，不要求逐字一致。不得因为输出提到相关章节或一般建议就判定命中。"
            "只返回合法JSON。"
        )
        expectation = (
            "该样本应判定为合规。平台必须明确表示符合要求，且不得新增实质性问题。"
            if expected_compliant
            else "该样本应检出下列全部人工问题。遗漏任意一个问题都不通过。"
        )
        user = f"""【验收要求】
{expectation}

【人工审批意见】
{json.dumps(expected_opinions, ensure_ascii=False, indent=2)}

【平台本次审核输出】
{self._to_text(platform_output)}

返回：
{{
  "passed": true或false,
  "observed_compliant": true或false,
  "covered_opinions": ["已明确覆盖的人工意见"],
  "missed_opinions": ["未覆盖的人工意见"],
  "reason": "严格且简短的验收理由"
}}
"""
        result = self._json_call(system, user)
        if not isinstance(result.get("passed"), bool):
            raise ValueError("LLM approval evaluation is missing boolean 'passed'")
        result.setdefault("covered_opinions", [])
        result.setdefault("missed_opinions", [])
        result.setdefault("reason", "")
        return result

    def evaluate_approval_group(
        self, cases: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        system = (
            "你是招标文件审核结果验收员。请逐个样本对照人工审批意见，严格判断平台输出是否达标。"
            "问题样本必须明确检出全部人工问题；合规样本必须明确保持合规且不得新增实质性问题。"
            "只看语义，不要求逐字一致；仅提到章节、一般建议或无法判断均不算命中。"
            "每个case_id必须且只能返回一次，只返回合法JSON。"
        )
        user = f"""【待验收样本】
{json.dumps(cases, ensure_ascii=False, indent=2, default=str)}

返回：
{{
  "cases": [
    {{
      "case_id": "原case_id",
      "passed": true或false,
      "observed_compliant": true或false,
      "covered_opinions": ["已覆盖意见"],
      "missed_opinions": ["未覆盖意见"],
      "reason": "严格且简短的验收理由"
    }}
  ]
}}
"""
        result = self._json_call(system, user)
        rows = result.get("cases")
        if not isinstance(rows, list):
            raise ValueError("LLM group evaluation is missing the 'cases' list")
        expected_ids = {str(case.get("case_id") or "") for case in cases}
        evaluated: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            case_id = str(row.get("case_id") or "")
            if case_id not in expected_ids or case_id in evaluated:
                raise ValueError(f"Unexpected or duplicate evaluation case_id: {case_id}")
            if not isinstance(row.get("passed"), bool):
                raise ValueError(f"Evaluation {case_id} is missing boolean 'passed'")
            row.setdefault("covered_opinions", [])
            row.setdefault("missed_opinions", [])
            row.setdefault("reason", "")
            evaluated[case_id] = row
        missing = expected_ids - set(evaluated)
        if missing:
            raise ValueError(
                "LLM group evaluation omitted cases: " + ", ".join(sorted(missing))
            )
        return evaluated

    def optimize_review_point_group(
        self,
        *,
        review_item: str,
        original_point: str,
        current_point: str,
        target_cases: list[dict[str, Any]],
        protection_cases: list[dict[str, Any]],
        latest_evaluation: dict[str, Any],
        previous_attempts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        system = (
            "你是公共资源交易招标文件审核规则专家。请优化一个可跨项目复用的审核要点。"
            "必须以原始审核要点为基底，只做解决漏检所必需的最小增补；不得删除或弱化原要求。"
            "一个审核要点会同时用于多份招标文件，因此必须覆盖全部目标问题，同时避免让原本合规的"
            "样本产生误报。不得把项目名称、文件名、页码、样本原句或答案写入审核要点。"
            "保持原有Markdown结构，只返回合法JSON。"
        )
        prompt_payload = {
            "review_item": review_item,
            "original_point": original_point,
            "current_point": current_point,
            "target_cases": target_cases,
            "regression_protection_cases": protection_cases,
            "latest_evaluation": latest_evaluation,
            "previous_failed_attempts": previous_attempts,
        }
        user = f"""请根据以下联合验收数据生成下一版审核要点：
{json.dumps(prompt_payload, ensure_ascii=False, indent=2, default=str)}

返回：
{{
  "optimized_point": "优化后的完整审核要点",
  "change_note": "本轮最小修改说明",
  "coverage_strategy": "如何覆盖全部目标并保护合规样本"
}}
"""
        result = self._json_call(system, user)
        if not str(result.get("optimized_point") or "").strip():
            raise ValueError("LLM group optimizer returned an empty review point")
        return result

    def _json_call(self, system: str, user: str) -> dict[str, Any]:
        last_error = ""
        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.session.post(
                    self.url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "temperature": 0.1,
                    },
                    timeout=120,
                )
                if response.status_code != 200:
                    raise RuntimeError(
                        f"LLM HTTP {response.status_code}: {(response.text or '')[:1000]}"
                    )
                raw = response.json()["choices"][0]["message"]["content"]
                return self._parse_json(raw)
            except (requests.RequestException, RuntimeError, ValueError, KeyError) as exc:
                last_error = str(exc)
                if attempt < max_attempts:
                    time.sleep(2 * attempt)
        raise RuntimeError(f"LLM request failed after retries: {last_error}")

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        try:
            value = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", cleaned)
            if not match:
                raise ValueError(f"LLM did not return JSON: {raw[:500]}")
            value = json.loads(match.group(0))
        if not isinstance(value, dict):
            raise ValueError("LLM JSON response must be an object")
        return value

    @staticmethod
    def _to_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
