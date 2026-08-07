from __future__ import annotations

import ast
import json
import re
from typing import Any


def select_review_items(report: dict[str, Any], review_item: str) -> list[dict[str, Any]]:
    return [
        item
        for item in (report.get("data") or [])
        if str(item.get("reviewItem") or "").strip() == str(review_item).strip()
    ]


def report_item_output(item: dict[str, Any]) -> dict[str, Any]:
    return repair_mojibake({
        "reviewItem": item.get("reviewItem"),
        "reviewPointId": item.get("reviewPointId"),
        "reviewStatus": item.get("reviewStatus"),
        "fileName": item.get("fileName") or item.get("reviewFileName"),
        "errorReason": item.get("errorReason"),
        "materialExcerpt": item.get("materialExcerpt"),
        "reasoning": item.get("reasoning"),
        "contentResult": item.get("contentResult"),
        "fileResult": item.get("fileResult"),
        "point": item.get("point"),
    })


def repair_mojibake(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: repair_mojibake(item) for key, item in value.items()}
    if isinstance(value, list):
        return [repair_mojibake(item) for item in value]
    if not isinstance(value, str):
        return value
    markers = ("ÖÐ", "ÎÄ", "Éó", "²»", "Ò»")
    if not any(marker in value for marker in markers):
        return value
    try:
        return value.encode("latin-1").decode("gbk")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


def deterministic_issue_evaluation(
    expected_issue: str, platform_output: Any
) -> dict[str, Any] | None:
    repaired = repair_mojibake(platform_output)
    ai_result_text = json.dumps(
        _collect_key(repaired, "ai_result"), ensure_ascii=False, default=str
    )
    if "企业材料完全符合审评要点" in ai_result_text and "不一致" not in ai_result_text:
        return {
            "hit": False,
            "method": "deterministic_strict",
            "reason": "平台最终 ai_result 明确判定材料符合要求，未检出人工已知问题。",
            "detected_issue": "",
        }

    text = json.dumps(
        _decision_projection(repaired), ensure_ascii=False, default=str
    )
    has_inconsistency = "不一致" in text or "冲突" in text
    has_scope = any(term in text for term in ("要求", "条件", "条款"))
    has_comparison = any(
        term in text for term in ("章节", "两处", "差异", "不同", "分别")
    )
    if has_inconsistency and has_scope and has_comparison:
        return {
            "hit": True,
            "method": "deterministic_strict",
            "reason": "平台明确指出不同章节或条款的要求存在具体差异。",
            "detected_issue": expected_issue,
        }
    if "企业材料完全符合审评要点" in text and not has_inconsistency:
        return {
            "hit": False,
            "method": "deterministic_strict",
            "reason": "平台明确判定材料符合要求，未检出人工已知问题。",
            "detected_issue": "",
        }
    return None


_DECISION_KEYS = {
    "ai_result",
    "errorReason",
    "reasoning",
    "AI审评结果",
    "AI审评推理",
    "AI审评理由_reasons",
    "reason",
    "reasons",
    "issue",
    "result",
    "detected_issue",
}


def _decision_projection(value: Any) -> Any:
    if isinstance(value, list):
        return [_decision_projection(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        key: _decision_projection(item)
        for key, item in value.items()
        if key in _DECISION_KEYS
    }


def _collect_key(value: Any, target: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, list):
        for item in value:
            found.extend(_collect_key(item, target))
    elif isinstance(value, dict):
        for key, item in value.items():
            if key == target:
                found.append(item)
            found.extend(_collect_key(item, target))
    return found


def parse_workflow_output(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {"raw": raw}
    cleaned = raw.strip().replace("```json", "").replace("```", "")
    candidates = [cleaned]
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match and match.group(0) != cleaned:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        try:
            value = ast.literal_eval(candidate)
            if isinstance(value, dict):
                return value
        except (ValueError, SyntaxError):
            pass
    return {"raw": raw}


def find_review_point_meta(upload: dict[str, Any], review_item: str) -> dict[str, Any]:
    candidates = [
        item
        for item in (upload.get("reviewPoints") or [])
        if str(item.get("reviewItem") or "").strip() == str(review_item).strip()
    ]
    if not candidates:
        raise ValueError(f"Upload response has no metadata for review item {review_item}")
    return candidates[0]
