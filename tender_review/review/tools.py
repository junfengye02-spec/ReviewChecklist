from __future__ import annotations

from collections.abc import Iterable

from tender_review.shared.contracts import ensure_call_active

from .models import (
    ComparisonResult,
    ComparisonToolInput,
    DateExtraction,
    DateRule,
    ExtractedField,
    NumberExtraction,
    NumericRangeExtraction,
    NumericRangeRule,
    SetExtraction,
    SetRule,
    TextExtraction,
    TextPresenceRule,
    ToolRequest,
    ToolResult,
)


TOOL_VERSION = "1.0.0"


def _field(input_value: ComparisonToolInput) -> ExtractedField:
    fields = tuple(
        field
        for field in input_value.extraction.fields
        if field.field_name == input_value.rule.field_name
    )
    if len(fields) != 1:
        raise ValueError(
            f"extraction must contain exactly one field named "
            f"{input_value.rule.field_name!r}"
        )
    return fields[0]


def _result(
    input_value: ComparisonToolInput,
    *,
    passed: bool,
    code: str,
    message: str,
    field: ExtractedField,
) -> ToolResult:
    result = ComparisonResult(
        review_item_id=input_value.rule.review_item_id,
        tool_name=input_value.rule.tool_name,
        tool_version=TOOL_VERSION,
        passed=passed,
        code=code,
        message=message,
        sources=field.sources,
    )
    return ToolResult(
        tool_name=input_value.rule.tool_name,
        tool_version=TOOL_VERSION,
        output_json=result.model_dump_json(),
    )


class DateComparisonTool:
    @property
    def name(self) -> str:
        return "date_compare"

    def execute(self, request: ToolRequest) -> ToolResult:
        input_value = self._input(request)
        if not isinstance(input_value.rule, DateRule):
            raise ValueError("date_compare requires a DateRule")
        field = _field(input_value)
        if not isinstance(field, DateExtraction):
            raise ValueError("date_compare requires a date extraction")
        actual = field.value
        expected = input_value.rule.expected
        operator = input_value.rule.operator
        passed = {
            "before": actual < expected,
            "on_or_before": actual <= expected,
            "equal": actual == expected,
            "on_or_after": actual >= expected,
            "after": actual > expected,
        }[operator]
        relation = "satisfies" if passed else "does not satisfy"
        return _result(
            input_value,
            passed=passed,
            code="date_requirement_met" if passed else "date_requirement_not_met",
            message=f"Observed date {actual.isoformat()} {relation} {operator} "
            f"{expected.isoformat()}.",
            field=field,
        )

    def _input(self, request: ToolRequest) -> ComparisonToolInput:
        ensure_call_active(request.call)
        if request.tool_name != self.name:
            raise ValueError(f"{self.name} cannot handle {request.tool_name!r}")
        return ComparisonToolInput.model_validate_json(request.input_json)


class SetComparisonTool:
    @property
    def name(self) -> str:
        return "set_compare"

    def execute(self, request: ToolRequest) -> ToolResult:
        input_value = self._input(request)
        if not isinstance(input_value.rule, SetRule):
            raise ValueError("set_compare requires a SetRule")
        field = _field(input_value)
        if not isinstance(field, SetExtraction):
            raise ValueError("set_compare requires a set extraction")
        actual = set(field.values)
        expected = set(input_value.rule.expected_values)
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        passed = not missing and (
            input_value.rule.mode == "contains_all" or not unexpected
        )
        details = f"missing={missing}"
        if input_value.rule.mode == "exact":
            details += f", unexpected={unexpected}"
        return _result(
            input_value,
            passed=passed,
            code="set_requirement_met" if passed else "set_requirement_not_met",
            message=f"Set comparison ({input_value.rule.mode}): {details}.",
            field=field,
        )

    def _input(self, request: ToolRequest) -> ComparisonToolInput:
        ensure_call_active(request.call)
        if request.tool_name != self.name:
            raise ValueError(f"{self.name} cannot handle {request.tool_name!r}")
        return ComparisonToolInput.model_validate_json(request.input_json)


class NumericRangeComparisonTool:
    @property
    def name(self) -> str:
        return "numeric_range"

    def execute(self, request: ToolRequest) -> ToolResult:
        input_value = self._input(request)
        if not isinstance(input_value.rule, NumericRangeRule):
            raise ValueError("numeric_range requires a NumericRangeRule")
        field = _field(input_value)
        if isinstance(field, NumberExtraction):
            actual_minimum = actual_maximum = field.value
            actual_unit = field.unit
        elif isinstance(field, NumericRangeExtraction):
            actual_minimum = field.minimum
            actual_maximum = field.maximum
            actual_unit = field.unit
        else:
            raise ValueError("numeric_range requires a number or range extraction")

        expected_minimum = input_value.rule.expected_minimum
        expected_maximum = input_value.rule.expected_maximum
        unit_matches = (
            input_value.rule.unit is None or input_value.rule.unit == actual_unit
        )
        lower_matches = expected_minimum is None or actual_minimum >= expected_minimum
        upper_matches = expected_maximum is None or actual_maximum <= expected_maximum
        passed = unit_matches and lower_matches and upper_matches
        return _result(
            input_value,
            passed=passed,
            code="numeric_range_met" if passed else "numeric_range_not_met",
            message=(
                f"Observed range [{actual_minimum}, {actual_maximum}] "
                f"unit={actual_unit!r}; required range "
                f"[{expected_minimum}, {expected_maximum}] "
                f"unit={input_value.rule.unit!r}."
            ),
            field=field,
        )

    def _input(self, request: ToolRequest) -> ComparisonToolInput:
        ensure_call_active(request.call)
        if request.tool_name != self.name:
            raise ValueError(f"{self.name} cannot handle {request.tool_name!r}")
        return ComparisonToolInput.model_validate_json(request.input_json)


class TextPresenceComparisonTool:
    @property
    def name(self) -> str:
        return "text_presence"

    def execute(self, request: ToolRequest) -> ToolResult:
        input_value = self._input(request)
        if not isinstance(input_value.rule, TextPresenceRule):
            raise ValueError("text_presence requires a TextPresenceRule")
        field = _field(input_value)
        if not isinstance(field, TextExtraction):
            raise ValueError("text_presence requires a text extraction")
        actual = field.value
        terms: Iterable[str] = input_value.rule.required_terms
        if not input_value.rule.case_sensitive:
            actual = actual.casefold()
            terms = tuple(term.casefold() for term in terms)
        presence = tuple(term in actual for term in terms)
        passed = all(presence) if input_value.rule.mode == "all" else any(presence)
        missing = tuple(
            original
            for original, is_present in zip(
                input_value.rule.required_terms, presence, strict=True
            )
            if not is_present
        )
        return _result(
            input_value,
            passed=passed,
            code="text_present" if passed else "text_missing",
            message=(
                f"Text presence ({input_value.rule.mode}) checked "
                f"{len(input_value.rule.required_terms)} term(s); missing={list(missing)}."
            ),
            field=field,
        )

    def _input(self, request: ToolRequest) -> ComparisonToolInput:
        ensure_call_active(request.call)
        if request.tool_name != self.name:
            raise ValueError(f"{self.name} cannot handle {request.tool_name!r}")
        return ComparisonToolInput.model_validate_json(request.input_json)


def default_review_tools() -> tuple[
    DateComparisonTool,
    SetComparisonTool,
    NumericRangeComparisonTool,
    TextPresenceComparisonTool,
]:
    return (
        DateComparisonTool(),
        SetComparisonTool(),
        NumericRangeComparisonTool(),
        TextPresenceComparisonTool(),
    )
