from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    base_url: str
    username: str
    llm_url: str
    llm_api_key: str
    llm_model: str
    excel_path: Path
    pdf_path: Path
    runs_dir: Path
    review_item: str = "example-rule"
    expected_issue: str = "招标文件不同章节中的要求不一致"
    product_property: str = "tender-review"
    applicant_name: str = "local-review-client"
    clinical_evaluation: str = "not-applicable"
    poll_interval_seconds: float = 8.0
    poll_timeout_seconds: float = 3600.0

    @classmethod
    def load(
        cls,
        legacy_config: Path | None = None,
        excel_path: Path | None = None,
        pdf_path: Path | None = None,
        runs_dir: Path | None = None,
    ) -> "Settings":
        config_value = legacy_config or os.environ.get("TENDER_REVIEW_LEGACY_CONFIG")
        if not config_value:
            raise ValueError(
                "Legacy config path is required via --legacy-config or "
                "TENDER_REVIEW_LEGACY_CONFIG"
            )
        config_path = Path(config_value).expanduser().resolve()
        data: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
        llm = data.get("llm") or {}

        excel_value = excel_path or os.environ.get("TENDER_REVIEW_EXCEL_PATH")
        pdf_value = pdf_path or os.environ.get("TENDER_REVIEW_PDF_PATH")
        if not excel_value or not pdf_value:
            raise ValueError(
                "Private input paths are required via explicit arguments or "
                "TENDER_REVIEW_EXCEL_PATH and TENDER_REVIEW_PDF_PATH"
            )
        resolved_excel = Path(excel_value).expanduser()
        resolved_pdf = Path(pdf_value).expanduser()

        settings = cls(
            base_url=str(data.get("base_url") or "").rstrip("/"),
            username=str(data.get("username") or "").strip(),
            llm_url=str(
                os.environ.get("TENDER_LLM_URL") or llm.get("base_url") or ""
            ).strip(),
            llm_api_key=str(
                os.environ.get("TENDER_LLM_API_KEY") or llm.get("api_key") or ""
            ).strip(),
            llm_model=str(
                os.environ.get("TENDER_LLM_MODEL") or llm.get("model") or ""
            ).strip(),
            excel_path=resolved_excel.resolve(),
            pdf_path=resolved_pdf.resolve(),
            runs_dir=(runs_dir or PROJECT_DIR / "runs").resolve(),
            review_item=str(data.get("review_item") or "example-rule").strip(),
            expected_issue=str(
                data.get("expected_issue")
                or "招标文件不同章节中的要求不一致"
            ).strip(),
            product_property=str(
                data.get("product_property") or "tender-review"
            ).strip(),
            applicant_name=str(
                data.get("applicant_name") or "local-review-client"
            ).strip(),
            clinical_evaluation=str(
                data.get("clinical_evaluation") or "not-applicable"
            ).strip(),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        missing = []
        if not self.base_url:
            missing.append("base_url")
        if not self.username:
            missing.append("username")
        if not self.llm_url:
            missing.append("llm.base_url")
        if not self.llm_api_key:
            missing.append("llm.api_key")
        if not self.llm_model:
            missing.append("llm.model")
        if missing:
            raise ValueError(f"Legacy config is missing: {', '.join(missing)}")
        if not self.excel_path.is_file():
            raise FileNotFoundError(f"Excel file not found: {self.excel_path}")
        if not self.pdf_path.is_file():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
