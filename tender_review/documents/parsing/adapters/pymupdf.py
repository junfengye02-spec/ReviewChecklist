from __future__ import annotations

import re
from statistics import median

import fitz

from tender_review.shared.contracts import ensure_call_active

from ..models import (
    BlockSource,
    BoundingBox,
    PageExtraction,
    PageImageRequest,
    PageParseStatus,
    PageQuality,
    ParseArtifact,
    ParseFailure,
    ParseRequest,
    ParsedBlock,
    ParserDescriptor,
    RenderedPage,
    TableCell,
    TableRegion,
    build_parse_artifact,
    build_parsed_document,
    sha256_bytes,
    sha256_text,
    stable_sha256,
)


_SECTION_NUMBER = re.compile(r"^(?:第[一二三四五六七八九十百千0-9]+[章节条]|\d+(?:\.\d+){0,5})")


class PyMuPDFStructuredParser:
    """PyMuPDF adapter that preserves page-local source locations."""

    name = "pymupdf-structured-parser"
    version = str(fitz.VersionBind)

    def __init__(self, *, min_text_characters: int = 20, render_dpi: int = 200) -> None:
        if min_text_characters < 1:
            raise ValueError("min_text_characters must be positive")
        if not 72 <= render_dpi <= 600:
            raise ValueError("render_dpi must be between 72 and 600")
        self.min_text_characters = min_text_characters
        self.render_dpi = render_dpi
        config = {
            "min_text_characters": min_text_characters,
            "reading_order": "pymupdf_get_text_dict_sort",
            "render_dpi": render_dpi,
            "table_detection": "page_find_tables",
        }
        self.descriptor = ParserDescriptor(
            name=self.name,
            version=self.version,
            config_sha256=stable_sha256(config),
        )

    def parse(self, request: ParseRequest) -> ParseArtifact:
        ensure_call_active(request.call)
        try:
            document = fitz.open(stream=request.pdf_bytes, filetype="pdf")
        except Exception as exc:
            raise ValueError("PDF could not be opened by PyMuPDF") from exc

        try:
            pages = tuple(
                self._extract_page(page, page_number=index + 1)
                for index, page in enumerate(document)
            )
        finally:
            document.close()

        parsed = build_parsed_document(
            document_id=request.document_id,
            document_sha256=request.document_sha256,
            parser=self.descriptor,
            pages=pages,
        )
        qualities = tuple(
            PageQuality(
                page_number=page.page_number,
                status=page.status,
                initial_status=page.status,
                text_characters=page.text_characters,
                ocr_candidate=page.candidate_reason is not None,
                candidate_reason=page.candidate_reason,
                table_warnings=page.table_warnings,
                extraction_failure=page.extraction_failure,
            )
            for page in pages
        )
        return build_parse_artifact(parsed, qualities)

    def render_page(self, request: PageImageRequest) -> RenderedPage:
        ensure_call_active(request.call)
        try:
            document = fitz.open(stream=request.pdf_bytes, filetype="pdf")
            try:
                if request.page_number > document.page_count:
                    raise ValueError(f"page {request.page_number} does not exist")
                page = document.load_page(request.page_number - 1)
                scale = request.dpi / 72
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(scale, scale), alpha=False
                )
                image = pixmap.tobytes("png")
            finally:
                document.close()
        except ValueError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"could not render PDF page {request.page_number}"
            ) from exc
        return RenderedPage(
            document_id=request.document_id,
            document_sha256=request.document_sha256,
            page_number=request.page_number,
            image_png=image,
            image_sha256=sha256_bytes(image),
            renderer=self.descriptor,
        )

    def _extract_page(self, page: fitz.Page, *, page_number: int) -> PageExtraction:
        try:
            width = max(float(page.rect.width), 1.0)
            height = max(float(page.rect.height), 1.0)
        except Exception:
            width = 1.0
            height = 1.0
        try:
            page_dict = page.get_text("dict", sort=True)
            raw_blocks = self._text_blocks(page_dict)
            table_specs, table_warnings = self._table_specs(page, page_number)
            font_sizes = [entry["font_size"] for entry in raw_blocks]
            median_font_size = median(font_sizes) if font_sizes else 0.0
            blocks = tuple(
                self._build_block(
                    entry,
                    page_number=page_number,
                    reading_order=index,
                    median_font_size=median_font_size,
                    table_specs=table_specs,
                )
                for index, entry in enumerate(raw_blocks)
            )
            tables = self._build_tables(
                table_specs,
                page_number=page_number,
                blocks=blocks,
            )
            text = "\n".join(block.text for block in blocks)
            text_characters = sum(not character.isspace() for character in text)
            status, reason = self._page_status(text, text_characters)
            return PageExtraction(
                page_number=page_number,
                width=width,
                height=height,
                blocks=blocks,
                tables=tables,
                text_sha256=sha256_text(text),
                text_characters=text_characters,
                status=status,
                candidate_reason=reason,
                table_detection_attempted=True,
                table_warnings=tuple(table_warnings),
            )
        except Exception as exc:
            failure = ParseFailure(
                stage="page_text_extraction",
                error_type=type(exc).__name__,
                message=str(exc)[:2048] or "PyMuPDF failed to extract the page",
            )
            return PageExtraction(
                page_number=page_number,
                width=width,
                height=height,
                text_sha256=sha256_text(""),
                text_characters=0,
                status=PageParseStatus.EXTRACTION_FAILED,
                candidate_reason="page_text_extraction_failed",
                extraction_failure=failure,
                table_warnings=("table detection was skipped after page extraction failed",),
            )

    @staticmethod
    def _text_blocks(page_dict: dict[str, object]) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        for block in page_dict.get("blocks", []):
            if not isinstance(block, dict) or block.get("type") != 0:
                continue
            lines = block.get("lines", [])
            text_lines: list[str] = []
            font_sizes: list[float] = []
            font_names: list[str] = []
            for line in lines if isinstance(lines, list) else []:
                if not isinstance(line, dict):
                    continue
                spans = line.get("spans", [])
                line_text = "".join(
                    str(span.get("text", ""))
                    for span in spans
                    if isinstance(span, dict)
                )
                if line_text:
                    text_lines.append(line_text)
                for span in spans if isinstance(spans, list) else []:
                    if isinstance(span, dict):
                        font_sizes.append(float(span.get("size", 0.0)))
                        font_names.append(str(span.get("font", "")))
            text = "\n".join(text_lines).strip()
            bbox = block.get("bbox")
            if not text or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            entries.append(
                {
                    "bbox": BoundingBox(
                        x0=float(bbox[0]),
                        y0=float(bbox[1]),
                        x1=float(bbox[2]),
                        y1=float(bbox[3]),
                    ),
                    "font_name": next((name for name in font_names if name), None),
                    "font_size": max(font_sizes, default=0.0) or None,
                    "text": text,
                }
            )
        return entries

    def _build_block(
        self,
        entry: dict[str, object],
        *,
        page_number: int,
        reading_order: int,
        median_font_size: float,
        table_specs: list[dict[str, object]],
    ) -> ParsedBlock:
        text = str(entry["text"])
        font_size = entry["font_size"]
        heading_level = self._heading_level(text, font_size, median_font_size)
        table_id = next(
            (
                str(table["table_id"])
                for table in table_specs
                if self._overlaps(entry["bbox"], table["bbox"])
            ),
            None,
        )
        digest = sha256_text(text)
        return ParsedBlock(
            block_id=f"p{page_number}:b{reading_order}:{digest[:16]}",
            page_number=page_number,
            reading_order=reading_order,
            text=text,
            text_sha256=digest,
            bbox=entry["bbox"],
            source=BlockSource.PDF_TEXT,
            font_name=entry["font_name"],
            font_size=font_size,
            title_candidate=heading_level is not None,
            heading_level=heading_level,
            table_id=table_id,
        )

    @staticmethod
    def _heading_level(
        text: str,
        font_size: object,
        median_font_size: float,
    ) -> int | None:
        normalized = " ".join(text.split())
        if len(normalized) > 120:
            return None
        numbered = _SECTION_NUMBER.match(normalized)
        if numbered:
            marker = numbered.group(0)
            if "." in marker:
                return min(marker.count(".") + 1, 6)
            return 1 if "章" in marker else 2
        if isinstance(font_size, float) and font_size >= median_font_size + 1.0:
            return 2
        return None

    def _page_status(
        self, text: str, text_characters: int
    ) -> tuple[PageParseStatus, str | None]:
        if text_characters == 0:
            return PageParseStatus.EMPTY_TEXT, "no_extractable_text"
        has_control_characters = any(
            ord(character) < 32 and not character.isspace() for character in text
        )
        if "\ufffd" in text or has_control_characters:
            return PageParseStatus.ANOMALOUS_TEXT, "anomalous_text_layer"
        if text_characters < self.min_text_characters:
            return PageParseStatus.LOW_TEXT, "below_text_threshold"
        return PageParseStatus.OK, None

    def _table_specs(
        self, page: fitz.Page, page_number: int
    ) -> tuple[list[dict[str, object]], list[str]]:
        try:
            finder = page.find_tables()
            tables = getattr(finder, "tables", [])
        except Exception as exc:
            return [], [f"PyMuPDF table detection failed: {type(exc).__name__}"]

        specs: list[dict[str, object]] = []
        warnings: list[str] = []
        for table_index, table in enumerate(tables):
            try:
                bbox_values = table.bbox
                bbox = BoundingBox(
                    x0=float(bbox_values[0]),
                    y0=float(bbox_values[1]),
                    x1=float(bbox_values[2]),
                    y1=float(bbox_values[3]),
                )
                rows = table.extract()
                cells: list[TableCell] = []
                for row_index, row in enumerate(rows):
                    for column_index, cell_text in enumerate(row):
                        text = str(cell_text or "")
                        cell_bbox = self._table_cell_bbox(table, row_index, column_index)
                        cells.append(
                            TableCell(
                                row_index=row_index,
                                column_index=column_index,
                                text=text,
                                text_sha256=sha256_text(text),
                                bbox=cell_bbox,
                            )
                        )
                specs.append(
                    {
                        "bbox": bbox,
                        "cells": tuple(cells),
                        "table_id": f"p{page_number}:t{table_index}",
                    }
                )
            except Exception as exc:
                warnings.append(
                    f"table {table_index} extraction failed: {type(exc).__name__}"
                )
        return specs, warnings

    @staticmethod
    def _table_cell_bbox(
        table: object, row_index: int, column_index: int
    ) -> BoundingBox | None:
        rows = getattr(table, "rows", [])
        if row_index >= len(rows):
            return None
        cells = getattr(rows[row_index], "cells", [])
        if column_index >= len(cells) or cells[column_index] is None:
            return None
        values = cells[column_index]
        return BoundingBox(
            x0=float(values[0]),
            y0=float(values[1]),
            x1=float(values[2]),
            y1=float(values[3]),
        )

    @staticmethod
    def _build_tables(
        table_specs: list[dict[str, object]],
        *,
        page_number: int,
        blocks: tuple[ParsedBlock, ...],
    ) -> tuple[TableRegion, ...]:
        tables: list[TableRegion] = []
        for spec in table_specs:
            bbox = spec["bbox"]
            block_ids = tuple(
                block.block_id
                for block in blocks
                if block.bbox is not None and PyMuPDFStructuredParser._overlaps(block.bbox, bbox)
            )
            cells = spec["cells"]
            tables.append(
                TableRegion(
                    table_id=str(spec["table_id"]),
                    page_number=page_number,
                    bbox=bbox,
                    cells=cells,
                    source_block_ids=block_ids,
                    extraction_method="pymupdf_page_find_tables",
                    text_sha256=sha256_text(
                        "\n".join(cell.text for cell in cells)
                    ),
                )
            )
        return tuple(tables)

    @staticmethod
    def _overlaps(left: object, right: object) -> bool:
        if not isinstance(left, BoundingBox) or not isinstance(right, BoundingBox):
            return False
        return (
            max(left.x0, right.x0) < min(left.x1, right.x1)
            and max(left.y0, right.y0) < min(left.y1, right.y1)
        )
