from __future__ import annotations

from dataclasses import dataclass

from .models import (
    ChunkBlockReference,
    ChunkSet,
    DocumentChunk,
    ParseArtifact,
    ParsedBlock,
    StrategyDescriptor,
    TableRegion,
    sha256_text,
    stable_sha256,
)


@dataclass(frozen=True)
class _Piece:
    text: str
    page_number: int
    source_blocks: tuple[ChunkBlockReference, ...]
    table_ids: tuple[str, ...] = ()


class StructuralChunker:
    """Bounded chunks that prefer section and table boundaries over fixed windows."""

    name = "structural-pdf-chunker"
    version = "1"

    def __init__(self, *, max_characters: int = 1600) -> None:
        if max_characters < 100:
            raise ValueError("max_characters must be at least 100")
        self.max_characters = max_characters
        self.descriptor = StrategyDescriptor(
            name=self.name,
            version=self.version,
            config_sha256=stable_sha256({"max_characters": max_characters}),
        )

    def chunk(self, artifact: ParseArtifact) -> ChunkSet:
        section_path: tuple[str, ...] = ()
        current: list[_Piece] = []
        chunks: list[DocumentChunk] = []
        current_section: tuple[str, ...] = ()
        tables = {
            table.table_id: table
            for page in artifact.document.pages
            for table in page.tables
        }
        blocks_by_id = {
            block.block_id: block
            for page in artifact.document.pages
            for block in page.blocks
        }
        seen_tables: set[str] = set()

        for page in artifact.document.pages:
            for block in page.blocks:
                if block.title_candidate:
                    if current:
                        chunks.append(
                            self._build_chunk(
                                artifact,
                                sequence=len(chunks),
                                section_path=current_section,
                                pieces=current,
                            )
                        )
                        current = []
                    section_path = self._next_section_path(section_path, block)
                    current_section = section_path
                    remainder = self._heading_remainder_piece(block)
                    if remainder is not None:
                        for bounded_piece in self._split_piece(remainder):
                            if current:
                                chunks.append(
                                    self._build_chunk(
                                        artifact,
                                        sequence=len(chunks),
                                        section_path=current_section,
                                        pieces=current,
                                    )
                                )
                                current = []
                            current.append(bounded_piece)
                    continue
                pieces = self._pieces_for_block(
                    block,
                    tables,
                    seen_tables,
                    blocks_by_id,
                )
                for piece in pieces:
                    for bounded_piece in self._split_piece(piece):
                        projected = (
                            sum(len(item.text) for item in current)
                            + len(current)
                            + len(bounded_piece.text)
                        )
                        if current and projected > self.max_characters:
                            chunks.append(
                                self._build_chunk(
                                    artifact,
                                    sequence=len(chunks),
                                    section_path=current_section,
                                    pieces=current,
                                )
                            )
                            current = []
                        current.append(bounded_piece)

            for table in page.tables:
                if table.table_id in seen_tables:
                    continue
                seen_tables.add(table.table_id)
                for piece in self._table_pieces(table, blocks_by_id):
                    for bounded_piece in self._split_piece(piece):
                        projected = (
                            sum(len(item.text) for item in current)
                            + len(current)
                            + len(bounded_piece.text)
                        )
                        if current and projected > self.max_characters:
                            chunks.append(
                                self._build_chunk(
                                    artifact,
                                    sequence=len(chunks),
                                    section_path=current_section,
                                    pieces=current,
                                )
                            )
                            current = []
                        current.append(bounded_piece)

        if current:
            chunks.append(
                self._build_chunk(
                    artifact,
                    sequence=len(chunks),
                    section_path=current_section,
                    pieces=current,
                )
            )
        payload = {
            "schema_version": 1,
            "document_id": artifact.document.document_id,
            "document_sha256": artifact.document.document_sha256,
            "parser": artifact.document.parser.model_dump(mode="json"),
            "strategy": self.descriptor.model_dump(mode="json"),
            "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
        }
        return ChunkSet(
            **payload,
            chunk_set_sha256=stable_sha256(payload),
        )

    def _pieces_for_block(
        self,
        block: ParsedBlock,
        tables: dict[str, TableRegion],
        seen_tables: set[str],
        blocks_by_id: dict[str, ParsedBlock],
    ) -> tuple[_Piece, ...]:
        if block.table_id is None:
            return (self._block_piece(block),)
        if block.table_id in seen_tables:
            return ()
        table = tables.get(block.table_id)
        if table is None:
            return (self._block_piece(block),)
        seen_tables.add(table.table_id)
        return tuple(self._table_pieces(table, blocks_by_id))

    @staticmethod
    def _block_piece(block: ParsedBlock) -> _Piece:
        return _Piece(
            text=block.text,
            page_number=block.page_number,
            source_blocks=(
                ChunkBlockReference(
                    block_id=block.block_id,
                    page_number=block.page_number,
                    text_start=0,
                    text_end=len(block.text),
                    bbox=block.bbox,
                ),
            ),
        )

    @staticmethod
    def _heading_remainder_piece(block: ParsedBlock) -> _Piece | None:
        first_line, separator, _ = block.text.partition("\n")
        if not separator:
            return None
        start = len(first_line) + len(separator)
        while start < len(block.text) and block.text[start].isspace():
            start += 1
        if start >= len(block.text):
            return None
        return _Piece(
            text=block.text[start:],
            page_number=block.page_number,
            source_blocks=(
                ChunkBlockReference(
                    block_id=block.block_id,
                    page_number=block.page_number,
                    text_start=start,
                    text_end=len(block.text),
                    bbox=block.bbox,
                ),
            ),
        )

    def _table_pieces(
        self,
        table: TableRegion,
        blocks_by_id: dict[str, ParsedBlock],
    ) -> list[_Piece]:
        rows = self._table_rows(table)
        source_blocks = tuple(
            ChunkBlockReference(
                block_id=block.block_id,
                page_number=block.page_number,
                text_start=0,
                text_end=len(block.text),
                bbox=block.bbox,
            )
            for block_id in table.source_block_ids
            if (block := blocks_by_id.get(block_id)) is not None
        )
        if not rows:
            return []
        full_text = "\n".join(rows)
        if len(full_text) <= self.max_characters:
            return [
                _Piece(
                    text=full_text,
                    page_number=table.page_number,
                    source_blocks=source_blocks,
                    table_ids=(table.table_id,),
                )
            ]

        header = rows[0]
        groups: list[_Piece] = []
        group_rows = [header]
        current_length = len(header)
        for row in rows[1:]:
            next_length = current_length + 1 + len(row)
            if len(group_rows) > 1 and next_length > self.max_characters:
                groups.append(
                    _Piece(
                        text="\n".join(group_rows),
                        page_number=table.page_number,
                        source_blocks=source_blocks,
                        table_ids=(table.table_id,),
                    )
                )
                group_rows = [header, row]
                current_length = len(header) + 1 + len(row)
            else:
                group_rows.append(row)
                current_length = next_length
        if group_rows:
            groups.append(
                _Piece(
                    text="\n".join(group_rows),
                    page_number=table.page_number,
                    source_blocks=source_blocks,
                    table_ids=(table.table_id,),
                )
            )
        return groups

    @staticmethod
    def _table_rows(table: TableRegion) -> list[str]:
        cells_by_row: dict[int, list[tuple[int, str]]] = {}
        for cell in table.cells:
            cells_by_row.setdefault(cell.row_index, []).append(
                (cell.column_index, cell.text)
            )
        return [
            "\t".join(text for _, text in sorted(cells))
            for _, cells in sorted(cells_by_row.items())
        ]

    def _split_piece(self, piece: _Piece) -> tuple[_Piece, ...]:
        if len(piece.text) <= self.max_characters:
            return (piece,)
        pieces: list[_Piece] = []
        start = 0
        while start < len(piece.text):
            end = min(start + self.max_characters, len(piece.text))
            if end < len(piece.text):
                breakpoint = max(
                    piece.text.rfind("\n", start + 1, end + 1),
                    piece.text.rfind(" ", start + 1, end + 1),
                )
                if breakpoint > start:
                    end = breakpoint
            segment = piece.text[start:end]
            if not segment:
                end = min(start + self.max_characters, len(piece.text))
                segment = piece.text[start:end]
            pieces.append(
                _Piece(
                    text=segment,
                    page_number=piece.page_number,
                    source_blocks=self._slice_references(
                        piece.source_blocks,
                        start=start,
                        end=end,
                        source_text_length=len(piece.text),
                    ),
                    table_ids=piece.table_ids,
                )
            )
            start = end
            while start < len(piece.text) and piece.text[start] in "\n ":
                start += 1
        return tuple(pieces)

    @staticmethod
    def _slice_references(
        references: tuple[ChunkBlockReference, ...],
        *,
        start: int,
        end: int,
        source_text_length: int,
    ) -> tuple[ChunkBlockReference, ...]:
        if len(references) != 1 or source_text_length == 0:
            return references
        reference = references[0]
        if reference.text_end - reference.text_start != source_text_length:
            return references
        return (
            reference.model_copy(
                update={
                    "text_start": reference.text_start + start,
                    "text_end": reference.text_start + end,
                }
            ),
        )

    def _build_chunk(
        self,
        artifact: ParseArtifact,
        *,
        sequence: int,
        section_path: tuple[str, ...],
        pieces: list[_Piece],
    ) -> DocumentChunk:
        raw_text = "\n".join(piece.text for piece in pieces)
        source_blocks = _deduplicate_references(
            reference for piece in pieces for reference in piece.source_blocks
        )
        table_ids = tuple(
            dict.fromkeys(table_id for piece in pieces for table_id in piece.table_ids)
        )
        digest = sha256_text(raw_text)
        return DocumentChunk(
            chunk_id=(
                f"{artifact.document.document_id}:chunk:{sequence}:{digest[:16]}"
            ),
            document_id=artifact.document.document_id,
            document_sha256=artifact.document.document_sha256,
            parser=artifact.document.parser,
            strategy=self.descriptor,
            page_start=min(piece.page_number for piece in pieces),
            page_end=max(piece.page_number for piece in pieces),
            section_path=section_path,
            raw_text=raw_text,
            text_sha256=digest,
            source_blocks=source_blocks,
            table_ids=table_ids,
        )

    @staticmethod
    def _next_section_path(
        current: tuple[str, ...], block: ParsedBlock) -> tuple[str, ...]:
        level = block.heading_level or 1
        title = " ".join(block.text.splitlines()[0].split())
        return (*current[: level - 1], title)


def _deduplicate_references(
    references: object,
) -> tuple[ChunkBlockReference, ...]:
    result: list[ChunkBlockReference] = []
    seen: set[tuple[str, int, int]] = set()
    for reference in references:
        if not isinstance(reference, ChunkBlockReference):
            continue
        key = (reference.block_id, reference.text_start, reference.text_end)
        if key not in seen:
            seen.add(key)
            result.append(reference)
    return tuple(result)
