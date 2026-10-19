"""Deterministic fixed-window and structure-aware chunking strategies."""

import hashlib
import re
from collections.abc import Iterable
from typing import Protocol

from repopulse.ingestion.models import Chunk, SourceDocument


class Chunker(Protocol):
    """Common interface for document chunking strategies."""

    @property
    def name(self) -> str:
        """Stable strategy identifier."""

    def chunk(self, document: SourceDocument) -> list[Chunk]:
        """Split a source document into citation-bearing passages."""


def _build_chunk(document: SourceDocument, strategy: str, index: int, text: str) -> Chunk:
    digest = hashlib.sha256(f"{document.source_id}|{strategy}|{index}|{text}".encode()).hexdigest()
    return Chunk(
        id=digest,
        source_id=document.source_id,
        repo=document.repo,
        kind=document.kind,
        number=document.number,
        url=document.url,
        author=document.author,
        created_at=document.created_at,
        updated_at=document.updated_at,
        labels=document.labels,
        strategy=strategy,  # type: ignore[arg-type]
        index=index,
        text=text,
    )


def _slices(text: str, size: int, overlap: int) -> Iterable[str]:
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end].strip()
        if piece:
            yield piece
        if end == len(text):
            break
        start = end - overlap


class FixedChunker:
    """Split rendered text by character windows with overlap."""

    def __init__(self, size: int = 500, overlap: int = 80) -> None:
        if size < 1 or overlap < 0 or overlap >= size:
            raise ValueError("Require size > 0 and 0 <= overlap < size")
        self.size = size
        self.overlap = overlap

    @property
    def name(self) -> str:
        """Return the fixed strategy identifier."""
        return "fixed"

    def chunk(self, document: SourceDocument) -> list[Chunk]:
        """Chunk the complete source text, including all comments."""
        return [
            _build_chunk(document, self.name, index, text)
            for index, text in enumerate(_slices(document.render_text(), self.size, self.overlap))
        ]


class StructureChunker:
    """Keep Markdown sections and comment blocks together where possible."""

    def __init__(self, max_chars: int = 500) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be positive")
        self.max_chars = max_chars

    @property
    def name(self) -> str:
        """Return the structure-aware strategy identifier."""
        return "structure"

    def _blocks(self, document: SourceDocument) -> list[str]:
        heading = f"{document.kind.replace('_', ' ').title()}: {document.title}"
        sections = re.split(r"(?=^#{1,6}\s+)", document.body.strip(), flags=re.MULTILINE)
        body_blocks = [
            block.strip() for section in sections for block in re.split(r"\n\s*\n", section)
        ]
        blocks = [heading, *(block for block in body_blocks if block)]
        blocks.extend(
            f"Comment by {comment.author} ({comment.url}):\n{comment.body.strip()}"
            for comment in document.comments
            if comment.body.strip()
        )
        return blocks

    def chunk(self, document: SourceDocument) -> list[Chunk]:
        """Pack paragraph-sized blocks, splitting only oversized blocks."""
        passages: list[str] = []
        current = ""
        for block in self._blocks(document):
            if len(block) > self.max_chars:
                if current:
                    passages.append(current)
                    current = ""
                passages.extend(_slices(block, self.max_chars, 0))
                continue
            candidate = f"{current}\n\n{block}" if current else block
            if len(candidate) <= self.max_chars:
                current = candidate
            else:
                passages.append(current)
                current = block
        if current:
            passages.append(current)
        return [
            _build_chunk(document, self.name, index, text) for index, text in enumerate(passages)
        ]
