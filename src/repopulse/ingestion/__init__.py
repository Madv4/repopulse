"""GitHub document normalization, chunking and local vector ingestion."""

from repopulse.ingestion.chunking import FixedChunker, StructureChunker
from repopulse.ingestion.models import Chunk, SourceComment, SourceDocument

__all__ = ["Chunk", "FixedChunker", "SourceComment", "SourceDocument", "StructureChunker"]
