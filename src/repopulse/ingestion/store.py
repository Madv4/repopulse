"""Persistent ChromaDB store for precomputed local embeddings."""

import json
from pathlib import Path
from typing import Any, cast

import chromadb
from chromadb.config import Settings as ChromaSettings

from repopulse.ingestion.embed import Embedder
from repopulse.ingestion.models import Chunk, SourceDocument
from repopulse.retrieval.models import SearchHit


class ChromaStore:
    """Keep each chunking strategy in its own persistent collection."""

    def __init__(self, directory: Path, strategy: str) -> None:
        if strategy not in {"fixed", "structure"}:
            raise ValueError("strategy must be fixed or structure")
        self.strategy = strategy
        self.client = chromadb.PersistentClient(
            path=str(directory), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(name=f"repopulse_{strategy}")

    @staticmethod
    def _metadata(chunk: Chunk) -> dict[str, str | int]:
        return {
            "source_id": chunk.source_id,
            "repo": chunk.repo,
            "type": chunk.kind,
            "number": chunk.number if chunk.number is not None else -1,
            "url": chunk.url,
            "author": chunk.author,
            "created_at": chunk.created_at.isoformat(),
            "updated_at": chunk.updated_at.isoformat() if chunk.updated_at else "",
            "labels": json.dumps(chunk.labels, ensure_ascii=False),
            "strategy": chunk.strategy,
            "chunk_index": chunk.index,
        }

    def replace_document(
        self, document: SourceDocument, chunks: list[Chunk], embedder: Embedder
    ) -> int:
        """Replace a source's old chunks so reingestion cannot leave stale text."""
        if any(
            chunk.source_id != document.source_id or chunk.strategy != self.strategy
            for chunk in chunks
        ):
            raise ValueError("chunks must all belong to this source and strategy")
        if not chunks:
            self.collection.delete(where={"source_id": document.source_id})
            return 0
        vectors = embedder.encode([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("embedder returned the wrong number of vectors")
        self.collection.delete(where={"source_id": document.source_id})
        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            embeddings=vectors,
            documents=[chunk.text for chunk in chunks],
            metadatas=[self._metadata(chunk) for chunk in chunks],
        )
        return len(chunks)

    def clear_repo(self, repo: str) -> None:
        """Remove one repository from this collection before a fixture rebuild."""
        self.collection.delete(where={"repo": repo})

    def search(
        self, query: str, embedder: Embedder, *, repo: str, limit: int = 20
    ) -> list[SearchHit]:
        """Run local vector search and return citation-bearing hits."""
        if limit < 1:
            raise ValueError("limit must be positive")
        if self.collection.count() == 0:
            return []
        vector = embedder.encode([query])
        if len(vector) != 1:
            raise ValueError("embedder returned the wrong number of query vectors")
        result = self.collection.query(
            query_embeddings=vector,
            n_results=limit,
            where={"repo": repo},
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        documents = (result["documents"] or [[]])[0]
        metadatas = (result["metadatas"] or [[]])[0]
        distances = (result["distances"] or [[]])[0]
        hits: list[SearchHit] = []
        for chunk_id, text, raw, distance in zip(ids, documents, metadatas, distances, strict=True):
            if text is None or raw is None:
                continue
            metadata = cast(dict[str, Any], raw)
            chunk = Chunk(
                id=chunk_id,
                source_id=str(metadata["source_id"]),
                repo=str(metadata["repo"]),
                kind=str(metadata["type"]),  # type: ignore[arg-type]
                number=int(metadata["number"]) if int(metadata["number"]) >= 0 else None,
                url=str(metadata["url"]),
                author=str(metadata["author"]),
                created_at=str(metadata["created_at"]),  # type: ignore[arg-type]
                updated_at=str(metadata["updated_at"]) or None,  # type: ignore[arg-type]
                labels=json.loads(str(metadata["labels"])),
                strategy=str(metadata["strategy"]),  # type: ignore[arg-type]
                index=int(metadata["chunk_index"]),
                text=text,
            )
            hits.append(SearchHit(chunk=chunk, distance=float(distance)))
        return hits
