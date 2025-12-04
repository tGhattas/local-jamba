"""High-level orchestration for retrieval-augmented generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from rich.console import Console  # type: ignore[import-not-found]

from .chunker import Chunk, chunk_pages
from .document import WebDocumentContent
from .embeddings import EmbeddingClient
from .store import IndexMetadata, IndexStore, LoadedIndex


@dataclass(slots=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


class RAGSession:
    """Wrapper that manages FAISS indexes and retrieval for a single chat."""

    def __init__(
        self,
        *,
        store: IndexStore,
        embedder: EmbeddingClient,
        console: Console | None = None,
        top_k: int,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.console = console
        self.top_k = top_k
        self.loaded: LoadedIndex | None = None
        self.slug: str | None = None

    def build_for_document(self, slug: str, document: WebDocumentContent) -> LoadedIndex:
        self.slug = slug
        if self.console:
            self.console.print(
                f"[meta]Building RAG index '{slug}' "
                f"({document.page_count} pages)…[/meta]"
            )
        chunks = chunk_pages(document.pages)
        embeddings = self.embedder.embed([chunk.content for chunk in chunks])
        if embeddings.size == 0:
            raise ValueError("No text chunks generated for indexing.")
        self.loaded = self.store.save(
            slug,
            url=document.url,
            pages=document.pages,
            chunks=chunks,
            embeddings=embeddings,
        )
        return self.loaded

    def attach_loaded(self, slug: str) -> LoadedIndex:
        self.slug = slug
        self.loaded = self.store.load(slug)
        if self.console:
            self.console.print(
                f"[meta]Loaded cached index '{slug}' "
                f"({self.loaded.metadata.chunk_count} chunks).[/meta]"
            )
        return self.loaded

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        if not self.loaded:
            return []
        query_vec = self.embedder.embed([query])
        if query_vec.size == 0:
            return []
        hits = self.store.search(self.loaded, query_vec, top_k=self.top_k)
        return [RetrievedChunk(chunk=chunk, score=score) for chunk, score in hits]

    def list_indexes(self) -> list[IndexMetadata]:
        return self.store.list_metadata()

    def delete_index(self, slug: str) -> bool:
        removed = self.store.delete(slug)
        if removed and self.slug == slug:
            self.loaded = None
            self.slug = None
        return removed

