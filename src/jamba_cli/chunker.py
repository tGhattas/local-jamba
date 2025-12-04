"""Utilities for splitting crawled documentation pages into dense chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .crawler import CrawledPage
from .settings import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE


@dataclass(slots=True)
class Chunk:
    """A fixed-size slice of documentation text ready for embeddings."""

    id: str
    url: str
    title: str
    content: str
    page_index: int
    chunk_index: int


def chunk_pages(
    pages: Sequence[CrawledPage],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split crawled pages into overlapping chunks."""
    all_chunks: list[Chunk] = []
    for page_idx, page in enumerate(pages):
        sections = list(_chunk_text(page.content, chunk_size, overlap))
        for chunk_idx, section in enumerate(sections):
            chunk_id = f"{page_idx:05d}-{chunk_idx:04d}"
            payload = f"{page.title}\nURL: {page.url}\n\n{section}".strip()
            all_chunks.append(
                Chunk(
                    id=chunk_id,
                    url=page.url,
                    title=page.title,
                    content=payload,
                    page_index=page_idx,
                    chunk_index=chunk_idx,
                )
            )
    return all_chunks


def _chunk_text(text: str, chunk_size: int, overlap: int) -> Iterable[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []

    tokens = normalized.split(" ")
    if not tokens:
        return []

    step = max(1, chunk_size - overlap)
    index = 0
    chunks: list[str] = []
    while index < len(tokens):
        window = tokens[index : index + chunk_size]
        chunk = " ".join(window).strip()
        if chunk:
            chunks.append(chunk)
        index += step
    return chunks

