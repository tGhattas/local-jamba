"""Persistence helpers for FAISS indexes and crawled pages."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import faiss  # type: ignore[import-not-found]
import numpy as np

from .chunker import Chunk
from .crawler import CrawledPage
from .document import WebDocumentContent
from .settings import INDEX_DIR


@dataclass(slots=True)
class IndexMetadata:
    slug: str
    url: str
    page_count: int
    chunk_count: int
    embedding_dim: int
    created_at: str


@dataclass(slots=True)
class LoadedIndex:
    slug: str
    url: str
    index: faiss.Index
    chunks: list[Chunk]
    pages: list[CrawledPage]
    metadata: IndexMetadata

    def document(self) -> WebDocumentContent:
        text = "\n\n".join(_format_page(page) for page in self.pages).strip()
        return WebDocumentContent(
            url=self.url,
            pages=self.pages,
            text=text,
            page_count=len(self.pages),
        )


class IndexStore:
    def __init__(self, root_dir: Path | None = None) -> None:
        self.root = (root_dir or INDEX_DIR).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths
    def slugify(self, source: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", source.lower())
        slug = slug.strip("-")
        return slug or "index"

    def _folder(self, slug: str) -> Path:
        return self.root / slug

    # ------------------------------------------------------------------ CRUD
    def exists(self, slug: str) -> bool:
        return self._folder(slug).exists()

    def save(
        self,
        slug: str,
        *,
        url: str,
        pages: Sequence[CrawledPage],
        chunks: Sequence[Chunk],
        embeddings: np.ndarray,
    ) -> LoadedIndex:
        if not chunks:
            raise ValueError("Cannot persist index without chunks.")

        folder = self._folder(slug)
        folder.mkdir(parents=True, exist_ok=True)

        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        faiss.write_index(index, str(folder / "index.faiss"))

        _write_json(folder / "meta.json", _meta_payload(slug, url, pages, chunks, dim))
        _write_json(folder / "chunks.json", [_chunk_payload(chunk) for chunk in chunks])
        _write_json(folder / "pages.json", [_page_payload(page) for page in pages])

        return self.load(slug)

    def load(self, slug: str) -> LoadedIndex:
        folder = self._folder(slug)
        if not folder.exists():
            raise FileNotFoundError(f"Index '{slug}' not found in {self.root}")

        index = faiss.read_index(str(folder / "index.faiss"))
        meta_dict = _read_json(folder / "meta.json")
        metadata = IndexMetadata(**meta_dict)
        chunks = [_chunk_from_dict(obj) for obj in _read_json(folder / "chunks.json")]
        pages = [_page_from_dict(obj) for obj in _read_json(folder / "pages.json")]
        return LoadedIndex(
            slug=slug,
            url=metadata.url,
            index=index,
            chunks=chunks,
            pages=pages,
            metadata=metadata,
        )

    def list_metadata(self) -> list[IndexMetadata]:
        results: list[IndexMetadata] = []
        for folder in sorted(self.root.glob("*")):
            meta_file = folder / "meta.json"
            if not meta_file.exists():
                continue
            meta_dict = _read_json(meta_file)
            try:
                results.append(IndexMetadata(**meta_dict))
            except TypeError:
                continue
        return results

    def delete(self, slug: str) -> bool:
        folder = self._folder(slug)
        if not folder.exists():
            return False
        shutil.rmtree(folder)
        return True

    def search(
        self,
        loaded: LoadedIndex,
        query_vectors: np.ndarray,
        *,
        top_k: int,
    ) -> list[tuple[Chunk, float]]:
        if query_vectors.ndim != 2:
            raise ValueError("Query vectors must be 2D.")
        faiss.normalize_L2(query_vectors)
        scores, indices = loaded.index.search(query_vectors, top_k)

        hits: list[tuple[Chunk, float]] = []
        for rank, chunk_idx in enumerate(indices[0]):
            if chunk_idx < 0 or chunk_idx >= len(loaded.chunks):
                continue
            hits.append((loaded.chunks[chunk_idx], float(scores[0][rank])))
        return hits


# --------------------------------------------------------------------------- helpers
def _meta_payload(
    slug: str,
    url: str,
    pages: Sequence[CrawledPage],
    chunks: Sequence[Chunk],
    dim: int,
) -> dict[str, object]:
    return {
        "slug": slug,
        "url": url,
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "embedding_dim": dim,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _chunk_payload(chunk: Chunk) -> dict[str, object]:
    return asdict(chunk)


def _chunk_from_dict(data: dict[str, object]) -> Chunk:
    return Chunk(
        id=str(data["id"]),
        url=str(data["url"]),
        title=str(data["title"]),
        content=str(data["content"]),
        page_index=int(data["page_index"]),
        chunk_index=int(data["chunk_index"]),
    )


def _page_payload(page: CrawledPage) -> dict[str, str]:
    return {"url": page.url, "title": page.title, "content": page.content}


def _page_from_dict(data: dict[str, str]) -> CrawledPage:
    return CrawledPage(url=data["url"], title=data["title"], content=data["content"])


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _read_json(path: Path) -> list | dict:
    return json.loads(path.read_text())


def _format_page(page: CrawledPage) -> str:
    header = f"### {page.title}\nURL: {page.url}"
    return f"{header}\n\n{page.content.strip()}"

