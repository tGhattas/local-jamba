"""Utilities for loading and preparing documents from PDFs or URLs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pypdf import PdfReader

from .crawler import CrawledPage, DocumentationCrawler

ProgressCallback = Callable[[int, int | None], None]


@dataclass(slots=True)
class DocumentContent:
    path: Path
    text: str
    page_count: int

    @property
    def characters(self) -> int:
        return len(self.text)

    def preview(self, chars: int = 320) -> str:
        snippet = self.text[:chars].strip().replace("\n", " ")
        return snippet + ("…" if len(self.text) > chars else "")


def load_document(pdf_path: Path) -> DocumentContent:
    """Load a PDF into a single concatenated text blob."""
    path = Path(pdf_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF not found at {path}")

    reader = PdfReader(str(path))
    pages: list[str] = []

    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to extract text from page {idx}") from exc
        cleaned = text.strip()
        if cleaned:
            pages.append(cleaned)

    merged = "\n\n".join(pages).strip()
    if not merged:
        raise ValueError(f"No text extracted from {path}")

    return DocumentContent(path=path, text=merged, page_count=len(reader.pages))


@dataclass(slots=True)
class WebDocumentContent:
    url: str
    pages: list[CrawledPage]
    text: str
    page_count: int

    @property
    def characters(self) -> int:
        return len(self.text)

    def preview(self, chars: int = 320) -> str:
        snippet = self.text[:chars].strip().replace("\n", " ")
        return snippet + ("…" if len(self.text) > chars else "")


def load_from_url(
    url: str,
    *,
    max_pages: int | None,
    max_depth: int,
    progress_callback: ProgressCallback | None = None,
) -> WebDocumentContent:
    """Crawl a documentation site and merge the content into a single blob."""
    crawler = DocumentationCrawler(
        url,
        max_pages=max_pages,
        max_depth=max_depth,
        progress_callback=progress_callback,
    )
    pages = crawler.crawl()
    if not pages:
        raise ValueError(f"No crawlable HTML content found at {url}")

    merged = "\n\n".join(_format_page(page) for page in pages).strip()
    if not merged:
        raise ValueError("Crawler returned pages without textual content.")

    return WebDocumentContent(
        url=url,
        pages=pages,
        text=merged,
        page_count=len(pages),
    )


def _format_page(page: CrawledPage) -> str:
    header = f"### {page.title}\nURL: {page.url}"
    return f"{header}\n\n{page.content.strip()}"

