"""Utilities for loading and preparing long PDF documents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


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

