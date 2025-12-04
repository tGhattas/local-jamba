"""Centralized runtime defaults for the Jamba CLI."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = Path.home() / ".jambashrimp"
_REPO_ASSETS = REPO_ROOT / "assets"
_CACHE_ASSETS = CACHE_DIR / "assets"
if _REPO_ASSETS.exists():
    ASSETS_DIR = _REPO_ASSETS
else:
    ASSETS_DIR = _CACHE_ASSETS
DOCS_DIR = ASSETS_DIR / "docs"
MODELS_DIR = ASSETS_DIR / "models"
INDEX_DIR = CACHE_DIR / "indexes"

for path in (DOCS_DIR, MODELS_DIR, INDEX_DIR):
    path.mkdir(parents=True, exist_ok=True)

DEFAULT_PDF_URL = (
    "https://www.oecd.org/content/dam/oecd/en/publications/reports/2025/09/"
    "key-findings-and-integration-strategies-on-the-impact-of-digital-technologies-on-students-learning_fad2ee0b/"
    "ab309c32-en.pdf"
)
DEFAULT_DOC_URL = "https://docs.ray.io/"
DEFAULT_DOC_PATH = DOCS_DIR / "oecd-digital-learning.pdf"
DEFAULT_MODEL_REPO = "ai21labs/AI21-Jamba-Reasoning-3B-GGUF"
DEFAULT_MODEL_FILE = "jamba-reasoning-3b-Q4_K_M.gguf"
DEFAULT_MODEL_PATH = MODELS_DIR / DEFAULT_MODEL_FILE

DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_CTX_LEN = 200_000
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95
DEFAULT_HISTORY_TURNS = 0
DEFAULT_MAX_CRAWL_PAGES = 0  # 0 = unlimited
DEFAULT_CRAWL_DEPTH = 3
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 250
DEFAULT_RAG_TOP_K = 5
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SYSTEM_PROMPT = (
    "You are Jamba, a witty but kind research assistant. You always ground your "
    "answers in the provided document, cite sections when possible, and keep the "
    "tone playful without being distracting. If the user asks something that is "
    "not in the document, gracefully say so."
)

CODING_SYSTEM_PROMPT = (
    "You are Jamba, an upbeat coding copilot grounded in the provided "
    "documentation. Offer concise explanations, cite the most relevant sections, "
    "and include runnable snippets when helpful. When the docs do not contain the "
    "answer, clearly state that and offer the closest related guidance instead."
)

