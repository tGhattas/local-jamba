"""Download the default GGUF model and sample PDF for JambaShrimp."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

from .settings import (
    DEFAULT_DOC_PATH,
    DEFAULT_MODEL_FILE,
    DEFAULT_MODEL_PATH,
    DEFAULT_MODEL_REPO,
    DEFAULT_PDF_URL,
    DOCS_DIR,
    MODELS_DIR,
)


def download_pdf(url: str, destination: Path, force: bool = False) -> Path:
    """Download the PDF if missing or force is True."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        print(f"[pdf] already present at {destination}")
        return destination

    print(f"[pdf] downloading from {url}")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(destination, "wb") as handle:
            for chunk in response.iter_content(1024 * 256):
                if chunk:
                    handle.write(chunk)
    size_mb = destination.stat().st_size / (1024 * 1024)
    print(f"[pdf] saved {size_mb:.2f} MB to {destination}")
    return destination


def download_model(
    repo_id: str,
    filename: str,
    destination: Path,
    force: bool = False,
    token: str | None = None,
) -> Path:
    """Download the GGUF file via huggingface_hub."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        print(f"[model] already present at {destination}")
        return destination

    print(f"[model] downloading {filename} from {repo_id}")
    downloaded_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
            local_dir=str(destination.parent),
            local_dir_use_symlinks=False,
        )
    )
    if downloaded_path != destination:
        downloaded_path.replace(destination)
    size_gb = destination.stat().st_size / (1024 * 1024 * 1024)
    print(f"[model] saved {size_gb:.2f} GB to {destination}")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-url", default=DEFAULT_PDF_URL, help="Document URL to download.")
    parser.add_argument(
        "--pdf-path",
        default=str(DEFAULT_DOC_PATH),
        help=f"Path to store the PDF (default inside {DOCS_DIR}).",
    )
    parser.add_argument(
        "--model-repo",
        default=DEFAULT_MODEL_REPO,
        help="Hugging Face repository containing the GGUF file.",
    )
    parser.add_argument(
        "--model-file",
        default=DEFAULT_MODEL_FILE,
        help="GGUF filename inside the repository.",
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help=f"Where to store the GGUF model (default inside {MODELS_DIR}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload assets even if they already exist.",
    )
    parser.add_argument(
        "--hf-token-env",
        default="HF_TOKEN",
        help="Environment variable that stores the Hugging Face token (optional).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hf_token = os.getenv(args.hf_token_env)
    pdf_path = download_pdf(args.pdf_url, Path(args.pdf_path), force=args.force)
    model_path = download_model(
        args.model_repo,
        args.model_file,
        Path(args.model_path),
        force=args.force,
        token=hf_token,
    )

    print("\nAssets ready:")
    print(f"  PDF  -> {pdf_path}")
    print(f"  Model -> {model_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

