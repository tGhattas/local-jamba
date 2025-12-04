## JambaShrimp (`jash`)

Playful command-line interface for chatting with any PDF or crawled documentation site using AI21's Jamba Reasoning 3B (GGUF build). The CLI can:

- load entire PDFs directly into the prompt for ad-hoc question answering
- crawl documentation sites (recursively) and ground answers with a local FAISS index (RAG)
- cache indexes per URL so you can come back later—even offline—and keep chatting

The product name is **JambaShrimp**, and its CLI entry point is the shortened `jash`.

### Requirements

- [uv](https://docs.astral.sh/uv/latest/) for environment management
- macOS with Metal (recommended) or CPU fallback
- Optional `HF_TOKEN` environment variable if your Hugging Face account is required for downloads

### Install with `pipx`

Publishing the package (named `jambashrimp`) to PyPI lets anyone install the CLI with an isolated virtual environment:

```bash
pipx install jambashrimp

# run it
jash --help

# download the sample PDF + model
jash-setup
```

While developing locally you can point pipx directly at the repo:

```bash
pipx install --spec . jambashrimp
```

Assets are stored under `~/.jambashrimp/assets` when the package is installed globally. You can override `--model`/`--pdf` if you keep files elsewhere.

### Dev Setup

```bash
uv sync
uv run scripts/setup_assets.py     # downloads the default OECD PDF + GGUF model
uv run python -m jamba_cli.cli
```

Pass `--pdf` or `--model` to point at different files. Run `uv run jamba-chat --help` to see all knobs (context window, temperature, GPU layers, etc.).

### CLI Shortcuts

- `/quit` or `/exit` – leave the chat
- `/history` – print the running conversation
- `/reload` – reload the PDF from disk (useful when editing)
- `/help` – show the command list

By default the CLI hides `<think>` traces but still streams answers token-by-token. Useful switches:

- `--history-turns N` – keep N previous Q/A pairs inside the prompt (defaults to 0 to maximize context for the document itself).
- `--cache-prompt` – reuse the KV cache between turns if you *really* need faster follow ups. Leave it off (default) if you encounter `llama_decode returned -1`; clearing the cache between turns prevents those context overflows.
- `--show-thinking` / `--no-stream` – reveal hidden reasoning traces or print whole answers at once.
