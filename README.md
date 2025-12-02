## Jamba PDF Chat CLI

Playful command-line interface for chatting with an entire PDF at once using AI21's Jamba Reasoning 3B (GGUF build). The CLI loads the whole document into the prompt (no RAG) to highlight the model's 128K+ context window.

### Requirements

- [uv](https://docs.astral.sh/uv/latest/) for environment management
- macOS with Metal (recommended) or CPU fallback
- Optional `HF_TOKEN` environment variable if your Hugging Face account is required for downloads

### Setup

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
