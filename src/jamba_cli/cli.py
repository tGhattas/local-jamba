"""Playful CLI to chat with PDFs or documentation sites using Jamba Reasoning 3B."""

from __future__ import annotations

import argparse
from argparse import BooleanOptionalAction
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence

from llama_cpp import Llama  # type: ignore[import-not-found]
from rich import box  # type: ignore[import-not-found]
from rich.align import Align  # type: ignore[import-not-found]
from rich.console import Console  # type: ignore[import-not-found]
from rich.live import Live  # type: ignore[import-not-found]
from rich.markdown import Markdown  # type: ignore[import-not-found]
from rich.panel import Panel  # type: ignore[import-not-found]
from rich.progress import (  # type: ignore[import-not-found]
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from prompt_toolkit import PromptSession  # type: ignore[import-not-found]
from prompt_toolkit.completion import (  # type: ignore[import-not-found]
    Completer,
    Completion,
    CompleteEvent,
    PathCompleter,
)
from prompt_toolkit.document import Document as PTDocument  # type: ignore[import-not-found]
from prompt_toolkit.formatted_text import FormattedText  # type: ignore[import-not-found]
from prompt_toolkit.patch_stdout import patch_stdout  # type: ignore[import-not-found]
from prompt_toolkit.styles import Style  # type: ignore[import-not-found]
from rich.prompt import Confirm, IntPrompt, Prompt  # type: ignore[import-not-found]
from rich.table import Table  # type: ignore[import-not-found]
from rich.text import Text  # type: ignore[import-not-found]
from rich.theme import Theme  # type: ignore[import-not-found]

from .document import (
    DocumentContent,
    WebDocumentContent,
    load_document,
    load_from_url,
)
from .embeddings import EmbeddingClient
from .rag import RAGSession, RetrievedChunk
from .store import IndexStore
from .settings import (
    CODING_SYSTEM_PROMPT,
    DEFAULT_CTX_LEN,
    DEFAULT_CRAWL_DEPTH,
    DEFAULT_DOC_URL,
    DEFAULT_DOC_PATH,
    DEFAULT_HISTORY_TURNS,
    DEFAULT_MAX_CRAWL_PAGES,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_PATH,
    DEFAULT_RAG_TOP_K,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    SYSTEM_PROMPT,
)
from .setup_assets import main as setup_assets_main

PLAYFUL_THEME = Theme(
    {
        "info": "bold bright_cyan",
        "warn": "gold3",
        "error": "bold red",
        "assistant": "medium_purple1",
        "assistant.border": "purple3",
        "user": "bright_cyan",
        "meta": "grey66",
        "highlight": "turquoise2",
        "banner.top": "#00c8ff",
        "banner.mid": "#a855f7",
        "banner.bottom": "#7c83fd",
    }
)

BANNER_LINES = (
    "     ___   _   __  __ ___   _      _____ _          _                🦐",
    "    |_  | /_\\ |  \\/  | _ ) /_\\    / ____| |        (_)               ",
    "     | | / _ \\| |\\/| | _ \\/ _ \\   | (___ | |__  _ __ _ _ __ ___  _ __   ",
    "     | |/ ___ \\ |  | |___/ ___ \\   \\___ \\| '_ \\| '__| | '_ ` _ \\| '_ \\  ",
    " /\\__/ /_/   \\_\\_|  |_|____/   \\_\\  ____) | | | | |  | | | | | | | |_) | ",
    " \\____/                           |_____/|_| |_|_|  |_|_| |_| |_| .__/  ",
    "                                                                | |     ",
    "                                                                |_|     ",
)

DocumentLike = DocumentContent | WebDocumentContent
ReloadCallable = Callable[[], DocumentLike]


def print_banner(console: Console) -> None:
    """Print the Jamba Shrimp ASCII art banner."""
    text = Text()
    for idx, line in enumerate(BANNER_LINES):
        color = ["banner.top", "banner.mid", "banner.bottom"][idx % 3]
        text.append(line + "\n", style=color)
    subtitle = Text(
        "Docs Copilot • grounded answers with runnable snippets",
        style="meta",
    )
    console.print(Align.center(text))
    console.print(Align.center(subtitle))
    console.print()


@dataclass(slots=True)
class ConversationTurn:
    role: str
    content: str


@dataclass(slots=True)
class SourceSelection:
    pdf: Path | None = None
    url: str | None = None
    index: str | None = None
    max_pages: int | None = None
    crawl_depth: int | None = None


class CLICompleter(Completer):
    COMMANDS = [
        "/quit",
        "/exit",
        "/history",
        "/reload",
        "/help",
        "/clear",
        "/indexes",
        "/delete-index",
        "/sources",
    ]

    def __init__(self, slug_provider: Callable[[], Sequence[str]]) -> None:
        self.slug_provider = slug_provider
        self.path_completer = PathCompleter(expanduser=True)

    def get_completions(
        self, document: PTDocument, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        stripped = document.text_before_cursor.lstrip()
        if stripped.startswith("/"):
            yield from self._command_completions(document, stripped)
            return
        token = document.get_word_before_cursor(pattern=re.compile(r"[^\s]+"))
        if token and (
            token.startswith(("/", ".", "~"))
            or "/" in token
            or "\\" in token
        ):
            yield from self._path_completions(token, complete_event)

    def _command_completions(
        self, document: PTDocument, stripped: str
    ) -> Iterable[Completion]:
        current = document.get_word_before_cursor(pattern=re.compile(r"[^\s]+")) or ""
        parts = stripped.split()
        if parts and parts[0] == "/delete-index" and len(parts) >= 2:
            prefix = current
            for slug in self.slug_provider():
                if slug.startswith(prefix):
                    yield Completion(slug, start_position=-len(prefix))
            return
        for command in self.COMMANDS:
            if command.startswith(current):
                yield Completion(command, start_position=-len(current))

    def _path_completions(
        self, token: str, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        temp_doc = PTDocument(text=token, cursor_position=len(token))
        path_event = CompleteEvent(
            completion_requested=getattr(complete_event, "completion_requested", False),
            text_inserted=getattr(complete_event, "text_inserted", False),
        )
        for completion in self.path_completer.get_completions(temp_doc, path_event):
            yield Completion(completion.text, start_position=-len(token))


class JambaCLI:
    def __init__(
        self,
        document: DocumentLike,
        *,
        system_prompt: str,
        reload_document: ReloadCallable,
        source_label: str,
        model_path: Path,
        ctx_len: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        gpu_layers: int,
        threads: int | None,
        hide_thinking: bool,
        stream: bool,
        history_turns: int,
        cache_prompt: bool,
        rag_session: RAGSession | None = None,
        rag_slug: str | None = None,
        index_store: IndexStore | None = None,
    ) -> None:
        self.console = Console(theme=PLAYFUL_THEME)
        self.document = document
        self.system_prompt = system_prompt
        self.reload_document = reload_document
        self.source_label = source_label
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.hide_thinking = hide_thinking
        self.stream = stream
        self.history_turns = max(history_turns, 0)
        self.history: list[ConversationTurn] = []
        self.stop_sequences = ["\nUser:", "\nAssistant:", "<|im_start|>user"]
        self.cache_prompt = cache_prompt
        self.ctx_len = ctx_len
        self.model_label = model_path.name
        self.session_started = datetime.now()
        self.rag = rag_session
        self.rag_slug = rag_slug
        self.index_store = index_store
        self.last_sources: list[RetrievedChunk] = []
        self._prompt_tokens = FormattedText(
            [("class:prompt.label", "You "), ("class:prompt.symbol", "› ")]
        )
        self._prompt_style = Style.from_dict(
            {
                "prompt.label": "ansibrightcyan bold",
                "prompt.symbol": "ansibrightblack",
            }
        )
        self.prompt_session = PromptSession(
            completer=CLICompleter(self._available_index_slugs),
            style=self._prompt_style,
            reserve_space_for_menu=6,
            complete_while_typing=True,
        )

        with self.console.status("[info]Summoning Jamba…[/info]", spinner="dots"):
            self.llm = Llama(
                model_path=str(model_path),
                n_ctx=ctx_len,
                n_gpu_layers=gpu_layers,
                n_threads=threads,
                flash_attn=True,
                logits_all=False,
                verbose=False,
            )
        self.console.print("[info]Model ready![/info]\n")

    def _build_prompt(self, new_user_question: str) -> str:
        context_title, context_body = self._context_section(new_user_question)
        doc_section = (
            f"<<SYS>>{self.system_prompt}<<SYS>>\n\n"
            f"{context_title}\n"
            f"{context_body}\n\n"
            "## Conversation\n"
        )

        convo_lines: list[str] = []
        for turn in self._history_window():
            role_label = "User" if turn.role == "user" else "Assistant"
            convo_lines.append(f"{role_label}: {turn.content.strip()}")
        convo_lines.append(f"User: {new_user_question.strip()}")

        prompt = doc_section + "\n".join(convo_lines) + "\nAssistant:"
        return prompt

    def _history_window(self) -> list[ConversationTurn]:
        if self.history_turns <= 0:
            return []
        window = self.history_turns * 2
        return self.history[-window:]

    def _context_section(self, question: str) -> tuple[str, str]:
        if self.rag:
            retrieved = self.rag.retrieve(question)
            if retrieved:
                self.last_sources = retrieved
                formatted = []
                for idx, ctx in enumerate(retrieved, start=1):
                    formatted.append(
                        f"[{idx}] Title: {ctx.chunk.title}\n"
                        f"URL: {ctx.chunk.url}\n"
                        f"{ctx.chunk.content}"
                    )
                return "## Retrieved Context", "\n\n".join(formatted)
        self.last_sources = []
        return "## Document Context", self.document.text

    def _print_intro(self) -> None:
        self._print_banner()
        self.console.print(self._document_panel())
        self._print_answer_meta()
        self.console.print(
            "[meta]Commands: /quit, /history, /reload, /help, /clear, "
            "/indexes, /sources, /delete-index <slug|number>[/meta]\n"
        )

    def _print_banner(self) -> None:
        print_banner(self.console)

    def _document_panel(self) -> Panel:
        stats = Table.grid(padding=(0, 2))
        stats.add_column(style="meta")
        stats.add_column()

        if isinstance(self.document, DocumentContent):
            stats.add_row("Source", str(self.document.path))
        else:
            stats.add_row("Source", self.document.url)

        stats.add_row("Pages", str(self.document.page_count))
        stats.add_row("Characters", f"{self.document.characters:,}")
        stats.add_row("Preview", self.document.preview())
        if self.rag_slug:
            stats.add_row("Index", self.rag_slug)

        return Panel(
            stats,
            title="Context primed",
            border_style="info",
            box=box.ROUNDED,
        )

    def _status_bar(self) -> Panel:
        history_pairs = len(self.history) // 2
        info = Text.assemble(
            ("Model ", "meta"),
            (self.model_label, "info"),
            ("  Source ", "meta"),
            (self.source_label, "highlight"),
            ("  Context ", "meta"),
            (f"{self.ctx_len:,}", "info"),
            ("  History ", "meta"),
            (str(history_pairs), "info"),
        )
        if self.rag:
            info.append("  RAG ", style="meta")
            info.append(self.rag_slug or "enabled", style="info")
        return Panel(info, border_style="meta", box=box.SQUARE, padding=(0, 1))

    def _clean_answer(self, raw_text: str) -> str:
        answer = raw_text.rsplit("</think>", 1)[-1] if self.hide_thinking else raw_text
        return answer.strip()

    def _answer_panel(self, text: str) -> Panel:
        body = Markdown(
            text if text.strip() else "_(no answer produced)_",
            code_theme="monokai",
        )
        return Panel(
            body,
            title="[assistant]Jamba[/assistant]",
            border_style="assistant.border",
            box=box.ROUNDED,
        )

    def _render_answer(self, cleaned: str) -> None:
        self.console.print(self._answer_panel(cleaned))

    def _print_answer_meta(self) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        meta = Text.assemble(
            ("⏱ ", "meta"),
            (timestamp, "meta"),
            ("  •  Model ", "meta"),
            (self.model_label, "info"),
            ("  •  Source ", "meta"),
            (self.source_label, "highlight"),
        )
        self.console.print(meta)
        self.console.print(self._status_bar())

    def _print_sources(self) -> None:
        if not self.last_sources:
            return
        table = Table(
            show_header=True,
            header_style="meta",
            padding=(0, 1),
            box=box.SIMPLE_HEAVY,
        )
        table.add_column("#", style="meta", width=3)
        table.add_column("Title", style="assistant", overflow="fold", ratio=2)
        table.add_column("URL", style="highlight", overflow="fold", ratio=3)
        table.add_column("Score", style="meta", width=8)

        for idx, ctx in enumerate(self.last_sources, start=1):
            table.add_row(
                str(idx),
                ctx.chunk.title,
                ctx.chunk.url,
                f"{ctx.score:.2f}",
            )

        self.console.print(
            Panel(
                table,
                title="Retrieved Sources",
                border_style="assistant.border",
                box=box.ROUNDED,
            )
        )

    def _available_index_slugs(self) -> list[str]:
        if self.rag:
            try:
                return [meta.slug for meta in self.rag.list_indexes()]
            except Exception:
                return []
        if self.index_store:
            try:
                return [meta.slug for meta in self.index_store.list_metadata()]
            except Exception:
                return []
        return []

    def _show_indexes(self) -> None:
        if not self.rag:
            self.console.print("[warn]RAG is disabled for this session.[/warn]")
            return
        entries = self.rag.list_indexes()
        if not entries:
            self.console.print("[meta](no cached indexes yet)[/meta]")
            return
        table = Table(
            show_header=True,
            header_style="meta",
            box=box.ROUNDED,
            padding=(0, 1),
        )
        table.add_column("#", style="meta", width=4, justify="right")
        table.add_column("Slug", style="highlight", overflow="fold", no_wrap=False)
        table.add_column("URL", style="assistant", overflow="fold")
        table.add_column("Pages", justify="right", style="info")
        table.add_column("Chunks", justify="right", style="info")
        table.add_column("Created", style="meta")

        for idx, meta in enumerate(entries, start=1):
            table.add_row(
                str(idx),
                meta.slug,
                meta.url,
                str(meta.page_count),
                str(meta.chunk_count),
                meta.created_at.split("T")[0],
            )
        self.console.print(Panel(table, border_style="info", title="Cached Indexes"))

    def _handle_delete_command(self, user_input: str) -> None:
        if not self.rag:
            self.console.print("[warn]RAG is disabled for this session.[/warn]")
            return
        parts = user_input.split(maxsplit=1)
        if len(parts) != 2:
            self.console.print("[warn]Usage: /delete-index <slug|number>[/warn]")
            return
        token = parts[1].strip()
        if not token:
            self.console.print("[warn]Usage: /delete-index <slug|number>[/warn]")
            return
        slug = self._resolve_index_token(token)
        if not slug:
            self.console.print(
                "[warn]No index matched that identifier. Run /indexes for the current list.[/warn]"
            )
            return
        removed = self.rag.delete_index(slug)
        if removed:
            self.console.print(f"[warn]Deleted index '{slug}'.[/warn]")
            if self.rag_slug == slug:
                self.rag_slug = None
        else:
            self.console.print(f"[warn]Index '{slug}' not found.[/warn]")

    def _resolve_index_token(self, token: str) -> str | None:
        if not self.rag:
            return None
        entries = self.rag.list_indexes()
        if not entries:
            return None
        if token.isdigit():
            idx = int(token)
            if 1 <= idx <= len(entries):
                return entries[idx - 1].slug
        for entry in entries:
            if entry.slug == token:
                return entry.slug
        matches = [entry.slug for entry in entries if entry.slug.startswith(token)]
        if len(matches) == 1:
            return matches[0]
        return None

    def _stream_completion(self, prompt: str) -> str:
        response_chunks: list[str] = []
        if self.stream:
            if not self.cache_prompt:
                self.llm.reset()
            visible_started = not self.hide_thinking
            hidden_buffer = ""
            display_text = ""
            with Live(console=self.console, refresh_per_second=12) as live:
                for chunk in self.llm.create_completion(
                    prompt=prompt,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stop=self.stop_sequences,
                    stream=True,
                ):
                    token = chunk["choices"][0]["text"]
                    response_chunks.append(token)
                    if visible_started:
                        display_text += token
                        live.update(self._answer_panel(display_text))
                    else:
                        hidden_buffer += token
                        marker = hidden_buffer.find("</think>")
                        if marker != -1:
                            visible_started = True
                            tail = hidden_buffer[marker + len("</think>") :]
                            display_text += tail
                            hidden_buffer = ""
                            live.update(self._answer_panel(display_text))
                if not visible_started and hidden_buffer.strip():
                    live.update(self._answer_panel(hidden_buffer))
                elif visible_started and not display_text.strip():
                    live.update(self._answer_panel(display_text))
            return "".join(response_chunks)

        if not self.cache_prompt:
            self.llm.reset()
        result = self.llm.create_completion(
            prompt=prompt,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=self.stop_sequences,
            stream=False,
        )
        return result["choices"][0]["text"]

    def ask(self, user_question: str) -> None:
        prompt = self._build_prompt(user_question)
        try:
            raw_answer = self._stream_completion(prompt)
        except RuntimeError as exc:
            message = str(exc)
            if "llama_decode returned" in message:
                self.console.print(
                    "[error]Model hit its context or GPU memory limit. "
                    "Try lowering --ctx/--max-new-tokens, reducing --history-turns, "
                    "or enabling --cache-prompt.[/error]"
                )
                return
            raise
        cleaned = self._clean_answer(raw_answer)
        if not self.stream:
            self._render_answer(cleaned)
        self.history.append(ConversationTurn("user", user_question))
        self.history.append(ConversationTurn("assistant", cleaned))

    def show_history(self) -> None:
        if not self.history:
            self.console.print("[meta](empty history)[/meta]")
            return

        for turn in self.history:
            role_style = "user" if turn.role == "user" else "assistant"
            label = "You" if turn.role == "user" else "Jamba"
            self.console.print(f"[{role_style}]{label}: {turn.content}[/{role_style}]")

    def repl(self) -> None:
        self.console.clear()
        self._print_intro()
        while True:
            try:
                with patch_stdout(raw=True):
                    user_input = self.prompt_session.prompt(self._prompt_tokens).strip()
            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[warn]Bye![/warn]")
                break

            if not user_input:
                continue
            if user_input in {"/quit", "/exit"}:
                self.console.print("[warn]Exiting chat.[/warn]")
                break
            if user_input == "/history":
                self.show_history()
                continue
            if user_input == "/reload":
                self.console.print("[meta]Refreshing source…[/meta]")
                try:
                    self.document = self.reload_document()
                    self.console.print("[info]Source refreshed![/info]")
                except Exception as exc:  # pragma: no cover - defensive logging
                    self.console.print(f"[error]{exc}[/error]")
                continue
            if user_input == "/help":
                self.console.print(
                    "[meta]Commands: /quit, /history, /reload, /help, /clear, "
                    "/indexes, /sources, /delete-index <slug|number>[/meta]"
                )
                continue
            if user_input == "/clear":
                self.console.clear()
                self._print_intro()
                continue
            if user_input == "/indexes":
                self._show_indexes()
                continue
            if user_input == "/sources":
                if self.last_sources:
                    self._print_sources()
                else:
                    self.console.print("[meta](no retrieved sources yet)[/meta]")
                continue
            if user_input.startswith("/delete-index"):
                self._handle_delete_command(user_input)
                continue

            self.ask(user_input)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="PDF to load (interactive prompt uses the bundled sample if omitted).",
    )
    source_group.add_argument(
        "--url",
        type=str,
        help="Documentation URL to crawl and ground answers in.",
    )
    source_group.add_argument(
        "--index",
        type=str,
        help="Slug of a cached documentation index (skips crawling).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the GGUF model file.",
    )
    parser.add_argument("--ctx", type=int, default=DEFAULT_CTX_LEN, help="Context window.")
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Generation budget."
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=-1,
        help="Layers to offload to GPU (-1 = all, best for Apple Silicon).",
    )
    parser.add_argument("--threads", type=int, default=None, help="CPU threads (default auto).")
    parser.add_argument(
        "--show-thinking",
        action="store_true",
        help="Display the model's <think> traces instead of hiding them.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output (prints full answer at once).",
    )
    parser.add_argument(
        "--history-turns",
        type=int,
        default=DEFAULT_HISTORY_TURNS,
        help="Number of previous Q/A pairs to keep in the prompt (default 0).",
    )
    parser.add_argument(
        "--cache-prompt",
        action="store_true",
        help="Reuse the KV cache between turns (disabled by default).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_MAX_CRAWL_PAGES,
        help="Maximum documentation pages to crawl when --url is provided (0 = all).",
    )
    parser.add_argument(
        "--crawl-depth",
        type=int,
        default=DEFAULT_CRAWL_DEPTH,
        help="Depth limit for the documentation crawler when --url is provided.",
    )
    parser.add_argument(
        "--rag",
        action=BooleanOptionalAction,
        default=None,
        help="Toggle retrieval-augmented mode (auto-enabled for --url/--index).",
    )
    parser.add_argument(
        "--rag-top-k",
        type=int,
        default=DEFAULT_RAG_TOP_K,
        help="Chunks to retrieve per question when RAG is enabled.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuilding the FAISS index when crawling a URL source.",
    )
    return parser.parse_args(argv)


def _crawl_with_progress(
    console: Console,
    url: str,
    *,
    max_pages: int | None,
    crawl_depth: int,
) -> WebDocumentContent:
    progress = Progress(
        SpinnerColumn(style="info"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TextColumn("{task.completed} pages"),
        TimeElapsedColumn(),
        console=console,
    )
    with progress:
        task_id = progress.add_task("Crawling docs", total=max_pages)

        def _update(completed: int, total: int | None) -> None:
            progress.update(task_id, completed=completed, total=total)

        return load_from_url(
            url,
            max_pages=max_pages,
            max_depth=crawl_depth,
            progress_callback=_update,
        )


def interactive_setup(console: Console, store: IndexStore) -> SourceSelection:
    console.clear()
    print_banner(console)
    console.print(
        "[meta]No --pdf/--url/--index provided. Launching interactive setup…[/meta]"
    )
    metadata = store.list_metadata()
    console.print("Select source type:")
    console.print("  1. PDF file (one-shot mode)")
    console.print("  2. Documentation URL (crawl & RAG mode)")
    has_indexes = bool(metadata)
    if has_indexes:
        console.print("  3. Cached index (RAG mode)")
    max_choice = 3 if has_indexes else 2

    def _ask_number(prompt: str, upper: int, *, default: int) -> int:
        while True:
            choice = IntPrompt.ask(prompt, default=default)
            if 1 <= choice <= upper:
                return choice
            console.print(f"[warn]Enter a number between 1 and {upper}.[/warn]")

    selection = _ask_number("Choice", max_choice, default=1)
    if selection == 1:
        pdf_default = str(DEFAULT_DOC_PATH)
        pdf_value = Prompt.ask("PDF path", default=pdf_default).strip() or pdf_default
        return SourceSelection(pdf=Path(pdf_value).expanduser())
    if selection == 2:
        url_default = DEFAULT_DOC_URL
        url_value = Prompt.ask("Documentation URL", default=url_default).strip()
        url_value = url_value or url_default
        max_pages = IntPrompt.ask(
            "Maximum pages to crawl (0 = unlimited)",
            default=DEFAULT_MAX_CRAWL_PAGES,
        )
        crawl_depth = IntPrompt.ask(
            "Crawl depth",
            default=DEFAULT_CRAWL_DEPTH,
        )
        return SourceSelection(
            url=url_value,
            max_pages=max(0, max_pages),
            crawl_depth=max(1, crawl_depth),
        )

    if not metadata:
        console.print("[warn]No cached indexes available. Falling back to PDF.[/warn]")
        return SourceSelection(pdf=DEFAULT_DOC_PATH)

    table = Table(
        show_header=True,
        header_style="meta",
        padding=(0, 1),
        box=box.ROUNDED,
    )
    table.add_column("#", justify="right", style="meta", width=4)
    table.add_column("Slug", style="highlight")
    table.add_column("URL", style="assistant", overflow="fold")
    table.add_column("Pages", justify="right", style="info")
    table.add_column("Chunks", justify="right", style="info")
    for idx, entry in enumerate(metadata, start=1):
        table.add_row(
            str(idx),
            entry.slug,
            entry.url,
            str(entry.page_count),
            str(entry.chunk_count),
        )
    console.print(table)
    index_choice = _ask_number("Select cached index", len(metadata), default=1)
    return SourceSelection(index=metadata[index_choice - 1].slug)


def _needs_default_assets(args: argparse.Namespace) -> tuple[bool, bool]:
    wants_default_model = Path(args.model) == DEFAULT_MODEL_PATH
    missing_model = wants_default_model and not DEFAULT_MODEL_PATH.exists()

    pdf_path = Path(args.pdf) if args.pdf else None
    wants_default_pdf = (
        pdf_path is not None
        and pdf_path == DEFAULT_DOC_PATH
        and not args.url
        and not args.index
    )
    missing_pdf = wants_default_pdf and not DEFAULT_DOC_PATH.exists()

    return missing_model, missing_pdf


def _maybe_run_setup(console: Console, args: argparse.Namespace) -> None:
    missing_model, missing_pdf = _needs_default_assets(args)
    if not (missing_model or missing_pdf):
        return

    parts = []
    if missing_model:
        parts.append(f"model at [highlight]{DEFAULT_MODEL_PATH}[/highlight]")
    if missing_pdf:
        parts.append(f"sample PDF at [highlight]{DEFAULT_DOC_PATH}[/highlight]")
    missing_text = " and ".join(parts)
    console.print(
        f"[warn]Missing required assets: {missing_text}. "
        "Run `jash-setup` to download them?[/warn]"
    )
    if Confirm.ask("Run setup now?", default=True):
        try:
            setup_assets_main()
        except KeyboardInterrupt:
            console.print("\n[warn]Setup interrupted by user.[/warn]")
        except Exception as exc:
            console.print(f"[error]Setup failed: {exc}[/error]")
    else:
        console.print("[warn]Skipping automatic setup. You can run `jash-setup` later.[/warn]")

    missing_model, missing_pdf = _needs_default_assets(args)
    if missing_model or missing_pdf:
        console.print(
            "[error]Default assets are still missing. "
            "Please run `jash-setup` before starting the chat.[/error]"
        )
        sys.exit(1)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    console = Console(theme=PLAYFUL_THEME)
    store = IndexStore()
    if not args.pdf and not args.url and not args.index:
        selection = interactive_setup(console, store)
        if selection.index:
            args.index = selection.index
        elif selection.url:
            args.url = selection.url
            if selection.max_pages is not None:
                args.max_pages = selection.max_pages
            if selection.crawl_depth is not None:
                args.crawl_depth = selection.crawl_depth
        else:
            args.pdf = selection.pdf or DEFAULT_DOC_PATH

    _maybe_run_setup(console, args)

    use_rag = args.rag if args.rag is not None else bool(args.url or args.index)
    rag_session: RAGSession | None = None
    rag_slug: str | None = None

    if args.index and not use_rag:
        console.print("[error]--index requires RAG mode. Remove --no-rag.[/error]")
        return 1

    if use_rag:
        rag_session = RAGSession(
            store=store,
            embedder=EmbeddingClient(console=console),
            console=console,
            top_k=max(1, args.rag_top_k),
        )

    loader: ReloadCallable
    system_prompt: str
    source_label: str

    if args.index:
        slug = args.index.strip()
        rag_slug = slug

        def _load_cached() -> WebDocumentContent:
            assert rag_session is not None
            loaded = rag_session.attach_loaded(slug)
            return loaded.document()

        loader = _load_cached
        system_prompt = CODING_SYSTEM_PROMPT
        source_label = slug
    elif args.url:
        normalized_max_pages = args.max_pages if args.max_pages and args.max_pages > 0 else None
        slug = store.slugify(args.url)
        rag_slug = slug if use_rag else None

        def _load_and_index() -> WebDocumentContent:
            doc = _crawl_with_progress(
                console,
                args.url,
                max_pages=normalized_max_pages,
                crawl_depth=args.crawl_depth,
            )
            if rag_session:
                rag_session.build_for_document(slug, doc)
            return doc

        if (
            rag_session
            and not args.rebuild_index
            and store.exists(slug)
        ):
            console.print(
                "[meta]Using cached index. Pass --rebuild-index to refresh the crawl.[/meta]"
            )

            def _load_cached() -> WebDocumentContent:
                assert rag_session is not None
                loaded = rag_session.attach_loaded(slug)
                return loaded.document()

            loader = _load_cached
        else:
            loader = _load_and_index

        system_prompt = CODING_SYSTEM_PROMPT
        source_label = args.url
    else:
        pdf_path = args.pdf

        def loader() -> DocumentContent:
            return load_document(pdf_path)

        system_prompt = SYSTEM_PROMPT
        source_label = str(pdf_path)

    try:
        document = loader()
    except Exception as exc:
        console.print(f"[error]{exc}[/error]")
        return 1
    if isinstance(document, WebDocumentContent):
        source_label = document.url

    cli = JambaCLI(
        document=document,
        system_prompt=system_prompt,
        reload_document=loader,
        source_label=source_label,
        model_path=args.model,
        ctx_len=args.ctx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        gpu_layers=args.gpu_layers,
        threads=args.threads,
        hide_thinking=not args.show_thinking,
        stream=not args.no_stream,
        history_turns=args.history_turns,
        cache_prompt=args.cache_prompt,
        rag_session=rag_session,
        rag_slug=rag_slug,
        index_store=store,
    )
    cli.repl()
    return 0


if __name__ == "__main__":
    sys.exit(main())
