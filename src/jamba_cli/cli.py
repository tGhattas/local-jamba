"""Playful CLI to chat with a long PDF using Jamba Reasoning 3B."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from llama_cpp import Llama  # type: ignore[import-not-found]
from rich.console import Console  # type: ignore[import-not-found]
from rich.panel import Panel  # type: ignore[import-not-found]
from rich.prompt import Prompt  # type: ignore[import-not-found]
from rich.table import Table  # type: ignore[import-not-found]
from rich.theme import Theme  # type: ignore[import-not-found]

from .document import DocumentContent, load_document
from .settings import (
    DEFAULT_CTX_LEN,
    DEFAULT_DOC_PATH,
    DEFAULT_HISTORY_TURNS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_PATH,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    SYSTEM_PROMPT,
)

PLAYFUL_THEME = Theme(
    {
        "info": "bold cyan",
        "warn": "yellow",
        "error": "bold red",
        "assistant": "magenta",
        "user": "deep_sky_blue3",
        "meta": "grey70",
    }
)


@dataclass(slots=True)
class ConversationTurn:
    role: str
    content: str


class JambaCLI:
    def __init__(
        self,
        document: DocumentContent,
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
    ) -> None:
        self.console = Console(theme=PLAYFUL_THEME)
        self.document = document
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.hide_thinking = hide_thinking
        self.stream = stream
        self.history_turns = max(history_turns, 0)
        self.history: list[ConversationTurn] = []
        self.stop_sequences = ["\nUser:", "\nAssistant:", "<|im_start|>user"]
        self.cache_prompt = cache_prompt

        self.console.print("[info]Loading Jamba model...[/info]")
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
        doc_section = (
            f"<<SYS>>{SYSTEM_PROMPT}<<SYS>>\n\n"
            "## Document Context\n"
            f"{self.document.text}\n\n"
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

    def _print_intro(self) -> None:
        stats = Table(title="Document loaded", show_header=False)
        stats.add_row("Path", str(self.document.path))
        stats.add_row("Pages", str(self.document.page_count))
        stats.add_row("Characters", f"{self.document.characters:,}")
        stats.add_row("Preview", self.document.preview())
        self.console.print(Panel(stats, border_style="info"))
        self.console.print(
            "[meta]Type your question. Commands: /quit, /reload, /history, /help[/meta]\n"
        )

    def _clean_answer(self, raw_text: str) -> str:
        answer = raw_text.rsplit("</think>", 1)[-1] if self.hide_thinking else raw_text
        return answer.strip()

    def _render_answer(self, cleaned: str) -> None:
        self.console.print("\n[assistant]" + cleaned + "[/assistant]\n")

    def _stream_completion(self, prompt: str) -> str:
        response_chunks: list[str] = []
        if self.stream:
            if not self.cache_prompt:
                self.llm.reset()
            visible_started = not self.hide_thinking
            hidden_buffer = ""
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
                    self.console.print(token, end="", style="assistant")
                else:
                    hidden_buffer += token
                    marker = hidden_buffer.find("</think>")
                    if marker != -1:
                        visible_started = True
                        tail = hidden_buffer[marker + len("</think>") :]
                        if tail:
                            self.console.print(tail, end="", style="assistant")
                        hidden_buffer = ""

            if not visible_started and hidden_buffer.strip():
                # No </think> tag – print everything we collected.
                self.console.print(hidden_buffer, end="", style="assistant")

            self.console.print()  # newline after streaming block
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
        self._print_intro()
        while True:
            try:
                user_input = Prompt.ask("[user]You[/user]").strip()
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
                self.console.print("[warn]Reloading document...[/warn]")
                self.document = load_document(self.document.path)
                self.console.print("[info]Document reloaded![/info]")
                continue
            if user_input == "/help":
                self.console.print(
                    "[meta]Commands: /quit, /history, /reload, /help[/meta]"
                )
                continue

            self.ask(user_input)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_DOC_PATH, help="PDF to load.")
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
        help="Reuse the KV cache between turns (disabled by default to avoid context overflows).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        document = load_document(args.pdf)
    except Exception as exc:
        Console(theme=PLAYFUL_THEME).print(f"[error]{exc}[/error]")
        return 1

    cli = JambaCLI(
        document=document,
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
    )
    cli.repl()
    return 0


if __name__ == "__main__":
    sys.exit(main())

