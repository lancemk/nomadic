#!/usr/bin/env python3
"""
nomadic CLI
────────────────────────────────────────────────────────────────────────────
BITSTRIC · L7 IC: AI Capacity Expert
Multimodal SKILL.md context extractor + Structured Tool-Calling Kernel

USAGE
  # One-shot — extract context from a SKILL.md path
  python -m nomadic extract /mnt/skills/public/docx/SKILL.md

  # One-shot with image alongside the SKILL.md text
  python -m nomadic extract /mnt/skills/public/docx/SKILL.md \
         --image /path/to/diagram.png

  # Interactive kernel (structured tool-calling agent loop)
  python -m nomadic kernel

  # Initialize config-backed database settings
  python -m nomadic initialize

  # Start the bundled mlx_vlm.server (helper, run once in a separate shell)
  python -m nomadic serve

REQUIRES (on Apple Silicon)
  pip install mlx-vlm        # mlx_vlm.server + mlx_vlm.generate
  pip install requests rich  # HTTP client + pretty output
  pip install prompt_toolkit # optional enhanced multiline kernel input
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pathlib
import re
import subprocess
import sys
import textwrap
import time
import readline  # noqa: F401 — enables arrow-key history in input()
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

# ─── optional rich ────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.rule import Rule
    from rich.syntax import Syntax
    _RICH = True
except ImportError:
    _RICH = False

try:
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import HSplit, Window
    from prompt_toolkit.layout.dimension import Dimension
    from prompt_toolkit.styles import Style
    from prompt_toolkit.widgets import Frame, Label, TextArea
    _PROMPT_TOOLKIT = True
except ImportError:
    _PROMPT_TOOLKIT = False

try:
    import requests
    _REQUESTS = True
except ImportError:
    _REQUESTS = False

from . import agent
from . import db as chatdb

_MEMORY_NOTE_PREFIX = "[KERNEL MEMORY SUMMARY]"
_KERNEL_INPUT_ACCENT = "#ea7100"
_KERNEL_EXIT_ARM_WINDOW_SECONDS = 2.0

# ══════════════════════════════════════════════════════════════════════════════
# SERVER CONFIG  (mlx_vlm.server inline params)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ServerConfig:
    """
    Server settings plus CLI-side context controls.
    Kept here so the tool is self-contained — no external config file needed.
    """
    # ── Model selection ──────────────────────────────────────────────────────
    model: str = "mlx-community/gemma-4-26b-a4b-it-4bit"
    # model: str = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
    #   Alternatives ranked by SKILL.md comprehension quality:
    #     "mlx-community/Pixtral-12B-2409-4bit"         ← best OCR of markdown
    #     "mlx-community/Qwen2-VL-7B-Instruct-4bit"     ← balanced
    #     "mlx-community/llava-1.6-mistral-7b-4bit"     ← fast, lower RAM
    #     "mlx-community/Phi-3.5-vision-instruct-4bit"  ← small, 6 GB

    # ── Server bind ─────────────────────────────────────────────────────────
    host: str = "127.0.0.1"
    port: int = 8080

    # ── Generation defaults (inline to every request) ────────────────────────
    max_tokens: int = 1024          # per completion turn
    temperature: float = 0.15       # low → deterministic; good for extraction
    top_p: float = 0.9
    repetition_penalty: float = 1.1 # mild repeat suppression

    # ── Context window management ────────────────────────────────────────────
    max_skill_chars: int = 12_000   # truncate SKILL.md if huge (fits 4-bit ctx)
    max_context_tokens: int = 12_000  # approximate history budget for kernel
    context_recompress_to: int = 8_000 # target after automatic history compression

    # ── Online metadata search ──────────────────────────────────────────────
    metadata_base_url: str = field(
        default_factory=lambda: os.getenv("METADATA_BASE_URL", "http://localhost:8888")
    )

    # ── Structured tool-calling kernel ──────────────────────────────────────
    cot_system_prompt: str = field(default_factory=agent.default_system_prompt)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def server_argv(self) -> list[str]:
        """Arguments passed to `python -m mlx_vlm.server`."""
        return [
            sys.executable, "-m", "mlx_vlm.server",
            "--model",       self.model,
            "--host",        self.host,
            "--port",        str(self.port),
            # mlx_vlm.server does not expose temp/top_p at CLI level;
            # those are per-request params sent in the JSON body.
        ]


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

cfg_global = ServerConfig()

class UI:
    """Thin wrapper — falls back to plain print when rich is absent."""

    def __init__(self):
        self.con = Console(highlight=False) if _RICH else None

    def rule(self, title: str = ""):
        if _RICH:
            self.con.print(Rule(title, style="dim"))
        else:
            print(f"\n{'─' * 60}  {title}")

    def info(self, msg: str):
        tag = "[dim cyan]·[/]" if _RICH else "·"
        if _RICH:
            self.con.print(f"{tag} {msg}")
        else:
            print(f"  · {msg}")

    def warn(self, msg: str):
        if _RICH:
            self.con.print(f"[yellow]⚠[/]  {msg}")
        else:
            print(f"  ⚠  {msg}")

    def error(self, msg: str):
        if _RICH:
            self.con.print(f"[bold red]✗[/]  {msg}")
        else:
            print(f"  ✗  {msg}", file=sys.stderr)

    def success(self, msg: str):
        if _RICH:
            self.con.print(f"[bold green]✔[/]  {msg}")
        else:
            print(f"  ✔  {msg}")

    def model_output(self, text: str):
        if _RICH:
            try:
                self.con.print(Markdown(text))
            except Exception:
                self.con.print(text)
        else:
            print(textwrap.fill(text, width=100))

    def code(self, text: str, lang: str = "python"):
        if _RICH:
            self.con.print(Syntax(text, lang, theme="monokai", line_numbers=True))
        else:
            print(text)

    def panel(self, title: str, body: str):
        if _RICH:
            self.con.print(Panel(body, title=title, border_style="dim"))
        else:
            print(f"\n[{title}]\n{body}\n")

    def prompt(self, label: str) -> str:
        if _RICH:
            return Prompt.ask(f"[bold cyan]{label}[/]")
        return input(f"{label}: ")


ui = UI()


@dataclass(frozen=True)
class KernelInputSubmission:
    raw: str
    stripped: str
    kind: str


@dataclass(frozen=True)
class KernelNextActionSelection:
    kind: str
    text: Optional[str] = None


def parse_kernel_input(raw: str) -> KernelInputSubmission:
    stripped = raw.strip()
    if not stripped:
        return KernelInputSubmission(raw=raw, stripped=stripped, kind="empty")
    if "\n" not in raw and stripped.startswith("/"):
        return KernelInputSubmission(raw=raw, stripped=stripped, kind="command")
    return KernelInputSubmission(raw=raw, stripped=stripped, kind="message")


@dataclass
class KernelInterruptController:
    window_seconds: float = _KERNEL_EXIT_ARM_WINDOW_SECONDS
    armed_at: Optional[float] = None

    def clear(self):
        self.armed_at = None

    def arm_or_exit(self, *, now: Optional[float] = None) -> bool:
        current = time.monotonic() if now is None else now
        if self.armed_at is not None and current - self.armed_at <= self.window_seconds:
            self.armed_at = None
            return True
        self.armed_at = current
        return False

    def is_armed(self, *, now: Optional[float] = None) -> bool:
        if self.armed_at is None:
            return False
        current = time.monotonic() if now is None else now
        if current - self.armed_at > self.window_seconds:
            self.armed_at = None
            return False
        return True


def parse_next_action_options(raw: str) -> list[str]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        candidates = re.findall(r"\{.*\}", raw, flags=re.DOTALL)
        for candidate in candidates:
            try:
                data = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue
        else:
            return []

    if not isinstance(data, dict):
        return []
    options = data.get("options")
    if not isinstance(options, list):
        return []

    cleaned: list[str] = []
    for option in options:
        if not isinstance(option, str):
            continue
        text = option.strip()
        if not text or text in cleaned:
            continue
        cleaned.append(text)
        if len(cleaned) == 3:
            break
    return cleaned if len(cleaned) == 3 else []


def resolve_next_action_selection(
    submission: KernelInputSubmission,
    options: list[str],
) -> KernelNextActionSelection:
    if submission.kind != "message":
        return KernelNextActionSelection(kind="none")

    choice = submission.stripped
    if choice in {"1", "2", "3"}:
        index = int(choice) - 1
        if index < len(options):
            return KernelNextActionSelection(kind="selected", text=options[index])
    if choice == "4":
        return KernelNextActionSelection(kind="other")
    return KernelNextActionSelection(kind="none")


class KernelInput:
    def __init__(self, ui: UI):
        self.ui = ui
        self._fallback_notice_shown = False

    def prompt(self, label: str) -> str:
        if _PROMPT_TOOLKIT:
            return self._prompt_multiline(label)
        return self._prompt_fallback(label)

    def _prompt_fallback(self, label: str) -> str:
        if not self._fallback_notice_shown:
            self._fallback_notice_shown = True
            if _RICH:
                self.ui.con.print(
                    Panel(
                        "Optional dependency `prompt_toolkit` is not installed.\n"
                        "Falling back to the classic single-line prompt.",
                        title="Kernel Input",
                        border_style=_KERNEL_INPUT_ACCENT,
                    )
                )
            else:
                self.ui.warn(
                    "prompt_toolkit is not installed; using the classic single-line prompt."
                )

        if _RICH:
            return Prompt.ask(f"[bold {_KERNEL_INPUT_ACCENT}]{label}[/]")
        return input(f"{label}: ")

    def _prompt_multiline(self, label: str) -> str:
        text_area = TextArea(
            multiline=True,
            wrap_lines=True,
            focus_on_click=True,
            scrollbar=False,
            style="class:input-body",
            height=Dimension(min=3, max=12),
        )
        help_text = Label(
            text="Enter submit  Shift+Enter newline  Ctrl-D exit",
            style="class:input-help",
            dont_extend_height=True,
        )
        frame = Frame(text_area, title=label, style="class:input-frame")
        root = HSplit(
            [
                frame,
                Window(height=1, char=" "),
                help_text,
            ]
        )
        style = Style.from_dict(
            {
                "input-frame.border": f"fg:{_KERNEL_INPUT_ACCENT}",
                "input-frame.label": f"bold fg:{_KERNEL_INPUT_ACCENT}",
                "input-body": "",
                "input-help": f"fg:{_KERNEL_INPUT_ACCENT} italic",
            }
        )
        kb = KeyBindings()

        def bind_optional(*keys: str):
            def decorator(func):
                try:
                    kb.add(*keys)(func)
                except Exception:
                    return func
                return func

            return decorator

        @kb.add("enter")
        def _submit(event):
            event.app.exit(result=text_area.text)

        @bind_optional("s-enter")
        def _newline_shift(event):
            text_area.buffer.insert_text("\n")

        @bind_optional("escape", "enter")
        def _newline_escape_enter(event):
            text_area.buffer.insert_text("\n")

        @bind_optional("c-j")
        def _newline_ctrl_j(event):
            text_area.buffer.insert_text("\n")

        @kb.add("c-c")
        def _interrupt(event):
            event.app.exit(exception=KeyboardInterrupt())

        @kb.add("c-d")
        def _eof(event):
            if text_area.buffer.text:
                text_area.buffer.delete()
                return
            event.app.exit(exception=EOFError())

        app = Application(
            layout=Layout(root, focused_element=text_area),
            key_bindings=kb,
            style=style,
            full_screen=False,
            mouse_support=False,
        )
        return app.run()


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE ENCODING
# ══════════════════════════════════════════════════════════════════════════════

def encode_image(path: str | pathlib.Path) -> str:
    """Return base64-encoded image string for the mlx_vlm.server /v1/chat request."""
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    raw = p.read_bytes()
    return base64.b64encode(raw).decode()


def image_media_type(path: str | pathlib.Path) -> str:
    ext = pathlib.Path(path).suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(ext, "image/png")


def _build_next_action_snapshot(history: list[dict], *, max_messages: int = 8, max_chars: int = 4_000) -> str:
    recent = history[-max_messages:]
    chunks: list[str] = []
    total = 0
    for message in recent:
        role = str(message.get("role", "user")).upper()
        text = _message_to_text(message).strip()
        if not text:
            continue
        text = text[:900]
        chunk = f"{role}:\n{text}"
        if total + len(chunk) > max_chars:
            remaining = max_chars - total
            if remaining > 32:
                chunks.append(chunk[:remaining].rstrip())
            break
        chunks.append(chunk)
        total += len(chunk) + 2
    return "\n\n".join(chunks)


def generate_next_action_options(cfg: ServerConfig, history: list[dict]) -> list[str]:
    snapshot = _build_next_action_snapshot(history)
    if not snapshot:
        return []

    messages = [
        {
            "role": "system",
            "content": (
                "You generate next-step options for an interactive coding assistant.\n"
                "Return JSON only in this exact shape:\n"
                '{"options":["...","...","..."]}\n'
                "Requirements:\n"
                "- exactly 3 options\n"
                "- option 1 must be the recommended next action\n"
                "- ground the options in the conversation snapshot\n"
                "- keep each option concise, actionable, and specific\n"
                "- each option should read like a user request the assistant can act on\n"
                "- do not include numbering, markdown, or commentary"
            ),
        },
        {
            "role": "user",
            "content": (
                "Conversation snapshot:\n"
                f"{snapshot}\n\n"
                "Generate the 3 most likely next user actions."
            ),
        },
    ]

    try:
        raw = vlm_complete(cfg, messages, stream=False)
    except Exception:
        return []
    return parse_next_action_options(raw)


def render_next_action_menu(options: list[str]) -> str:
    if len(options) != 3:
        return ""
    return "\n".join(
        [
            f"1. {options[0]}  [recommended]",
            f"2. {options[1]}",
            f"3. {options[2]}",
            "4. Other...",
        ]
    )


# ══════════════════════════════════════════════════════════════════════════════
# SKILL.md READER
# ══════════════════════════════════════════════════════════════════════════════

def read_skill(path: str | pathlib.Path, max_chars: int = cfg_global.max_skill_chars) -> str:
    """
    Read a SKILL.md from a local path.
    Truncates to max_chars to stay within the VLM context window.
    """
    payload = agent.read_skill_file(path, max_chars=max_chars)
    if payload["truncated"]:
        ui.warn(
            f"SKILL.md is {payload['original_chars']:,} chars — "
            f"truncating to {max_chars:,}"
        )
    return str(payload["content"])


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _truncate(text: str, limit: int) -> str:
    text = _normalize_whitespace(text)
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text":
                parts.append(item.get("text", ""))
            elif item_type == "image_url":
                parts.append("[image attached]")
            else:
                parts.append(f"[{item_type or 'content'}]")
        return "\n".join(part for part in parts if part)
    return str(content)


def _message_to_text(message: dict) -> str:
    return _content_to_text(message.get("content", ""))


def _is_memory_note(message: dict) -> bool:
    content = message.get("content", "")
    return (
        message.get("role") == "system"
        and isinstance(content, str)
        and content.startswith(_MEMORY_NOTE_PREFIX)
    )


def _memory_note_body(message: dict) -> str:
    content = message.get("content", "")
    if not isinstance(content, str):
        return _message_to_text(message)
    if content.startswith(_MEMORY_NOTE_PREFIX):
        body = content[len(_MEMORY_NOTE_PREFIX):].lstrip()
        return body or _MEMORY_NOTE_PREFIX
    return content


def _estimate_text_tokens(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _estimate_message_tokens(message: dict) -> int:
    role = message.get("role", "user")
    content = message.get("content", "")
    total = 6  # message framing overhead

    if role == "system":
        total += 4
    elif role == "assistant":
        total += 3

    if isinstance(content, str):
        total += _estimate_text_tokens(content)
    elif isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text":
                total += _estimate_text_tokens(item.get("text", ""))
            elif item_type == "image_url":
                total += 700
            else:
                total += 32
    else:
        total += _estimate_text_tokens(str(content))

    return max(1, total)


def _estimate_history_tokens(
    history: list[dict],
    *,
    include_system: bool = False,
) -> int:
    start = 0
    if not include_system and history and history[0].get("role") == "system":
        start = 1
    return sum(_estimate_message_tokens(msg) for msg in history[start:])


def _summarize_message(message: dict) -> str:
    role = message.get("role", "user")
    if _is_memory_note(message):
        body = _truncate(_memory_note_body(message), 60)
        return f"previous memory: {body}"

    text = _truncate(_message_to_text(message), 60)
    if not text:
        text = "[empty]"
    if role == "assistant":
        return f"assistant: {text}"
    if role == "system":
        return f"system: {text}"
    return f"user: {text}"


def _build_memory_note(messages: list[dict], *, max_lines: int, max_chars: int) -> str:
    lines = [
        _MEMORY_NOTE_PREFIX,
        "Older turns were compressed to stay within budget.",
    ]
    for idx, message in enumerate(messages):
        if idx >= max_lines:
            remaining = len(messages) - idx
            lines.append(f"- ... {remaining} more message(s) omitted")
            break
        lines.append(f"- {_summarize_message(message)}")

        body = "\n".join(lines)
        if len(body) >= max_chars:
            return body[: max(0, max_chars - 20)].rstrip() + "\n- ..."

    return "\n".join(lines)


def _compact_history_pass(history: list[dict], *, target_tokens: int) -> list[dict]:
    if len(history) <= 2:
        return history

    system = history[0]
    source = history[1:]
    existing_memory: list[dict] = []
    if source and _is_memory_note(source[0]):
        existing_memory = [source[0]]
        source = source[1:]

    if not source:
        return history

    reserve = max(60, min(1200, max(60, target_tokens // 4)))
    tail_budget = max(0, target_tokens - reserve)

    tail: list[dict] = []
    tail_tokens = 0
    for msg in reversed(source):
        msg_tokens = _estimate_message_tokens(msg)
        if tail and tail_tokens + msg_tokens > tail_budget:
            break
        tail.append(msg)
        tail_tokens += msg_tokens

    tail.reverse()
    compressed = source[: len(source) - len(tail)]
    if not compressed and source:
        compressed = source[:-1]
        tail = source[-1:]

    if existing_memory:
        compressed = existing_memory + compressed

    memory_note = {
        "role": "system",
        "content": _build_memory_note(
            compressed,
            max_lines=8,
            max_chars=max(120, reserve * 2),
        ),
    }

    return [system, memory_note] + tail


# ══════════════════════════════════════════════════════════════════════════════
# VLM CLIENT  (mlx_vlm.server OpenAI-compatible /v1/chat/completions)
# ══════════════════════════════════════════════════════════════════════════════

def _check_server(cfg: ServerConfig, timeout: float = 3.0) -> bool:
    if not _REQUESTS:
        ui.error("requests not installed — pip install requests")
        return False
    try:
        r = requests.get(f"{cfg.base_url}/v1/models", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _build_user_content(
    text: str,
    image_path: Optional[str] = None,
) -> list[dict]:
    """Build the content array for a user message (text + optional image)."""
    content: list[dict] = []
    if image_path:
        b64 = encode_image(image_path)
        mt  = image_media_type(image_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mt};base64,{b64}"},
        })
    content.append({"type": "text", "text": text})
    return content


def vlm_complete(
    cfg: ServerConfig,
    messages: list[dict],
    stream: bool = True,
) -> str:
    """
    POST to mlx_vlm.server /v1/chat/completions.
    Returns the full assistant reply as a string.
    Streams tokens to stdout if stream=True.
    """
    if not _REQUESTS:
        raise RuntimeError("pip install requests")

    payload = {
        "model":              cfg.model,
        "messages":           messages,
        "max_tokens":         cfg.max_tokens,
        "temperature":        cfg.temperature,
        "top_p":              cfg.top_p,
        "repetition_penalty": cfg.repetition_penalty,
        "stream":             stream,
    }

    url = f"{cfg.base_url}/v1/chat/completions"
    full_reply = ""

    if stream:
        with requests.post(url, json=payload, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8")
                if line.startswith("data: "):
                    chunk_str = line[6:]
                    if chunk_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(chunk_str)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            print(delta, end="", flush=True)
                            full_reply += delta
                    except (json.JSONDecodeError, KeyError):
                        pass
        print()  # newline after streamed output
    else:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        full_reply = data["choices"][0]["message"]["content"]

    return full_reply


# ══════════════════════════════════════════════════════════════════════════════
# COMMAND: serve
# ══════════════════════════════════════════════════════════════════════════════

def cmd_serve(cfg: ServerConfig):
    """
    Launch mlx_vlm.server as a subprocess with inlined params.
    Blocks until Ctrl-C.
    """
    ui.rule("mlx_vlm.server")
    ui.info(f"Model : {cfg.model}")
    ui.info(f"Bind  : {cfg.host}:{cfg.port}")
    ui.info(f"Cmd   : {' '.join(cfg.server_argv)}")
    ui.rule()
    print()
    try:
        proc = subprocess.run(cfg.server_argv)
        sys.exit(proc.returncode)
    except KeyboardInterrupt:
        ui.info("Server stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# COMMAND: extract  (one-shot SKILL.md context extraction)
# ══════════════════════════════════════════════════════════════════════════════

_EXTRACT_PROMPT_TEMPLATE = """\
You are reading a SKILL.md specification file that describes tool capabilities \
available in a local AI development environment.

SKILL.md PATH: {path}

SKILL.md CONTENT:
──────────────────────────────────────────────────────────
{content}
──────────────────────────────────────────────────────────

{image_note}

Extract and return a structured context summary with these sections:
1. **Skill Name & Purpose** — one sentence.
2. **Trigger Conditions** — bullet list of when this skill should be invoked.
3. **Key Capabilities** — what this skill produces or enables.
4. **Input Requirements** — file types, data formats, environment constraints.
5. **Output Artefacts** — what files/objects are produced.
6. **Integration Notes** — how this skill connects to QLoRA training, RAG, or agentic pipelines.
7. **Compliance / Robotics Relevance** — specific applicability to BITSTRIC domains.
8. **Recommended LoRA Training Data** — if this skill relates to document types, \
describe how its outputs could become training records (text/JSONL format).
"""

def cmd_extract(
    skill_path: str,
    image_path: Optional[str],
    cfg: ServerConfig,
):
    ui.rule("SKILL.md Context Extraction")

    if not _check_server(cfg):
        ui.error(
            f"mlx_vlm.server not reachable at {cfg.base_url}\n"
            "  Run first:  python -m nomadic serve"
        )
        sys.exit(1)

    ui.info(f"Reading: {skill_path}")
    skill_text = read_skill(skill_path, cfg.max_skill_chars)

    image_note = ""
    if image_path:
        ui.info(f"Image  : {image_path}")
        image_note = (
            "An image has also been provided alongside the SKILL.md. "
            "Analyse it and incorporate any relevant visual information "
            "(diagrams, architecture charts, screenshots) into the summary."
        )

    prompt = _EXTRACT_PROMPT_TEMPLATE.format(
        path=skill_path,
        content=skill_text,
        image_note=image_note,
    )

    messages = [
        {
            "role": "user",
            "content": _build_user_content(prompt, image_path),
        }
    ]

    ui.rule("VLM Response")
    t0 = time.perf_counter()
    reply = vlm_complete(cfg, messages, stream=True)
    elapsed = time.perf_counter() - t0

    ui.rule()
    ui.success(f"Done in {elapsed:.1f}s — {len(reply.split()):,} words")

    # Persist to evals/ for QLoRA pipeline consumption
    out_dir = pathlib.Path("evals")
    out_dir.mkdir(exist_ok=True)
    safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", pathlib.Path(skill_path).stem)
    out_path = out_dir / f"skill_context_{safe_name}.json"
    out_path.write_text(json.dumps({
        "skill_path": skill_path,
        "image_path": image_path,
        "model": cfg.model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "context": reply,
    }, indent=2))
    ui.info(f"Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# COMMAND: kernel  (interactive structured tool-calling loop)
# ══════════════════════════════════════════════════════════════════════════════

_KERNEL_BANNER = r"""
 ███╗   ██╗ ██████╗ ███╗   ███╗ █████╗ ██████╗ ██╗ ██████╗
 ████╗  ██║██╔═══██╗████╗ ████║██╔══██╗██╔══██╗██║██╔════╝
 ██╔██╗ ██║██║   ██║██╔████╔██║███████║██║  ██║██║██║
 ██║╚██╗██║██║   ██║██║╚██╔╝██║██╔══██║██║  ██║██║██║
 ██║ ╚████║╚██████╔╝██║ ╚═╝ ██║██║  ██║██████╔╝██║╚██████╗
 ╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝ ╚═════╝
                              by Bitstric Labs
"""

_KERNEL_HELP = """
┌─ Kernel Commands ───────────────────────────────────────────┐
│  /skill <path>      Load a SKILL.md into context            │
│  /image <path>      Attach an image to next message         │
│  /exec <cmd>        Run a shell command (output → context)  │
│  /clear             Clear conversation history              │
│  /history           Print conversation history              │
│  /config            Show current ServerConfig               │
│  /session           Show active Session metadata            │
│  /model <name>      Switch model (restarts server needed)   │
│  /temp <0.0–1.0>    Set temperature                         │
│  /tokens <n>        Set max_tokens per turn                 │
│  /save <file>       Save conversation to JSON               │
│  /help              This message                            │
│  /quit  or  Ctrl-D  Exit                                    │
│  Ctrl-C             Press twice quickly to exit             │
└─────────────────────────────────────────────────────────────┘
Input editor:
  Enter submits the current message.
  Shift+Enter inserts a newline in the enhanced editor.
  If prompt_toolkit is unavailable, the kernel falls back to the
  classic single-line prompt.
Post-response next steps:
  After each agent reply, the kernel suggests:
  1) recommended next action
  2) alternate next action
  3) alternate next action
  4) other (free text)
  Type 1-4 at the next prompt to use them.
Model tool-call protocol:
  {"tool_call":{"name":"read_skill","arguments":{"path":"..."}}}
  {"tool_call":{"name":"exec_shell","arguments":{"command":"..."}}}
  {"tool_call":{"name":"metadata_search","arguments":{"query":"..."}}}
Context management:
  Automatic compression keeps the system prompt + newest turns
  when the approximate non-system history budget is exceeded.
  Configure with --max-skill-chars, --max-context-tokens, and
  --context-recompress-to.
"""

@dataclass
class KernelState:
    cfg: ServerConfig
    conn: Any = None
    session: Optional[dict[str, Any]] = None
    history: list[dict] = field(default_factory=list)
    pending_image: Optional[str] = None
    next_actions: list[str] = field(default_factory=list)
    turn: int = 0
    last_message_id: Optional[int] = None
    last_user_message_id: Optional[int] = None
    last_assistant_message_id: Optional[int] = None

    @property
    def session_id(self) -> Optional[str]:
        return self.session.get("id") if self.session else None

    @property
    def chat_name(self) -> str:
        return self.session.get("chat_name", "Chat") if self.session else "Chat"

    def reset_history(self):
        self.history = [{
            "role": "system",
            "content": self.cfg.cot_system_prompt,
        }]
        self.next_actions = []

    def add_user(
        self,
        text: str,
        image_path: Optional[str] = None,
        *,
        auto_title: bool = True,
    ):
        parent_id = self.last_message_id
        content = _build_user_content(text, image_path)
        self.history.append({
            "role": "user",
            "content": content,
        })
        self.pending_image = None
        if self.conn and self.session_id:
            msg_id = chatdb.append_message(
                self.conn,
                self.session_id,
                role="user",
                content_text=text,
                content_json=content,
                content_type="multimodal" if image_path else "text",
                parent_message_id=parent_id,
                metadata={
                    "source": "kernel",
                    "turn": self.turn,
                    "has_image": bool(image_path),
                },
            )
            self.last_message_id = msg_id
            self.last_user_message_id = msg_id
            if image_path:
                chatdb.record_attachment(
                    self.conn,
                    self.session_id,
                    message_id=msg_id,
                    kind="image",
                    file_path=image_path,
                    metadata={"source": "kernel", "turn": self.turn},
                )
            if auto_title and self.session and not self.session.get("started_at") and text.strip():
                before_title = self.session.get("chat_name", "")
                self.session = dict(
                    chatdb.finalize_session_title(
                        self.conn,
                        self.session_id,
                        user_text=text,
                    )
                )
                if self.session.get("chat_name") != before_title:
                    ui.success(f"Chat Name → {self.session['chat_name']}")
            return msg_id
        return None

    def add_assistant(self, text: str):
        parent_id = self.last_message_id
        self.history.append({
            "role": "assistant",
            "content": text,
        })
        if self.conn and self.session_id:
            msg_id = chatdb.append_message(
                self.conn,
                self.session_id,
                role="assistant",
                content_text=text,
                content_type="text",
                parent_message_id=parent_id,
                metadata={"source": "kernel", "turn": self.turn},
            )
            self.last_message_id = msg_id
            self.last_assistant_message_id = msg_id
            return msg_id
        return None

    def record_tool_invocation(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        depth: int,
        raw_json: str = "",
    ):
        if self.conn and self.session_id:
            chatdb.record_event(
                self.conn,
                self.session_id,
                event_type="tool",
                event_name="call",
                status="ok",
                payload={
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "depth": depth,
                    "raw_json": raw_json,
                    "turn": self.turn,
                },
                message_id=self.last_assistant_message_id,
            )

    def add_tool_result(
        self,
        label: str,
        result: str,
        *,
        status: str = "ok",
        payload: Optional[dict[str, Any]] = None,
        tool_name: Optional[str] = None,
        arguments: Optional[dict[str, Any]] = None,
        depth: Optional[int] = None,
    ):
        """Inject tool/EXEC output as a user-role system note."""
        note = f"[KERNEL TOOL RESULT — {label}]\n{result}\n[END TOOL RESULT]"
        self.history.append({"role": "user", "content": note})
        if self.conn and self.session_id:
            metadata = {
                "source": "kernel",
                "label": label,
                "turn": self.turn,
                "status": status,
            }
            if tool_name is not None:
                metadata["tool_name"] = tool_name
            if arguments is not None:
                metadata["arguments"] = arguments
            if depth is not None:
                metadata["depth"] = depth
            msg_id = chatdb.append_message(
                self.conn,
                self.session_id,
                role="tool",
                content_text=note,
                content_type="tool_result",
                parent_message_id=self.last_message_id,
                metadata=metadata,
            )
            chatdb.record_event(
                self.conn,
                self.session_id,
                event_type="tool",
                event_name="result",
                status=status,
                payload={
                    "label": label,
                    "bytes": len(result.encode("utf-8")),
                    "turn": self.turn,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "depth": depth,
                    **(payload or {}),
                },
                message_id=msg_id,
            )
            self.last_message_id = msg_id
            return msg_id
        return None

    def inject_context(self, text: str, *, source: str, metadata: Optional[dict[str, Any]] = None):
        self.history.append({
            "role": "system",
            "content": text,
        })
        if self.conn and self.session_id:
            msg_id = chatdb.append_message(
                self.conn,
                self.session_id,
                role="system",
                content_text=text,
                content_type="system_note",
                parent_message_id=self.last_message_id,
                metadata={
                    "source": source,
                    **(metadata or {}),
                },
            )
            self.last_message_id = msg_id
            chatdb.record_event(
                self.conn,
                self.session_id,
                event_type="session",
                event_name=source,
                status="ok",
                payload={
                    "chars": len(text),
                    **(metadata or {}),
                },
                message_id=msg_id,
            )
            return msg_id
        return None

    def compact_history_if_needed(self) -> bool:
        """
        Compress older turns into a memory note when the approximate context
        budget is exceeded. Returns True when compaction happened.
        """
        budget = max(0, self.cfg.max_context_tokens)
        if budget <= 0 or len(self.history) <= 2:
            return False

        current_tokens = _estimate_history_tokens(self.history)
        if current_tokens <= budget:
            return False

        target = min(budget, max(1, self.cfg.context_recompress_to))
        working_history = self.history

        for _ in range(6):
            new_history = _compact_history_pass(working_history, target_tokens=target)
            if new_history == working_history:
                break

            working_history = new_history
            if _estimate_history_tokens(working_history) <= budget:
                break

            target = max(1, target // 2)

        if working_history == self.history:
            return False

        self.history = working_history
        summary_text = ""
        if len(self.history) > 1 and isinstance(self.history[1].get("content"), str):
            summary_text = self.history[1]["content"]
        after_tokens = _estimate_history_tokens(self.history)
        if self.conn and self.session_id:
            self.session = dict(
                chatdb.update_session(
                    self.conn,
                    self.session_id,
                    summary=summary_text or None,
                )
            )
            chatdb.record_event(
                self.conn,
                self.session_id,
                event_type="context",
                event_name="compressed",
                status="ok",
                payload={
                    "before_tokens": current_tokens,
                    "after_tokens": after_tokens,
                    "budget": budget,
                    "recompress_to": target,
                    "summary": summary_text[:500],
                },
            )
        ui.info(
            "[context] compressed older turns "
            f"→ ~{after_tokens:,}/{budget:,} tokens"
        )
        return True

def _load_kernel_state(
    conn,
    session_row: Any,
    cfg: ServerConfig,
) -> KernelState:
    messages = chatdb.load_messages(conn, session_row["id"])
    history = chatdb.build_runtime_history(conn, session_row)
    last_message_id = messages[-1]["id"] if messages else None
    last_user_message_id = next(
        (row["id"] for row in reversed(messages) if row["role"] == "user"),
        None,
    )
    last_assistant_message_id = next(
        (row["id"] for row in reversed(messages) if row["role"] == "assistant"),
        None,
    )
    return KernelState(
        cfg=cfg,
        conn=conn,
        session=dict(session_row),
        history=history,
        turn=int(session_row["user_turn_count"] or 0),
        last_message_id=last_message_id,
        last_user_message_id=last_user_message_id,
        last_assistant_message_id=last_assistant_message_id,
    )


def _prompt_with_default(label: str, default: Optional[str] = None) -> str:
    prompt = label if default in (None, "") else f"{label} [{default}]"
    value = ui.prompt(prompt).strip()
    return value or (default or "")


def _resolve_initialize_settings(
    *,
    db_kind: Optional[str],
    db_path: Optional[str],
    postgres_dsn: Optional[str],
    config_path: Optional[str],
) -> chatdb.DatabaseSettings:
    existing = chatdb.load_database_settings_from_config(config_path)
    chosen_kind = db_kind or (existing.kind if existing else None)

    while chosen_kind not in chatdb.DATABASE_KINDS:
        chosen_kind = _prompt_with_default("Database kind", "sqlite3").lower()
        if chosen_kind not in chatdb.DATABASE_KINDS:
            ui.warn("Choose either `sqlite3` or `postgres`.")

    if chosen_kind == "sqlite3":
        default_path = (
            db_path
            or (existing.sqlite_path if existing else None)
            or str(chatdb.DEFAULT_DB_PATH)
        )
        sqlite_path = db_path or _prompt_with_default("SQLite database path", default_path)
        return chatdb.DatabaseSettings(kind="sqlite3", sqlite_path=sqlite_path)

    default_dsn = postgres_dsn or (existing.postgres_dsn if existing else None) or ""
    dsn = postgres_dsn or _prompt_with_default("Postgres DSN", default_dsn)
    while not dsn.strip():
        ui.warn("Postgres DSN is required when `--db-kind postgres` is selected.")
        dsn = _prompt_with_default("Postgres DSN", default_dsn)
    return chatdb.DatabaseSettings(kind="postgres", postgres_dsn=dsn.strip())


def cmd_initialize(
    *,
    db_kind: Optional[str],
    db_path: Optional[str],
    postgres_dsn: Optional[str],
    config_path: Optional[str],
):
    """Create ~/.config/nomadic/config.jsonc and validate the selected database."""

    settings = _resolve_initialize_settings(
        db_kind=db_kind,
        db_path=db_path,
        postgres_dsn=postgres_dsn,
        config_path=config_path,
    )
    config_file = chatdb.write_config_file(settings, config_path)

    conn = chatdb.connect(settings)
    try:
        chatdb.init_db(conn)
    finally:
        conn.close()

    ui.success("Nomadic database configuration initialized.")
    ui.info(f"Config Path : {config_file}")
    ui.info(f"DB Kind     : {settings.kind}")
    ui.info(f"DB Target   : {chatdb.describe_database_target(settings)}")


def cmd_kernel(cfg: ServerConfig, db_settings: chatdb.DatabaseSettings, session_id: Optional[str]):
    """Interactive structured tool-calling kernel."""

    conn = chatdb.connect(db_settings)
    db_target = chatdb.describe_database_target(db_settings)
    state: Optional[KernelState] = None
    tool_registry = agent.build_default_registry()
    input_layer = KernelInput(ui)
    interrupt_controller = KernelInterruptController()
    try:
        chatdb.init_db(conn)

        if not _check_server(cfg):
            ui.error(
                f"mlx_vlm.server not reachable at {cfg.base_url}\n"
                "  Run first:  python -m nomadic serve"
            )
            sys.exit(1)

        try:
            if session_id:
                session = chatdb.resume_session(conn, session_id)
                mode = "resumed"
            else:
                session = chatdb.create_session(
                    conn,
                    system_prompt=cfg.cot_system_prompt,
                    config=cfg,
                )
                mode = "created"
        except KeyError as exc:
            ui.error(str(exc))
            return

        state = _load_kernel_state(conn, session, cfg)
        chatdb.record_event(
            conn,
            state.session_id,
            event_type="session",
            event_name=mode,
            status="ok",
            payload={
                "chat_name": state.chat_name,
                "database_kind": db_settings.kind,
                "database_target": db_target,
            },
        )

        if _RICH:
            ui.con.print(_KERNEL_BANNER, style="bold green")
        else:
            print(_KERNEL_BANNER)
        print(_KERNEL_HELP)
        ui.info(f"Session   : {state.session_id}")
        ui.info(f"Chat Name : {state.chat_name}")
        ui.info(f"DB Kind   : {db_settings.kind}")
        ui.info(f"DB Target : {db_target}")
        ui.info("Agent tools: read_skill, exec_shell, metadata_search")
        ui.info(f"Metadata  : {cfg.metadata_base_url}")
        ui.info(
            "Context budget (non-system history): "
            f"~{cfg.max_context_tokens:,} tokens "
            f"(recompress to ~{cfg.context_recompress_to:,})"
        )
        ui.info(f"SKILL.md cap: {cfg.max_skill_chars:,} chars")

        while True:
            # ── Prompt ──────────────────────────────────────────────────────────
            prompt_turn = state.turn + 1
            try:
                raw = input_layer.prompt(f"[{prompt_turn}] you")
            except EOFError:
                print()
                ui.info("Kernel exit.")
                break
            except KeyboardInterrupt:
                print()
                if interrupt_controller.arm_or_exit():
                    ui.info("Kernel exit.")
                    break
                ui.warn(
                    "Press Ctrl-C again within "
                    f"{int(interrupt_controller.window_seconds)}s to exit."
                )
                continue

            interrupt_controller.clear()
            submission = parse_kernel_input(raw)
            if submission.kind == "empty":
                continue

            selection = resolve_next_action_selection(submission, state.next_actions)
            if selection.kind == "selected" and selection.text:
                ui.info(f"Next step  : {selection.text}")
                submission = parse_kernel_input(selection.text)
            elif selection.kind == "other":
                try:
                    raw = input_layer.prompt(f"[{prompt_turn}] you · other")
                except EOFError:
                    print()
                    ui.info("Kernel exit.")
                    break
                except KeyboardInterrupt:
                    print()
                    if interrupt_controller.arm_or_exit():
                        ui.info("Kernel exit.")
                        break
                    ui.warn(
                        "Press Ctrl-C again within "
                        f"{int(interrupt_controller.window_seconds)}s to exit."
                    )
                    continue
                interrupt_controller.clear()
                submission = parse_kernel_input(raw)
                if submission.kind == "empty":
                    continue

            state.turn = prompt_turn

            # ── Built-in commands ────────────────────────────────────────────
            if submission.kind == "command":
                parts = submission.stripped.split(maxsplit=1)
                cmd = parts[0].lower()
                rest = parts[1] if len(parts) > 1 else ""

                if cmd in ("/quit", "/exit", "/q"):
                    ui.info("Kernel exit.")
                    break

                elif cmd == "/help":
                    print(_KERNEL_HELP)

                elif cmd == "/clear":
                    prev_turn = state.turn
                    state.reset_history()
                    state.turn = 0
                    state.pending_image = None
                    state.last_message_id = None
                    state.last_user_message_id = None
                    state.last_assistant_message_id = None
                    if state.session_id:
                        chatdb.record_event(
                            state.conn,
                            state.session_id,
                            event_type="session",
                            event_name="history_cleared",
                            status="ok",
                            payload={"turn": prev_turn},
                        )
                    ui.success("History cleared.")

                elif cmd == "/history":
                    for i, msg in enumerate(state.history):
                        role = msg["role"]
                        content = msg["content"]
                        if isinstance(content, list):
                            content = " ".join(
                                c.get("text", "") for c in content if c.get("type") == "text"
                            )
                        ui.panel(f"[{i}] {role}", content[:500])

                elif cmd == "/config":
                    ui.panel("ServerConfig", json.dumps({
                        "model":                 cfg.model,
                        "host":                  cfg.host,
                        "port":                  cfg.port,
                        "max_tokens":            cfg.max_tokens,
                        "temperature":           cfg.temperature,
                        "top_p":                 cfg.top_p,
                        "max_skill_chars":       cfg.max_skill_chars,
                        "max_context_tokens":    cfg.max_context_tokens,
                        "context_recompress_to": cfg.context_recompress_to,
                        "metadata_base_url":     cfg.metadata_base_url,
                        "database_kind":         db_settings.kind,
                        "database_target":       db_target,
                    }, indent=2))
                    if state.session:
                        ui.panel("Session", json.dumps(state.session, indent=2, default=str))

                elif cmd == "/session":
                    if state.session:
                        session_payload = {
                            "session_id": state.session_id,
                            "chat_name": state.chat_name,
                            "status": state.session.get("status"),
                            "created_at": state.session.get("created_at"),
                            "started_at": state.session.get("started_at"),
                            "ended_at": state.session.get("ended_at"),
                            "message_count": state.session.get("message_count"),
                            "user_turn_count": state.session.get("user_turn_count"),
                            "assistant_turn_count": state.session.get("assistant_turn_count"),
                            "tool_event_count": state.session.get("tool_event_count"),
                            "last_message_at": state.session.get("last_message_at"),
                            "summary": state.session.get("summary"),
                            "database_kind": db_settings.kind,
                            "database_target": db_target,
                        }
                        ui.panel("Session", json.dumps(session_payload, indent=2, default=str))
                    else:
                        ui.warn("No active session loaded.")

                elif cmd == "/model":
                    if rest:
                        cfg.model = rest.strip()
                        ui.success(f"Model → {cfg.model}  (restart server to apply)")

                elif cmd == "/temp":
                    try:
                        cfg.temperature = float(rest)
                        ui.success(f"Temperature → {cfg.temperature}")
                    except ValueError:
                        ui.warn("Usage: /temp 0.15")

                elif cmd == "/tokens":
                    try:
                        cfg.max_tokens = int(rest)
                        ui.success(f"max_tokens → {cfg.max_tokens}")
                    except ValueError:
                        ui.warn("Usage: /tokens 1024")

                elif cmd == "/skill":
                    if not rest:
                        ui.warn("Usage: /skill <path>")
                        continue
                    skill_result = agent.run_named_tool(
                        "read_skill",
                        {"path": rest.strip()},
                        runtime=agent.AgentRuntime(state=state, ui=ui),
                        registry=tool_registry,
                    )
                    if skill_result.status != "ok":
                        ui.error(skill_result.payload.get("error", skill_result.content))
                        continue
                    try:
                        skill_text = str(skill_result.payload["content"])
                        inject = (
                            f"[SKILL.md LOADED — {rest.strip()}]\n"
                            f"{skill_text}\n"
                            "[Use this context for subsequent questions.]"
                        )
                        state.inject_context(
                            inject,
                            source="skill_loaded",
                            metadata={
                                "path": str(skill_result.payload["path"]),
                                "chars": len(skill_text),
                                "truncated": bool(skill_result.payload.get("truncated")),
                            },
                        )
                        ui.success(f"SKILL.md injected into context ({len(skill_text):,} chars).")
                    except Exception as e:
                        ui.error(str(e))

                elif cmd == "/image":
                    if not rest:
                        ui.warn("Usage: /image <path>")
                        continue
                    p = pathlib.Path(rest.strip())
                    if not p.exists():
                        ui.error(f"Image not found: {p}")
                    else:
                        state.pending_image = str(p)
                        ui.success(f"Image queued for next message: {p}")

                elif cmd == "/exec":
                    if not rest:
                        ui.warn("Usage: /exec <shell command>")
                        continue
                    try:
                        result = agent.run_named_tool(
                            "exec_shell",
                            {"command": rest},
                            runtime=agent.AgentRuntime(state=state, ui=ui),
                            registry=tool_registry,
                        )
                    except KeyboardInterrupt:
                        ui.warn("Interrupted current tool run; returning to the kernel prompt.")
                        continue
                    print(result.payload.get("preview") or "[no output]")
                    state.add_tool_result(
                        result.label,
                        result.content,
                        status=result.status,
                        payload=result.payload,
                        tool_name=result.tool_name,
                        arguments={"command": rest},
                    )
                    if result.status == "ok":
                        ui.success("Result injected into context.")
                    else:
                        ui.warn("Command failed; result injected into context.")

                elif cmd == "/save":
                    fname = rest.strip() or f"kernel_session_{int(time.time())}.json"
                    export_payload = {
                        "session": state.session,
                        "history": state.history,
                    }
                    pathlib.Path(fname).write_text(
                        json.dumps(export_payload, indent=2, default=str)
                    )
                    ui.success(f"Saved → {fname}")

                else:
                    ui.warn(f"Unknown command: {cmd}  — type /help")

                continue  # skip VLM call for built-in commands

            # ── Normal user turn → VLM completion ───────────────────────────
            state.next_actions = []
            state.add_user(submission.raw, state.pending_image)
            state.compact_history_if_needed()

            ui.rule(f"turn {state.turn} · agent response")
            t0 = time.perf_counter()

            try:
                reply = agent.run_agent_loop(
                    state=state,
                    complete_fn=vlm_complete,
                    ui=ui,
                    registry=tool_registry,
                )
            except KeyboardInterrupt:
                ui.warn("Interrupted current run; returning to the kernel prompt.")
                continue
            except Exception as e:
                ui.error(f"VLM error: {e}")
                continue

            elapsed = time.perf_counter() - t0

            ui.rule()
            ui.info(
                f"{elapsed:.1f}s · {len(reply.split()):,} words · "
                f"history ~{_estimate_history_tokens(state.history):,} tokens · "
                f"{len(state.history)} messages"
            )
            state.next_actions = generate_next_action_options(cfg, state.history)
            menu = render_next_action_menu(state.next_actions)
            if menu:
                ui.panel("User Action On Next Step", menu)
    finally:
        if state and state.session_id:
            try:
                chatdb.close_session(
                    conn,
                    state.session_id,
                    summary=state.session.get("summary") if state.session else None,
                )
            except Exception:
                pass
        conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# COMMAND: list-skills
# ══════════════════════════════════════════════════════════════════════════════

def cmd_list_skills(root: str = "/mnt/skills"):
    """Walk the skills tree and print every SKILL.md found."""
    root_p = pathlib.Path(root)
    found = sorted(root_p.rglob("SKILL.md"))
    if not found:
        ui.warn(f"No SKILL.md files found under {root}")
        return
    ui.rule(f"SKILL.md inventory ({len(found)} found)")
    for p in found:
        rel = p.relative_to(root_p)
        ui.info(str(rel))
    ui.rule()


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nomadic",
        description="BITSTRIC · mlx_vlm.server multimodal SKILL.md extractor + structured agent kernel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          # One-shot extraction
          python -m nomadic extract /mnt/skills/public/docx/SKILL.md

          # One-shot with an architecture diagram
          python -m nomadic extract /mnt/skills/public/docx/SKILL.md \\
                 --image /tmp/arch.png

          # Interactive kernel
          python -m nomadic kernel

          # Initialize config-backed database settings
          python -m nomadic initialize --db-kind sqlite3 --db-path kernel.sqlite3

          # Start the VLM server (separate terminal)
          python -m nomadic serve

          # Initialize or upgrade the session DB directly
          nomadic-db init --db-kind postgres --postgres-dsn postgresql://user:pass@host/db

          # List all SKILL.md files
          python -m nomadic list-skills

          # Override model inline
          python -m nomadic kernel \\
                 --model mlx-community/Pixtral-12B-2409-4bit \\
                 --port 8081 --temperature 0.05

          # Adjust the context budget
          python -m nomadic kernel \\
                 --max-context-tokens 12000 \\
                 --context-recompress-to 8000 \\
                 --max-skill-chars 12000

          # Resume an existing chat session
          python -m nomadic kernel --session-id <uuidv4>
        """),
    )

    def add_global_overrides(parser: argparse.ArgumentParser, *, suppress_defaults: bool = False):
        """Register shared VLM overrides on a parser."""
        default = argparse.SUPPRESS if suppress_defaults else None
        parser.add_argument("--model",       default=default, metavar="MODEL_ID",
                            help="mlx-community model to load (overrides default in ServerConfig)")
        parser.add_argument("--host",        default=default, help="Server host (default: 127.0.0.1)")
        parser.add_argument("--port",        default=default, type=int, help="Server port (default: 8080)")
        parser.add_argument("--temperature", default=default, type=float, help="Sampling temperature")
        parser.add_argument("--top-p",       default=default, type=float, dest="top_p")
        parser.add_argument("--max-tokens",  default=default, type=int,  dest="max_tokens")
        parser.add_argument("--max-skill-chars", default=default, type=int, dest="max_skill_chars",
                            help="Maximum SKILL.md characters to inject at once")
        parser.add_argument("--max-context-tokens", default=default, type=int, dest="max_context_tokens",
                            help="Approximate non-system history budget before auto-compression")
        parser.add_argument("--context-recompress-to", default=default, type=int,
                            dest="context_recompress_to",
                            help="Approximate non-system history target after compression")
        parser.add_argument(
            "--metadata-base-url",
            default=default,
            dest="metadata_base_url",
            help=(
                "Metadata search service base URL "
                "(default: METADATA_BASE_URL or http://localhost:8888)"
            ),
        )
        chatdb.add_db_selection_args(parser, suppress_defaults=suppress_defaults)
        parser.add_argument("--session-id", default=default, dest="session_id",
                            help="Resume a specific chat session UUID")

    # Accept globals before the command.
    add_global_overrides(p)

    sub = p.add_subparsers(dest="command", required=True)

    # Accept globals after the command as well, without overwriting values
    # already parsed by the root parser.
    sub_common = argparse.ArgumentParser(add_help=False)
    add_global_overrides(sub_common, suppress_defaults=True)

    # serve
    sub.add_parser("serve", help="Start mlx_vlm.server with inlined config params",
                   parents=[sub_common])

    # extract
    ext = sub.add_parser("extract", help="One-shot SKILL.md context extraction",
                         parents=[sub_common])
    ext.add_argument("skill_path", help="Absolute or relative path to SKILL.md")
    ext.add_argument("--image", default=None, metavar="IMAGE_PATH",
                     help="Optional image to attach (diagram, screenshot, etc.)")

    # kernel
    sub.add_parser("kernel", help="Launch interactive structured tool-calling kernel",
                   parents=[sub_common])

    # initialize
    sub.add_parser("initialize", help="Write config.jsonc and bootstrap the selected database",
                   parents=[sub_common])

    # list-skills
    ls = sub.add_parser("list-skills", help="Walk and list all SKILL.md files",
                        parents=[sub_common])
    ls.add_argument("--root", default="/mnt/skills", help="Root directory to scan")

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    # ── Apply global overrides to ServerConfig ────────────────────────────────
    cfg = ServerConfig()
    if args.model is not None:       cfg.model       = args.model
    if args.host is not None:        cfg.host        = args.host
    if args.port is not None:        cfg.port        = args.port
    if args.temperature is not None: cfg.temperature = args.temperature
    if args.top_p is not None:       cfg.top_p       = args.top_p
    if args.max_tokens is not None:  cfg.max_tokens  = args.max_tokens
    if args.max_skill_chars is not None:
        cfg.max_skill_chars = args.max_skill_chars
    if args.max_context_tokens is not None:
        cfg.max_context_tokens = args.max_context_tokens
    if args.context_recompress_to is not None:
        cfg.context_recompress_to = args.context_recompress_to
    if args.metadata_base_url is not None:
        cfg.metadata_base_url = args.metadata_base_url

    if cfg.context_recompress_to > cfg.max_context_tokens:
        ui.warn(
            "context_recompress_to exceeds max_context_tokens; "
            "clamping recompress target to the max budget."
        )
        cfg.context_recompress_to = cfg.max_context_tokens

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if args.command == "serve":
        cmd_serve(cfg)

    elif args.command == "extract":
        cmd_extract(args.skill_path, args.image, cfg)

    elif args.command == "initialize":
        cmd_initialize(
            db_kind=getattr(args, "db_kind", None),
            db_path=getattr(args, "db_path", None),
            postgres_dsn=getattr(args, "postgres_dsn", None),
            config_path=getattr(args, "config_path", None),
        )

    elif args.command == "kernel":
        try:
            db_settings = chatdb.resolve_db_settings_from_args(args, use_config=True)
        except ValueError as exc:
            parser.error(str(exc))
        cmd_kernel(cfg, db_settings, args.session_id)

    elif args.command == "list-skills":
        cmd_list_skills(args.root)


if __name__ == "__main__":
    main()
