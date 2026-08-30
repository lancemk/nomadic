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
import os
import readline  # noqa: F401 — enables arrow-key history in input()
from dataclasses import dataclass, field
from typing import Any, Generator, Iterable, Optional

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
    from prompt_toolkit.layout.containers import HSplit, Window, FloatContainer, Float
    from prompt_toolkit.layout.dimension import Dimension
    from prompt_toolkit.layout.menus import CompletionsMenu
    from prompt_toolkit.styles import Style
    from prompt_toolkit.widgets import Frame, Label, TextArea
    from prompt_toolkit.completion import WordCompleter, Completer, Completion, PathCompleter
    from prompt_toolkit.document import Document
    from prompt_toolkit.history import FileHistory
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
from .hitl import HITLManager, SafetyProfile, ReviewDecision
from .tasks import TaskRegistry, TaskStatus
from .orchestrator import Orchestrator, AgentInstance, AgentStatus

_MEMORY_NOTE_PREFIX = "[KERNEL MEMORY SUMMARY]"
_KERNEL_INPUT_ACCENT = "#ea7100"
_KERNEL_EXIT_ARM_WINDOW_SECONDS = 2.0


def normalize_model_name(model_name: str) -> str:
    """
    Normalize model name to match huggingface-cli / hf download repo-id.
    Strips hf:// prefix and optional repo type prefix (e.g. models/).
    """
    cleaned = model_name.strip()
    if cleaned.startswith("hf://"):
        cleaned = cleaned[5:]
    if cleaned.startswith("models/"):
        cleaned = cleaned[7:]
    return cleaned


def get_safe_memory_budget() -> float:
    """
    Returns the safe memory budget in GB for inference,
    reserving at least 8GB or 25% of total unified memory for the OS.
    """
    import subprocess
    try:
        mem_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]))
        mem_gb = mem_bytes / (1024 ** 3)
    except Exception:
        mem_gb = 16.0
    os_reserve = max(8.0, mem_gb * 0.25)
    return max(4.0, mem_gb - os_reserve)


def render_token_grid(used_pct: float, user_pct: float, assistant_pct: float, tool_pct: float, system_pct: float) -> str:
    total_blocks = 100
    user_blocks = int(round(user_pct * total_blocks))
    assistant_blocks = int(round(assistant_pct * total_blocks))
    tool_blocks = int(round(tool_pct * total_blocks))
    system_blocks = int(round(system_pct * total_blocks))
    
    used_blocks = user_blocks + assistant_blocks + tool_blocks + system_blocks
    free_blocks = max(0, total_blocks - used_blocks)
    
    blocks = []
    blocks.extend(["[blue]●[/blue]"] * user_blocks)
    blocks.extend(["[green]●[/green]"] * assistant_blocks)
    blocks.extend(["[yellow]●[/yellow]"] * tool_blocks)
    blocks.extend(["[white]●[/white]"] * system_blocks)
    blocks.extend(["□"] * free_blocks)
    
    grid = []
    for i in range(0, 100, 10):
        grid.append(" ".join(blocks[i:i+10]))
    return "\n".join(grid)


def estimate_parameters_from_config(cfg_dict: dict) -> Optional[float]:
    """Estimate model parameter count in billions from its config dictionary."""
    if "num_parameters" in cfg_dict:
        return cfg_dict["num_parameters"] / 1e9

    h = cfg_dict.get("hidden_size") or cfg_dict.get("d_model")
    l = cfg_dict.get("num_hidden_layers") or cfg_dict.get("num_layers") or cfg_dict.get("n_layer")
    ff = cfg_dict.get("intermediate_size") or cfg_dict.get("ffn_dim")
    v = cfg_dict.get("vocab_size") or 32000

    if not h or not l:
        return None

    if not ff:
        ff = 4 * h

    is_swiglu = cfg_dict.get("hidden_act") in ("silu", "swish") or cfg_dict.get("mlp_bias") is False
    mlp_factor = 3 if is_swiglu else 2

    attn_params = 4 * (h ** 2) * l
    mlp_params = mlp_factor * h * ff * l
    embed_params = v * h

    total_params = attn_params + mlp_params + embed_params
    return total_params / 1e9


def detect_parameters_from_name(model_name: str) -> Optional[float]:
    """Extract parameter count in billions from model name string (e.g. 'qwen-32b' -> 32.0)."""
    import re
    match = re.search(r'(?i)\b(\d+(?:\.\d+)?)[bm]\b', model_name)
    if match:
        val = float(match.group(1))
        if 'm' in match.group(0).lower():
            return val / 1000.0
        return val
    return None


def determine_optimal_q_bits(model_path: str) -> int:
    """Determine optimal quantization bits based on physical memory budget and model parameter count."""
    m_avail = get_safe_memory_budget()

    p = None
    # Try local path first
    local_path = pathlib.Path(model_path)
    config_file = None
    if local_path.is_dir():
        config_file = local_path / "config.json"
    else:
        try:
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(repo_id=model_path, filename="config.json")
            config_file = pathlib.Path(config_path)
        except Exception:
            pass

    if config_file and config_file.exists():
        try:
            with open(config_file, "r") as f:
                cfg_dict = json.load(f)
            p = estimate_parameters_from_config(cfg_dict)
        except Exception:
            pass

    if p is None:
        p = detect_parameters_from_name(model_path)

    if p is None:
        return 4

    # P * Q/8 * 1.15 + 4.0 <= M_avail => Q <= (M_avail - 4.0) * 8 / (P * 1.15)
    limit = m_avail - 4.0
    if limit <= 0:
        return 2

    q_max = (limit * 8) / (p * 1.15)
    if q_max >= 8.0:
        return 8
    elif q_max >= 4.0:
        return 4
    elif q_max >= 3.0:
        return 3
    else:
        return 2


def run_quantization(model_path: str, q_bits: int, cache_dir: pathlib.Path) -> str:
    """Execute mlx_lm convert / mlx_vlm.convert as a subprocess to quantize the model."""
    import shutil
    import subprocess

    is_text = is_text_only_model(model_path)
    module_name = "mlx_lm" if is_text else "mlx_vlm"

    cmd = [
        # Calling `python -m mlx_lm convert ...` directly is deprecated. Use `mlx_lm convert ...` or `python -m mlx_lm convert ...` instead.
        # sys.executable, "-m", f"{module_name}.convert",
        # 
        sys.executable, "-m", f"{module_name} convert",
        "--hf-path", model_path,
        "-q",
        "--q-bits", str(q_bits),
        "-o", str(cache_dir)
    ]

    ui.rule(f"QUANTIZATION IN PROGRESS ({q_bits}-bit)")
    ui.info(f"Source Model : {model_path}")
    ui.info(f"Target Cache : {cache_dir}")
    ui.info(f"Command      : {' '.join(cmd)}")
    ui.rule()

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        if process.stdout:
            for line in process.stdout:
                print(line, end="")
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"Quantization failed with exit code {process.returncode}")
        ui.success("Quantization completed successfully!")
    except Exception as e:
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
        raise RuntimeError(f"Quantization failed: {e}")
    return str(cache_dir)


def is_text_only_model(model_path: str) -> bool:
    """Check if model is text-only (non-multimodal) based on its configuration."""
    import json
    import pathlib
    model_path = normalize_model_name(model_path)
    local_path = pathlib.Path(model_path)
    config_file = None
    if local_path.is_dir():
        config_file = local_path / "config.json"
    else:
        try:
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(repo_id=model_path, filename="config.json")
            config_file = pathlib.Path(config_path)
        except Exception:
            pass

    if config_file and config_file.exists():
        try:
            with open(config_file, "r") as f:
                cfg = json.load(f)
            if cfg.get("language_model_only") is True:
                return True
            # Check vision configuration presence in root config or text_config
            has_vision = "vision_config" in cfg or "vision_tower" in cfg
            if not has_vision:
                return True
        except Exception:
            pass
    return False


# ══════════════════════════════════════════════════════════════════════════════
# KNOWN MODELS  (candidates offered by `/model` with no arguments)
# ══════════════════════════════════════════════════════════════════════════════

KNOWN_MODELS: list[tuple[str, str]] = [
    ("mlx-community/gemma-4-26b-a4b-it-4bit", "best stable so far, on local"),
    ("mlx-community/gemma-4-12B-bf16", "caution: prone to loops"),
    ("mlx-community/gpt-oss-20b-MXFP4-Q8", ""),
    ("mlx-community/Qwen-AgentWorld-35B-A3B-bf16", "caution: needs ~70GB RAM, OOMs on 48GB machines"),
    ("mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit", ""),
    ("mlx-community/Qwen3.6-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking-8bit", ""),
]


# ══════════════════════════════════════════════════════════════════════════════
# SERVER CONFIG  (mlx_vlm.server inline params)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ServerConfig:
    """
    Server settings plus CLI-side context controls.

│Make a indepth comparison reports, of diff in JEPA based world model, that how it address the "domain knowledge" limitation│
│ of robotics usecases, and what VLM extended model, does help in contribute (if it is even used here. COrrect if wrong) 

    # ---- Out of memory
    # model: str = "mlx-community/Qwen-AgentWorld-35B-A3B-bf16"

    # ---- This model is suceptible to "Loop of the Death"
    #
    # model: str = "mlx-community/gemma-4-12B-bf16" 

    # model: str = "mlx-community/gpt-oss-20b-MXFP4-Q8"
    
    # model: str = "mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit"
    # model: str = "mlx-community/Qwen3.6-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking-8bit"

    """
    # model: str = "mlx-community/"....

    # ── Model selection ──────────────────────────────────────────────────────
    model: str = "mlx-community/gemma-4-26b-a4b-it-4bit" # !! Best stable so far. On local

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

    # ── Profiles & Presets ──────────────────────────────────────────────────
    profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    presets: dict[str, str] = field(default_factory=dict)
    templates: dict[str, str] = field(default_factory=dict)

    # ── Runtime Quantization & Cache ─────────────────────────────────────────
    quantize: bool = False
    q_bits: str = "auto"
    cache_dir: Optional[str] = None

    # ── Structured tool-calling kernel ──────────────────────────────────────
    cot_system_prompt: str = field(default_factory=agent.default_system_prompt)

    def __setattr__(self, name: str, value: Any):
        if name == "model" and isinstance(value, str):
            value = normalize_model_name(value)
        super().__setattr__(name, value)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def server_argv(self) -> list[str]:
        """Arguments passed to the local server command."""
        if is_text_only_model(self.model):
            return [
                sys.executable, "-m", "mlx_lm", "server",
                "--model",       self.model,
                "--host",        self.host,
                "--port",        str(self.port),
            ]
        else:
            return [
                sys.executable, "-m", "mlx_vlm", "server",
                "--model",       self.model,
                "--host",        self.host,
                "--port",        str(self.port),
            ]

    def apply_profile(self, name: str) -> bool:
        """Override current settings with values from a named profile."""
        if name not in self.profiles:
            return False
        overrides = self.profiles[name]
        for key, value in overrides.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return True

    def apply_preset(self, name: str) -> bool:
        """Override system prompt with a named preset."""
        if name not in self.presets:
            return False
        self.cot_system_prompt = self.presets[name]
        return True

    def format_template(self, name: str, args_string: str) -> str | None:
        """Format a named template with positional arguments."""
        if name not in self.templates:
            return None
        template = self.templates[name]
        # Very simple positional replacement: {0}, {1}, etc.
        args = [arg.strip() for arg in args_string.split(",")] if args_string else []
        try:
            return template.format(*args)
        except IndexError:
            return None


def load_server_config(config_path: str | pathlib.Path | None = None) -> ServerConfig:
    """Load ServerConfig from config.jsonc, merging with defaults."""
    cfg = ServerConfig()
    try:
        payload = chatdb.load_config_file(config_path)
        server = payload.get("server", {})
        if isinstance(server, dict):
            for key, value in server.items():
                if key in ("profiles", "presets", "templates"):
                    setattr(cfg, key, value)
                elif hasattr(cfg, key) and not key.startswith("_"):
                    setattr(cfg, key, value)
        
        kernel = payload.get("kernel", {})
        if isinstance(kernel, dict):
            for key, value in kernel.items():
                if hasattr(cfg, key) and not key.startswith("_"):
                    setattr(cfg, key, value)
                    
    except Exception:
        pass
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

cfg_global = ServerConfig()

class UI:
    """Thin wrapper — falls back to plain print when rich is absent."""

    def __init__(self):
        self.con = Console(highlight=False) if _RICH else None
        self._current_status = None

    def start_status(self, msg: str):
        if _RICH and self.con:
            self.stop_status()
            self._current_status = self.con.status(f"[cyan]{msg}[/]")
            self._current_status.start()
        else:
            print(f"[{msg}]")

    def stop_status(self):
        if _RICH and self._current_status:
            self._current_status.stop()
            self._current_status = None

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


if _PROMPT_TOOLKIT:
    class BoundedFileHistory(FileHistory):
        """FileHistory that truncates to a maximum length."""
        
        def __init__(self, filename: str, max_length: int = 1000):
            super().__init__(filename)
            self.max_length = max_length
            self._append_count = 0

        def load_history_strings(self) -> Iterable[str]:
            # parent returns newest first
            strings = list(super().load_history_strings())
            if len(strings) > self.max_length:
                return strings[:self.max_length]
            return strings

        def store_string(self, string: str) -> None:
            super().store_string(string)
            self._append_count += 1
            
            # Rewrite file occasionally to prevent infinite growth
            if self._append_count >= 50:
                self._rewrite()

        def _rewrite(self) -> None:
            strings = list(super().load_history_strings())
            if len(strings) > self.max_length:
                strings = strings[:self.max_length]
                strings.reverse() # oldest first for writing
                
                with open(self.filename, 'wb') as f:
                    pass # clear
                    
                for s in strings:
                    super().store_string(s)
                    
            self._append_count = 0


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
        commands = [
            "/goal", "/go", "/grill-me", "/agents", "/tasks", "/browser", 
            "/fork", "/rewind", "/undo", "/resume", "/switch", "/clear", "/rename", 
            "/btw", "/diff", "/artifacts", "/add-dir", "/context", "/usage", "/config", 
            "/settings", "/export", "/help", "/exit", "/quit", "/skill", "/image", "/exec", 
            "/profile", "/preset", "/template", "/model", "/temp", "/tokens", "/save", "/chat", 
            "/safety", "/review", "/task"
        ]
        
        placeholders = {
            "/goal", "/go", "/grill-me", "/agents", "/tasks", "/browser", 
            "/fork", "/rewind", "/undo", "/switch", "/rename", "/btw", 
            "/diff", "/artifacts", "/add-dir", "/context", "/usage", 
            "/settings", "/export"
        }
        
        
        display_dict = {
            cmd: f"{cmd} (TBD)" if cmd in placeholders else cmd 
            for cmd in commands
        }
        
        class KernelCompleter(Completer):
            def __init__(self):
                self.cmd_completer = WordCompleter(commands, ignore_case=True, display_dict=display_dict)
                self.path_completer = PathCompleter(expanduser=True)
                
            def get_completions(self, document, complete_event):
                word = document.get_word_before_cursor(WORD=True)
                if word.startswith("/"):
                    yield from self.cmd_completer.get_completions(document, complete_event)
                elif word.startswith("@"):
                    fake_doc = Document(
                        document.text_before_cursor[:-len(word)] + word[1:], 
                        cursor_position=document.cursor_position - 1
                    )
                    for c in self.path_completer.get_completions(fake_doc, complete_event):
                        yield Completion(c.text, start_position=c.start_position, display=c.display)
                        
        completer = KernelCompleter()

        # Setup history
        try:
            max_hist = int(os.environ.get("NOMADIC_HISTORY_SIZE", "1000"))
        except ValueError:
            max_hist = 1000
        
        history_path = chatdb.default_config_dir() / "history.txt"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history = BoundedFileHistory(str(history_path), max_length=max_hist)

        text_area = TextArea(
            multiline=True,
            wrap_lines=True,
            focus_on_click=True,
            scrollbar=False,
            style="class:input-body",
            height=Dimension(min=3, max=12),
            completer=completer,
            complete_while_typing=True,
            history=history,
        )
        help_text = Label(
            text="Enter submit  Shift+Enter newline  Ctrl-D exit",
            style="class:input-help",
            dont_extend_height=True,
        )
        frame = Frame(text_area, title=label, style="class:input-frame")
        root = FloatContainer(
            content=HSplit(
                [
                    frame,
                    Window(height=1, char=" "),
                    help_text,
                ]
            ),
            floats=[
                Float(
                    xcursor=True,
                    ycursor=True,
                    content=CompletionsMenu(max_height=8, scroll_offset=1),
                )
            ],
        )
        style = Style.from_dict(
            {
                "input-frame.border": f"fg:{_KERNEL_INPUT_ACCENT}",
                "input-frame.label": f"bold fg:{_KERNEL_INPUT_ACCENT}",
                "input-body": "",
                "input-help": f"fg:{_KERNEL_INPUT_ACCENT} italic",
                "completion-menu": "bg:#333333 fg:white",
                "completion-menu.completion.current": f"bg:{_KERNEL_INPUT_ACCENT} fg:black",
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


def _generate_semantic_memory(cfg: ServerConfig, compressed_messages: list[dict], existing_memory: str = "") -> str:
    """Uses the VLM to generate a dense semantic summary of compressed messages."""
    raw_text = "\n".join(_summarize_message(msg) for msg in compressed_messages)
    
    prompt = (
        "You are an expert at extracting dense, highly relevant context from conversation logs. "
        "Summarize the following past conversation turns into a compact 'memory note'. "
        "Focus ONLY on established facts, user preferences, code structure, paths, and unresolved problems. "
        "DO NOT summarize greetings or pleasantries. "
        "Return ONLY the summary text, no formatting, no preamble."
    )
    
    if existing_memory:
        prompt += f"\n\nExisting memory context to integrate:\n{existing_memory}"
        
    prompt += f"\n\nRecent turns to compress:\n{raw_text}"

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    
    try:
        summary = vlm_complete(cfg, messages, stream=False)
        return summary.strip()
    except Exception as e:
        ui.warn(f"Semantic summarization failed: {e}")
        return ""


def _compact_history_pass(
    history: list[dict], 
    *, 
    target_tokens: int,
    cfg: Optional[ServerConfig] = None,
) -> list[dict]:
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

    existing_mem_str = _memory_note_body(existing_memory[0]) if existing_memory else ""
    
    # Generate semantic memory if we have a config (meaning we can call the VLM)
    semantic_summary = ""
    if cfg is not None and compressed:
        semantic_summary = _generate_semantic_memory(cfg, compressed, existing_memory=existing_mem_str)
        
    if semantic_summary:
        content = f"{_MEMORY_NOTE_PREFIX}\n{semantic_summary}"
    else:
        # Fallback to simple truncation
        if existing_memory:
            compressed = existing_memory + compressed
        content = _build_memory_note(
            compressed,
            max_lines=8,
            max_chars=max(120, reserve * 2),
        )

    memory_note = {
        "role": "system",
        "content": content,
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
        message = data["choices"][0]["message"]
        content = message.get("content")
        if content is None:
            hint = (
                "If you just switched models with /model, restart `nomadic serve` "
                "first — the running server keeps serving whatever model it was "
                "started with until it's restarted."
            )
            if message.get("tool_calls"):
                raise RuntimeError(
                    f"Model returned native tool_calls with no text content, which "
                    f"nomadic's structured tool-calling loop does not consume. {hint}"
                )
            raise RuntimeError(f"Model server returned an empty response (content was null). {hint}")
        full_reply = content

    return full_reply


# ══════════════════════════════════════════════════════════════════════════════
# COMMAND: serve
# ══════════════════════════════════════════════════════════════════════════════

def cmd_serve(cfg: ServerConfig):
    """
    Launch mlx_vlm.server or mlx_lm.server as a subprocess with inlined params.
    Blocks until Ctrl-C.
    """
    server_type = "mlx_lm" if is_text_only_model(cfg.model) else "mlx_vlm"
    ui.rule(f"{server_type}.server")
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
        server_type = "mlx_lm" if is_text_only_model(cfg.model) else "mlx_vlm"
        ui.error(
            f"{server_type} server not reachable at {cfg.base_url}\n"
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

_KERNEL_HELP_BASE = """
┌─ Kernel Commands ───────────────────────────────────────────┐
│ 🛠️  Execution & Agent Control                                │
│  /goal              Run autonomously to task completion     │
│  /go                Execute next steps without prompt       │
│  /grill-me          Force agent to ask clarifying questions │
│  /agents            Agent Manager Panel for subagents       │
│  /tasks             Track background processes              │
│  /browser           Opt-into browser capabilities           │
│                                                             │
│ 💬 Conversation & Session State                             │
│  /fork              Branch conversation to new session      │
│  /rewind, /undo     Rollback conversation history           │
│  /resume, /switch   Resume previous conversation            │
│  /clear             Clear conversation history              │
│  /rename <name>     Rename active conversation thread       │
│  /btw <query>       Ask background side question            │
│                                                             │
│ 📁 Workspace & System Utilities                             │
│  /diff              View workspace diff                     │
│  /artifacts         Open generated components panel         │
│  /add-dir <path>    Add directory to active workspace       │
│  /context           Live token usage & checkpoints          │
│  /usage             Real-time quota & rate-limit status     │
│  /config, /settings Interactive Settings Editor             │
│  /export            Push session state to GUI               │
│  /help              This message                            │
│  /exit, /quit       Exit TUI safely                         │
│                                                             │
│ ⚙️  Advanced (Legacy)                                       │
│  /skill <path>      Load a SKILL.md into context            │
│  /image <path>      Attach an image to next message         │
│  /exec <cmd>        Run a shell command                     │
│  /profile <name>    Switch to a named config profile        │
│  /preset <name>     Switch to a named system prompt preset  │
│  /template <name>   Execute a parameterized prompt template │
│  /model <name>      Switch model                            │
│  /temp <0.0–1.0>    Set temperature                         │
│  /tokens <n>        Set max_tokens per turn                 │
│  /save <name>       Name current chat in DB or export JSON  │
│  /chat ...          Session management subcommands          │
└─────────────────────────────────────────────────────────────┘
"""

_KERNEL_HELP_INPUT = """
Input editor:
  Enter submits the current message.
  Shift+Enter inserts a newline in the enhanced editor.
  If prompt_toolkit is unavailable, the kernel falls back to the
  classic single-line prompt.
"""

_KERNEL_HELP_RESPONSE = """
Post-response next steps:
  After each agent reply, the kernel suggests:
  1) recommended next action
  2) alternate next action
  3) alternate next action
  4) other (free text)
  Type 1-4 at the next prompt to use them.
"""

_KERNEL_HELP_TOOL_CALL_PROTOCOL = """
Model tool-call protocol:
  {"tool_call":{"name":"read_skill","arguments":{"path":"..."}}}
  {"tool_call":{"name":"exec_shell","arguments":{"command":"..."}}}
  {"tool_call":{"name":"metadata_search","arguments":{"query":"..."}}}
"""

_KERNEL_HELP_CONTEXT = """
Context management:
  Automatic compression keeps the system prompt + newest turns
  when the approximate non-system history budget is exceeded.
  Configure with --max-skill-chars, --max-context-tokens, and
  --context-recompress-to.
"""

# + _KERNEL_HELP_RESPONSE \
_KERNEL_HELP = _KERNEL_HELP_BASE \
    + _KERNEL_HELP_INPUT \
    + _KERNEL_HELP_TOOL_CALL_PROTOCOL \
    + _KERNEL_HELP_CONTEXT \

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
            new_history = _compact_history_pass(working_history, target_tokens=target, cfg=self.cfg)
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


def cmd_kernel(
    cfg: ServerConfig,
    db_settings: chatdb.DatabaseSettings,
    session_id: Optional[str],
    profile: Optional[str] = None,
):
    """Interactive structured tool-calling kernel."""

    # ── Merge config from file ──────────────────────────────────────────────
    cfg_file = load_server_config()
    # Merge cfg_file into cfg (which came from CLI flags)
    # CLI flags take precedence over config file, but profiles are only in file.
    cfg.profiles = cfg_file.profiles
    cfg.presets = cfg_file.presets

    if profile:
        if not cfg.apply_profile(profile):
            ui.warn(f"Profile not found: {profile}")
        else:
            ui.info(f"Using profile: {profile}")

    conn = chatdb.connect(db_settings)
    db_target = chatdb.describe_database_target(db_settings)
    state: Optional[KernelState] = None
    
    # ── Initialize Tool Registry ────────────────────────────────────────────
    tool_registry = agent.build_default_registry()
    # Load user-defined tool packs from ~/.config/nomadic/tools
    user_tools_dir = chatdb.default_config_dir() / "tools"
    if user_tools_dir.exists():
        tool_registry.load_from_directory(user_tools_dir)
    
    input_layer = KernelInput(ui)
    interrupt_controller = KernelInterruptController()
    
    # ── Orchestrator & TaskRegistry & HITL Manager ──
    hitl_manager = HITLManager(SafetyProfile.SEMI_AUTONOMOUS)
    task_registry = TaskRegistry(conn)
    orchestrator = Orchestrator(
        default_system_prompt=cfg.cot_system_prompt,
        default_tool_registry=tool_registry,
        hitl_manager=hitl_manager,
        task_registry=task_registry,
    )
    
    # ── Browser State & Handlers ──
    browser_state = {"playwright": None, "browser": None, "page": None}
    
    def handle_browser_navigate(args, runtime):
        url = args.get("url")
        if not url:
            return agent.ToolExecutionResult("browser_navigate", "browser", "error", "Missing 'url' argument.")
        if not browser_state.get("page"):
            return agent.ToolExecutionResult("browser_navigate", "browser", "error", "Browser is not active.")
        try:
            browser_state["page"].goto(url)
            title = browser_state["page"].title()
            return agent.ToolExecutionResult(
                "browser_navigate",
                "browser",
                "ok",
                f"Successfully navigated to {url}. Page Title: {title}",
                payload={"url": url, "title": title}
            )
        except Exception as e:
            return agent.ToolExecutionResult("browser_navigate", "browser", "error", f"Failed to navigate: {e}")

    def handle_browser_screenshot(args, runtime):
        if not browser_state.get("page"):
            return agent.ToolExecutionResult("browser_screenshot", "browser", "error", "Browser is not active.")
        try:
            import time
            screenshot_path = chatdb.default_config_dir() / f"screenshot_{int(time.time())}.png"
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            browser_state["page"].screenshot(path=str(screenshot_path))
            runtime.state.pending_image = str(screenshot_path)
            return agent.ToolExecutionResult(
                "browser_screenshot",
                "browser",
                "ok",
                f"Screenshot saved to {screenshot_path}. Queued for next message.",
                payload={"path": str(screenshot_path)}
            )
        except Exception as e:
            return agent.ToolExecutionResult("browser_screenshot", "browser", "error", f"Failed to capture screenshot: {e}")

    def handle_browser_extract_text(args, runtime):
        if not browser_state.get("page"):
            return agent.ToolExecutionResult("browser_extract_text", "browser", "error", "Browser is not active.")
        try:
            text = browser_state["page"].locator("body").inner_text()
            return agent.ToolExecutionResult(
                "browser_extract_text",
                "browser",
                "ok",
                text[:12000],
                payload={"text_len": len(text)}
            )
        except Exception as e:
            return agent.ToolExecutionResult("browser_extract_text", "browser", "error", f"Failed to extract text: {e}")

    async def run_agent_interactive(orchestrator, agent_id, cfg, state, ui):
        def complete_fn(history, stream=False):
            return vlm_complete(cfg, history, stream=stream)
        instance = orchestrator.get_agent(agent_id)
        if not instance:
            return
        while not instance.is_terminal:
            res = await instance.run(complete_fn, ui)
            if instance.status == AgentStatus.AWAITING_REVIEW:
                req = instance.hitl_manager.pending_reviews.get(instance.id)
                if req:
                    prompt_text = instance.hitl_manager.format_review_prompt(req)
                    print(prompt_text)
                    user_choice = ui.prompt("HITL Choice (a/r/m/s/q)")
                    if user_choice.strip().lower() in ("q", "quit", "exit"):
                        ui.warn("Aborted agent execution.")
                        instance.status = AgentStatus.CANCELLED
                        if instance.task:
                            orchestrator.task_registry.cancel_task(instance.task.id)
                        break
                    result = instance.hitl_manager.parse_review_input(user_choice, req)
                    instance.resolve_review(
                        decision=result.decision,
                        modified_args=result.modified_arguments,
                        feedback=result.feedback,
                        ui=ui
                    )
            else:
                break
    try:
        chatdb.init_db(conn)

        if not _check_server(cfg):
            server_type = "mlx_lm" if is_text_only_model(cfg.model) else "mlx_vlm"
            ui.error(
                f"{server_type} server not reachable at {cfg.base_url}\n"
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
                    metadata={"pwd": os.getcwd()},
                )
                mode = "created"
        except KeyError as exc:
            ui.error(str(exc))
            return

        state = _load_kernel_state(conn, session, cfg)
        task_registry.load_from_db(state.session_id)
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

                elif cmd == "/profile":
                    if not rest:
                        if cfg.profiles:
                            ui.info(f"Available profiles: {', '.join(cfg.profiles.keys())}")
                        else:
                            ui.warn("No profiles defined in config.jsonc.")
                        ui.warn("Usage: /profile <name>")
                        continue
                    if cfg.apply_profile(rest.strip()):
                        ui.success(f"Applied profile: {rest.strip()}")
                    else:
                        ui.error(f"Profile not found: {rest.strip()}")

                elif cmd == "/preset":
                    if not rest:
                        if cfg.presets:
                            ui.info(f"Available presets: {', '.join(cfg.presets.keys())}")
                        else:
                            ui.warn("No presets defined in config.jsonc.")
                        ui.warn("Usage: /preset <name>")
                        continue
                    if cfg.apply_preset(rest.strip()):
                        ui.success(f"Applied preset: {rest.strip()}")
                    else:
                        ui.error(f"Preset not found: {rest.strip()}")

                elif cmd == "/template":
                    subparts = rest.split(maxsplit=1)
                    if not subparts:
                        if cfg.templates:
                            ui.info(f"Available templates: {', '.join(cfg.templates.keys())}")
                        else:
                            ui.warn("No templates defined in config.jsonc.")
                        ui.warn("Usage: /template <name> [arg0, arg1, ...]")
                        continue
                    
                    tname = subparts[0]
                    targs = subparts[1] if len(subparts) > 1 else ""
                    formatted = cfg.format_template(tname, targs)
                    if formatted is None:
                        ui.error(f"Failed to format template: {tname}. Check name and arguments.")
                    else:
                        ui.success(f"Template '{tname}' evaluated. Submitting as user prompt...")
                        # Pretend the user typed this and skip to the next loop iteration.
                        # We must modify the submission object and break out of the built-in commands block.
                        submission = KernelInputSubmission(raw=formatted, stripped=formatted.strip(), kind="message")
                        # We do not `continue` here; we let it fall through to the VLM completion phase
                        # by essentially rewriting the submission.
                        pass # Fallthrough is safe here because it's the last check if we break out of the elif chain.
                        
                        # Wait, we are in an `if submission.kind == "command":` block.
                        # If we break out, we will hit the "continue" at the end of the block.
                        # Instead, let's process it immediately as if it were a message turn.
                        
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
                        
                        continue
                        
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
                    arg = rest.strip()
                    if not arg or arg.lower() == "select":
                        ui.rule("Select Model")
                        for idx, (name, note) in enumerate(KNOWN_MODELS, start=1):
                            marker = "→" if name == cfg.model else " "
                            suffix = f"  ({note})" if note else ""
                            ui.info(f"{marker} {idx}. {name}{suffix}")
                        ui.info(f"\nCurrent: {cfg.model}")
                        ui.warn("Usage: /model <number>  or  /model <repo-id>  (restart server to apply)")
                        continue
                    if arg.isdigit():
                        index = int(arg) - 1
                        if 0 <= index < len(KNOWN_MODELS):
                            cfg.model = KNOWN_MODELS[index][0]
                            ui.success(f"Model → {cfg.model}  (restart server to apply)")
                        else:
                            ui.error(f"No model at index {arg}. Run /model to see the list.")
                        continue
                    cfg.model = arg
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
                    if rest and not rest.strip().endswith(".json"):
                        state.session = dict(chatdb.update_session(conn, state.session_id, chat_name=rest.strip()))
                        ui.success(f"Session saved in DB as: {state.chat_name}")
                    else:
                        fname = rest.strip() or f"kernel_session_{int(time.time())}.json"
                        export_payload = {
                            "session": state.session,
                            "history": state.history,
                        }
                        pathlib.Path(fname).write_text(
                            json.dumps(export_payload, indent=2, default=str)
                        )
                        ui.success(f"Saved to file → {fname}")

                elif cmd == "/chat":
                    subparts = rest.split(maxsplit=1)
                    subcmd = subparts[0].lower() if subparts else "list"
                    subrest = subparts[1] if len(subparts) > 1 else ""

                    if subcmd == "list":
                        pwd = os.getcwd()
                        sessions = chatdb.list_sessions(conn, pwd=pwd)
                        if not sessions:
                            ui.warn(f"No sessions found for {pwd}")
                        else:
                            ui.rule(f"Sessions in {pwd}")
                            for s in sessions:
                                print(f"  {s['id'][:8]}... | {s['chat_name']:30} | {s['updated_at']}")

                    elif subcmd == "resume":
                        if not subrest:
                            ui.warn("Usage: /chat resume <id|name>")
                            continue
                        target_id = subrest.strip()
                        target_session = chatdb.get_session(conn, target_id)
                        if not target_session:
                            target_session = chatdb.get_session_by_name(conn, target_id)

                        if target_session:
                            session = chatdb.resume_session(conn, target_session["id"])
                            state = _load_kernel_state(conn, session, cfg)
                            ui.success(f"Resumed session: {state.chat_name}")
                        else:
                            ui.error(f"Session not found: {subrest}")

                    elif subcmd == "rename":
                        if not subrest:
                            ui.warn("Usage: /chat rename <new_name>")
                            continue
                        state.session = dict(chatdb.update_session(conn, state.session_id, chat_name=subrest.strip()))
                        ui.success(f"Session renamed to: {state.chat_name}")

                    else:
                        ui.warn("Usage: /chat [list|resume|rename]")

                elif cmd == "/resume":
                    if not rest:
                        ui.warn("Usage: /resume <id|name>")
                        continue
                    target_id = rest.strip()
                    target_session = chatdb.get_session(conn, target_id)
                    if not target_session:
                        target_session = chatdb.get_session_by_name(conn, target_id)

                    if target_session:
                        session = chatdb.resume_session(conn, target_session["id"])
                        state = _load_kernel_state(conn, session, cfg)
                        ui.success(f"Resumed session: {state.chat_name}")
                    else:
                        ui.error(f"Session not found: {rest}")

                elif cmd in ("/settings", "/switch", "/undo"):
                    # Aliases for existing commands
                    if cmd == "/settings":
                        ui.warn("Alias /settings mapped to /config (run /config manually for now)")
                    elif cmd == "/switch":
                        ui.warn("Alias /switch mapped to /resume (run /resume manually for now)")
                    elif cmd == "/undo":
                        ui.warn("Alias /undo mapped to /rewind (run /rewind manually for now)")
                        
                elif cmd == "/goal":
                    objective = rest.strip()
                    if not objective:
                        objective = "Continue working on the current task based on conversation history."
                    ui.info(f"Starting goal-oriented execution: {objective}")
                    task = orchestrator.task_registry.create_task(
                        objective=objective,
                        session_id=state.session_id,
                    )
                    agent_inst = orchestrator.spawn_agent(
                        task=task,
                        system_prompt=cfg.cot_system_prompt,
                        tool_registry=tool_registry,
                    )
                    existing_msgs = [m for m in state.history if m["role"] != "system"]
                    agent_inst.history = [{"role": "system", "content": agent_inst.system_prompt}] + existing_msgs + [{"role": "user", "content": f"New Objective: {objective}"}]
                    asyncio.run(run_agent_interactive(orchestrator, agent_inst.id, cfg, state, ui))
                    state.history = list(agent_inst.history)
                    if agent_inst.task and agent_inst.task.result:
                        ui.success(f"Goal achieved: {agent_inst.task.result}")
                    elif agent_inst.status == AgentStatus.FAILED:
                        ui.error(f"Goal failed: {agent_inst.error}")

                elif cmd == "/go":
                    ui.info("Resuming agent execution on current context...")
                    state.next_actions = []
                    ui.rule(f"turn {state.turn} · agent response")
                    t0 = time.perf_counter()
                    def status_callback(event: agent.StatusEvent):
                        ui.start_status(event.message)
                    try:
                        reply = agent.run_agent_loop(
                            state=state,
                            complete_fn=vlm_complete,
                            ui=ui,
                            registry=tool_registry,
                            status_callback=status_callback,
                        )
                    except KeyboardInterrupt:
                        ui.warn("Interrupted current run; returning to the kernel prompt.")
                        continue
                    except Exception as e:
                        ui.error(f"VLM error: {e}")
                        continue
                    finally:
                        ui.stop_status()
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

                elif cmd == "/grill-me":
                    topic = rest.strip()
                    ui.info("Entering interrogative mode. Let's clarify requirements...")
                    grill_prompt = (
                        "[INTERROGATIVE MODE] Your goal is to grill the user with clarifying questions "
                        "about their request to resolve any ambiguity. Ask one question at a time. "
                        "Do not use tools. When you believe all ambiguity is resolved and you have "
                        "enough detail to proceed, respond with exactly: 'ALL_RESOLVED'."
                    )
                    state.inject_context(grill_prompt, source="grill_mode")
                    if topic:
                        state.add_user(f"I want to: {topic}")
                    else:
                        state.add_user("Please ask clarifying questions to help me define my goal.")
                    while True:
                        ui.rule("Grill Turn")
                        reply = vlm_complete(cfg, state.history, stream=True)
                        print()
                        if "ALL_RESOLVED" in reply:
                            ui.success("All requirements clarified! Exiting grill mode.")
                            state.add_assistant("All requirements clarified.")
                            break
                        state.add_assistant(reply)
                        try:
                            user_ans = ui.prompt("Your answer (type 'done' to finish grilling)")
                        except (KeyboardInterrupt, EOFError):
                            ui.warn("Grill mode aborted.")
                            break
                        if user_ans.strip().lower() in ("done", "stop", "exit"):
                            ui.success("Exiting grill mode.")
                            break
                        state.add_user(user_ans)

                elif cmd == "/agents":
                    subparts = rest.split(maxsplit=1)
                    subcmd = subparts[0].lower() if subparts else "list"
                    subrest = subparts[1] if len(subparts) > 1 else ""
                    if subcmd in ("list", "show"):
                        dashboard = orchestrator.format_dashboard()
                        ui.panel("Orchestrator Agents Dashboard", dashboard)
                    elif subcmd == "cancel":
                        if not subrest:
                            ui.warn("Usage: /agents cancel <agent_id>")
                            continue
                        aid = subrest.strip()
                        if orchestrator.cancel_agent(aid):
                            ui.success(f"Cancelled agent {aid}")
                        else:
                            ui.error(f"Agent {aid} not found or not active")
                    elif subcmd == "spawn":
                        if not subrest:
                            ui.warn("Usage: /agents spawn <objective>")
                            continue
                        obj = subrest.strip()
                        task = orchestrator.task_registry.create_task(
                            objective=obj,
                            session_id=state.session_id,
                        )
                        agent_inst = orchestrator.spawn_agent(
                            task=task,
                            system_prompt=cfg.cot_system_prompt,
                            tool_registry=tool_registry,
                        )
                        def run_bg():
                            loop = asyncio.get_event_loop()
                            loop.create_task(orchestrator.run_agent(
                                agent_inst.id,
                                lambda hist, stream=False: vlm_complete(cfg, hist, stream=stream),
                                ui=None,
                            ))
                        run_bg()
                        ui.success(f"Spawned background agent {agent_inst.id} for task: {obj}")
                    else:
                        ui.warn("Usage: /agents [list|cancel <id>|spawn <objective>]")

                elif cmd == "/tasks":
                    subparts = rest.split(maxsplit=1)
                    subcmd = subparts[0].lower() if subparts else "list"
                    subrest = subparts[1] if len(subparts) > 1 else ""
                    if subcmd in ("list", "show"):
                        tasks = orchestrator.task_registry.tasks
                        if not tasks:
                            ui.info("No tasks recorded in this session.")
                        else:
                            lines = []
                            lines.append(f"{'Task ID':<10} | {'Status':<12} | {'Agent ID':<15} | {'Objective'}")
                            lines.append("─" * 80)
                            for t in tasks:
                                aid = t.agent_id or "—"
                                obj = t.objective if len(t.objective) < 40 else t.objective[:37] + "..."
                                lines.append(f"{t.id[:8]:<10} | {t.status.value:<12} | {aid:<15} | {obj}")
                            ui.panel("Task Registry", "\n".join(lines))
                    elif subcmd == "cancel":
                        if not subrest:
                            ui.warn("Usage: /tasks cancel <task_id>")
                            continue
                        tid = subrest.strip()
                        matched_task = None
                        for t in orchestrator.task_registry.tasks:
                            if t.id == tid or t.id.startswith(tid):
                                matched_task = t
                                break
                        if matched_task:
                            orchestrator.task_registry.cancel_task(matched_task.id)
                            if matched_task.agent_id:
                                orchestrator.cancel_agent(matched_task.agent_id)
                            ui.success(f"Cancelled task {matched_task.id}")
                        else:
                            ui.error(f"Task {tid} not found")
                    elif subcmd == "inspect":
                        if not subrest:
                            ui.warn("Usage: /tasks inspect <task_id>")
                            continue
                        tid = subrest.strip()
                        matched_task = None
                        for t in orchestrator.task_registry.tasks:
                            if t.id == tid or t.id.startswith(tid):
                                matched_task = t
                                break
                        if matched_task:
                            body = [
                                f"Task ID    : {matched_task.id}",
                                f"Objective  : {matched_task.objective}",
                                f"Status     : {matched_task.status.value}",
                                f"Agent ID   : {matched_task.agent_id or '—'}",
                                f"Created At : {matched_task.created_at}",
                                f"Updated At : {matched_task.updated_at}",
                            ]
                            if matched_task.result:
                                body.append(f"Result     : {matched_task.result}")
                            if matched_task.error:
                                body.append(f"Error      : {matched_task.error}")
                            if matched_task.artifacts:
                                body.append("\nArtifacts:")
                                for art in matched_task.artifacts:
                                    body.append(f"  · {art.path} ({art.action})")
                            ui.panel(f"Inspect Task {matched_task.id[:8]}", "\n".join(body))
                        else:
                            ui.error(f"Task {tid} not found")
                    else:
                        ui.warn("Usage: /tasks [list|cancel <id>|inspect <id>]")

                elif cmd == "/browser":
                    subparts = rest.split(maxsplit=1)
                    subcmd = subparts[0].lower() if subparts else "status"
                    if subcmd == "on":
                        try:
                            from playwright.sync_api import sync_playwright
                            if not browser_state.get("playwright"):
                                ui.info("Launching headless browser...")
                                browser_state["playwright"] = sync_playwright().start()
                                browser_state["browser"] = browser_state["playwright"].chromium.launch(headless=True)
                                browser_state["page"] = browser_state["browser"].new_page()
                                tool_registry._tools["browser_navigate"] = agent.ToolDefinition(
                                    name="browser_navigate",
                                    description="Navigate the browser to a given URL.",
                                    parameters={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
                                    handler=handle_browser_navigate
                                )
                                tool_registry._tools["browser_screenshot"] = agent.ToolDefinition(
                                    name="browser_screenshot",
                                    description="Take a screenshot of the current browser page.",
                                    parameters={"type": "object", "properties": {}},
                                    handler=handle_browser_screenshot
                                )
                                tool_registry._tools["browser_extract_text"] = agent.ToolDefinition(
                                    name="browser_extract_text",
                                    description="Extract visible text from the browser page.",
                                    parameters={"type": "object", "properties": {}},
                                    handler=handle_browser_extract_text
                                )
                                ui.success("Browser is ON. Tools registered.")
                            else:
                                ui.info("Browser is already running.")
                        except ImportError:
                            ui.error("Playwright is not installed. Run: pip install playwright && playwright install")
                        except Exception as e:
                            ui.error(f"Failed to start browser: {e}")
                    elif subcmd == "off":
                        if browser_state.get("playwright"):
                            ui.info("Closing browser session...")
                            try:
                                if browser_state.get("browser"):
                                    browser_state["browser"].close()
                                if browser_state.get("playwright"):
                                    browser_state["playwright"].stop()
                            except Exception:
                                pass
                            browser_state["playwright"] = None
                            browser_state["browser"] = None
                            browser_state["page"] = None
                            tool_registry._tools.pop("browser_navigate", None)
                            tool_registry._tools.pop("browser_screenshot", None)
                            tool_registry._tools.pop("browser_extract_text", None)
                            ui.success("Browser session closed. Tools unregistered.")
                        else:
                            ui.info("Browser is already OFF.")
                    elif subcmd == "status":
                        if browser_state.get("page"):
                            try:
                                url = browser_state["page"].url
                                title = browser_state["page"].title()
                                ui.info(f"Browser Status: ACTIVE\n  URL: {url}\n  Title: {title}")
                            except Exception:
                                ui.info("Browser Status: ACTIVE (Page not loaded)")
                        else:
                            ui.info("Browser Status: INACTIVE")
                    else:
                        url = rest.strip()
                        if url:
                            if not browser_state.get("page"):
                                ui.info("Auto-turning ON browser...")
                                try:
                                    from playwright.sync_api import sync_playwright
                                    browser_state["playwright"] = sync_playwright().start()
                                    browser_state["browser"] = browser_state["playwright"].chromium.launch(headless=True)
                                    browser_state["page"] = browser_state["browser"].new_page()
                                    tool_registry._tools["browser_navigate"] = agent.ToolDefinition(
                                        name="browser_navigate",
                                        description="Navigate the browser to a given URL.",
                                        parameters={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
                                        handler=handle_browser_navigate
                                    )
                                    tool_registry._tools["browser_screenshot"] = agent.ToolDefinition(
                                        name="browser_screenshot",
                                        description="Take a screenshot of the current browser page.",
                                        parameters={"type": "object", "properties": {}},
                                        handler=handle_browser_screenshot
                                    )
                                    tool_registry._tools["browser_extract_text"] = agent.ToolDefinition(
                                        name="browser_extract_text",
                                        description="Extract visible text from the browser page.",
                                        parameters={"type": "object", "properties": {}},
                                        handler=handle_browser_extract_text
                                    )
                                    ui.success("Browser turned ON.")
                                except Exception as e:
                                    ui.error(f"Failed to turn ON browser: {e}")
                                    continue
                            ui.info(f"Navigating to {url}...")
                            res = handle_browser_navigate({"url": url}, agent.AgentRuntime(state=state, ui=ui))
                            if res.status == "ok":
                                ui.success(res.content)
                            else:
                                ui.error(res.content)
                        else:
                            ui.warn("Usage: /browser [on|off|<url>]")

                elif cmd == "/fork":
                    new_name = rest.strip()
                    if not new_name:
                        new_name = f"{state.chat_name}-fork"
                    curr_sess = chatdb.get_session(conn, state.session_id)
                    if not curr_sess:
                        ui.error("Cannot fork: active session not loaded.")
                        continue
                    new_id = str(uuid.uuid4())
                    now = chatdb.utc_now()
                    try:
                        columns = [
                            "id", "chat_name", "status", "created_at", "updated_at", "started_at", "ended_at",
                            "model", "host", "port", "temperature", "top_p", "max_tokens", "max_skill_chars",
                            "max_context_tokens", "context_recompress_to", "system_prompt", "summary",
                            "message_count", "user_turn_count", "assistant_turn_count", "tool_event_count",
                            "last_message_at", "metadata_json"
                        ]
                        vals = []
                        for col in columns:
                            if col == "id":
                                vals.append(new_id)
                            elif col == "chat_name":
                                vals.append(new_name)
                            elif col in ("created_at", "updated_at"):
                                vals.append(now)
                            else:
                                vals.append(curr_sess.get(col))
                        placeholders = ", ".join([conn.placeholder] * len(columns))
                        with chatdb._transaction(conn):
                            conn.execute(
                                f"INSERT INTO chat_sessions ({', '.join(columns)}) VALUES ({placeholders})",
                                tuple(vals)
                            )
                            old_messages = chatdb.load_messages(conn, state.session_id)
                            id_map = {}
                            for msg in old_messages:
                                parent_new_id = id_map.get(msg.get("parent_message_id")) if msg.get("parent_message_id") is not None else None
                                msg_fields = {
                                    "session_id": new_id,
                                    "message_index": msg.get("message_index"),
                                    "role": msg.get("role"),
                                    "content_type": msg.get("content_type"),
                                    "content_text": msg.get("content_text"),
                                    "content_json": msg.get("content_json"),
                                    "raw_request_json": msg.get("raw_request_json"),
                                    "raw_response_json": msg.get("raw_response_json"),
                                    "token_estimate": msg.get("token_estimate"),
                                    "created_at": msg.get("created_at"),
                                    "parent_message_id": parent_new_id,
                                    "metadata_json": msg.get("metadata_json")
                                }
                                col_names = ", ".join(msg_fields.keys())
                                p_holders = ", ".join([conn.placeholder] * len(msg_fields))
                                cursor = conn.execute(
                                    f"INSERT INTO chat_messages ({col_names}) VALUES ({p_holders})",
                                    tuple(msg_fields.values())
                                )
                                if hasattr(cursor, "lastrowid"):
                                    id_map[msg["id"]] = cursor.lastrowid
                        session = chatdb.get_session(conn, new_id)
                        state = _load_kernel_state(conn, session, cfg)
                        ui.success(f"Session forked and switched to: {state.chat_name} (ID: {state.session_id})")
                    except Exception as e:
                        ui.error(f"Failed to fork session: {e}")

                elif cmd in ("/rewind", "/undo"):
                    try:
                        n = int(rest.strip()) if rest.strip() else 1
                    except ValueError:
                        ui.warn("Usage: /rewind [<N>]")
                        continue
                    msgs = chatdb.load_messages(conn, state.session_id)
                    if not msgs:
                        ui.warn("No messages to rewind.")
                        continue
                    user_msgs = [m for m in msgs if m["role"] == "user"]
                    if len(user_msgs) < n:
                        ui.error(f"Cannot rewind {n} turns; only {len(user_msgs)} user turns exist.")
                        continue
                    target_user_msg = user_msgs[-n]
                    target_index = target_user_msg["message_index"]
                    with chatdb._transaction(conn):
                        conn.execute(
                            f"DELETE FROM chat_messages WHERE session_id = {conn.placeholder} AND message_index >= {conn.placeholder}",
                            (state.session_id, target_index)
                        )
                        conn.execute(
                            f"DELETE FROM chat_events WHERE session_id = {conn.placeholder} AND created_at >= {conn.placeholder}",
                            (state.session_id, target_user_msg["created_at"])
                        )
                    remaining_msgs = chatdb.load_messages(conn, state.session_id)
                    message_count = len(remaining_msgs)
                    user_turn_count = len([m for m in remaining_msgs if m["role"] == "user"])
                    assistant_turn_count = len([m for m in remaining_msgs if m["role"] == "assistant"])
                    tool_event_count = len([m for m in remaining_msgs if m["role"] == "tool"])
                    chatdb.update_session(
                        conn,
                        state.session_id,
                        message_count=message_count,
                        user_turn_count=user_turn_count,
                        assistant_turn_count=assistant_turn_count,
                        tool_event_count=tool_event_count
                    )
                    session = chatdb.get_session(conn, state.session_id)
                    state = _load_kernel_state(conn, session, cfg)
                    ui.success(f"Rewound {n} turn(s). Reverted to turn {state.turn}.")

                elif cmd == "/rename":
                    new_name = rest.strip()
                    if not new_name:
                        ui.warn("Usage: /rename <new_name>")
                        continue
                    state.session = dict(chatdb.update_session(conn, state.session_id, chat_name=new_name))
                    ui.success(f"Session renamed to: {state.chat_name}")

                elif cmd == "/btw":
                    question = rest.strip()
                    if not question:
                        ui.warn("Usage: /btw <question>")
                        continue
                    ui.success(f"Dispatched background question: {question}")
                    task = orchestrator.task_registry.create_task(
                        objective=question,
                        session_id=state.session_id,
                        metadata={"btw": True}
                    )
                    agent_inst = orchestrator.spawn_agent(
                        task=task,
                        system_prompt=cfg.cot_system_prompt + "\nBe extremely concise. You are answering a side question.",
                        tool_registry=tool_registry,
                    )
                    recent_history = [m for m in state.history if m["role"] != "system"][-6:]
                    agent_inst.history = [{"role": "system", "content": agent_inst.system_prompt}] + recent_history + [{"role": "user", "content": question}]
                    def run_btw_bg():
                        async def run_and_notify():
                            try:
                                result = await orchestrator.run_agent(
                                    agent_inst.id,
                                    lambda hist, stream=False: vlm_complete(cfg, hist, stream=stream),
                                    ui=None
                                )
                                print()
                                ui.panel("Background Side-Question Answered", f"Q: {question}\n\nA: {result}")
                            except Exception as e:
                                print()
                                ui.error(f"Background side-question failed: {e}")
                        loop = asyncio.get_event_loop()
                        loop.create_task(run_and_notify())
                    run_btw_bg()

                elif cmd == "/diff":
                    path = rest.strip()
                    workspace_root = os.getcwd()
                    git_cmd = ["git", "diff"]
                    if path:
                        git_cmd.append(path)
                    try:
                        res = subprocess.run(
                            git_cmd,
                            capture_output=True,
                            text=True,
                            cwd=workspace_root,
                            check=True
                        )
                        diff_output = res.stdout
                        if not diff_output.strip():
                            ui.info("No modifications in the current active workspace.")
                        else:
                            ui.rule(f"Git Diff {path}")
                            ui.code(diff_output, lang="diff")
                    except Exception as e:
                        ui.error(f"Failed to generate git diff: {e}")

                elif cmd == "/artifacts":
                    all_artifacts = []
                    for t in orchestrator.task_registry.tasks:
                        for art in t.artifacts:
                            all_artifacts.append((art.path, art.action, art.timestamp, t.id))
                    if not all_artifacts:
                        ui.info("No artifacts tracked in this session.")
                    else:
                        lines = []
                        lines.append(f"{'Path':<40} | {'Action':<10} | {'Task ID':<10} | {'Timestamp'}")
                        lines.append("─" * 80)
                        for path, action, ts, tid in all_artifacts:
                            lines.append(f"{path:<40} | {action:<10} | {tid[:8]:<10} | {ts}")
                        ui.panel("Artifact Manager", "\n".join(lines))

                elif cmd == "/add-dir":
                    path = rest.strip()
                    if not path:
                        ui.warn("Usage: /add-dir <path>")
                        continue
                    resolved_path = os.path.abspath(path)
                    if not os.path.exists(resolved_path):
                        ui.error(f"Path does not exist: {resolved_path}")
                        continue
                    if not hasattr(state, "workspace_paths"):
                        state.workspace_paths = [os.getcwd()]
                    if resolved_path not in state.workspace_paths:
                        state.workspace_paths.append(resolved_path)
                    ui.success(f"Added directory to active workspace: {resolved_path}")
                    state.inject_context(
                        f"[WORKSPACE UPDATE] The directory {resolved_path} has been added to the active workspace. You are now permitted to access files in this directory.",
                        source="workspace_update"
                    )

                elif cmd == "/context":
                    subparts = rest.split(maxsplit=1)
                    subcmd = subparts[0].lower() if subparts else "status"
                    subrest = subparts[1] if len(subparts) > 1 else ""
                    if not hasattr(state, "checkpoints"):
                        state.checkpoints = {}
                    if subcmd == "checkpoint":
                        name = subrest.strip() or f"checkpoint_{len(state.checkpoints) + 1}"
                        state.checkpoints[name] = {
                            "history": list(state.history),
                            "turn": state.turn,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        ui.success(f"Checkpoint '{name}' created at turn {state.turn}.")
                    elif subcmd == "restore":
                        name = subrest.strip()
                        if not name:
                            ui.warn("Usage: /context restore <name>")
                            continue
                        if name not in state.checkpoints:
                            ui.error(f"Checkpoint '{name}' not found.")
                            continue
                        chk = state.checkpoints[name]
                        state.history = list(chk["history"])
                        state.turn = chk["turn"]
                        ui.success(f"Restored to checkpoint '{name}' (turn {state.turn}).")
                    elif subcmd == "clear":
                        state.checkpoints = {}
                        ui.success("All checkpoints cleared.")
                    elif subcmd in ("status", "list"):
                        total_budget = cfg.max_context_tokens
                        user_len = sum(len(json.dumps(m["content"])) for m in state.history if m["role"] == "user")
                        assistant_len = sum(len(m["content"]) for m in state.history if m["role"] == "assistant")
                        system_len = sum(len(m["content"]) for m in state.history if m["role"] == "system")
                        user_tokens = user_len // 4
                        assistant_tokens = assistant_len // 4
                        system_tokens = system_len // 4
                        total_used = user_tokens + assistant_tokens + system_tokens
                        user_pct = user_tokens / total_budget if total_budget > 0 else 0
                        assistant_pct = assistant_tokens / total_budget if total_budget > 0 else 0
                        system_pct = system_tokens / total_budget if total_budget > 0 else 0
                        total_used_pct = total_used / total_budget if total_budget > 0 else 0
                        grid_str = render_token_grid(total_used_pct, user_pct, assistant_pct, 0.05, system_pct)
                        panel_body = f"""{grid_str}

Model: {cfg.model}
Total usage: {total_used:,}/{total_budget:,} tokens ({total_used_pct*100:.1f}%)

Token usage by category:
  ● User messages: {user_tokens:,} tokens ({user_pct*100:.1f}%)
  ● Agent responses: {assistant_tokens:,} tokens ({assistant_pct*100:.1f}%)
  ● System prompt: {system_tokens:,} tokens ({system_pct*100:.1f}%)
  ● Free space: {max(0, total_budget - total_used):,} tokens ({(1-total_used_pct)*100:.1f}%)

Checkpoints ({len(state.checkpoints)}):
"""
                        for name, chk in state.checkpoints.items():
                            panel_body += f"  · {name} (turn {chk['turn']}) - {chk['timestamp']}\n"
                        ui.panel("Context & Token Usage Analytics", panel_body)
                    else:
                        ui.warn("Usage: /context [checkpoint <name>|restore <name>|clear|status]")

                elif cmd == "/usage":
                    try:
                        query = """
                            SELECT 
                                model, 
                                COUNT(*) as session_count, 
                                SUM(message_count) as total_messages,
                                SUM(user_turn_count) as total_user_turns,
                                SUM(assistant_turn_count) as total_assistant_turns
                            FROM chat_sessions 
                            GROUP BY model
                        """
                        rows = chatdb._query_all(conn, query, ())
                        if not rows:
                            ui.info("No usage stats recorded yet.")
                        else:
                            body = []
                            body.append(f"{'Model':<40} | {'Sessions':<10} | {'Messages':<10} | {'User Turns':<10} | {'Agent Turns'}")
                            body.append("─" * 85)
                            for r in rows:
                                model = r.get("model") or "unknown"
                                body.append(f"{model:<40} | {r.get('session_count'):<10} | {r.get('total_messages'):<10} | {r.get('total_user_turns'):<10} | {r.get('total_assistant_turns')}")
                            ui.panel("Model Usage Statistics", "\n".join(body))
                    except Exception as e:
                        ui.error(f"Failed to query usage statistics: {e}")

                elif cmd == "/export":
                    subparts = rest.split(maxsplit=1)
                    format_type = subparts[0].lower() if subparts else "json"
                    dest = subparts[1].strip() if len(subparts) > 1 else ""
                    export_payload = {
                        "session": state.session,
                        "history": state.history,
                    }
                    if format_type == "json":
                        fname = dest or f"export_session_{state.session_id[:8]}.json"
                        try:
                            pathlib.Path(fname).write_text(
                                json.dumps(export_payload, indent=2, default=str)
                            )
                            ui.success(f"Session successfully exported to JSON: {fname}")
                        except Exception as e:
                            ui.error(f"Failed to write JSON export: {e}")
                    elif format_type == "markdown":
                        fname = dest or f"export_session_{state.session_id[:8]}.md"
                        try:
                            lines = [f"# Session: {state.chat_name}", f"ID: {state.session_id}", ""]
                            for msg in state.history:
                                role = msg["role"].upper()
                                content = msg["content"]
                                if isinstance(content, list):
                                    content = "\n".join(
                                        c.get("text", "") for c in content if c.get("type") == "text"
                                    )
                                lines.append(f"### {role}")
                                lines.append(f"{content}")
                                lines.append("")
                            pathlib.Path(fname).write_text("\n".join(lines))
                            ui.success(f"Session successfully exported to Markdown: {fname}")
                        except Exception as e:
                            ui.error(f"Failed to write Markdown export: {e}")
                    elif format_type == "push":
                        gui_url = dest or "http://localhost:7878/api/session/import"
                        ui.info(f"Pushing session to Antigravity GUI at {gui_url}...")
                        try:
                            resp = requests.post(gui_url, json=export_payload, timeout=5)
                            if resp.status_code == 200:
                                ui.success("Successfully pushed session to Antigravity GUI.")
                            else:
                                ui.error(f"GUI returned error: {resp.status_code} - {resp.text}")
                        except Exception as e:
                            ui.error(f"Could not connect to GUI endpoint: {e}")
                    else:
                        ui.warn("Usage: /export [json|markdown|push] [<destination>]")
                    
                else:
                    ui.warn(f"Unknown command: {cmd}  — type /help")

                continue  # skip VLM call for built-in commands

            # ── Normal user turn → VLM completion ───────────────────────────
            state.next_actions = []
            
            resolved_msg = submission.raw
            import re
            attachments = []
            for match in set(re.findall(r'(?<!\S)@([^\s]+)', submission.raw)):
                path = match.strip()
                p = pathlib.Path(path)
                if p.exists() and p.is_file():
                    try:
                        content = p.read_text(encoding="utf-8", errors="replace")
                        if len(content) > cfg.max_skill_chars:
                            content = content[:cfg.max_skill_chars] + "\n\n[... truncated ...]"
                        attachments.append(f'<file path="{path}">\n{content}\n</file>')
                    except Exception as e:
                        ui.warn(f"Failed to attach {path}: {e}")
                        
            if attachments:
                resolved_msg += "\n\n[Attached Context:]\n" + "\n\n".join(attachments)

            state.add_user(resolved_msg, state.pending_image)
            state.compact_history_if_needed()

            ui.rule(f"turn {state.turn} · agent response")
            t0 = time.perf_counter()
            
            def status_callback(event: agent.StatusEvent):
                ui.start_status(event.message)

            try:
                reply = agent.run_agent_loop(
                    state=state,
                    complete_fn=vlm_complete,
                    ui=ui,
                    registry=tool_registry,
                    status_callback=status_callback,
                )
            except KeyboardInterrupt:
                ui.warn("Interrupted current run; returning to the kernel prompt.")
                continue
            except Exception as e:
                ui.error(f"VLM error: {e}")
                continue
            finally:
                ui.stop_status()

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
        parser.add_argument("--profile", default=default, dest="profile",
                            help="Named config profile to load from config.jsonc")
        parser.add_argument("--quantize",    action="store_true", default=default,
                            help="Enable runtime model quantization check and conversion")
        parser.add_argument("--q-bits",      default=default, help="Target bits for quantization (8, 4, 3, 2, or auto)")
        parser.add_argument("--cache-dir",   default=default, help="Output folder for quantized weights")

    # Accept globals before the command.
    add_global_overrides(p)

    sub = p.add_subparsers(dest="command", required=True)

    # Accept globals after the command as well, without overwriting values
    # already parsed by the root parser.
    sub_common = argparse.ArgumentParser(add_help=False)
    add_global_overrides(sub_common, suppress_defaults=True)

    # serve
    sub.add_parser("serve", help="Start the local MLX server (mlx_vlm or mlx_lm) with inlined config params",
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
    args, extras = parser.parse_known_args()

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
    if getattr(args, "quantize", False):
        cfg.quantize = True
    if getattr(args, "q_bits", None) is not None:
        cfg.q_bits = args.q_bits
    if getattr(args, "cache_dir", None) is not None:
        cfg.cache_dir = args.cache_dir

    # Process inline key-value pairs (e.g. model=xxx or other_key=val)
    unrecognized = []
    for extra in extras:
        if "=" not in extra:
            unrecognized.append(extra)
        else:
            key, val = extra.split("=", 1)
            key_norm = key.strip().lower().replace("-", "_")
            if key.startswith("-") or not hasattr(cfg, key_norm) or key_norm.startswith("_"):
                unrecognized.append(extra)
            else:
                orig_val = getattr(cfg, key_norm)
                try:
                    if isinstance(orig_val, bool):
                        setattr(cfg, key_norm, val.strip().lower() in ("true", "1", "yes"))
                    elif isinstance(orig_val, int):
                        setattr(cfg, key_norm, int(val.strip()))
                    elif isinstance(orig_val, float):
                        setattr(cfg, key_norm, float(val.strip()))
                    else:
                        setattr(cfg, key_norm, val.strip())
                except Exception:
                    setattr(cfg, key_norm, val.strip())

    if unrecognized:
        parser.error(f"unrecognized arguments: {' '.join(unrecognized)}")

    if cfg.quantize:
        # Resolve target q_bits
        q_bits_str = str(cfg.q_bits).strip().lower()
        if q_bits_str == "auto" or not q_bits_str:
            q_bits = determine_optimal_q_bits(cfg.model)
        else:
            try:
                q_bits = int(q_bits_str)
            except ValueError:
                q_bits = 4
                ui.warn(f"Invalid q_bits value '{cfg.q_bits}'; defaulting to 4-bit.")

        # Resolve cache dir
        if cfg.cache_dir:
            target_cache = pathlib.Path(cfg.cache_dir)
        else:
            safe_model_name = cfg.model.replace("/", "--")
            target_cache = chatdb.default_config_dir() / "quantized" / safe_model_name / f"{q_bits}bit"

        # Check if already quantized
        if pathlib.Path(cfg.model).is_dir():
            ui.info(f"Model is a local directory: {cfg.model}. Skipping runtime quantization.")
        elif (target_cache / "config.json").exists():
            ui.info(f"Using cached {q_bits}-bit quantized model: {target_cache}")
            cfg.model = str(target_cache)
        else:
            ui.info(f"Model {cfg.model} needs {q_bits}-bit quantization. Starting converter...")
            try:
                run_quantization(cfg.model, q_bits, target_cache)
                cfg.model = str(target_cache)
            except Exception as e:
                ui.error(f"Failed to perform runtime quantization: {e}")
                sys.exit(1)

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
        cmd_kernel(cfg, db_settings, args.session_id, profile=args.profile)

    elif args.command == "list-skills":
        cmd_list_skills(args.root)


if __name__ == "__main__":
    main()
