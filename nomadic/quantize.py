"""Runtime quantization pipeline (FC-13 – FC-16).

``audit`` (FC-13) → ``detect`` parameters (FC-14) → choose Q → cache hit?
→ ``convert`` (FC-15, managed subprocess streaming ``StatusEvent``s) → the
caller redirects ``ServerConfig.model`` to the result (FC-16).
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Any, Callable, Optional

from . import paths
from .agent import LifecycleState, StatusEvent

StatusSink = Callable[[StatusEvent], None]

DEFAULT_KV_CTX_LEN = 8192
FALLBACK_KV_CACHE_GB = 4.0
SUPPORTED_Q_BITS = (8, 4, 3, 2)


# ── helpers ────────────────────────────────────────────────────────────────


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


def runtime_python() -> str:
    """Interpreter that hosts the MLX runtime (mlx_lm / mlx_vlm).

    From source this is the current interpreter. In a PyInstaller binary
    ``sys.executable`` is the binary itself, so resolve the managed runtime
    venv created by ``install.sh --with-mlx`` (or ``$NOMADIC_PYTHON``).
    """
    override = os.environ.get("NOMADIC_PYTHON")
    if override:
        return override
    if not getattr(sys, "frozen", False):
        return sys.executable
    home = paths.data_dir()
    for candidate in (home / "venv" / "bin" / "python", home / "venv" / "Scripts" / "python.exe"):
        if candidate.exists():
            return str(candidate)
    raise RuntimeError(
        "MLX runtime not found. Run `install.sh --with-mlx` (Apple Silicon) "
        "or set NOMADIC_PYTHON to a Python with mlx-lm/mlx-vlm installed."
    )


# ── FC-13: system resource auditor ─────────────────────────────────────────


def get_safe_memory_budget() -> float:
    """
    Returns the safe memory budget in GB for inference,
    reserving at least 8GB or 25% of total unified memory for the OS.
    """
    try:
        mem_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]))
        mem_gb = mem_bytes / (1024 ** 3)
    except Exception:
        mem_gb = 16.0
    os_reserve = max(8.0, mem_gb * 0.25)
    return max(4.0, mem_gb - os_reserve)


# ── FC-14: model parameter detector ────────────────────────────────────────


def hf_token() -> Optional[str]:
    """Hugging Face auth token from the environment (for gated/private repos)."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def load_model_config(model_path: str) -> Optional[dict[str, Any]]:
    """Read ``config.json`` from a local directory or the Hugging Face Hub."""
    model_path = normalize_model_name(model_path)
    local_path = pathlib.Path(model_path)
    config_file: Optional[pathlib.Path] = None
    if local_path.is_dir():
        config_file = local_path / "config.json"
    else:
        try:
            from huggingface_hub import hf_hub_download

            config_file = pathlib.Path(
                hf_hub_download(repo_id=model_path, filename="config.json", token=hf_token())
            )
        except Exception:
            return None
    if config_file is None or not config_file.exists():
        return None
    try:
        return json.loads(config_file.read_text(encoding="utf-8"))
    except Exception:
        return None


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


def estimate_kv_cache_gb(cfg_dict: Optional[dict], ctx_len: int = DEFAULT_KV_CTX_LEN) -> float:
    """``M_kv_cache = 2 × L × H × D × ctx_len × 2 bytes`` in GB (H = KV heads, GQA-aware)."""
    if not cfg_dict:
        return FALLBACK_KV_CACHE_GB
    text_cfg = cfg_dict.get("text_config") if isinstance(cfg_dict.get("text_config"), dict) else cfg_dict
    layers = text_cfg.get("num_hidden_layers") or text_cfg.get("num_layers") or text_cfg.get("n_layer")
    hidden = text_cfg.get("hidden_size") or text_cfg.get("d_model")
    heads = text_cfg.get("num_attention_heads") or text_cfg.get("n_head")
    kv_heads = text_cfg.get("num_key_value_heads") or heads
    head_dim = text_cfg.get("head_dim") or (hidden // heads if hidden and heads else None)
    if not (layers and kv_heads and head_dim):
        return FALLBACK_KV_CACHE_GB
    return 2 * layers * kv_heads * head_dim * ctx_len * 2 / (1024 ** 3)


def detect_parameters(model_path: str, cfg_dict: Optional[dict] = None) -> Optional[float]:
    p = estimate_parameters_from_config(cfg_dict) if cfg_dict else None
    return p if p is not None else detect_parameters_from_name(model_path)


def determine_optimal_q_bits(model_path: str, ctx_len: int = DEFAULT_KV_CTX_LEN) -> int:
    """Pick the highest Q whose ``P × Q/8 × 1.15 + M_kv_cache`` fits the memory budget."""
    m_avail = get_safe_memory_budget()
    cfg_dict = load_model_config(model_path)
    p = detect_parameters(model_path, cfg_dict)
    if p is None:
        return 4

    limit = m_avail - estimate_kv_cache_gb(cfg_dict, ctx_len)
    if limit <= 0:
        return 2

    q_max = (limit * 8) / (p * 1.15)
    for bits in SUPPORTED_Q_BITS:
        if q_max >= bits:
            return bits
    return 2


def is_text_only_model(model_path: str) -> bool:
    """Check if model is text-only (non-multimodal) based on its configuration."""
    cfg = load_model_config(model_path)
    if cfg is None:
        return False
    if cfg.get("language_model_only") is True:
        return True
    return not ("vision_config" in cfg or "vision_tower" in cfg)


# ── FC-15: runtime quantizer engine ────────────────────────────────────────


def build_convert_command(model_path: str, q_bits: int, out_dir: pathlib.Path) -> list[str]:
    module_name = "mlx_lm" if is_text_only_model(model_path) else "mlx_vlm"
    return [
        runtime_python(), "-m", module_name, "convert",
        "--hf-path", model_path,
        "-q",
        "--q-bits", str(q_bits),
        "-o", str(out_dir),
    ]


def verify_converter(module_name: str) -> None:
    """Fail early with an actionable message when ``mlx-lm`` / ``mlx-vlm`` is missing."""
    probe = subprocess.run(
        [runtime_python(), "-c", f"import importlib.util, sys; sys.exit(importlib.util.find_spec('{module_name}') is None)"],
        capture_output=True,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            f"{module_name} is not installed in the MLX runtime ({runtime_python()}). "
            'Install it with `pip install -e ".[mlx]"` or `install.sh --with-mlx`.'
        )


def _emit(sink: Optional[StatusSink], message: str) -> None:
    if sink is not None:
        sink(StatusEvent(LifecycleState.EXECUTING, message))


async def run_quantization_async(
    model_path: str,
    q_bits: int,
    cache_dir: pathlib.Path,
    *,
    status: Optional[StatusSink] = None,
    log: Optional[Callable[[str], None]] = None,
) -> str:
    """Run the converter as a managed async subprocess, streaming log lines as ``StatusEvent``s."""
    cmd = build_convert_command(model_path, q_bits, cache_dir)
    verify_converter(cmd[2])
    _emit(status, f"Quantizing {model_path} to {q_bits}-bit...")
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    try:
        assert process.stdout is not None
        async for raw in process.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if log is not None:
                log(line)
            if line.strip():
                _emit(status, f"Quantizing {model_path}: {line.strip()[:100]}")
        code = await process.wait()
        if code != 0:
            raise RuntimeError(f"Quantization failed with exit code {code}")
    except BaseException:
        if process.returncode is None:
            process.kill()
        shutil.rmtree(cache_dir, ignore_errors=True)
        raise
    return str(cache_dir)


def run_quantization(
    model_path: str,
    q_bits: int,
    cache_dir: pathlib.Path,
    *,
    status: Optional[StatusSink] = None,
    log: Optional[Callable[[str], None]] = print,
) -> str:
    """Synchronous wrapper around :func:`run_quantization_async` for the CLI."""
    try:
        return asyncio.run(
            run_quantization_async(model_path, q_bits, cache_dir, status=status, log=lambda l: log(l) if log else None)
        )
    except Exception as exc:
        shutil.rmtree(cache_dir, ignore_errors=True)
        raise RuntimeError(f"Quantization failed: {exc}") from exc


# ── FC-16: server bootstrapper helper ──────────────────────────────────────


def resolve_q_bits(value: Any, model: str, *, warn: Optional[Callable[[str], None]] = None) -> int:
    """Interpret ``--q-bits`` (``auto`` | 8 | 4 | 3 | 2)."""
    text = str(value).strip().lower()
    if text in ("", "auto"):
        return determine_optimal_q_bits(model)
    try:
        bits = int(text)
    except ValueError:
        bits = 0
    if bits not in SUPPORTED_Q_BITS:
        if warn:
            warn(f"Invalid q_bits value '{value}'; defaulting to 4-bit.")
        return 4
    return bits


def cached_model_dir(model: str, q_bits: int, cache_dir: Optional[str] = None) -> pathlib.Path:
    return pathlib.Path(cache_dir) if cache_dir else paths.quantized_cache_dir(model, q_bits)
