"""Client-side filesystem layout for a deployed nomadic.

Resolution order, per directory kind:
  1. ``$NOMADIC_HOME`` (single root; kinds become ``config/ data/ cache/ state/`` beneath it)
  2. the matching ``$XDG_*_HOME`` variable
  3. the default under ``~``

Defaults:
  config  ~/.config/nomadic       config.jsonc, tools/ (user tool packs)
  data    ~/.local/share/nomadic  nomadic.sqlite3 (sessions/tasks), venv/ (MLX runtime), screenshots/
  cache   ~/.cache/nomadic        quantized/ (regenerable converted models; can be tens of GB)
  state   ~/.local/state/nomadic  history.txt (REPL input history)

Model weights downloaded from Hugging Face stay in the Hugging Face cache
(``$HF_HOME`` / ``~/.cache/huggingface``), which nomadic does not manage.
"""

from __future__ import annotations

import os
import pathlib

APP = "nomadic"
DB_FILENAME = "nomadic.sqlite3"
LEGACY_DB_FILENAME = "kernel.sqlite3"


def _resolve(kind: str, xdg_var: str, default_rel: tuple[str, ...], home: str | pathlib.Path | None) -> pathlib.Path:
    if home is not None:
        return pathlib.Path(home).expanduser().joinpath(*default_rel, APP)
    root = os.environ.get("NOMADIC_HOME")
    if root:
        return pathlib.Path(root).expanduser() / kind
    xdg = os.environ.get(xdg_var)
    if xdg:
        return pathlib.Path(xdg).expanduser() / APP
    return pathlib.Path.home().joinpath(*default_rel, APP)


def config_dir(home: str | pathlib.Path | None = None) -> pathlib.Path:
    return _resolve("config", "XDG_CONFIG_HOME", (".config",), home)


def data_dir(home: str | pathlib.Path | None = None) -> pathlib.Path:
    return _resolve("data", "XDG_DATA_HOME", (".local", "share"), home)


def cache_dir(home: str | pathlib.Path | None = None) -> pathlib.Path:
    return _resolve("cache", "XDG_CACHE_HOME", (".cache",), home)


def state_dir(home: str | pathlib.Path | None = None) -> pathlib.Path:
    return _resolve("state", "XDG_STATE_HOME", (".local", "state"), home)


def default_db_path() -> pathlib.Path:
    """Default SQLite file: absolute, under the data dir.

    A ``kernel.sqlite3`` already present in the working directory (the
    pre-standalone default) keeps being used so existing setups don't lose sessions.
    """
    legacy = pathlib.Path(LEGACY_DB_FILENAME)
    if legacy.is_file():
        return legacy
    return data_dir() / DB_FILENAME


def quantized_cache_dir(model: str, q_bits: int) -> pathlib.Path:
    """Where a runtime-quantized copy of ``model`` lives (legacy config-dir copies are reused)."""
    rel = pathlib.Path("quantized") / model.replace("/", "--") / f"{q_bits}bit"
    legacy = config_dir() / rel
    if (legacy / "config.json").exists():
        return legacy
    return cache_dir() / rel


def history_path() -> pathlib.Path:
    """REPL history file; migrates the legacy ``<config>/history.txt`` on first use."""
    new = state_dir() / "history.txt"
    old = config_dir() / "history.txt"
    if not new.exists() and old.is_file():
        try:
            new.parent.mkdir(parents=True, exist_ok=True)
            new.write_bytes(old.read_bytes())
        except OSError:
            return old
    return new


def describe() -> dict[str, str]:
    """Resolved locations, for `nomadic paths` / diagnostics."""
    return {
        "config": str(config_dir()),
        "data": str(data_dir()),
        "cache": str(cache_dir()),
        "state": str(state_dir()),
        "database (default)": str(default_db_path()),
        "mlx runtime venv": str(data_dir() / "venv"),
        "user tool packs": str(config_dir() / "tools"),
        "history": str(history_path()),
    }
