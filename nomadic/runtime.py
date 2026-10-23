"""Locate the Python interpreter that hosts the MLX runtime (``mlx_lm`` / ``mlx_vlm``).

nomadic itself ships as a standalone binary, so the MLX packages live in a
*different* environment. They are expected to be installed explicitly, e.g.::

    uv tool install mlx-lm            # text models
    uv tool install mlx-vlm           # vision-language models
    uv tool install mlx-lm --with mlx-vlm

``resolve`` finds the interpreter of that environment, per module, in order:

1. ``$NOMADIC_PYTHON``                       explicit override (trusted as given)
2. ``<config>/runtime.json`` ``{"python": …}``  pinned by ``nomadic runtime set``
3. the running interpreter, when not frozen and the module is importable
4. the managed venv ``<data>/venv``          (``install.sh --with-mlx``)
5. ``uv tool`` environments                   (``$UV_TOOL_DIR`` / ``uv tool dir`` / XDG default)
6. entry-point scripts on ``$PATH``           (``mlx_lm.server`` …): the shebang names the venv
   interpreter, which also covers pipx and other tool installers.

Steps 3–6 only accept an environment whose site-packages really contains the module.
"""

from __future__ import annotations

import glob
import json
import os
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterator, Optional

from . import paths

MLX_MODULES = ("mlx_lm", "mlx_vlm")
_SCRIPT_SUFFIXES = ("server", "convert", "generate")
_INSTALL_HINT = {
    "mlx_lm": "uv tool install mlx-lm",
    "mlx_vlm": "uv tool install mlx-vlm",
}


@dataclass(frozen=True)
class Runtime:
    python: str
    source: str  # env | pinned | current | managed-venv | uv-tool | path-script


class RuntimeNotFound(RuntimeError):
    """No interpreter with the requested MLX module could be located."""


# ── pinned configuration ──────────────────────────────────────────────────


def config_file() -> pathlib.Path:
    return paths.config_dir() / "runtime.json"


def pinned_python() -> Optional[str]:
    try:
        value = json.loads(config_file().read_text(encoding="utf-8")).get("python")
    except (OSError, ValueError, AttributeError):
        return None
    return value if isinstance(value, str) and value else None


def pin_python(python: Optional[str]) -> pathlib.Path:
    """Persist (or with ``None`` clear) the pinned runtime interpreter."""
    path = config_file()
    if python is None:
        path.unlink(missing_ok=True)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"python": python}, indent=2) + "\n", encoding="utf-8")
    return path


# ── environment inspection ────────────────────────────────────────────────


def _venv_python(prefix: pathlib.Path) -> Optional[pathlib.Path]:
    for rel in ("bin/python", "Scripts/python.exe"):
        cand = prefix / rel
        if cand.exists():
            return cand
    return None


def has_module(python: str | os.PathLike, module: str) -> bool:
    """True when ``python``'s venv has ``module`` installed (filesystem check, no subprocess)."""
    prefix = pathlib.Path(python).parent.parent
    for pattern in (f"lib/python3*/site-packages/{module}", f"lib/python3*/site-packages/{module}.py",
                    f"Lib/site-packages/{module}"):
        if glob.glob(str(prefix / pattern)):
            return True
    return False


def _is_venv(python: str | os.PathLike) -> bool:
    prefix = pathlib.Path(python).parent.parent
    return bool(glob.glob(str(prefix / "lib/python3*/site-packages")) or glob.glob(str(prefix / "Lib/site-packages")))


def _uv_tool_dirs() -> Iterator[pathlib.Path]:
    seen: set[pathlib.Path] = set()

    def emit(p: pathlib.Path) -> Iterator[pathlib.Path]:
        if p not in seen:
            seen.add(p)
            yield p

    if os.environ.get("UV_TOOL_DIR"):
        yield from emit(pathlib.Path(os.environ["UV_TOOL_DIR"]).expanduser())
    xdg = os.environ.get("XDG_DATA_HOME")
    yield from emit((pathlib.Path(xdg).expanduser() if xdg else pathlib.Path.home() / ".local" / "share") / "uv" / "tools")
    if shutil.which("uv"):
        try:
            out = subprocess.run(["uv", "tool", "dir"], capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                yield from emit(pathlib.Path(out.stdout.strip()))
        except (OSError, subprocess.SubprocessError):
            pass


def _uv_tool_candidates(module: str) -> Iterator[pathlib.Path]:
    # Tool dir names are the distribution name; any tool env may carry the module via --with.
    preferred = module.replace("_", "-")
    for tools in _uv_tool_dirs():
        if not tools.is_dir():
            continue
        names = [preferred] + sorted(p.name for p in tools.iterdir() if p.is_dir() and p.name != preferred)
        for name in names:
            py = _venv_python(tools / name)
            if py is not None:
                yield py


def _shebang_python(script: pathlib.Path) -> Optional[pathlib.Path]:
    try:
        with open(script, "rb") as fh:
            first = fh.readline(512).decode("utf-8", "replace").strip()
    except OSError:
        return None
    if not first.startswith("#!"):
        return None
    parts = first[2:].split()
    if not parts:
        return None
    exe = parts[1] if os.path.basename(parts[0]) == "env" and len(parts) > 1 else parts[0]
    path = pathlib.Path(exe)
    return path if path.is_absolute() and path.exists() else None


def _path_script_candidates(module: str) -> Iterator[pathlib.Path]:
    for suffix in _SCRIPT_SUFFIXES:
        found = shutil.which(f"{module}.{suffix}")
        if found:
            py = _shebang_python(pathlib.Path(found))
            if py is not None:
                yield py


def _candidates(module: str) -> Iterator[Runtime]:
    override = os.environ.get("NOMADIC_PYTHON")
    if override:
        yield Runtime(override, "env")
    pinned = pinned_python()
    if pinned:
        yield Runtime(pinned, "pinned")
    if not getattr(sys, "frozen", False):
        yield Runtime(sys.executable, "current")
    managed = _venv_python(paths.data_dir() / "venv")
    if managed is not None:
        yield Runtime(str(managed), "managed-venv")
    for py in _uv_tool_candidates(module):
        yield Runtime(str(py), "uv-tool")
    for py in _path_script_candidates(module):
        yield Runtime(str(py), "path-script")


def resolve(module: str) -> Runtime:
    """Interpreter for ``module``; raises :class:`RuntimeNotFound` with install guidance."""
    tried: list[str] = []
    for cand in _candidates(module):
        if not (pathlib.Path(cand.python).exists() or shutil.which(cand.python)):
            tried.append(f"{cand.source}: {cand.python} (does not exist)")
        elif has_module(cand.python, module):
            return cand
        elif cand.source in ("env", "pinned") and not _is_venv(cand.python):
            return cand  # explicit non-venv interpreter: trusted, the server reports a missing module itself
        else:
            tried.append(f"{cand.source}: {cand.python} (no {module})")
    hint = _INSTALL_HINT.get(module, f"uv tool install {module.replace('_', '-')}")
    detail = ("\n  searched: " + "; ".join(tried)) if tried else ""
    raise RuntimeNotFound(
        f"{module} not found. Install it with `{hint}` (nomadic finds uv tool environments automatically), "
        "or point nomadic at an existing environment: `nomadic runtime set /path/to/python` "
        f"or NOMADIC_PYTHON=/path/to/python.{detail}"
    )


def runtime_python(module: Optional[str] = None) -> str:
    """Interpreter path for ``module``; without one, any MLX module (or the current interpreter from source)."""
    if module is not None:
        return resolve(module).python
    for name in MLX_MODULES:
        try:
            return resolve(name).python
        except RuntimeNotFound:
            continue
    if not getattr(sys, "frozen", False):
        return sys.executable
    return resolve(MLX_MODULES[0]).python  # raises with guidance


def report() -> dict[str, Optional[Runtime]]:
    out: dict[str, Optional[Runtime]] = {}
    for name in MLX_MODULES:
        try:
            out[name] = resolve(name)
        except RuntimeNotFound:
            out[name] = None
    return out
