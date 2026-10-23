"""Version resolution, kept import-light so ``--version`` never loads the CLI."""

from __future__ import annotations

import pathlib


def get_version() -> str:
    """Resolve the version: stamped file (binaries) → source pyproject → package metadata."""
    try:
        from ._version import __version__
        return __version__
    except ImportError:
        pass
    try:
        import tomllib
        root = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
        return tomllib.loads(root.read_text(encoding="utf-8"))["project"]["version"]
    except Exception:
        pass
    try:
        from importlib.metadata import version
        return version("nomadic")
    except Exception:
        return "0.0.0+unknown"
