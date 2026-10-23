"""File system tools (FC-10): ``read_file``, ``write_file``, ``replace_content``, ``list_dir``.

Handlers emit ``READING`` / ``WRITING`` / ``EDITING`` status events (FC-17)
through :meth:`AgentRuntime.emit` and report the files they touch in
``ToolExecutionResult.payload`` so the orchestrator can track task artifacts (FC-4).
"""

from __future__ import annotations

import ast
import os
import pathlib
import textwrap
from typing import Any, Mapping

from ..agent import (
    AgentRuntime,
    LifecycleState,
    ToolDefinition,
    ToolExecutionResult,
)

MAX_READ_BYTES = 5_000_000
LIST_DIR_LIMIT = 500
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", ".mypy_cache", ".pytest_cache"}


def _resolve(path: Any, name: str) -> pathlib.Path:
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"{name} requires a non-empty string path")
    return pathlib.Path(os.path.expanduser(path.strip()))


def _int_arg(arguments: Mapping[str, Any], key: str) -> int | None:
    value = arguments.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _read_lines(p: pathlib.Path, start: int, end: int | None, max_chars: int) -> tuple[list[str], int | None, bool]:
    """Stream *p* and return ``(lines, total_lines_or_None, truncated)``.

    Only the requested window is held in memory, so large files are safe to
    read as long as a range (or the character budget) bounds the result.
    """
    lines: list[str] = []
    chars = 0
    truncated = False
    total = 0
    with p.open("r", encoding="utf-8", errors="replace") as handle:
        for total, line in enumerate(handle, start=1):
            if total < start:
                continue
            if end is not None and total > end:
                # keep counting only for small files; stop early on big ones
                if p.stat().st_size > MAX_READ_BYTES:
                    return lines, None, truncated
                continue
            if truncated:
                continue
            text = line.rstrip("\n")
            if chars + len(text) > max_chars:
                lines.append(text[: max(0, max_chars - chars)])
                truncated = True
                continue
            lines.append(text)
            chars += len(text) + 1
    return lines, total, truncated


def handle_read_file(arguments: Mapping[str, Any], runtime: AgentRuntime) -> ToolExecutionResult:
    p = _resolve(arguments.get("path"), "read_file")
    start = _int_arg(arguments, "start_line")
    end = _int_arg(arguments, "end_line")
    if start is not None and start < 1:
        raise ValueError("start_line must be >= 1")
    if start is not None and end is not None and end < start:
        raise ValueError("end_line must be >= start_line")
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")

    runtime.emit(LifecycleState.READING, f"Reading {p}")
    lines, total, truncated = _read_lines(p, start or 1, end, runtime.max_skill_chars)
    first = start or 1
    last = first + len(lines) - 1 if lines else first
    content = "\n".join(lines)
    notes: list[str] = []
    if truncated:
        notes.append(
            f"[... truncated at {runtime.max_skill_chars} chars; "
            "use start_line/end_line to read further ...]"
        )
    span = f"lines {first}-{last}" + (f" of {total}" if total is not None else "")
    body = content + ("\n\n" + "\n".join(notes) if notes else "")
    return ToolExecutionResult(
        tool_name="read_file",
        label=f"read_file: {p}",
        status="ok",
        content=f"Path: {p}\nRange: {span}\nContent:\n{body}",
        payload={"path": str(p), "content": content, "start_line": first, "end_line": last,
                 "total_lines": total, "truncated": truncated},
    )


def handle_write_file(arguments: Mapping[str, Any], runtime: AgentRuntime) -> ToolExecutionResult:
    p = _resolve(arguments.get("path"), "write_file")
    content = arguments.get("content")
    if not isinstance(content, str):
        raise ValueError("write_file requires path and content")
    action = "modified" if p.exists() else "created"
    runtime.emit(LifecycleState.WRITING, f"Writing {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return ToolExecutionResult(
        tool_name="write_file",
        label=f"write_file: {p}",
        status="ok",
        content=f"Successfully wrote to {p}",
        payload={"path": str(p), "action": action, "bytes": len(content.encode("utf-8"))},
    )


def _find_symbol(tree: ast.AST, dotted: str) -> ast.AST:
    node: ast.AST = tree
    for part in dotted.split("."):
        for child in getattr(node, "body", []):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and child.name == part:
                node = child
                break
        else:
            raise ValueError(f"Symbol not found: {dotted}")
    if node is tree:
        raise ValueError("symbol must name a function, class, or Class.method")
    return node


def _replace_symbol(text: str, symbol: str, replacement: str) -> tuple[str, int]:
    """AST-aware edit: swap the full source of a Python definition."""
    node = _find_symbol(ast.parse(text), symbol)
    # include decorators in the replaced span
    first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    last = node.end_lineno or node.lineno
    lines = text.splitlines(keepends=True)
    indent = " " * node.col_offset
    body = textwrap.dedent(replacement).rstrip("\n")
    new_lines = [(indent + ln if ln.strip() else ln) + "\n" for ln in body.splitlines()]
    lines[first - 1:last] = new_lines
    return "".join(lines), first


def handle_replace_content(arguments: Mapping[str, Any], runtime: AgentRuntime) -> ToolExecutionResult:
    p = _resolve(arguments.get("path"), "replace_content")
    target = arguments.get("target")
    symbol = arguments.get("symbol")
    replacement = arguments.get("replacement")
    replace_all = bool(arguments.get("replace_all", False))
    if type(replacement) is not str or not (target or symbol):
        raise ValueError("replace_content requires path, replacement, and a target or symbol")
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")

    text = p.read_text(encoding="utf-8")
    if symbol:
        if p.suffix != ".py":
            raise ValueError("symbol edits are only supported for .py files")
        new_text, line = _replace_symbol(text, str(symbol), replacement)
        detail = f"symbol {symbol}"
    else:
        count = text.count(target)
        if count == 0:
            raise ValueError("Target string not found in file")
        if count > 1 and not replace_all:
            raise ValueError(
                f"Target matches {count} places; add context to make it unique or set replace_all"
            )
        line = text.count("\n", 0, text.index(target)) + 1
        new_text = text.replace(target, replacement) if replace_all else text.replace(target, replacement, 1)
        detail = f"{count if replace_all else 1} occurrence(s)"

    if p.suffix == ".py":
        try:
            compile(new_text, str(p), "exec")
        except SyntaxError as exc:
            raise ValueError(f"Edit would leave {p.name} with a syntax error: {exc.msg} (line {exc.lineno})")

    runtime.emit(LifecycleState.EDITING, f"Editing line {line} of {p}")
    p.write_text(new_text, encoding="utf-8")
    return ToolExecutionResult(
        tool_name="replace_content",
        label=f"replace_content: {p}",
        status="ok",
        content=f"Successfully replaced {detail} in {p} (first change at line {line})",
        payload={"path": str(p), "action": "modified", "line": line},
    )


def handle_list_dir(arguments: Mapping[str, Any], runtime: AgentRuntime) -> ToolExecutionResult:
    raw = arguments.get("path", ".")
    p = _resolve(raw if raw else ".", "list_dir")
    if not p.is_dir():
        raise FileNotFoundError(f"Directory not found: {p}")
    recursive = bool(arguments.get("recursive", True))
    runtime.emit(LifecycleState.READING, f"Listing {p}")

    items: list[str] = []
    if recursive:
        for root, dirs, files in os.walk(p):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            rel = pathlib.Path(root).relative_to(p)
            items.extend(str(rel / f) for f in sorted(files))
            if len(items) > LIST_DIR_LIMIT:
                break
    else:
        items = [e.name + ("/" if e.is_dir() else "") for e in sorted(p.iterdir())]
    shown = items[:LIST_DIR_LIMIT]
    suffix = f"\n[... {len(items) - LIST_DIR_LIMIT}+ more ...]" if len(items) > LIST_DIR_LIMIT else ""
    return ToolExecutionResult(
        tool_name="list_dir",
        label=f"list_dir: {p}",
        status="ok",
        content=f"Directory contents of {p}{' (recursive files)' if recursive else ''}:\n" + "\n".join(shown) + suffix,
        payload={"path": str(p), "count": len(items)},
    )


def get_tools() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="read_file",
            description="Read a file, optionally a 1-indexed inclusive line range. Large files are read incrementally.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=handle_read_file,
        ),
        ToolDefinition(
            name="write_file",
            description="Create or overwrite a file with content.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            handler=handle_write_file,
        ),
        ToolDefinition(
            name="replace_content",
            description=(
                "Edit a file. Either replace an exact `target` string (must be unique unless replace_all), "
                "or for .py files replace a whole definition by `symbol` (e.g. 'func' or 'Class.method')."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "target": {"type": "string"},
                    "symbol": {"type": "string"},
                    "replacement": {"type": "string"},
                    "replace_all": {"type": "boolean"},
                },
                "required": ["path", "replacement"],
                "additionalProperties": False,
            },
            handler=handle_replace_content,
        ),
        ToolDefinition(
            name="list_dir",
            description="List files in a directory (recursive by default; skips .git, __pycache__, node_modules).",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}, "recursive": {"type": "boolean"}},
                "additionalProperties": False,
            },
            handler=handle_list_dir,
        ),
    ]
