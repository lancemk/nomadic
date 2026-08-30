#!/usr/bin/env python3
"""Structured tool-calling agent loop helpers for the CLI kernel."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional, List

from .search import metadata as metadata_search

class LifecycleState(Enum):
    THINKING = "thinking"
    READING = "reading"
    WRITING = "writing"
    EDITING = "editing"
    EXECUTING = "executing"
    REVIEWING = "reviewing"

@dataclass(frozen=True)
class StatusEvent:
    state: LifecycleState
    message: str
    depth: int = 0

DEFAULT_AGENT_MAX_DEPTH = 5
DEFAULT_SHELL_TIMEOUT_SECONDS = 30
DEFAULT_TOOL_OUTPUT_CHARS = 4_000


@dataclass(frozen=True)
class ToolCall:
    """Validated tool call emitted by the model."""

    name: str
    arguments: dict[str, Any]
    raw_json: str


@dataclass(frozen=True)
class ToolExecutionResult:
    """Normalized tool execution result for context injection and logging."""

    tool_name: str
    label: str
    status: str
    content: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentRuntime:
    """Runtime state exposed to tools."""

    state: Any
    ui: Any
    loop_depth: int = 0
    shell_timeout_seconds: int = DEFAULT_SHELL_TIMEOUT_SECONDS
    max_tool_output_chars: int = DEFAULT_TOOL_OUTPUT_CHARS
    status_callback: Callable[[StatusEvent], None] | None = None

    @property
    def max_skill_chars(self) -> int:
        return int(getattr(self.state.cfg, "max_skill_chars", 12_000))

    @property
    def metadata_base_url(self) -> str:
        return str(
            getattr(
                self.state.cfg,
                "metadata_base_url",
                metadata_search.DEFAULT_METADATA_BASE_URL,
            )
        )


ToolHandler = Callable[[Mapping[str, Any], AgentRuntime], ToolExecutionResult]


@dataclass(frozen=True)
class ToolDefinition:
    """Definition for one tool exposed to the model."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


def read_skill_file(
    path: str | pathlib.Path,
    *,
    max_chars: int,
) -> dict[str, Any]:
    """Read and optionally truncate a SKILL.md file."""
    skill_path = pathlib.Path(path)
    if not skill_path.exists():
        raise FileNotFoundError(f"SKILL.md not found: {skill_path}")

    text = skill_path.read_text(encoding="utf-8", errors="replace")
    original_chars = len(text)
    truncated = original_chars > max_chars
    if truncated:
        text = text[:max_chars] + "\n\n[… truncated …]"

    return {
        "path": str(skill_path),
        "content": text,
        "original_chars": original_chars,
        "returned_chars": len(text),
        "truncated": truncated,
    }


def execute_shell_command(
    command: str,
    *,
    timeout_seconds: int = DEFAULT_SHELL_TIMEOUT_SECONDS,
    max_output_chars: int = DEFAULT_TOOL_OUTPUT_CHARS,
) -> dict[str, Any]:
    """Run a shell command and capture normalized output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        combined = stdout + stderr
        preview = combined[:max_output_chars]
        return {
            "command": command,
            "ok": result.returncode == 0,
            "status": "ok" if result.returncode == 0 else "error",
            "exit_code": result.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "combined_output": combined,
            "preview": preview,
            "timeout": False,
            "truncated": len(combined) > len(preview),
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        combined = (stdout or "") + (stderr or "")
        preview = combined[:max_output_chars]
        return {
            "command": command,
            "ok": False,
            "status": "error",
            "exit_code": None,
            "stdout": stdout,
            "stderr": stderr,
            "combined_output": combined,
            "preview": preview,
            "timeout": True,
            "truncated": len(combined) > len(preview),
            "error": f"timeout after {timeout_seconds}s",
        }


def _render_read_skill_result(payload: Mapping[str, Any]) -> str:
    lines = [
        "Tool: read_skill",
        f"Path: {payload['path']}",
        f"Truncated: {'yes' if payload['truncated'] else 'no'}",
        f"Characters: {payload['returned_chars']}",
        "",
        "Content:",
        str(payload["content"]),
    ]
    return "\n".join(lines)


def _render_exec_shell_result(payload: Mapping[str, Any]) -> str:
    lines = [
        "Tool: exec_shell",
        f"Command: {payload['command']}",
        f"Status: {payload['status']}",
        f"Exit Code: {payload['exit_code']}",
        f"Timeout: {'yes' if payload['timeout'] else 'no'}",
    ]
    if payload.get("truncated"):
        lines.append("Output Truncated: yes")
    lines.extend(
        [
            "",
            "Output:",
            str(payload.get("preview") or "[no output]"),
        ]
    )
    return "\n".join(lines)


def _render_metadata_search_result(payload: Mapping[str, Any]) -> str:
    return metadata_search.render_metadata_search_result(dict(payload))

def _handle_read_file(arguments: Mapping[str, Any], runtime: AgentRuntime) -> ToolExecutionResult:
    path = arguments.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("read_file requires a string path")
    start_line = arguments.get("start_line")
    end_line = arguments.get("end_line")
    
    p = pathlib.Path(path.strip())
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
        
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    if isinstance(start_line, int) and isinstance(end_line, int):
        content = "\n".join(lines[start_line-1:end_line])
    else:
        content = "\n".join(lines)
        
    if len(content) > runtime.max_skill_chars:
        content = content[:runtime.max_skill_chars] + "\n\n[... truncated ...]"
        
    return ToolExecutionResult(
        tool_name="read_file",
        label=f"read_file: {path}",
        status="ok",
        content=f"Path: {path}\nContent:\n{content}",
        payload={"path": path, "content": content},
    )

def _handle_write_file(arguments: Mapping[str, Any], runtime: AgentRuntime) -> ToolExecutionResult:
    path = arguments.get("path")
    content = arguments.get("content")
    if not path or not isinstance(content, str):
        raise ValueError("write_file requires path and content")
    p = pathlib.Path(path.strip())
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return ToolExecutionResult(
        tool_name="write_file",
        label=f"write_file: {path}",
        status="ok",
        content=f"Successfully wrote to {path}",
        payload={"path": str(path)}
    )

def _handle_replace_content(arguments: Mapping[str, Any], runtime: AgentRuntime) -> ToolExecutionResult:
    path = arguments.get("path")
    target = arguments.get("target")
    replacement = arguments.get("replacement")
    if not path or not target or type(replacement) is not str:
        raise ValueError("replace_content requires path, target, and replacement")
    p = pathlib.Path(path.strip())
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    text = p.read_text(encoding="utf-8")
    if target not in text:
        raise ValueError("Target string not found in file")
    text = text.replace(target, replacement)
    p.write_text(text, encoding="utf-8")
    return ToolExecutionResult(
        tool_name="replace_content",
        label=f"replace_content: {path}",
        status="ok",
        content=f"Successfully replaced content in {path}",
        payload={"path": str(path)}
    )

def _handle_list_dir(arguments: Mapping[str, Any], runtime: AgentRuntime) -> ToolExecutionResult:
    path = arguments.get("path", ".")
    p = pathlib.Path(path.strip())
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Directory not found: {p}")
    items = sorted([str(item.relative_to(p)) for item in p.rglob("*") if item.is_file()])
    return ToolExecutionResult(
        tool_name="list_dir",
        label=f"list_dir: {path}",
        status="ok",
        content=f"Directory contents of {path} (recursive files):\n" + "\n".join(items[:500]),
        payload={"path": str(path)}
    )


def _handle_read_skill(arguments: Mapping[str, Any], runtime: AgentRuntime) -> ToolExecutionResult:
    path = arguments.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("read_skill requires a non-empty string argument: path")

    payload = read_skill_file(path.strip(), max_chars=runtime.max_skill_chars)
    return ToolExecutionResult(
        tool_name="read_skill",
        label=f"read_skill: {payload['path']}",
        status="ok",
        content=_render_read_skill_result(payload),
        payload=dict(payload),
    )


def _handle_exec_shell(arguments: Mapping[str, Any], runtime: AgentRuntime) -> ToolExecutionResult:
    command = arguments.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError("exec_shell requires a non-empty string argument: command")

    payload = execute_shell_command(
        command.strip(),
        timeout_seconds=runtime.shell_timeout_seconds,
        max_output_chars=runtime.max_tool_output_chars,
    )
    return ToolExecutionResult(
        tool_name="exec_shell",
        label=f"exec_shell: {payload['command']}",
        status=str(payload["status"]),
        content=_render_exec_shell_result(payload),
        payload=dict(payload),
    )


def _handle_metadata_search(arguments: Mapping[str, Any], runtime: AgentRuntime) -> ToolExecutionResult:
    query = arguments.get("query")
    limit = arguments.get("limit")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("metadata_search requires a non-empty string argument: query")

    payload = metadata_search.search_metadata(
        query=query.strip(),
        limit=limit if isinstance(limit, int) else 5,
        base_url=runtime.metadata_base_url,
    )
    status = "ok" if payload.get("ok") else "error"
    label = f"metadata_search: {payload.get('query') or '[empty]'}"
    return ToolExecutionResult(
        tool_name="metadata_search",
        label=label,
        status=status,
        content=_render_metadata_search_result(payload),
        payload=dict(payload),
    )


class ToolRegistry:
    """Registry plus parsing and dispatch helpers for structured tool calls."""

    def __init__(self, tools: list[ToolDefinition]):
        self._tools = {tool.name: tool for tool in tools}

    @property
    def tools(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def register(self, tool: ToolDefinition):
        """Add a new tool to the registry."""
        self._tools[tool.name] = tool

    def unregister(self, name: str):
        """Remove a tool from the registry."""
        if name in self._tools:
            del self._tools[name]

    def load_from_directory(self, directory: str | pathlib.Path):
        """
        Scan a directory for .py files and load ToolDefinitions from them.
        Expects a module-level `get_tools` function that returns List[ToolDefinition].
        """
        path = pathlib.Path(directory).expanduser()
        if not path.exists() or not path.is_dir():
            return

        for py_file in path.glob("*.py"):
            if py_file.name == "__init__.py":
                continue
            
            try:
                module_name = f"nomadic.tools.{py_file.stem}"
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if hasattr(module, "get_tools"):
                        tools = module.get_tools()
                        for tool in tools:
                            if isinstance(tool, ToolDefinition):
                                self.register(tool)
            except Exception:
                # In a real tool, we might want to log this to the UI
                pass

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self.tools]

    def instruction_block(self) -> str:
        schema_json = json.dumps(self.schemas(), indent=2, ensure_ascii=False)
        return (
            "Tool Calling Protocol:\n"
            "When you need a tool, respond with ONLY valid JSON and no prose.\n"
            "Use exactly this envelope:\n"
            '{"tool_call":{"name":"<tool_name>","arguments":{...}}}\n'
            "Call at most one tool per message.\n"
            "When you do not need a tool, answer normally in plain text.\n"
            "Available tools:\n"
            f"{schema_json}"
        )

    def execute(self, tool_call: ToolCall, runtime: AgentRuntime) -> ToolExecutionResult:
        tool = self._tools.get(tool_call.name)
        if tool is None:
            raise ValueError(f"Unknown tool: {tool_call.name}")
            
        if runtime and runtime.status_callback:
            if tool.name in ("read_skill", "metadata_search"):
                runtime.status_callback(StatusEvent(LifecycleState.READING, f"Reading {tool_call.arguments}", runtime.loop_depth))
            elif tool.name == "exec_shell":
                runtime.status_callback(StatusEvent(LifecycleState.EXECUTING, f"Running {tool_call.arguments.get('command')}", runtime.loop_depth))
            elif tool.name in ("write_file", "replace_content"):
                runtime.status_callback(StatusEvent(LifecycleState.WRITING, f"Writing {tool_call.arguments.get('path')}", runtime.loop_depth))
            else:
                runtime.status_callback(StatusEvent(LifecycleState.EXECUTING, f"Running {tool.name}...", runtime.loop_depth))
                
        return tool.handler(tool_call.arguments, runtime)


def build_default_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ToolDefinition(
                name="read_skill",
                description="Read a local SKILL.md file and return its contents.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute or relative filesystem path to a SKILL.md file.",
                        }
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
                handler=_handle_read_skill,
            ),
            ToolDefinition(
                name="exec_shell",
                description="Run one shell command and return stdout and stderr.",
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to execute.",
                        }
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
                handler=_handle_exec_shell,
            ),
            ToolDefinition(
                name="metadata_search",
                description="Search online metadata using the configured metadata service.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Metadata search query string.",
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Optional maximum number of results to request.",
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                handler=_handle_metadata_search,
            ),
            ToolDefinition(
                name="read_file",
                description="Read contents of a file.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer"},
                        "end_line": {"type": "integer"}
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
                handler=_handle_read_file,
            ),
            ToolDefinition(
                name="write_file",
                description="Create or overwrite a file with content.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
                handler=_handle_write_file,
            ),
            ToolDefinition(
                name="replace_content",
                description="Replace exact string matches in a file.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "target": {"type": "string"},
                        "replacement": {"type": "string"}
                    },
                    "required": ["path", "target", "replacement"],
                    "additionalProperties": False,
                },
                handler=_handle_replace_content,
            ),
            ToolDefinition(
                name="list_dir",
                description="List all files in a directory.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"}
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
                handler=_handle_list_dir,
            ),
        ]
    )


def default_system_prompt() -> str:
    """Default system prompt for the structured tool-calling kernel."""
    return (
        "You are BITSTRIC's L7 IC AI Capacity Expert operating in a structured "
        "tool-calling kernel.\n\n"
        "Rules:\n"
        "1. Think privately before acting; do not reveal hidden reasoning.\n"
        "2. Use tools only when they help answer the current request.\n"
        "3. If a tool is needed, emit one structured tool_call JSON object.\n"
        "4. After tool results arrive, continue from that evidence.\n"
        "5. Conclude normal replies with a concise result summary when useful.\n"
        "6. Stay within the IT Compliance / Robotics / Legal-Finance domains.\n\n"
        f"{build_default_registry().instruction_block()}"
    )


def _balanced_json_objects(text: str) -> list[str]:
    """Extract balanced JSON object candidates from free-form text."""
    candidates: list[str] = []
    start: Optional[int] = None
    depth = 0
    in_string = False
    escaped = False

    for idx, char in enumerate(text):
        if start is None:
            if char == "{":
                start = idx
                depth = 1
                in_string = False
                escaped = False
            continue

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidates.append(text[start : idx + 1])
                start = None

    return candidates


def _extract_json_candidates(reply: str) -> list[str]:
    stripped = reply.strip()
    candidates: list[str] = []

    if stripped.startswith("```"):
        fence_parts = stripped.split("```")
        for part in fence_parts:
            part = part.strip()
            if not part:
                continue
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                candidates.append(part)

    candidates.extend(_balanced_json_objects(reply))
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        if candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique


def _normalize_tool_call_object(data: Any) -> Optional[dict[str, Any]]:
    if not isinstance(data, dict):
        return None
    if "tool_call" in data and isinstance(data["tool_call"], dict):
        return data["tool_call"]
    if isinstance(data.get("name"), str) and isinstance(data.get("arguments"), dict):
        return data
    return None


def parse_tool_call(reply: str) -> tuple[Optional[ToolCall], Optional[str]]:
    """
    Parse a structured tool call from the model reply.

    Returns `(tool_call, error_message)`. When no tool call is present, both
    values are `None`.
    """
    if not isinstance(reply, str):
        return None, None
    candidates = _extract_json_candidates(reply)
    saw_toolish_json = False
    last_error: Optional[str] = None

    for candidate in candidates:
        candidate_looks_like_tool = (
            '"tool_call"' in candidate
            or ('"name"' in candidate and '"arguments"' in candidate)
        )
        if candidate_looks_like_tool:
            saw_toolish_json = True
        try:
            decoded = json.loads(candidate)
        except json.JSONDecodeError as exc:
            if candidate_looks_like_tool:
                last_error = f"Malformed tool_call JSON: {exc.msg}"
            continue

        call_obj = _normalize_tool_call_object(decoded)
        if call_obj is None:
            continue

        name = call_obj.get("name")
        arguments = call_obj.get("arguments", {})
        if not isinstance(name, str) or not name.strip():
            return None, "Invalid tool_call: missing string field 'name'"
        if not isinstance(arguments, dict):
            return None, "Invalid tool_call: 'arguments' must be a JSON object"

        return ToolCall(name=name.strip(), arguments=dict(arguments), raw_json=candidate), None

    if saw_toolish_json:
        return None, last_error or "Invalid tool_call JSON envelope"

    return None, None


def _build_tool_error_result(
    *,
    label: str,
    error: str,
    tool_name: Optional[str] = None,
    arguments: Optional[Mapping[str, Any]] = None,
) -> ToolExecutionResult:
    payload = {
        "error": error,
        "tool_name": tool_name,
        "arguments": dict(arguments or {}),
    }
    lines = [
        f"Tool Error: {label}",
        f"Status: error",
        "",
        error,
    ]
    return ToolExecutionResult(
        tool_name=tool_name or "tool_error",
        label=label,
        status="error",
        content="\n".join(lines),
        payload=payload,
    )


def run_named_tool(
    tool_name: str,
    arguments: Mapping[str, Any],
    *,
    runtime: AgentRuntime,
    registry: Optional[ToolRegistry] = None,
) -> ToolExecutionResult:
    active_registry = registry or build_default_registry()
    tool_call = ToolCall(name=tool_name, arguments=dict(arguments), raw_json="")
    try:
        return active_registry.execute(tool_call, runtime)
    except Exception as exc:
        return _build_tool_error_result(
            label=f"{tool_name} error",
            error=str(exc),
            tool_name=tool_name,
            arguments=arguments,
        )


def run_agent_loop(
    *,
    state: Any,
    complete_fn: Callable[[Any, list[dict[str, Any]], bool], str],
    ui: Any,
    registry: Optional[ToolRegistry] = None,
    max_depth: int = DEFAULT_AGENT_MAX_DEPTH,
    status_callback: Callable[[StatusEvent], None] | None = None,
) -> str:
    """Drive the structured tool-calling loop until a final answer is produced."""
    active_registry = registry or build_default_registry()
    runtime = AgentRuntime(state=state, ui=ui, status_callback=status_callback)

    depth = 0
    while True:
        if status_callback:
            status_callback(StatusEvent(LifecycleState.THINKING, "Thinking...", depth))

        reply = complete_fn(state.cfg, state.history, stream=False)

        state.add_assistant(reply)
        tool_call, parse_error = parse_tool_call(reply)

        if tool_call is None and parse_error is None:
            ui.model_output(reply)
            return reply

        if depth >= max_depth:
            note = (
                f"Agent loop stopped after reaching the tool-call limit ({max_depth}). "
                "Please refine the request or continue from the latest evidence."
            )
            ui.warn(note)
            state.add_assistant(note)
            ui.model_output(note)
            return note

        depth += 1
        runtime.loop_depth = depth

        if tool_call is None:
            ui.warn(f"[agent] invalid tool_call JSON (depth {depth})")
            result = _build_tool_error_result(
                label="tool_call error",
                error=parse_error or "Invalid tool_call JSON",
            )
            state.add_tool_result(
                result.label,
                result.content,
                status=result.status,
                payload={**result.payload, "depth": depth},
                tool_name=result.tool_name,
                depth=depth,
            )
        else:
            ui.info(f"[agent] tool_call → {tool_call.name} (depth {depth})")
            if hasattr(state, "record_tool_invocation"):
                state.record_tool_invocation(
                    tool_name=tool_call.name,
                    arguments=tool_call.arguments,
                    depth=depth,
                    raw_json=tool_call.raw_json,
                )

            try:
                result = active_registry.execute(tool_call, runtime)
            except Exception as exc:
                result = _build_tool_error_result(
                    label=f"{tool_call.name} error",
                    error=str(exc),
                    tool_name=tool_call.name,
                    arguments=tool_call.arguments,
                )

            state.add_tool_result(
                result.label,
                result.content,
                status=result.status,
                payload={**result.payload, "depth": depth},
                tool_name=result.tool_name,
                arguments=tool_call.arguments,
                depth=depth,
            )

        state.compact_history_if_needed()
        ui.info(f"[agent] tool result → re-completing (depth {depth})")
        ui.rule(f"turn {state.turn}.{depth} · agent follow-up")
        reply = complete_fn(state.cfg, state.history, stream=False)
