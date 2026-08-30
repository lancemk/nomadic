"""
Agentic Orchestration Engine.

FC-1 (Standalone Single-Agent Orchestration) and FC-2 (Multi-Agent
Parallelism & Async Execution) from SRS-ext-v1.1.

Provides:
- ``AgentInstance``: A standalone agentic loop with its own context,
  tool registry, and session state.
- ``Orchestrator``: Top-level manager for task decomposition, agent
  assignment, and lifecycle management of multiple agent instances.
"""

from __future__ import annotations

import asyncio
import enum
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from . import agent as agent_mod
from .hitl import HITLManager, ReviewDecision, ReviewResult, SafetyProfile
from .tasks import Task, TaskArtifact, TaskRegistry, TaskStatus


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT INSTANCE  (FC-1 · M1.1.1)
# ═══════════════════════════════════════════════════════════════════════════════


class AgentStatus(enum.Enum):
    """Runtime status of an ``AgentInstance``."""

    IDLE = "idle"
    RUNNING = "running"
    AWAITING_REVIEW = "awaiting_review"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentInstance:
    """A standalone agentic loop with its own context, tool registry, and state.

    Extracted from the monolithic ``run_agent_loop`` to support:
    - independent lifecycle management per agent
    - HITL review gates between tool calls
    - async execution via the orchestrator
    """

    id: str
    system_prompt: str
    tool_registry: agent_mod.ToolRegistry
    hitl_manager: HITLManager | None = None
    task: Task | None = None

    history: list[dict[str, Any]] = field(default_factory=list)
    status: AgentStatus = AgentStatus.IDLE
    current_tool_call: agent_mod.ToolCall | None = None
    result: str | None = None
    error: str | None = None
    depth: int = 0
    max_depth: int = agent_mod.DEFAULT_AGENT_MAX_DEPTH

    _cancel_requested: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if not self.history:
            self.history = [{"role": "system", "content": self.system_prompt}]

    # -- Predicates ----------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            AgentStatus.COMPLETED,
            AgentStatus.FAILED,
            AgentStatus.CANCELLED,
        )

    # -- Lifecycle -----------------------------------------------------------

    def cancel(self) -> None:
        self._cancel_requested = True
        self.status = AgentStatus.CANCELLED

    def add_user_message(self, content: str) -> None:
        """Append a user message to the agent's history."""
        self.history.append({"role": "user", "content": content})

    # -- Core step -----------------------------------------------------------

    async def step(
        self,
        complete_fn: Callable[..., str],
        ui: Any,
    ) -> tuple[str | None, bool]:
        """Execute a single Thought → Action → Observation cycle.

        Returns
        -------
        (reply_text, is_final)
            *reply_text* is the model output (or ``None`` when the step
            is waiting for HITL review).  *is_final* is ``True`` when the
            agent has reached a terminal state.
        """
        if self._cancel_requested:
            self.status = AgentStatus.CANCELLED
            return None, True

        self.status = AgentStatus.RUNNING

        # Run the LLM completion in an executor so we never block the loop.
        loop = asyncio.get_running_loop()
        reply: str = await loop.run_in_executor(
            None,
            lambda: complete_fn(self.history, stream=False),
        )

        self.history.append({"role": "assistant", "content": reply})

        tool_call, parse_error = agent_mod.parse_tool_call(reply)

        # ── Final answer (no tool call) ────────────────────────────────────
        if tool_call is None and parse_error is None:
            self.status = AgentStatus.COMPLETED
            self.result = reply
            return reply, True

        # ── Depth guard ────────────────────────────────────────────────────
        if self.depth >= self.max_depth:
            note = (
                f"Agent loop stopped after reaching the tool-call limit "
                f"({self.max_depth}). Please refine the request or continue "
                f"from the latest evidence."
            )
            self.history.append({"role": "assistant", "content": note})
            self.status = AgentStatus.COMPLETED
            self.result = note
            return note, True

        self.depth += 1
        runtime = agent_mod.AgentRuntime(state=None, ui=ui)
        runtime.loop_depth = self.depth

        # ── Parse error ────────────────────────────────────────────────────
        if tool_call is None:
            result = agent_mod._build_tool_error_result(
                label="tool_call error",
                error=parse_error or "Invalid tool_call JSON",
            )
            self._inject_tool_result(result)
            return None, False

        self.current_tool_call = tool_call

        # ── HITL interception (FC-3) ───────────────────────────────────────
        if self.hitl_manager and self.hitl_manager.requires_review(
            tool_call.name, tool_call.arguments,
        ):
            self.status = AgentStatus.AWAITING_REVIEW
            self.hitl_manager.create_review_request(
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                raw_json=tool_call.raw_json,
                agent_id=self.id,
                depth=self.depth,
            )
            # Caller must resolve the review before continuing.
            return None, False

        # ── Execute tool ───────────────────────────────────────────────────
        result = self._execute_tool(tool_call, runtime)
        self._inject_tool_result(result)
        self._track_artifact(tool_call, result)
        self.current_tool_call = None
        return None, False

    # -- Full run loop -------------------------------------------------------

    async def run(
        self,
        complete_fn: Callable[..., str],
        ui: Any,
    ) -> str:
        """Drive the full agent loop until a final answer or terminal state."""
        self.status = AgentStatus.RUNNING

        while not self.is_terminal:
            reply, is_final = await self.step(complete_fn, ui)

            if is_final:
                if ui is not None and reply:
                    try:
                        ui.model_output(reply)
                    except Exception:
                        pass
                return reply or self.result or ""

            if self.status == AgentStatus.AWAITING_REVIEW:
                # In single-agent sync mode the caller is responsible for
                # resolving the review and then re-entering ``run``.
                return ""

        return self.result or ""

    # -- HITL resolution -----------------------------------------------------

    def resolve_review(
        self,
        decision: ReviewDecision,
        *,
        modified_args: dict[str, Any] | None = None,
        feedback: str = "",
        ui: Any = None,
    ) -> None:
        """Resume after a HITL review decision."""
        if self.status != AgentStatus.AWAITING_REVIEW:
            return

        if self.hitl_manager:
            self.hitl_manager.resolve_review(self.id)

        if decision == ReviewDecision.APPROVE:
            # Execute the previously intercepted tool call.
            if self.current_tool_call:
                runtime = agent_mod.AgentRuntime(state=None, ui=ui)
                runtime.loop_depth = self.depth
                result = self._execute_tool(self.current_tool_call, runtime)
                self._inject_tool_result(result)
                self._track_artifact(self.current_tool_call, result)
            self.current_tool_call = None
            self.status = AgentStatus.RUNNING

        elif decision == ReviewDecision.MODIFY and modified_args and self.current_tool_call:
            modified_call = agent_mod.ToolCall(
                name=self.current_tool_call.name,
                arguments=modified_args,
                raw_json=self.current_tool_call.raw_json,
            )
            runtime = agent_mod.AgentRuntime(state=None, ui=ui)
            runtime.loop_depth = self.depth
            result = self._execute_tool(modified_call, runtime)
            self._inject_tool_result(result)
            self._track_artifact(modified_call, result)
            self.current_tool_call = None
            self.status = AgentStatus.RUNNING

        else:
            # REJECT or SKIP — inject feedback as tool result
            msg = feedback or "Tool call rejected by operator."
            self.history.append({
                "role": "tool",
                "content": f"[HITL] {msg}",
            })
            self.current_tool_call = None
            self.status = AgentStatus.RUNNING

    # -- Internal helpers ----------------------------------------------------

    def _execute_tool(
        self,
        tool_call: agent_mod.ToolCall,
        runtime: agent_mod.AgentRuntime,
    ) -> agent_mod.ToolExecutionResult:
        try:
            return self.tool_registry.execute(tool_call, runtime)
        except Exception as exc:
            return agent_mod._build_tool_error_result(
                label=f"{tool_call.name} error",
                error=str(exc),
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
            )

    def _inject_tool_result(self, result: agent_mod.ToolExecutionResult) -> None:
        self.history.append({
            "role": "tool",
            "content": f"[{result.label}] {result.content}",
        })

    def _track_artifact(
        self,
        tool_call: agent_mod.ToolCall,
        result: agent_mod.ToolExecutionResult,
    ) -> None:
        """Record artifacts produced by tool execution in the linked task."""
        if self.task is None:
            return
        if result.status != "ok":
            return
        # Heuristic: track file-producing or shell tools
        if tool_call.name in ("write_file", "exec_shell"):
            path = tool_call.arguments.get(
                "path",
                tool_call.arguments.get("command", ""),
            )
            if path:
                self.task.artifacts.append(
                    TaskArtifact(path=str(path), action="modified"),
                )


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR  (FC-2 · M1.1.3)
# ═══════════════════════════════════════════════════════════════════════════════


class Orchestrator:
    """Top-level manager for task decomposition, agent assignment, and lifecycle.

    The orchestrator holds references to the shared ``HITLManager`` and
    ``TaskRegistry``, and provides helpers for spawning agents, running
    them to completion (including in parallel via ``asyncio``), and
    cleaning up terminal instances.
    """

    def __init__(
        self,
        *,
        default_system_prompt: str | None = None,
        default_tool_registry: agent_mod.ToolRegistry | None = None,
        hitl_manager: HITLManager | None = None,
        task_registry: TaskRegistry | None = None,
        max_parallel: int = 4,
    ):
        self._default_prompt = (
            default_system_prompt or agent_mod.default_system_prompt()
        )
        self._default_registry = (
            default_tool_registry or agent_mod.build_default_registry()
        )
        self._hitl = hitl_manager or HITLManager()
        self._tasks = task_registry or TaskRegistry()
        self._agents: dict[str, AgentInstance] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_parallel)
        self._max_parallel = max_parallel

    # -- Properties ----------------------------------------------------------

    @property
    def active_agents(self) -> dict[str, AgentInstance]:
        return dict(self._agents)

    @property
    def task_registry(self) -> TaskRegistry:
        return self._tasks

    @property
    def hitl_manager(self) -> HITLManager:
        return self._hitl

    # -- Agent lifecycle -----------------------------------------------------

    def spawn_agent(
        self,
        task: Task,
        *,
        system_prompt: str | None = None,
        tool_registry: agent_mod.ToolRegistry | None = None,
        max_depth: int = agent_mod.DEFAULT_AGENT_MAX_DEPTH,
    ) -> AgentInstance:
        """Create and register a new ``AgentInstance`` for *task*."""
        agent_id = f"agent-{uuid.uuid4().hex[:8]}"
        instance = AgentInstance(
            id=agent_id,
            system_prompt=system_prompt or self._default_prompt,
            tool_registry=tool_registry or self._default_registry,
            hitl_manager=self._hitl,
            task=task,
            max_depth=max_depth,
        )

        # Seed the conversation with the task objective.
        instance.add_user_message(task.objective)

        self._agents[agent_id] = instance
        self._tasks.assign_agent(task.id, agent_id)
        return instance

    async def run_agent(
        self,
        agent_id: str,
        complete_fn: Callable[..., str],
        ui: Any,
    ) -> str:
        """Run a single agent to completion and update its task status."""
        instance = self._agents.get(agent_id)
        if instance is None:
            raise ValueError(f"No active agent with id={agent_id!r}")

        try:
            result = await instance.run(complete_fn, ui)
            if instance.task:
                self._tasks.update_status(
                    instance.task.id,
                    TaskStatus.COMPLETED,
                    result=result,
                )
            return result
        except Exception as exc:
            instance.status = AgentStatus.FAILED
            instance.error = str(exc)
            if instance.task:
                self._tasks.update_status(
                    instance.task.id,
                    TaskStatus.FAILED,
                    error=str(exc),
                )
            raise

    async def run_parallel(
        self,
        tasks: list[Task],
        complete_fn: Callable[..., str],
        ui: Any,
    ) -> list[str]:
        """Run multiple tasks concurrently via ``asyncio.gather``."""
        agents = [self.spawn_agent(t) for t in tasks]
        results = await asyncio.gather(
            *(self.run_agent(a.id, complete_fn, ui) for a in agents),
            return_exceptions=True,
        )
        return [
            str(r) if not isinstance(r, BaseException) else f"Error: {r}"
            for r in results
        ]

    def cancel_agent(self, agent_id: str) -> bool:
        instance = self._agents.get(agent_id)
        if instance is None:
            return False
        instance.cancel()
        if instance.task:
            self._tasks.cancel_task(instance.task.id)
        return True

    def get_agent(self, agent_id: str) -> AgentInstance | None:
        return self._agents.get(agent_id)

    def cleanup_completed(self) -> int:
        """Remove terminal agents from the active registry."""
        to_remove = [
            aid for aid, inst in self._agents.items() if inst.is_terminal
        ]
        for aid in to_remove:
            del self._agents[aid]
        return len(to_remove)

    # -- Diagnostics ---------------------------------------------------------

    def status_summary(self) -> dict[str, Any]:
        agent_statuses: dict[str, int] = {}
        for inst in self._agents.values():
            key = inst.status.value
            agent_statuses[key] = agent_statuses.get(key, 0) + 1

        return {
            "active_agents": len(self._agents),
            "agent_statuses": agent_statuses,
            "task_summary": self._tasks.summary(),
        }

    def format_dashboard(self) -> str:
        """Render a plain-text status dashboard for the REPL."""
        lines: list[str] = []
        lines.append("═" * 62)
        lines.append("  ORCHESTRATOR DASHBOARD")
        lines.append("═" * 62)

        summary = self.status_summary()
        lines.append(f"  Active agents : {summary['active_agents']}")
        if summary["agent_statuses"]:
            for status, count in sorted(summary["agent_statuses"].items()):
                lines.append(f"    {status:<20} {count}")

        task_sum = summary["task_summary"]
        if task_sum:
            lines.append("")
            lines.append("  Task summary:")
            for status, count in sorted(task_sum.items()):
                lines.append(f"    {status:<20} {count}")

        lines.append("═" * 62)

        if self._agents:
            lines.append("")
            lines.append("  AGENTS:")
            lines.append(f"  {'ID':<20} {'Status':<18} {'Depth':<6} Task")
            lines.append("  " + "─" * 58)
            for aid, inst in self._agents.items():
                task_obj = (
                    inst.task.objective[:30] + "…"
                    if inst.task and len(inst.task.objective) > 30
                    else (inst.task.objective if inst.task else "—")
                )
                lines.append(
                    f"  {aid:<20} {inst.status.value:<18} {inst.depth:<6} {task_obj}"
                )

        tasks = self._tasks.tasks
        if tasks:
            lines.append("")
            lines.append("  TASKS:")
            lines.append(
                f"  {'ID':<10} {'Status':<12} {'Agent':<18} Objective"
            )
            lines.append("  " + "─" * 58)
            for t in tasks:
                lines.append(f"  {t.short_label()}")

        return "\n".join(lines)
