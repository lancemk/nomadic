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
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .status import StatusBus

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

    # Wiring supplied by the Orchestrator (all optional for standalone use).
    executor: Any = None
    status_bus: StatusBus | None = None
    status_callback: Callable[[agent_mod.StatusEvent], None] | None = None
    task_registry: TaskRegistry | None = None
    db_conn: Any = None

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

    # -- Runtime wiring / persistence -----------------------------------------

    def _runtime(self, ui: Any) -> agent_mod.AgentRuntime:
        runtime = agent_mod.AgentRuntime(state=None, ui=ui)
        runtime.loop_depth = self.depth
        runtime.status_callback = self.status_callback
        runtime.status_bus = self.status_bus
        return runtime

    def _emit(self, state: agent_mod.LifecycleState, message: str) -> None:
        self._runtime(None).emit(state, message)

    def to_state(self) -> dict[str, Any]:
        """Serialisable snapshot for the ``agent_states`` table."""
        call = self.current_tool_call
        return {
            "agent_id": self.id,
            "task_id": self.task.id if self.task else None,
            "session_id": self.task.session_id if self.task else None,
            "status": self.status.value,
            "depth": self.depth,
            "max_depth": self.max_depth,
            "system_prompt": self.system_prompt,
            "history": self.history,
            "current_tool_call": (
                {"name": call.name, "arguments": dict(call.arguments), "raw_json": call.raw_json}
                if call else None
            ),
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_state(
        cls,
        state: dict[str, Any],
        tool_registry: agent_mod.ToolRegistry,
        *,
        task: Task | None = None,
        hitl_manager: HITLManager | None = None,
    ) -> "AgentInstance":
        """Rebuild an instance from :meth:`to_state` output (e.g. after a restart)."""
        instance = cls(
            id=state["agent_id"],
            system_prompt=state.get("system_prompt", ""),
            tool_registry=tool_registry,
            hitl_manager=hitl_manager,
            task=task,
            history=list(state.get("history") or []),
            status=AgentStatus(state.get("status", "idle")),
            result=state.get("result"),
            error=state.get("error"),
            depth=int(state.get("depth", 0)),
            max_depth=int(state.get("max_depth", agent_mod.DEFAULT_AGENT_MAX_DEPTH)),
        )
        call = state.get("current_tool_call")
        if call:
            instance.current_tool_call = agent_mod.ToolCall(
                name=call["name"], arguments=call.get("arguments") or {}, raw_json=call.get("raw_json", ""),
            )
        return instance

    def persist(self) -> None:
        """Write the current state to the database (best-effort, like the task registry)."""
        if self.db_conn is None:
            return
        try:
            from .storage import db as chatdb

            chatdb.upsert_agent_state(self.db_conn, self.to_state())
        except Exception:
            pass

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
        self._emit(agent_mod.LifecycleState.THINKING, "Thinking...")

        # Run the LLM completion in an executor so we never block the loop.
        loop = asyncio.get_running_loop()
        reply: str = await loop.run_in_executor(
            self.executor,
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
        runtime = self._runtime(ui)

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
            self._emit(
                agent_mod.LifecycleState.REVIEWING,
                f"Awaiting user review of {tool_call.name}...",
            )
            self.hitl_manager.create_review_request(
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                raw_json=tool_call.raw_json,
                agent_id=self.id,
                depth=self.depth,
            )
            self.persist()
            # Caller must resolve the review before continuing.
            return None, False

        # ── Execute tool (blocking tools run off the event loop) ───────────
        result = await loop.run_in_executor(
            self.executor, self._execute_tool, tool_call, runtime,
        )
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
                runtime = self._runtime(ui)
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
            runtime = self._runtime(ui)
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
        """Record files created/modified by file tools in the linked task (FC-4)."""
        if self.task is None or result.status != "ok":
            return
        if tool_call.name not in ("write_file", "replace_content"):
            return
        payload = result.payload or {}
        path = payload.get("path") or tool_call.arguments.get("path")
        if not path:
            return
        action = str(payload.get("action") or "modified")
        if self.task_registry is not None:
            self.task_registry.add_artifact(self.task.id, str(path), action)
        else:
            self.task.artifacts.append(TaskArtifact(path=str(path), action=action))


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
        db_conn: Any = None,
        status_bus: StatusBus | None = None,
        status_callback: Callable[[agent_mod.StatusEvent], None] | None = None,
    ):
        self._default_prompt = (
            default_system_prompt or agent_mod.default_system_prompt()
        )
        self._default_registry = (
            default_tool_registry or agent_mod.build_default_registry()
        )
        self._hitl = hitl_manager or HITLManager()
        self._tasks = task_registry or TaskRegistry(db_conn)
        self._agents: dict[str, AgentInstance] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_parallel)
        self._process_pool: Any = None  # created lazily by run_blocking(use_process=True)
        self._max_parallel = max_parallel
        self._db_conn = db_conn
        self._status_bus = status_bus if status_bus is not None else StatusBus()
        self._status_callback = status_callback

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

    @property
    def status_bus(self) -> StatusBus:
        return self._status_bus

    def _wire(self, instance: AgentInstance) -> AgentInstance:
        instance.executor = self._executor
        instance.status_bus = self._status_bus
        instance.status_callback = self._status_callback
        instance.task_registry = self._tasks
        instance.db_conn = self._db_conn
        return instance

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

        self._wire(instance)
        self._agents[agent_id] = instance
        self._tasks.assign_agent(task.id, agent_id)
        instance.persist()
        return instance

    def restore_agents(self, session_id: str | None = None) -> list[AgentInstance]:
        """Rebuild non-terminal agents from the ``agent_states`` table after a restart."""
        if self._db_conn is None:
            return []
        from .storage import db as chatdb

        restored: list[AgentInstance] = []
        for state in chatdb.list_agent_states(self._db_conn, session_id=session_id):
            if state["agent_id"] in self._agents:
                continue
            if AgentStatus(state["status"]) in (
                AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.CANCELLED,
            ):
                continue
            task = self._tasks.get_task(state["task_id"]) if state.get("task_id") else None
            instance = AgentInstance.from_state(
                state, self._default_registry, task=task, hitl_manager=self._hitl,
            )
            self._agents[instance.id] = self._wire(instance)
            call = instance.current_tool_call
            if instance.status == AgentStatus.AWAITING_REVIEW and call is not None:
                self._hitl.create_review_request(
                    tool_name=call.name,
                    arguments=call.arguments,
                    raw_json=call.raw_json,
                    agent_id=instance.id,
                    depth=instance.depth,
                )
            restored.append(instance)
        return restored

    async def run_blocking(self, fn: Callable[..., Any], *args: Any, use_process: bool = False) -> Any:
        """Run a long local job off the event loop (thread pool, or process pool when picklable)."""
        loop = asyncio.get_running_loop()
        if use_process:
            if self._process_pool is None:
                from concurrent.futures import ProcessPoolExecutor

                self._process_pool = ProcessPoolExecutor(max_workers=self._max_parallel)
            return await loop.run_in_executor(self._process_pool, fn, *args)
        return await loop.run_in_executor(self._executor, fn, *args)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self._process_pool is not None:
            self._process_pool.shutdown(wait=False, cancel_futures=True)

    # -- Agent-to-agent hand-off (FC-2) --------------------------------------

    def handoff(
        self,
        from_agent_id: str,
        objective: str,
        *,
        evidence: str = "",
    ) -> AgentInstance:
        """Delegate a sub-task from one agent to a new child agent.

        A child task is registered under the parent's task, and the child is
        seeded with the objective plus the *evidence* the parent has gathered.
        """
        parent = self._agents.get(from_agent_id)
        if parent is None:
            raise ValueError(f"No active agent with id={from_agent_id!r}")
        child_task = self._tasks.create_task(
            objective,
            session_id=parent.task.session_id if parent.task else None,
            parent_task_id=parent.task.id if parent.task else None,
            metadata={"handoff_from": from_agent_id},
        )
        child = self.spawn_agent(child_task)
        if evidence:
            child.add_user_message(f"Evidence from {from_agent_id}:\n{evidence}")
        return child

    async def run_handoff(
        self,
        from_agent_id: str,
        objective: str,
        complete_fn: Callable[..., str],
        ui: Any,
        *,
        evidence: str = "",
    ) -> str:
        """Hand off a sub-task, run the child to completion, and return its result to the parent."""
        child = self.handoff(from_agent_id, objective, evidence=evidence)
        result = await self.run_agent(child.id, complete_fn, ui)
        parent = self._agents[from_agent_id]
        parent.history.append({
            "role": "tool",
            "content": f"[handoff {child.id}] {result}",
        })
        parent.persist()
        return result

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
            instance.persist()
            if instance.status == AgentStatus.AWAITING_REVIEW:
                return result  # paused for HITL; the task stays ACTIVE
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
            instance.persist()
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
        instance.persist()
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
