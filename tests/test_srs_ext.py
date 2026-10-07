"""Tests for SRS-ext-v1.1 modules: hitl, tasks, orchestrator.

Covers FC-1 (Agent Instance), FC-2 (Orchestrator), FC-3 (HITL),
and FC-4 (Task Management).  No mlx_vlm import; complete_fn is
mocked wherever an LLM call would occur.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# HITL imports
# ---------------------------------------------------------------------------
from nomadic.hitl import (
    DEFAULT_REVIEW_POLICIES,
    HITLManager,
    ReviewDecision,
    ReviewPolicy,
    ReviewRequest,
    ReviewResult,
    SafetyProfile,
)

# ---------------------------------------------------------------------------
# Task imports
# ---------------------------------------------------------------------------
from nomadic.tasks import Task, TaskArtifact, TaskRegistry, TaskStatus

# ---------------------------------------------------------------------------
# Orchestrator imports (agent module types used via the public surface)
# ---------------------------------------------------------------------------
from nomadic.agent import (
    AgentRuntime,
    ToolCall,
    ToolDefinition,
    ToolExecutionResult,
    ToolRegistry,
)
from nomadic.orchestrator import AgentInstance, AgentStatus, Orchestrator


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_tool_registry() -> ToolRegistry:
    """Build a minimal ToolRegistry with a no-op tool for testing."""
    noop_tool = ToolDefinition(
        name="noop",
        description="No-op",
        parameters={"type": "object", "properties": {}},
        handler=lambda args, rt: ToolExecutionResult(
            tool_name="noop",
            label="noop",
            status="ok",
            content="ok",
        ),
    )
    return ToolRegistry([noop_tool])


def _make_agent(
    *,
    hitl: HITLManager | None = None,
    task: Task | None = None,
    system_prompt: str = "test",
) -> AgentInstance:
    """Create an AgentInstance for unit tests."""
    return AgentInstance(
        id=f"agent-{uuid.uuid4().hex[:8]}",
        system_prompt=system_prompt,
        tool_registry=_make_tool_registry(),
        hitl_manager=hitl,
        task=task,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HITL TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestSafetyProfileFullyAutonomous:
    """FULLY_AUTONOMOUS profile should never require review."""

    def test_never_requires_review_for_any_tool(self):
        mgr = HITLManager(profile=SafetyProfile.FULLY_AUTONOMOUS)
        assert mgr.requires_review("exec_shell") is False
        assert mgr.requires_review("read_skill") is False
        assert mgr.requires_review("anything_else") is False

    def test_never_requires_review_even_with_policies(self):
        mgr = HITLManager(
            profile=SafetyProfile.FULLY_AUTONOMOUS,
            policies=[ReviewPolicy(tool_name=None)],  # wildcard policy
        )
        assert mgr.requires_review("exec_shell", {"command": "rm -rf /"}) is False


class TestSafetyProfileStrictlyHuman:
    """STRICTLY_HUMAN profile should always require review."""

    def test_always_requires_review(self):
        mgr = HITLManager(
            profile=SafetyProfile.STRICTLY_HUMAN,
            policies=[],  # even with no policies
        )
        assert mgr.requires_review("exec_shell") is True
        assert mgr.requires_review("read_skill") is True
        assert mgr.requires_review("unknown_tool") is True


class TestSafetyProfileSemiAutonomousWithPolicy:
    """SEMI_AUTONOMOUS profile should only trigger review for matching policies."""

    def test_matches_only_policy_tool(self):
        mgr = HITLManager(
            profile=SafetyProfile.SEMI_AUTONOMOUS,
            policies=[ReviewPolicy(tool_name="exec_shell")],
        )
        assert mgr.requires_review("exec_shell") is True
        assert mgr.requires_review("read_skill") is False

    def test_wildcard_policy_matches_all(self):
        mgr = HITLManager(
            profile=SafetyProfile.SEMI_AUTONOMOUS,
            policies=[ReviewPolicy(tool_name=None)],
        )
        assert mgr.requires_review("exec_shell") is True
        assert mgr.requires_review("read_skill") is True

    def test_no_policies_means_no_review(self):
        mgr = HITLManager(
            profile=SafetyProfile.SEMI_AUTONOMOUS,
            policies=[],
        )
        assert mgr.requires_review("exec_shell") is False


class TestReviewPolicyMatching:
    """ReviewPolicy.matches() with tool_name and argument_pattern."""

    def test_exact_tool_name_match(self):
        p = ReviewPolicy(tool_name="exec_shell")
        assert p.matches("exec_shell") is True
        assert p.matches("read_skill") is False

    def test_wildcard_tool_name(self):
        p = ReviewPolicy(tool_name=None)
        assert p.matches("exec_shell") is True
        assert p.matches("anything") is True

    def test_argument_pattern_match(self):
        p = ReviewPolicy(tool_name="exec_shell", argument_pattern="rm -rf")
        assert p.matches("exec_shell", {"command": "rm -rf /"}) is True
        assert p.matches("exec_shell", {"command": "ls -la"}) is False

    def test_argument_pattern_no_args_passes(self):
        """When arguments is None, the argument_pattern check is skipped."""
        p = ReviewPolicy(tool_name="exec_shell", argument_pattern="rm -rf")
        # arguments=None → pattern check is bypassed, tool_name still matches
        assert p.matches("exec_shell", None) is True

    def test_argument_pattern_wrong_tool(self):
        p = ReviewPolicy(tool_name="exec_shell", argument_pattern="rm")
        assert p.matches("read_skill", {"command": "rm -rf /"}) is False


class TestCreateAndResolveReview:
    """create_review_request + resolve_review round-trip."""

    def test_round_trip(self):
        mgr = HITLManager()
        req = mgr.create_review_request(
            tool_name="exec_shell",
            arguments={"command": "ls"},
            raw_json='{"tool_call":{"name":"exec_shell","arguments":{"command":"ls"}}}',
            agent_id="agent-001",
            depth=1,
        )
        assert isinstance(req, ReviewRequest)
        assert mgr.has_pending() is True
        assert "agent-001" in mgr.pending_reviews

        resolved = mgr.resolve_review("agent-001")
        assert resolved is req
        assert mgr.has_pending() is False

    def test_resolve_missing_agent_returns_none(self):
        mgr = HITLManager()
        assert mgr.resolve_review("nonexistent") is None


class TestParseReviewInput:
    """parse_review_input for approve, reject, skip, and modify."""

    @pytest.fixture()
    def dummy_request(self):
        return ReviewRequest(
            tool_name="exec_shell",
            arguments={"command": "echo hi"},
            raw_json="{}",
            agent_id="a1",
            depth=1,
        )

    @pytest.mark.parametrize("text", ["a", "A", "approve", "APPROVE", " approve "])
    def test_approve(self, dummy_request, text):
        result = HITLManager.parse_review_input(text, dummy_request)
        assert result.decision == ReviewDecision.APPROVE

    @pytest.mark.parametrize("text", ["r", "R", "reject", "REJECT"])
    def test_reject(self, dummy_request, text):
        result = HITLManager.parse_review_input(text, dummy_request)
        assert result.decision == ReviewDecision.REJECT
        assert "Rejected by operator" in result.feedback

    @pytest.mark.parametrize("text", ["s", "S", "skip", "SKIP"])
    def test_skip(self, dummy_request, text):
        result = HITLManager.parse_review_input(text, dummy_request)
        assert result.decision == ReviewDecision.SKIP

    def test_modify_valid_json(self, dummy_request):
        result = HITLManager.parse_review_input(
            'm {"command": "echo modified"}',
            dummy_request,
        )
        assert result.decision == ReviewDecision.MODIFY
        assert result.modified_arguments == {"command": "echo modified"}

    def test_modify_with_long_prefix(self, dummy_request):
        result = HITLManager.parse_review_input(
            'modify {"command": "echo modified"}',
            dummy_request,
        )
        assert result.decision == ReviewDecision.MODIFY

    def test_modify_invalid_json_falls_back_to_reject(self, dummy_request):
        result = HITLManager.parse_review_input(
            "m {not-valid-json}",
            dummy_request,
        )
        assert result.decision == ReviewDecision.REJECT
        assert "Failed to parse" in result.feedback

    def test_unknown_input_treated_as_reject_with_feedback(self, dummy_request):
        result = HITLManager.parse_review_input(
            "please don't run that",
            dummy_request,
        )
        assert result.decision == ReviewDecision.REJECT
        assert result.feedback == "please don't run that"


class TestAddRemovePolicy:
    """add_policy / remove_policy management."""

    def test_add_policy(self):
        mgr = HITLManager(policies=[])
        assert len(mgr.policies) == 0
        mgr.add_policy(ReviewPolicy(tool_name="exec_shell"))
        assert len(mgr.policies) == 1
        assert mgr.policies[0].tool_name == "exec_shell"

    def test_remove_policy_by_tool_name(self):
        mgr = HITLManager(
            policies=[
                ReviewPolicy(tool_name="exec_shell"),
                ReviewPolicy(tool_name="read_skill"),
            ]
        )
        removed = mgr.remove_policy("exec_shell")
        assert removed is True
        assert len(mgr.policies) == 1
        assert mgr.policies[0].tool_name == "read_skill"

    def test_remove_policy_nonexistent(self):
        mgr = HITLManager(policies=[])
        removed = mgr.remove_policy("nope")
        assert removed is False

    def test_clear_policies(self):
        mgr = HITLManager()  # has DEFAULT_REVIEW_POLICIES
        assert len(mgr.policies) > 0
        mgr.clear_policies()
        assert len(mgr.policies) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# TASK TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestTaskCreation:
    """Task creation sets correct defaults."""

    def test_defaults(self):
        t = Task(id="t1", objective="do stuff")
        assert t.status == TaskStatus.PENDING
        assert t.agent_id is None
        assert t.session_id is None
        assert t.parent_task_id is None
        assert t.artifacts == []
        assert t.result is None
        assert t.error is None
        assert isinstance(t.created_at, datetime)
        assert isinstance(t.updated_at, datetime)
        assert t.metadata == {}


class TestTaskStatusLifecycle:
    """PENDING → ACTIVE → COMPLETED lifecycle."""

    def test_lifecycle_transitions(self):
        reg = TaskRegistry()
        task = reg.create_task("build widget")
        assert task.status == TaskStatus.PENDING

        reg.assign_agent(task.id, "agent-42")
        task = reg.get_task(task.id)
        assert task.status == TaskStatus.ACTIVE
        assert task.agent_id == "agent-42"

        reg.update_status(task.id, TaskStatus.COMPLETED, result="done")
        task = reg.get_task(task.id)
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "done"


class TestTaskAssignAgent:
    """assign_agent sets agent_id and transitions to ACTIVE."""

    def test_assign_agent(self):
        reg = TaskRegistry()
        task = reg.create_task("test task")
        reg.assign_agent(task.id, "agent-99")
        task = reg.get_task(task.id)
        assert task.agent_id == "agent-99"
        assert task.status == TaskStatus.ACTIVE

    def test_assign_agent_nonexistent(self):
        reg = TaskRegistry()
        assert reg.assign_agent("missing-id", "agent-99") is None


class TestTaskArtifactTracking:
    """add_artifact appends a TaskArtifact to the task."""

    def test_add_artifact(self):
        reg = TaskRegistry()
        task = reg.create_task("produce files")
        reg.add_artifact(task.id, "/tmp/output.txt", action="created")
        task = reg.get_task(task.id)
        assert len(task.artifacts) == 1
        assert task.artifacts[0].path == "/tmp/output.txt"
        assert task.artifacts[0].action == "created"

    def test_add_artifact_nonexistent_task(self):
        reg = TaskRegistry()
        assert reg.add_artifact("missing", "/tmp/x.txt") is None


class TestTaskCancel:
    """cancel_task sets status to CANCELLED."""

    def test_cancel(self):
        reg = TaskRegistry()
        task = reg.create_task("cancel me")
        reg.cancel_task(task.id)
        task = reg.get_task(task.id)
        assert task.status == TaskStatus.CANCELLED

    def test_cancel_nonexistent(self):
        reg = TaskRegistry()
        assert reg.cancel_task("missing") is None


class TestTaskSerializationRoundtrip:
    """to_dict / from_dict preserves all fields."""

    def test_round_trip_basic(self):
        task = Task(
            id="t-42",
            objective="round-trip test",
            session_id="sess-1",
            parent_task_id="parent-0",
            metadata={"key": "value"},
        )
        d = task.to_dict()
        restored = Task.from_dict(d)
        assert restored.id == task.id
        assert restored.objective == task.objective
        assert restored.status == task.status
        assert restored.session_id == task.session_id
        assert restored.parent_task_id == task.parent_task_id
        assert restored.metadata == task.metadata

    def test_round_trip_with_artifacts(self):
        task = Task(id="t-43", objective="with artifact")
        task.artifacts.append(TaskArtifact(path="/foo/bar.py", action="modified"))
        d = task.to_dict()
        restored = Task.from_dict(d)
        assert len(restored.artifacts) == 1
        assert restored.artifacts[0].path == "/foo/bar.py"
        assert restored.artifacts[0].action == "modified"

    def test_round_trip_timestamps(self):
        task = Task(id="t-44", objective="ts test")
        d = task.to_dict()
        restored = Task.from_dict(d)
        # Compare to second precision (isoformat round-trip may shift micros)
        assert restored.created_at.replace(microsecond=0) == task.created_at.replace(microsecond=0)

    def test_from_dict_handles_missing_optional_fields(self):
        minimal = {"id": "t-99", "objective": "minimal"}
        restored = Task.from_dict(minimal)
        assert restored.status == TaskStatus.PENDING
        assert restored.artifacts == []
        assert restored.metadata == {}


class TestTaskRegistryQueries:
    """list_by_status, list_by_session, summary."""

    @pytest.fixture()
    def populated_registry(self):
        reg = TaskRegistry()
        t1 = reg.create_task("alpha", session_id="s1")
        t2 = reg.create_task("beta", session_id="s1")
        t3 = reg.create_task("gamma", session_id="s2")
        reg.assign_agent(t1.id, "agent-a")
        reg.update_status(t2.id, TaskStatus.COMPLETED, result="done")
        return reg

    def test_list_by_status_all(self, populated_registry):
        all_tasks = populated_registry.list_by_status()
        assert len(all_tasks) == 3

    def test_list_by_status_filtered(self, populated_registry):
        active = populated_registry.list_by_status(TaskStatus.ACTIVE)
        assert len(active) == 1
        completed = populated_registry.list_by_status(TaskStatus.COMPLETED)
        assert len(completed) == 1
        pending = populated_registry.list_by_status(TaskStatus.PENDING)
        assert len(pending) == 1

    def test_list_by_session(self, populated_registry):
        s1 = populated_registry.list_by_session("s1")
        assert len(s1) == 2
        s2 = populated_registry.list_by_session("s2")
        assert len(s2) == 1
        s3 = populated_registry.list_by_session("s3")
        assert len(s3) == 0

    def test_summary(self, populated_registry):
        summary = populated_registry.summary()
        assert summary["active"] == 1
        assert summary["completed"] == 1
        assert summary["pending"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestAgentInstanceInit:
    """AgentInstance initialisation and defaults."""

    def test_default_state(self):
        inst = _make_agent()
        assert inst.status == AgentStatus.IDLE
        assert inst.is_terminal is False
        assert inst.depth == 0
        assert inst.result is None
        assert inst.error is None
        assert inst._cancel_requested is False

    def test_history_seeded_with_system_prompt(self):
        inst = _make_agent(system_prompt="you are helpful")
        assert len(inst.history) == 1
        assert inst.history[0]["role"] == "system"
        assert inst.history[0]["content"] == "you are helpful"

    def test_add_user_message(self):
        inst = _make_agent()
        inst.add_user_message("hello")
        assert inst.history[-1] == {"role": "user", "content": "hello"}


class TestAgentInstanceCancel:
    """AgentInstance.cancel() transitions to CANCELLED."""

    def test_cancel_sets_status(self):
        inst = _make_agent()
        inst.cancel()
        assert inst.status == AgentStatus.CANCELLED
        assert inst._cancel_requested is True
        assert inst.is_terminal is True


class TestOrchestratorSpawnAgent:
    """Orchestrator.spawn_agent creates and registers an AgentInstance."""

    def test_spawn_agent(self):
        orch = Orchestrator(
            default_system_prompt="test",
            default_tool_registry=_make_tool_registry(),
        )
        task = orch.task_registry.create_task("test objective")
        inst = orch.spawn_agent(task)

        assert inst.id in orch.active_agents
        assert inst.task is task
        assert inst.status == AgentStatus.IDLE
        # spawn_agent should add a user message with the task objective
        assert any(
            m["role"] == "user" and m["content"] == "test objective"
            for m in inst.history
        )

    def test_spawn_agent_assigns_task(self):
        orch = Orchestrator(
            default_system_prompt="test",
            default_tool_registry=_make_tool_registry(),
        )
        task = orch.task_registry.create_task("job")
        inst = orch.spawn_agent(task)
        refreshed_task = orch.task_registry.get_task(task.id)
        assert refreshed_task.agent_id == inst.id
        assert refreshed_task.status == TaskStatus.ACTIVE


class TestOrchestratorCancelAgent:
    """Orchestrator.cancel_agent cancels agent and its task."""

    def test_cancel_existing_agent(self):
        orch = Orchestrator(
            default_system_prompt="test",
            default_tool_registry=_make_tool_registry(),
        )
        task = orch.task_registry.create_task("cancel me")
        inst = orch.spawn_agent(task)
        assert orch.cancel_agent(inst.id) is True
        assert inst.status == AgentStatus.CANCELLED
        assert orch.task_registry.get_task(task.id).status == TaskStatus.CANCELLED

    def test_cancel_nonexistent_agent(self):
        orch = Orchestrator(
            default_system_prompt="test",
            default_tool_registry=_make_tool_registry(),
        )
        assert orch.cancel_agent("nonexistent") is False


class TestOrchestratorCleanupCompleted:
    """cleanup_completed removes terminal agents."""

    def test_removes_terminal_agents(self):
        orch = Orchestrator(
            default_system_prompt="test",
            default_tool_registry=_make_tool_registry(),
        )
        t1 = orch.task_registry.create_task("done")
        t2 = orch.task_registry.create_task("still going")
        a1 = orch.spawn_agent(t1)
        a2 = orch.spawn_agent(t2)

        a1.status = AgentStatus.COMPLETED
        removed = orch.cleanup_completed()
        assert removed == 1
        assert a1.id not in orch.active_agents
        assert a2.id in orch.active_agents

    def test_cleanup_when_none_terminal(self):
        orch = Orchestrator(
            default_system_prompt="test",
            default_tool_registry=_make_tool_registry(),
        )
        task = orch.task_registry.create_task("active")
        orch.spawn_agent(task)
        assert orch.cleanup_completed() == 0


class TestOrchestratorStatusSummary:
    """status_summary returns correct counts."""

    def test_summary_structure(self):
        orch = Orchestrator(
            default_system_prompt="test",
            default_tool_registry=_make_tool_registry(),
        )
        t1 = orch.task_registry.create_task("alpha")
        t2 = orch.task_registry.create_task("beta")
        a1 = orch.spawn_agent(t1)
        a2 = orch.spawn_agent(t2)
        a1.status = AgentStatus.RUNNING

        summary = orch.status_summary()
        assert summary["active_agents"] == 2
        assert "running" in summary["agent_statuses"]
        assert summary["agent_statuses"]["running"] == 1
        assert "idle" in summary["agent_statuses"]
        assert summary["agent_statuses"]["idle"] == 1
        assert isinstance(summary["task_summary"], dict)


class TestOrchestratorFormatDashboard:
    """format_dashboard returns a non-empty string with expected sections."""

    def test_dashboard_output(self):
        orch = Orchestrator(
            default_system_prompt="test",
            default_tool_registry=_make_tool_registry(),
        )
        t1 = orch.task_registry.create_task("alpha objective")
        orch.spawn_agent(t1)

        dashboard = orch.format_dashboard()
        assert isinstance(dashboard, str)
        assert "ORCHESTRATOR DASHBOARD" in dashboard
        assert "AGENTS:" in dashboard
        assert "TASKS:" in dashboard
        assert "alpha objective" in dashboard

    def test_dashboard_empty_orchestrator(self):
        orch = Orchestrator(
            default_system_prompt="test",
            default_tool_registry=_make_tool_registry(),
        )
        dashboard = orch.format_dashboard()
        assert "ORCHESTRATOR DASHBOARD" in dashboard
        assert "Active agents : 0" in dashboard
