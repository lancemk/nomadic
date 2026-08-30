"""
Human-In-The-Loop (HITL) Manager and Safety Profiles.

FC-3 from SRS-ext-v1.1: Implements action interception, review UI,
and configurable safety profiles for agent tool execution.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from typing import Any, Optional


class SafetyProfile(enum.Enum):
    """Configurable safety levels for agent execution.

    - STRICTLY_HUMAN: every tool call requires explicit human approval.
    - SEMI_AUTONOMOUS: only tool calls matching a ReviewPolicy require review.
    - FULLY_AUTONOMOUS: no review required; all tool calls execute immediately.
    """

    STRICTLY_HUMAN = "strictly_human"
    SEMI_AUTONOMOUS = "semi_autonomous"
    FULLY_AUTONOMOUS = "fully_autonomous"


class ReviewDecision(enum.Enum):
    """Possible outcomes of a HITL review."""

    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    SKIP = "skip"


@dataclass
class ReviewPolicy:
    """Defines which tool calls require human review.

    When *tool_name* is ``None`` the policy matches **all** tool names.
    An optional *argument_pattern* (plain substring match for now) can
    further narrow the scope.
    """

    tool_name: str | None = None
    argument_pattern: str | None = None
    description: str = ""

    def matches(self, tool_name: str, arguments: dict[str, Any] | None = None) -> bool:
        if self.tool_name is not None and self.tool_name != tool_name:
            return False
        if self.argument_pattern is not None and arguments is not None:
            args_str = json.dumps(arguments)
            if self.argument_pattern not in args_str:
                return False
        return True


@dataclass
class ReviewRequest:
    """A pending review request created when a tool call is intercepted."""

    tool_name: str
    arguments: dict[str, Any]
    raw_json: str
    agent_id: str
    depth: int


@dataclass
class ReviewResult:
    """The result of a HITL review decision."""

    decision: ReviewDecision
    modified_arguments: dict[str, Any] | None = None
    feedback: str = ""


# ---------------------------------------------------------------------------
# Default policies applied in SEMI_AUTONOMOUS mode
# ---------------------------------------------------------------------------

DEFAULT_REVIEW_POLICIES: list[ReviewPolicy] = [
    ReviewPolicy(
        tool_name="exec_shell",
        description="Shell execution requires human review",
    ),
]


class HITLManager:
    """Manages human-in-the-loop review for agent tool calls.

    The manager holds a *safety profile* and a list of *review policies*.
    It exposes helpers for checking whether a tool call needs review,
    creating / resolving review requests, and formatting a review prompt
    suitable for the interactive REPL.
    """

    def __init__(
        self,
        profile: SafetyProfile = SafetyProfile.SEMI_AUTONOMOUS,
        policies: list[ReviewPolicy] | None = None,
    ):
        self._profile = profile
        self._policies: list[ReviewPolicy] = (
            policies if policies is not None else list(DEFAULT_REVIEW_POLICIES)
        )
        self._pending_reviews: dict[str, ReviewRequest] = {}

    # -- Properties ----------------------------------------------------------

    @property
    def profile(self) -> SafetyProfile:
        return self._profile

    @profile.setter
    def profile(self, value: SafetyProfile) -> None:
        self._profile = value

    @property
    def policies(self) -> list[ReviewPolicy]:
        return list(self._policies)

    # -- Policy management ---------------------------------------------------

    def add_policy(self, policy: ReviewPolicy) -> None:
        self._policies.append(policy)

    def remove_policy(self, tool_name: str) -> bool:
        before = len(self._policies)
        self._policies = [p for p in self._policies if p.tool_name != tool_name]
        return len(self._policies) < before

    def clear_policies(self) -> None:
        self._policies.clear()

    # -- Review logic --------------------------------------------------------

    def requires_review(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> bool:
        """Return ``True`` if the tool call requires human review."""
        if self._profile == SafetyProfile.FULLY_AUTONOMOUS:
            return False
        if self._profile == SafetyProfile.STRICTLY_HUMAN:
            return True
        # SEMI_AUTONOMOUS — check policies
        return any(p.matches(tool_name, arguments) for p in self._policies)

    def create_review_request(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        raw_json: str,
        agent_id: str,
        depth: int,
    ) -> ReviewRequest:
        """Create and register a pending review request."""
        request = ReviewRequest(
            tool_name=tool_name,
            arguments=arguments,
            raw_json=raw_json,
            agent_id=agent_id,
            depth=depth,
        )
        self._pending_reviews[agent_id] = request
        return request

    def resolve_review(self, agent_id: str) -> ReviewRequest | None:
        """Remove and return a pending review request for *agent_id*."""
        return self._pending_reviews.pop(agent_id, None)

    @property
    def pending_reviews(self) -> dict[str, ReviewRequest]:
        return dict(self._pending_reviews)

    def has_pending(self) -> bool:
        return bool(self._pending_reviews)

    # -- REPL display helpers ------------------------------------------------

    def format_review_prompt(self, request: ReviewRequest) -> str:
        """Format a review request for display in the terminal REPL."""
        args_pretty = json.dumps(request.arguments, indent=2)
        lines = [
            "═" * 62,
            "⚠️  HITL REVIEW REQUIRED",
            f"  Agent   : {request.agent_id}",
            f"  Tool    : {request.tool_name}",
            f"  Depth   : {request.depth}",
            f"  Args    :",
            *(f"    {l}" for l in args_pretty.splitlines()),
            "═" * 62,
            "  [A]pprove   [R]eject   [M]odify <json>   [S]kip",
        ]
        return "\n".join(lines)

    @staticmethod
    def parse_review_input(
        user_input: str,
        request: ReviewRequest,
    ) -> ReviewResult:
        """Parse raw user input into a ``ReviewResult``."""
        choice = user_input.strip().lower()

        if choice in ("a", "approve"):
            return ReviewResult(decision=ReviewDecision.APPROVE)

        if choice in ("r", "reject"):
            return ReviewResult(
                decision=ReviewDecision.REJECT,
                feedback="Rejected by operator.",
            )

        if choice in ("s", "skip"):
            return ReviewResult(decision=ReviewDecision.SKIP)

        if choice.startswith(("m ", "modify ")):
            try:
                payload = choice.split(maxsplit=1)[1]
                modified = json.loads(payload)
                return ReviewResult(
                    decision=ReviewDecision.MODIFY,
                    modified_arguments=modified,
                )
            except (json.JSONDecodeError, IndexError):
                return ReviewResult(
                    decision=ReviewDecision.REJECT,
                    feedback="Failed to parse modified arguments — original rejected.",
                )

        # Anything else is treated as operator feedback / soft rejection
        return ReviewResult(
            decision=ReviewDecision.REJECT,
            feedback=user_input.strip(),
        )
