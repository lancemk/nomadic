"""
Task & Objective Management.

FC-4 from SRS-ext-v1.1: Implements persistent task tracking,
status management, and artifact collection for agent workflows.
"""

from __future__ import annotations

import enum
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


class TaskStatus(enum.Enum):
    """Lifecycle status of a task."""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskArtifact:
    """A file or resource produced or modified during task execution."""

    path: str
    action: str  # "created", "modified", "deleted"
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "action": self.action,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskArtifact:
        ts = data.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        else:
            ts = datetime.now(timezone.utc)
        return cls(path=data["path"], action=data["action"], timestamp=ts)


@dataclass
class Task:
    """A discrete unit of work with objective and status tracking."""

    id: str
    objective: str
    status: TaskStatus = TaskStatus.PENDING
    agent_id: str | None = None
    session_id: str | None = None
    parent_task_id: str | None = None
    artifacts: list[TaskArtifact] = field(default_factory=list)
    result: str | None = None
    error: str | None = None
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- Serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "objective": self.objective,
            "status": self.status.value,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "parent_task_id": self.parent_task_id,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        artifacts_raw = data.get("artifacts")
        if isinstance(artifacts_raw, str):
            try:
                artifacts_raw = json.loads(artifacts_raw)
            except (json.JSONDecodeError, TypeError):
                artifacts_raw = []
        artifacts = [
            TaskArtifact.from_dict(a) for a in (artifacts_raw or [])
        ]

        def _parse_dt(val: Any) -> datetime:
            if isinstance(val, str):
                return datetime.fromisoformat(val)
            if isinstance(val, datetime):
                return val
            return datetime.now(timezone.utc)

        metadata_raw = data.get("metadata") or data.get("metadata_json", "{}")
        if isinstance(metadata_raw, str):
            try:
                metadata_raw = json.loads(metadata_raw)
            except (json.JSONDecodeError, TypeError):
                metadata_raw = {}

        return cls(
            id=data["id"],
            objective=data.get("objective", ""),
            status=TaskStatus(data.get("status", "pending")),
            agent_id=data.get("agent_id"),
            session_id=data.get("session_id"),
            parent_task_id=data.get("parent_task_id"),
            artifacts=artifacts,
            result=data.get("result"),
            error=data.get("error"),
            created_at=_parse_dt(data.get("created_at")),
            updated_at=_parse_dt(data.get("updated_at")),
            metadata=metadata_raw if isinstance(metadata_raw, dict) else {},
        )

    # -- Short display -------------------------------------------------------

    def short_label(self, max_obj: int = 50) -> str:
        obj = self.objective
        if len(obj) > max_obj:
            obj = obj[: max_obj - 1] + "…"
        agent = self.agent_id or "—"
        return f"{self.id[:8]}  {self.status.value:<10}  {agent:<16}  {obj}"


class TaskRegistry:
    """In-memory task registry with optional database persistence.

    The registry is the authoritative store for the current session's tasks.
    When a *db_conn* is supplied, changes are written through to the database
    on a best-effort basis.
    """

    def __init__(self, db_conn: Any = None):
        self._tasks: dict[str, Task] = {}
        self._db_conn = db_conn

    @property
    def tasks(self) -> list[Task]:
        return list(self._tasks.values())

    # -- CRUD ----------------------------------------------------------------

    def create_task(
        self,
        objective: str,
        *,
        session_id: str | None = None,
        parent_task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Task:
        task = Task(
            id=str(uuid.uuid4()),
            objective=objective,
            session_id=session_id,
            parent_task_id=parent_task_id,
            metadata=metadata or {},
        )
        self._tasks[task.id] = task
        self._persist(task)
        return task

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: str | None = None,
        error: str | None = None,
    ) -> Task | None:
        task = self._tasks.get(task_id)
        if task is None:
            return None
        task.status = status
        task.updated_at = datetime.now(timezone.utc)
        if result is not None:
            task.result = result
        if error is not None:
            task.error = error
        self._persist(task)
        return task

    def assign_agent(self, task_id: str, agent_id: str) -> Task | None:
        task = self._tasks.get(task_id)
        if task is None:
            return None
        task.agent_id = agent_id
        task.status = TaskStatus.ACTIVE
        task.updated_at = datetime.now(timezone.utc)
        self._persist(task)
        return task

    def add_artifact(
        self,
        task_id: str,
        path: str,
        action: str = "created",
    ) -> Task | None:
        task = self._tasks.get(task_id)
        if task is None:
            return None
        task.artifacts.append(TaskArtifact(path=path, action=action))
        task.updated_at = datetime.now(timezone.utc)
        self._persist(task)
        return task

    def cancel_task(self, task_id: str) -> Task | None:
        return self.update_status(task_id, TaskStatus.CANCELLED)

    # -- Queries -------------------------------------------------------------

    def list_by_status(self, status: TaskStatus | None = None) -> list[Task]:
        if status is None:
            return self.tasks
        return [t for t in self._tasks.values() if t.status == status]

    def list_by_session(self, session_id: str) -> list[Task]:
        return [
            t for t in self._tasks.values() if t.session_id == session_id
        ]

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for task in self._tasks.values():
            key = task.status.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    # -- DB persistence (best-effort) ----------------------------------------

    def _persist(self, task: Task) -> None:
        if self._db_conn is None:
            return
        try:
            from .storage import db as chatdb

            chatdb.upsert_task(self._db_conn, task)
        except Exception:
            pass

    def load_from_db(self, session_id: str | None = None) -> None:
        """Reload tasks from the database into memory."""
        if self._db_conn is None:
            return
        try:
            from .storage import db as chatdb

            rows = chatdb.list_tasks(self._db_conn, session_id=session_id)
            for row in rows:
                task = Task.from_dict(row)
                self._tasks[task.id] = task
        except Exception:
            pass
