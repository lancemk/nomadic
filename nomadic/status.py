"""Status event bus (FC-17).

``StatusEvent`` producers (tools, HITL, quantizer) call ``bus.publish``; the UI
drains the bus without blocking.  The bus is thread-safe and can be used from
asyncio tasks, executor threads, and the REPL thread alike.
"""

from __future__ import annotations

import queue
import threading
from typing import Callable

from .agent import LifecycleState, StatusEvent

__all__ = ["LifecycleState", "StatusBus", "StatusEvent"]


class StatusBus:
    """Thread-safe FIFO of :class:`StatusEvent` with optional subscribers."""

    def __init__(self, maxsize: int = 1000) -> None:
        self._queue: queue.Queue[StatusEvent] = queue.Queue(maxsize=maxsize)
        self._subscribers: list[Callable[[StatusEvent], None]] = []
        self._lock = threading.Lock()

    def subscribe(self, callback: Callable[[StatusEvent], None]) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def publish(self, event: StatusEvent) -> None:
        """Enqueue *event* (dropping the oldest when full) and notify subscribers."""
        while True:
            try:
                self._queue.put_nowait(event)
                break
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
        with self._lock:
            subscribers = list(self._subscribers)
        for callback in subscribers:
            try:
                callback(event)
            except Exception:
                pass

    def emit(self, state: LifecycleState, message: str, depth: int = 0) -> StatusEvent:
        event = StatusEvent(state, message, depth)
        self.publish(event)
        return event

    def poll(self) -> StatusEvent | None:
        """Return the next pending event, or ``None`` without blocking."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def drain(self) -> list[StatusEvent]:
        events: list[StatusEvent] = []
        while (event := self.poll()) is not None:
            events.append(event)
        return events
