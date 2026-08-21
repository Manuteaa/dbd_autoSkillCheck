"""Thread-safe event helpers shared by inference workers and the UI."""

from collections.abc import Callable
from queue import Queue
from typing import Any

WorkerEvent = dict[str, Any]
EventQueue = Queue[WorkerEvent]
EventCallback = Callable[[str, Any, str | None], None]


def put_event(
    event_queue: EventQueue,
    event_type: str,
    data: Any = None,
    level: str | None = None,
) -> None:
    """Put one worker event on a thread-safe queue."""
    event: WorkerEvent = {"type": event_type, "data": data}
    if level is not None:
        event["level"] = level
    event_queue.put(event)


def make_event_emitter(event_queue: EventQueue) -> EventCallback:
    """Return a callback that publishes events to ``event_queue``."""

    def emit(event_type: str, data: Any = None, level: str | None = None) -> None:
        put_event(event_queue, event_type, data, level)

    return emit
