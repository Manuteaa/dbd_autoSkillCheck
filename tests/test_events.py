"""Tests for worker event queue helpers."""

import unittest
from queue import Queue

from dbd.inference.events import make_event_emitter, put_event


class EventHelpersTests(unittest.TestCase):
    def test_put_event_omits_unused_level(self) -> None:
        event_queue = Queue()

        put_event(event_queue, "fps", 72)

        self.assertEqual(event_queue.get_nowait(), {"type": "fps", "data": 72})

    def test_emitter_adds_log_level(self) -> None:
        event_queue = Queue()
        emit = make_event_emitter(event_queue)

        emit("log", "failed", "error")

        self.assertEqual(
            event_queue.get_nowait(),
            {"type": "log", "data": "failed", "level": "error"},
        )


if __name__ == "__main__":
    unittest.main()
