"""Tests for the EventBus pub/sub system."""

import pytest
from emms.core.events import EventBus


@pytest.fixture
def bus():
    return EventBus()


class TestEmit:
    def test_emit_no_listeners(self, bus):
        count = bus.emit("test.event", {"key": "value"})
        assert count == 0

    def test_emit_with_listener(self, bus):
        received = []
        bus.on("test.event", lambda data: received.append(data))
        bus.emit("test.event", {"key": "value"})
        assert len(received) == 1
        assert received[0]["key"] == "value"

    def test_emit_multiple_listeners(self, bus):
        counts = {"a": 0, "b": 0}
        bus.on("test.event", lambda _: counts.__setitem__("a", counts["a"] + 1))
        bus.on("test.event", lambda _: counts.__setitem__("b", counts["b"] + 1))
        count = bus.emit("test.event")
        assert count == 2
        assert counts["a"] == 1
        assert counts["b"] == 1

    def test_emit_returns_listener_count(self, bus):
        bus.on("x", lambda _: None)
        bus.on("x", lambda _: None)
        bus.on("x", lambda _: None)
        assert bus.emit("x") == 3

    def test_emit_different_events_isolated(self, bus):
        a_received = []
        b_received = []
        bus.on("a", lambda data: a_received.append(data))
        bus.on("b", lambda data: b_received.append(data))
        bus.emit("a", {"from": "a"})
        assert len(a_received) == 1
        assert len(b_received) == 0


class TestOn:
    def test_on_registers_callback(self, bus):
        bus.on("event", lambda _: None)
        assert bus.listener_counts["event"] == 1

    def test_on_multiple_events(self, bus):
        bus.on("event1", lambda _: None)
        bus.on("event2", lambda _: None)
        assert "event1" in bus.listener_counts
        assert "event2" in bus.listener_counts


class TestOff:
    def test_off_removes_callback(self, bus):
        cb = lambda _: None
        bus.on("event", cb)
        bus.off("event", cb)
        assert bus.listener_counts.get("event", 0) == 0

    def test_off_nonexistent_callback(self, bus):
        bus.off("event", lambda _: None)  # should not raise


class TestOnce:
    def test_once_fires_only_once(self, bus):
        count = [0]
        bus.once("event", lambda _: count.__setitem__(0, count[0] + 1))
        bus.emit("event")
        bus.emit("event")
        bus.emit("event")
        assert count[0] == 1


class TestHistory:
    def test_history_records_events(self, bus):
        bus.emit("a", {"x": 1})
        bus.emit("b", {"y": 2})
        assert len(bus.history) == 2
        assert bus.history[0][0] == "a"
        assert bus.history[1][0] == "b"

    def test_history_bounded(self, bus):
        bus._max_history = 5
        for i in range(10):
            bus.emit("event", {"i": i})
        assert len(bus.history) == 5

    def test_listener_counts(self, bus):
        bus.on("a", lambda _: None)
        bus.on("a", lambda _: None)
        bus.on("b", lambda _: None)
        counts = bus.listener_counts
        assert counts["a"] == 2
        assert counts["b"] == 1


class TestClear:
    def test_clear_removes_everything(self, bus):
        bus.on("event", lambda _: None)
        bus.emit("event")
        bus.clear()
        assert bus.listener_counts == {}
        assert bus.history == []


class TestErrorHandling:
    def test_listener_error_does_not_crash(self, bus):
        """Exceptions in listeners should be caught."""
        def bad_listener(data):
            raise ValueError("oops")

        ok_received = []
        bus.on("event", bad_listener)
        bus.on("event", lambda data: ok_received.append(data))
        bus.emit("event", {"safe": True})
        # The second listener should still fire
        assert len(ok_received) == 1
