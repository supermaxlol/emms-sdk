"""Tests for WorldSensor (Gap 5: AGI Roadmap — Grounding)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from emms.memory.world_sensor import SensorType, WorldReading, WorldSensor


# ---------------------------------------------------------------------------
# WorldReading
# ---------------------------------------------------------------------------


class TestWorldReading:
    def test_creation(self):
        r = WorldReading(sensor_type=SensorType.TEMPORAL, key="time", value={"unix": 123})
        assert r.sensor_type == SensorType.TEMPORAL
        assert r.key == "time"
        assert r.confidence == 1.0

    def test_serialization(self):
        r = WorldReading(sensor_type=SensorType.SYSTEM, key="sys", value={"os": "Linux"})
        d = r.to_dict()
        r2 = WorldReading.from_dict(d)
        assert r2.sensor_type == r.sensor_type
        assert r2.key == r.key
        assert r2.value == r.value


# ---------------------------------------------------------------------------
# WorldSensor — Built-in Sensors
# ---------------------------------------------------------------------------


class TestWorldSensorBuiltins:
    def test_sense_time(self):
        ws = WorldSensor()
        r = ws.sense_time()
        assert r.sensor_type == SensorType.TEMPORAL
        assert "unix" in r.value
        assert "iso" in r.value
        assert "weekday" in r.value
        assert ws.total_reads == 1

    def test_sense_filesystem_exists(self):
        ws = WorldSensor()
        with tempfile.NamedTemporaryFile() as f:
            r = ws.sense_filesystem(f.name)
            assert r.value["exists"] is True
            assert r.value["is_file"] is True

    def test_sense_filesystem_not_exists(self):
        ws = WorldSensor()
        r = ws.sense_filesystem("/nonexistent/path/xyz")
        assert r.value["exists"] is False

    def test_sense_system(self):
        ws = WorldSensor()
        r = ws.sense_system()
        assert r.sensor_type == SensorType.SYSTEM
        assert "os" in r.value
        assert "hostname" in r.value
        assert "python" in r.value

    def test_sense_command(self):
        ws = WorldSensor()
        r = ws.sense_command("echo hello")
        assert r.sensor_type == SensorType.COMMAND
        assert r.value["returncode"] == 0
        assert "hello" in r.value["stdout"]

    def test_sense_command_timeout(self):
        ws = WorldSensor()
        r = ws.sense_command("sleep 10", timeout=0.1)
        assert r.value["returncode"] == -1
        assert "timeout" in r.value["stderr"]


# ---------------------------------------------------------------------------
# WorldSensor — Custom Sensors
# ---------------------------------------------------------------------------


class TestCustomSensors:
    def test_register_and_invoke(self):
        ws = WorldSensor()
        ws.register_sensor("battery", lambda: {"level": 85, "charging": True})
        r = ws.sense_custom("battery")
        assert r is not None
        assert r.value["level"] == 85
        assert r.sensor_type == SensorType.CUSTOM

    def test_sense_custom_missing(self):
        ws = WorldSensor()
        assert ws.sense_custom("nonexistent") is None

    def test_custom_sensor_error(self):
        ws = WorldSensor()
        ws.register_sensor("broken", lambda: 1 / 0)
        r = ws.sense_custom("broken")
        assert r is not None
        assert "error" in r.value


# ---------------------------------------------------------------------------
# WorldSensor — Scan & History
# ---------------------------------------------------------------------------


class TestWorldSensorScanHistory:
    def test_scan_returns_readings(self):
        ws = WorldSensor()
        readings = ws.scan()
        assert len(readings) >= 2  # time + system at minimum
        types = {r.sensor_type for r in readings}
        assert SensorType.TEMPORAL in types
        assert SensorType.SYSTEM in types

    def test_scan_includes_custom(self):
        ws = WorldSensor()
        ws.register_sensor("test", lambda: {"ok": True})
        readings = ws.scan()
        assert any(r.sensor_type == SensorType.CUSTOM for r in readings)

    def test_history(self):
        ws = WorldSensor()
        ws.sense_time()
        ws.sense_system()
        ws.sense_time()
        hist = ws.history()
        assert len(hist) == 3
        # Filter by key
        time_hist = ws.history(key="current_time")
        assert len(time_hist) == 2

    def test_latest(self):
        ws = WorldSensor()
        ws.sense_time()
        ws.sense_system()
        ws.sense_time()
        latest = ws.latest(key="current_time")
        assert latest is not None
        assert latest.sensor_type == SensorType.TEMPORAL

    def test_latest_by_type(self):
        ws = WorldSensor()
        ws.sense_time()
        ws.sense_system()
        latest = ws.latest(sensor_type=SensorType.SYSTEM)
        assert latest is not None
        assert latest.key == "system_info"

    def test_history_capacity(self):
        ws = WorldSensor(max_history=5)
        for _ in range(10):
            ws.sense_time()
        assert len(ws.history(limit=100)) <= 5
        assert ws.total_reads == 10


# ---------------------------------------------------------------------------
# WorldSensor — Persistence
# ---------------------------------------------------------------------------


class TestWorldSensorPersistence:
    def test_save_load_roundtrip(self):
        ws = WorldSensor()
        ws.sense_time()
        ws.sense_system()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sensor.json"
            ws.save_state(path)

            ws2 = WorldSensor()
            assert ws2.load_state(path)
            assert ws2.total_reads == 2
            assert len(ws2.history()) == 2

    def test_load_nonexistent_returns_false(self):
        ws = WorldSensor()
        assert ws.load_state("/nonexistent/path.json") is False

    def test_summary(self):
        ws = WorldSensor()
        ws.sense_time()
        ws.sense_system()
        s = ws.summary()
        assert "WorldSensor" in s
        assert "2 total reads" in s
