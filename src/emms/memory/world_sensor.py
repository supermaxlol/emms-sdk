"""Gap 5 — WorldSensor: API-based perception of the real world.

Provides structured readings from the environment (filesystem, time,
system state, network reachability) without any external dependencies.
Each reading is timestamped and typed so downstream modules can verify
beliefs against reality.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class SensorType(str, Enum):
    """Categories of world perception."""
    FILESYSTEM = "filesystem"
    TEMPORAL = "temporal"
    SYSTEM = "system"
    NETWORK = "network"
    COMMAND = "command"
    CUSTOM = "custom"


@dataclass
class WorldReading:
    """A single observation from the environment."""
    sensor_type: SensorType
    key: str
    value: Any
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0  # 1.0 = direct observation, lower for inferred
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "sensor_type": self.sensor_type.value,
            "key": self.key,
            "value": self.value,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> WorldReading:
        return cls(
            sensor_type=SensorType(d["sensor_type"]),
            key=d["key"],
            value=d["value"],
            timestamp=d.get("timestamp", time.time()),
            confidence=d.get("confidence", 1.0),
            metadata=d.get("metadata", {}),
        )


class WorldSensor:
    """Collects structured observations from the real world.

    All sensing is local / passive — no external API calls, no internet
    dependencies.  Sensors are intentionally simple so the module works
    on any POSIX or macOS host.
    """

    def __init__(self, *, max_history: int = 500) -> None:
        self._history: list[WorldReading] = []
        self._max_history = max_history
        self._custom_sensors: dict[str, Any] = {}  # name → callable
        self._total_reads = 0

    # -- built-in sensors ---------------------------------------------------

    def sense_time(self) -> WorldReading:
        """Current wall-clock time, timezone, and uptime."""
        import datetime as _dt
        now = time.time()
        dt = _dt.datetime.now()
        reading = WorldReading(
            sensor_type=SensorType.TEMPORAL,
            key="current_time",
            value={
                "unix": now,
                "iso": dt.isoformat(),
                "weekday": dt.strftime("%A"),
                "hour": dt.hour,
                "tz": str(_dt.datetime.now(_dt.timezone.utc).astimezone().tzinfo),
            },
            timestamp=now,
        )
        self._record(reading)
        return reading

    def sense_filesystem(self, path: str) -> WorldReading:
        """Stat a file or directory — exists, size, mtime, type."""
        p = Path(path)
        now = time.time()
        if p.exists():
            stat = p.stat()
            value: dict[str, Any] = {
                "exists": True,
                "is_file": p.is_file(),
                "is_dir": p.is_dir(),
                "size_bytes": stat.st_size,
                "mtime": stat.st_mtime,
                "age_seconds": now - stat.st_mtime,
            }
        else:
            value = {"exists": False}
        reading = WorldReading(
            sensor_type=SensorType.FILESYSTEM,
            key=f"fs:{path}",
            value=value,
            timestamp=now,
        )
        self._record(reading)
        return reading

    def sense_system(self) -> WorldReading:
        """Basic system info: OS, hostname, Python version, PID."""
        now = time.time()
        reading = WorldReading(
            sensor_type=SensorType.SYSTEM,
            key="system_info",
            value={
                "os": platform.system(),
                "hostname": socket.gethostname(),
                "python": platform.python_version(),
                "pid": os.getpid(),
                "cwd": os.getcwd(),
            },
            timestamp=now,
        )
        self._record(reading)
        return reading

    def sense_network(self, host: str = "8.8.8.8", port: int = 53,
                      timeout: float = 2.0) -> WorldReading:
        """Check if a TCP connection to host:port succeeds (basic reachability)."""
        now = time.time()
        reachable = False
        latency_ms = -1.0
        try:
            t0 = time.time()
            s = socket.create_connection((host, port), timeout=timeout)
            s.close()
            reachable = True
            latency_ms = round((time.time() - t0) * 1000, 2)
        except (OSError, socket.timeout):
            pass
        reading = WorldReading(
            sensor_type=SensorType.NETWORK,
            key=f"net:{host}:{port}",
            value={"reachable": reachable, "latency_ms": latency_ms},
            timestamp=now,
        )
        self._record(reading)
        return reading

    def sense_command(self, cmd: str, *, timeout: float = 5.0,
                      shell: bool = True) -> WorldReading:
        """Run a shell command and capture stdout (truncated to 4 KB).

        Only meant for lightweight inspection (``uptime``, ``df -h``,
        ``git status``, etc.).  Not for long-running or interactive commands.
        """
        now = time.time()
        try:
            result = subprocess.run(
                cmd, shell=shell, capture_output=True, text=True,
                timeout=timeout,
            )
            value: dict[str, Any] = {
                "returncode": result.returncode,
                "stdout": result.stdout[:4096],
                "stderr": result.stderr[:1024],
            }
        except subprocess.TimeoutExpired:
            value = {"returncode": -1, "stdout": "", "stderr": "timeout"}
        except Exception as exc:
            value = {"returncode": -1, "stdout": "", "stderr": str(exc)[:256]}
        reading = WorldReading(
            sensor_type=SensorType.COMMAND,
            key=f"cmd:{cmd[:80]}",
            value=value,
            timestamp=now,
        )
        self._record(reading)
        return reading

    # -- custom sensor registration -----------------------------------------

    def register_sensor(self, name: str, fn: Any) -> None:
        """Register a callable as a named custom sensor.

        The callable must return a dict (the reading value).
        """
        self._custom_sensors[name] = fn

    def sense_custom(self, name: str) -> WorldReading | None:
        """Invoke a custom sensor by name."""
        fn = self._custom_sensors.get(name)
        if fn is None:
            return None
        now = time.time()
        try:
            value = fn()
        except Exception as exc:
            value = {"error": str(exc)[:256]}
        reading = WorldReading(
            sensor_type=SensorType.CUSTOM,
            key=f"custom:{name}",
            value=value,
            timestamp=now,
        )
        self._record(reading)
        return reading

    # -- scan (all built-in sensors at once) --------------------------------

    def scan(self) -> list[WorldReading]:
        """Run all built-in sensors and return the combined readings."""
        readings = [
            self.sense_time(),
            self.sense_system(),
        ]
        # Custom sensors
        for name in self._custom_sensors:
            r = self.sense_custom(name)
            if r:
                readings.append(r)
        return readings

    # -- history & retrieval ------------------------------------------------

    def latest(self, key: str | None = None,
               sensor_type: SensorType | None = None) -> WorldReading | None:
        """Return the most recent reading matching the filter."""
        for r in reversed(self._history):
            if key and r.key != key:
                continue
            if sensor_type and r.sensor_type != sensor_type:
                continue
            return r
        return None

    def history(self, key: str | None = None, limit: int = 20) -> list[WorldReading]:
        """Return recent readings, optionally filtered by key."""
        out = []
        for r in reversed(self._history):
            if key and r.key != key:
                continue
            out.append(r)
            if len(out) >= limit:
                break
        return list(reversed(out))

    @property
    def total_reads(self) -> int:
        return self._total_reads

    # -- persistence --------------------------------------------------------

    def save_state(self, path: str | Path) -> None:
        """Save recent reading history to JSON."""
        data = {
            "version": "0.28.0",
            "total_reads": self._total_reads,
            "history": [r.to_dict() for r in self._history[-self._max_history:]],
        }
        Path(path).write_text(json.dumps(data, default=str), encoding="utf-8")

    def load_state(self, path: str | Path) -> bool:
        """Load reading history from JSON. Returns True on success."""
        p = Path(path)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self._total_reads = data.get("total_reads", 0)
            self._history = [WorldReading.from_dict(r) for r in data.get("history", [])]
            return True
        except Exception:
            return False

    def summary(self) -> str:
        """Human-readable summary."""
        types = {}
        for r in self._history:
            types[r.sensor_type.value] = types.get(r.sensor_type.value, 0) + 1
        parts = [f"WorldSensor: {self._total_reads} total reads, "
                 f"{len(self._history)} in history"]
        if types:
            parts.append("  by type: " + ", ".join(f"{k}={v}" for k, v in sorted(types.items())))
        if self._custom_sensors:
            parts.append(f"  custom sensors: {', '.join(self._custom_sensors.keys())}")
        return "\n".join(parts)

    # -- internal -----------------------------------------------------------

    def _record(self, reading: WorldReading) -> None:
        self._history.append(reading)
        self._total_reads += 1
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
