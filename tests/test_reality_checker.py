"""Tests for RealityChecker (Gap 5: AGI Roadmap — Grounding)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from emms.memory.reality_checker import (
    RealityChecker,
    VerificationResult,
    VerificationStatus,
)
from emms.memory.world_sensor import WorldSensor


# ---------------------------------------------------------------------------
# VerificationResult
# ---------------------------------------------------------------------------


class TestVerificationResult:
    def test_creation(self):
        r = VerificationResult(
            belief_text="test belief",
            status=VerificationStatus.CONFIRMED,
            confidence_delta=0.1,
        )
        assert r.status == VerificationStatus.CONFIRMED
        assert r.confidence_delta == 0.1

    def test_serialization(self):
        r = VerificationResult(
            belief_text="test",
            status=VerificationStatus.CONTRADICTED,
            confidence_delta=-0.2,
            evidence=["found mismatch"],
            domain="finance",
        )
        d = r.to_dict()
        r2 = VerificationResult.from_dict(d)
        assert r2.status == r.status
        assert r2.confidence_delta == r.confidence_delta
        assert r2.evidence == r.evidence


# ---------------------------------------------------------------------------
# RealityChecker — File Verification
# ---------------------------------------------------------------------------


class TestRealityCheckerFileVerification:
    def test_existing_file_confirms(self):
        ws = WorldSensor()
        rc = RealityChecker(sensor=ws)

        with tempfile.NamedTemporaryFile(suffix=".txt") as f:
            result = rc.check(f"The file {f.name} exists and is valid")
            assert result.confidence_delta > 0
            assert any("exists" in e for e in result.evidence)

    def test_missing_file_contradicts(self):
        ws = WorldSensor()
        rc = RealityChecker(sensor=ws)

        result = rc.check("The file /nonexistent/fakefile.txt has the data")
        assert result.confidence_delta < 0
        assert result.status == VerificationStatus.CONTRADICTED

    def test_no_file_reference_neutral(self):
        ws = WorldSensor()
        rc = RealityChecker(sensor=ws)

        result = rc.check("The market is bullish today")
        # No file refs = no file-based delta
        assert abs(result.confidence_delta) <= 0.5


# ---------------------------------------------------------------------------
# RealityChecker — Sensor Overlap
# ---------------------------------------------------------------------------


class TestRealityCheckerSensorOverlap:
    def test_sensor_overlap_boosts_confidence(self):
        ws = WorldSensor()
        # Create a reading that shares tokens
        ws.sense_system()  # adds OS, hostname, python version tokens
        rc = RealityChecker(sensor=ws)

        import platform
        os_name = platform.system().lower()
        result = rc.check(f"This system runs {os_name} operating system version")
        # Should find token overlap
        assert result.confidence_delta >= 0

    def test_no_sensor_no_overlap(self):
        rc = RealityChecker()  # no sensor
        result = rc.check("Some random belief about the system")
        assert result.status == VerificationStatus.UNCERTAIN


# ---------------------------------------------------------------------------
# RealityChecker — Temporal Staleness
# ---------------------------------------------------------------------------


class TestRealityCheckerStaleness:
    def test_old_timestamp_is_stale(self):
        rc = RealityChecker()
        # Reference a timestamp from ~60 days ago
        import time
        old_ts = int(time.time() - 86400 * 60)
        result = rc.check(f"The data was updated at {old_ts}")
        assert result.confidence_delta < 0
        assert any("stale" in e.lower() for e in result.evidence)

    def test_recent_timestamp_not_stale(self):
        rc = RealityChecker()
        import time
        recent_ts = int(time.time() - 3600)  # 1 hour ago
        result = rc.check(f"Last check was at {recent_ts}")
        # No staleness penalty for recent timestamp
        stale_evidence = [e for e in result.evidence if "stale" in e.lower()]
        assert len(stale_evidence) == 0


# ---------------------------------------------------------------------------
# RealityChecker — Batch & Reporting
# ---------------------------------------------------------------------------


class TestRealityCheckerBatchAndReporting:
    def test_batch_check(self):
        rc = RealityChecker()
        beliefs = [
            {"text": "The system is healthy"},
            {"text": "Memory usage is normal", "domain": "system"},
        ]
        results = rc.batch_check(beliefs)
        assert len(results) == 2

    def test_accuracy_report(self):
        rc = RealityChecker()
        rc.check("test belief 1")
        rc.check("test belief 2")
        report = rc.accuracy_report()
        assert report["total_checks"] == 2
        assert "status_counts" in report

    def test_results_history(self):
        rc = RealityChecker()
        rc.check("belief 1")
        rc.check("belief 2")
        rc.check("belief 3")
        results = rc.results(limit=2)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# RealityChecker — Persistence
# ---------------------------------------------------------------------------


class TestRealityCheckerPersistence:
    def test_save_load_roundtrip(self):
        rc = RealityChecker()
        rc.check("test belief")
        rc.check("another belief")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checker.json"
            rc.save_state(path)

            rc2 = RealityChecker()
            assert rc2.load_state(path)
            assert rc2._total_checks == 2
            assert len(rc2.results()) == 2

    def test_load_nonexistent_returns_false(self):
        rc = RealityChecker()
        assert rc.load_state("/nonexistent.json") is False

    def test_summary(self):
        rc = RealityChecker()
        rc.check("test belief")
        s = rc.summary()
        assert "RealityChecker" in s
        assert "1 checks" in s
