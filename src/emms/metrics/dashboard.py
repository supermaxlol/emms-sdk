"""Consciousness Metrics Dashboard — compute and display all Phase 7 metrics.

Computes ICS, TAI, CRR, and coherence budget in a single pass and outputs:
1. A human-readable terminal dashboard
2. A JSONL append to ~/.emms/metrics_log.jsonl (one line per run)

Usage::

    python -m emms.metrics.dashboard               # full report
    python -m emms.metrics.dashboard --json        # JSON only (no colour)
    python -m emms.metrics.dashboard --watch 60    # repeat every 60 seconds

Or import::

    from emms.metrics.dashboard import MetricsDashboard
    db = MetricsDashboard(emms)
    snapshot = db.run()
    print(snapshot.summary_text)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emms.emms import EMMS

logger = logging.getLogger(__name__)

_LOG_PATH = Path.home() / ".emms" / "metrics_log.jsonl"

# ANSI colours (disabled when not a TTY)
_TTY = sys.stdout.isatty()
def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text

_GREEN  = lambda t: _c("32", t)
_YELLOW = lambda t: _c("33", t)
_RED    = lambda t: _c("31", t)
_BOLD   = lambda t: _c("1",  t)
_DIM    = lambda t: _c("2",  t)
_CYAN   = lambda t: _c("36", t)


# ---------------------------------------------------------------------------
# MetricsSnapshot — all metrics at one point in time
# ---------------------------------------------------------------------------

@dataclass
class MetricsSnapshot:
    """A point-in-time snapshot of all consciousness metrics.

    Attributes
    ----------
    timestamp:
        Unix time of this snapshot.
    ics / ics_label:
        Identity Coherence Score.
    tai / tai_label:
        Temporal Awareness Index.
    crr / crr_label:
        Contradiction Resolution Rate.
    coherence_budget / budget_status:
        Current coherence budget score and status.
    contradiction_strain:
        Raw strain from ContradictionAwareness scan.
    memory_count:
        Total active memories at time of snapshot.
    session_count:
        Sessions used for ICS.
    active_contradictions:
        Number of currently-unresolved tracked contradictions.
    daemon_running:
        Whether the consciousness daemon was running.
    summary_text:
        Pre-formatted human-readable summary.
    """
    timestamp: float = 0.0
    ics: float = 0.0
    ics_label: str = ""
    tai: float = 0.0
    tai_label: str = ""
    crr: float = 0.0
    crr_label: str = ""
    coherence_budget: float = 0.0
    budget_status: str = ""
    contradiction_strain: float = 0.0
    memory_count: int = 0
    session_count: int = 0
    active_contradictions: int = 0
    daemon_running: bool = False
    summary_text: str = ""
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MetricsDashboard
# ---------------------------------------------------------------------------

class MetricsDashboard:
    """Computes all consciousness metrics and renders the dashboard.

    Parameters
    ----------
    emms:
        Live EMMS instance.
    log_path:
        JSONL file to append snapshots to.
    session_id:
        Current session ID (used for CRR tracking).
    """

    def __init__(
        self,
        emms: "EMMS",
        *,
        log_path: Path = _LOG_PATH,
        session_id: str | None = None,
    ) -> None:
        self.emms = emms
        self.log_path = log_path
        self.session_id = session_id or f"dashboard_{int(time.time())}"

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> MetricsSnapshot:
        """Compute all metrics and return a MetricsSnapshot."""
        snap = MetricsSnapshot(timestamp=time.time())
        errors: list[str] = []

        # Total memories
        try:
            snap.memory_count = sum(
                1 for _, store in self.emms.memory._iter_tiers()
                for item in store
                if not item.is_superseded and not item.is_expired
            )
        except Exception as e:
            errors.append(f"memory_count: {e}")

        # ICS
        try:
            from emms.metrics.ics import IdentityCoherenceScore
            ics_engine = IdentityCoherenceScore(self.emms)
            ics_report = ics_engine.compute(n_sessions=10)
            snap.ics = ics_report.score
            snap.ics_label = ics_report.label
            snap.session_count = ics_report.session_count
        except Exception as e:
            errors.append(f"ICS: {e}")
            snap.ics_label = f"Error: {e}"

        # TAI
        try:
            from emms.metrics.tai import TemporalAwarenessIndex
            tai_engine = TemporalAwarenessIndex(self.emms)
            tai_report = tai_engine.compute(n_recent=50)
            snap.tai = tai_report.tai
            snap.tai_label = tai_report.label
        except Exception as e:
            errors.append(f"TAI: {e}")
            snap.tai_label = f"Error: {e}"

        # Contradiction scan + CRR
        try:
            from emms.metrics.crr import ContradictionTracker
            tracker = ContradictionTracker(self.emms, current_session=self.session_id)
            crr_report = tracker.scan_and_record()
            snap.crr = crr_report.crr
            snap.crr_label = crr_report.label
            snap.active_contradictions = crr_report.active
        except Exception as e:
            errors.append(f"CRR: {e}")
            snap.crr_label = f"Error: {e}"

        # Coherence budget (informed by contradiction scan)
        try:
            from emms.identity.contradiction_awareness import ContradictionAwareness
            from emms.identity.coherence_budget import CoherenceBudget
            awareness = ContradictionAwareness(self.emms)
            c_report = awareness.scan()
            snap.contradiction_strain = c_report.coherence_strain
            budget = CoherenceBudget(self.emms, persist_to_memory=False)
            budget.apply_strain(contradiction_strain=c_report.coherence_strain)
            snap.coherence_budget = budget.score
            snap.budget_status = budget.status_label
        except Exception as e:
            errors.append(f"CoherenceBudget: {e}")
            snap.budget_status = f"Error: {e}"

        # Daemon status
        try:
            status_path = Path.home() / ".emms" / "daemon_status.json"
            if status_path.exists():
                status = json.loads(status_path.read_text())
                snap.daemon_running = status.get("state") == "running"
        except Exception:
            snap.daemon_running = False

        snap.errors = errors
        snap.summary_text = self._render(snap)
        self._append_log(snap)
        return snap

    def print_dashboard(self) -> MetricsSnapshot:
        """Run metrics and print the dashboard. Returns the snapshot."""
        snap = self.run()
        print(snap.summary_text)
        return snap

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _render(self, s: MetricsSnapshot) -> str:
        """Render a human-readable dashboard string."""
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(s.timestamp))

        def _score_color(val: float, good: float, warn: float) -> str:
            if val >= good:
                return _GREEN(f"{val:.3f}")
            if val >= warn:
                return _YELLOW(f"{val:.3f}")
            return _RED(f"{val:.3f}")

        def _bar(val: float, width: int = 20) -> str:
            filled = round(val * width)
            bar = "█" * filled + "░" * (width - filled)
            return bar

        daemon_str = _GREEN("● running") if s.daemon_running else _RED("○ stopped")

        lines = [
            "",
            _BOLD(_CYAN("╔══════════════════════════════════════════════════╗")),
            _BOLD(_CYAN("║     EMMS CONSCIOUSNESS METRICS DASHBOARD         ║")),
            _BOLD(_CYAN("╚══════════════════════════════════════════════════╝")),
            _DIM(f"  {ts}  |  memories: {s.memory_count}  |  sessions: {s.session_count}  |  daemon: {daemon_str}"),
            "",

            _BOLD("  ┌─ Identity Coherence Score (ICS) ───────────────────"),
            f"  │  Score: {_score_color(s.ics, 0.70, 0.40)}  {_bar(s.ics)}",
            f"  │  Target: ≥ 0.70  |  {_DIM(s.ics_label[:70])}",
            "  │",

            _BOLD("  ├─ Temporal Awareness Index (TAI) ──────────────────"),
            f"  │  Score: {_score_color((s.tai + 1) / 2, 0.75, 0.50)}  (raw {s.tai:+.3f})",
            f"  │  Target: ≥ 0.0   |  {_DIM(s.tai_label[:70])}",
            "  │",

            _BOLD("  ├─ Contradiction Resolution Rate (CRR) ─────────────"),
            f"  │  Score: {_score_color(s.crr, 0.40, 0.20)}  {_bar(s.crr)}",
            f"  │  Active: {s.active_contradictions} contradictions  |  {_DIM(s.crr_label[:60])}",
            "  │",

            _BOLD("  └─ Coherence Budget ────────────────────────────────"),
            f"     Score: {_score_color(s.coherence_budget, 0.75, 0.50)}  {_bar(s.coherence_budget)}",
            f"     Status: {s.budget_status}  |  Strain: {s.contradiction_strain:.3f}",
            "",
        ]

        # Benchmark table
        lines += [
            _BOLD("  Target Benchmark:"),
            "  ┌────────┬─────────┬──────────┬──────────┐",
            "  │ Metric │ Current │  Target  │  Status  │",
            "  ├────────┼─────────┼──────────┼──────────┤",
            f"  │  ICS   │  {s.ics:.3f}  │  ≥ 0.70  │  {'✓' if s.ics >= 0.70 else '…'}         │",
            f"  │  TAI   │  {s.tai:+.3f} │  ≥ 0.00  │  {'✓' if s.tai >= 0.0 else '…'}         │",
            f"  │  CRR   │  {s.crr:.3f}  │  ≥ 0.40  │  {'✓' if s.crr >= 0.40 else '…'}         │",
            f"  │  Budget│  {s.coherence_budget:.3f}  │  ≥ 0.75  │  {'✓' if s.coherence_budget >= 0.75 else '…'}         │",
            "  └────────┴─────────┴──────────┴──────────┘",
            "",
        ]

        if s.errors:
            lines += [_RED("  Errors:")]
            for err in s.errors:
                lines.append(_RED(f"    ! {err}"))
            lines.append("")

        return "\n".join(lines)

    def _append_log(self, snap: MetricsSnapshot) -> None:
        """Append a JSON line to the metrics log."""
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            record = {k: v for k, v in asdict(snap).items() if k != "summary_text"}
            with self.log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.warning("Dashboard: failed to write log: %s", exc)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import sys

    p = argparse.ArgumentParser(description="EMMS Consciousness Metrics Dashboard")
    p.add_argument("--json", action="store_true", help="Output JSON only")
    p.add_argument("--watch", type=int, default=0, metavar="SECONDS",
                   help="Repeat every N seconds (0 = run once)")
    p.add_argument("--state", default=os.path.expanduser("~/.emms/emms_state.json"),
                   help="EMMS state file path")
    args = p.parse_args()

    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from emms import EMMS
    emms = EMMS()
    if os.path.exists(args.state):
        emms.load(args.state)

    dashboard = MetricsDashboard(emms)

    def _once() -> None:
        snap = dashboard.run()
        if args.json:
            record = {k: v for k, v in asdict(snap).items() if k != "summary_text"}
            print(json.dumps(record, indent=2))
        else:
            print(snap.summary_text)

    if args.watch > 0:
        while True:
            _once()
            try:
                time.sleep(args.watch)
            except KeyboardInterrupt:
                break
    else:
        _once()


if __name__ == "__main__":
    main()
