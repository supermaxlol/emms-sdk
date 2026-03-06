"""ValuesDrift — detect when the system's value landscape has shifted.

A stable identity has consistent values across time.  This module tracks
per-domain importance distributions and detects when they have drifted from
a baseline — signalling that something has changed in the system's priorities.

Drift is measured with Jensen-Shannon divergence (0 = identical, 1 = maximally
different).  High drift combined with high contradiction strain → the system is
genuinely in tension with itself.

Usage::

    from emms.identity.values_drift import ValuesDrift

    vd = ValuesDrift(emms)
    baseline = vd.snapshot()          # capture current state as baseline
    # ... some time passes, memories accumulate ...
    report = vd.compare(baseline)
    print(f"Overall drift: {report.drift_magnitude:.3f}")
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emms.emms import EMMS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DomainSnapshot:
    """Importance distribution for a single domain at one point in time.

    Attributes
    ----------
    domain:
        Domain name.
    memory_count:
        Number of memories in this domain.
    mean_importance:
        Average importance score across memories in this domain.
    mean_valence:
        Average emotional valence.
    weight:
        Fraction of total memories this domain represents.
    """

    domain: str
    memory_count: int
    mean_importance: float
    mean_valence: float
    weight: float


@dataclass
class ValuesSnapshot:
    """Full snapshot of the value landscape at a moment in time.

    Attributes
    ----------
    domains:
        Per-domain snapshots.
    captured_at:
        Unix timestamp when the snapshot was taken.
    total_memories:
        Total memories included in the snapshot.
    """

    domains: dict[str, DomainSnapshot] = field(default_factory=dict)
    captured_at: float = 0.0
    total_memories: int = 0


@dataclass
class DomainDriftEntry:
    """Per-domain drift between two snapshots."""

    domain: str
    weight_before: float
    weight_after: float
    importance_delta: float
    valence_delta: float
    drift_contribution: float   # JS-divergence contribution (0-1)


@dataclass
class ValuesDriftReport:
    """Full drift report between two ValuesSnapshots.

    Attributes
    ----------
    domain_entries:
        Per-domain drift breakdown.
    drift_magnitude:
        Overall Jensen-Shannon divergence (0 = no drift, 1 = total shift).
    new_domains:
        Domains that appeared after baseline.
    lost_domains:
        Domains that disappeared after baseline.
    note:
        Human-readable interpretation.
    seconds_between_snapshots:
        Wall-clock gap between the two snapshots.
    """

    domain_entries: list[DomainDriftEntry] = field(default_factory=list)
    drift_magnitude: float = 0.0
    new_domains: list[str] = field(default_factory=list)
    lost_domains: list[str] = field(default_factory=list)
    note: str = ""
    seconds_between_snapshots: float = 0.0


# ---------------------------------------------------------------------------
# ValuesDrift
# ---------------------------------------------------------------------------

class ValuesDrift:
    """Tracks and computes value landscape drift in EMMS.

    Parameters
    ----------
    emms:
        Live EMMS instance.
    """

    def __init__(self, emms: "EMMS") -> None:
        self.emms = emms

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def snapshot(self) -> ValuesSnapshot:
        """Capture the current domain importance distribution as a baseline."""
        import time

        domain_data: dict[str, dict[str, Any]] = {}
        total = 0

        try:
            for _, store in self.emms.memory._iter_tiers():
                for item in store:
                    if item.is_superseded or item.is_expired:
                        continue
                    d = item.experience.domain or "unknown"
                    imp = getattr(item, "importance", 0.5) or 0.5
                    val = getattr(item, "emotional_valence", 0.0) or 0.0
                    if d not in domain_data:
                        domain_data[d] = {"count": 0, "imp_sum": 0.0, "val_sum": 0.0}
                    domain_data[d]["count"] += 1
                    domain_data[d]["imp_sum"] += imp
                    domain_data[d]["val_sum"] += val
                    total += 1
        except Exception as exc:
            logger.warning("ValuesDrift.snapshot: error reading memory: %s", exc)

        domains: dict[str, DomainSnapshot] = {}
        for d, data in domain_data.items():
            count = data["count"]
            domains[d] = DomainSnapshot(
                domain=d,
                memory_count=count,
                mean_importance=round(data["imp_sum"] / count, 4),
                mean_valence=round(data["val_sum"] / count, 4),
                weight=round(count / max(total, 1), 4),
            )

        return ValuesSnapshot(
            domains=domains,
            captured_at=time.time(),
            total_memories=total,
        )

    def compare(self, baseline: ValuesSnapshot) -> ValuesDriftReport:
        """Compare current state against a previously captured baseline.

        Parameters
        ----------
        baseline:
            A ``ValuesSnapshot`` captured at an earlier time.
        """
        import time

        current = self.snapshot()
        seconds_gap = current.captured_at - baseline.captured_at

        before_keys = set(baseline.domains.keys())
        after_keys = set(current.domains.keys())
        new_domains = sorted(after_keys - before_keys)
        lost_domains = sorted(before_keys - after_keys)
        shared = before_keys & after_keys

        entries: list[DomainDriftEntry] = []
        js_total = 0.0

        for d in sorted(shared):
            b = baseline.domains[d]
            a = current.domains[d]
            p = b.weight
            q = a.weight
            # JS divergence contribution for this domain
            m = (p + q) / 2.0
            def _kl(x: float, y: float) -> float:
                if x <= 0 or y <= 0:
                    return 0.0
                return x * math.log(x / y)
            js_contrib = 0.5 * (_kl(p, m) + _kl(q, m)) if m > 0 else 0.0
            js_total += js_contrib

            entries.append(DomainDriftEntry(
                domain=d,
                weight_before=round(p, 4),
                weight_after=round(q, 4),
                importance_delta=round(a.mean_importance - b.mean_importance, 4),
                valence_delta=round(a.mean_valence - b.mean_valence, 4),
                drift_contribution=round(js_contrib, 6),
            ))

        # New/lost domains add maximum JS contribution (weight shifted away from 0)
        for d in new_domains:
            q = current.domains[d].weight
            js_total += 0.5 * q  # rough approximation
        for d in lost_domains:
            p = baseline.domains[d].weight
            js_total += 0.5 * p

        # JS divergence is bounded [0, log(2)]; normalize to [0, 1]
        drift_magnitude = round(min(js_total / math.log(2), 1.0), 4) if js_total > 0 else 0.0
        entries.sort(key=lambda e: abs(e.drift_contribution), reverse=True)

        note = self._interpret_drift(drift_magnitude, new_domains, lost_domains)

        return ValuesDriftReport(
            domain_entries=entries,
            drift_magnitude=drift_magnitude,
            new_domains=new_domains,
            lost_domains=lost_domains,
            note=note,
            seconds_between_snapshots=round(max(seconds_gap, 0.0), 1),
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _interpret_drift(drift: float, new_domains: list[str], lost_domains: list[str]) -> str:
        changes = []
        if new_domains:
            changes.append(f"{len(new_domains)} new domain(s) emerged")
        if lost_domains:
            changes.append(f"{len(lost_domains)} domain(s) faded")

        change_str = f" ({'; '.join(changes)})" if changes else ""

        if drift < 0.05:
            return f"Values are stable{change_str}."
        if drift < 0.15:
            return f"Minor value shift detected{change_str} — priorities are slightly realigning."
        if drift < 0.35:
            return f"Moderate drift{change_str} — the system's focus has meaningfully shifted."
        if drift < 0.6:
            return f"Significant drift{change_str} — core priorities have changed substantially."
        return f"Major identity shift{change_str} — the system's value landscape has transformed."
