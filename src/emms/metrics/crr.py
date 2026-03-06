"""Contradiction Resolution Rate (CRR) — tracks if detected contradictions get resolved.

Of the contradictions detected in session N, what fraction are no longer
detectable by session N+3?

CRR = resolved / total_tracked

Target: CRR > 40% within 3 sessions of first detection.

How it works:
- ContradictionTracker persists detected contradictions to EMMS as tagged memories.
- On each scan, previously-tracked contradictions are re-checked.
- If a contradiction is no longer detectable (topic pair no longer co-occurs with
  opposing valence), it is marked resolved.

Usage::

    from emms.metrics.crr import ContradictionTracker

    tracker = ContradictionTracker(emms)
    tracker.scan_and_record()      # detect + save contradictions

    # ... later (session N+3) ...
    tracker.scan_and_record()
    report = tracker.compute_crr()
    print(report.crr)             # e.g. 0.45
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emms.emms import EMMS

logger = logging.getLogger(__name__)

_STORE_PATH = Path.home() / ".emms" / "contradiction_tracker.json"


# ---------------------------------------------------------------------------
# TrackedContradiction
# ---------------------------------------------------------------------------

@dataclass
class TrackedContradiction:
    """A contradiction that was detected and is being tracked."""
    id: str                      # hash of (memory_id_a, memory_id_b)
    memory_id_a: str
    memory_id_b: str
    domain: str
    tension_score: float
    detected_at: float           # Unix timestamp
    detected_session: str | None
    resolved_at: float | None = None
    resolved_session: str | None = None
    is_resolved: bool = False
    check_count: int = 0         # how many times we've re-checked


@dataclass
class CRRReport:
    """Contradiction Resolution Rate report.

    Attributes
    ----------
    crr:
        Fraction of tracked contradictions that are now resolved (0-1).
    total_tracked:
        All contradictions ever tracked.
    resolved:
        Number now resolved.
    active:
        Number still unresolved.
    new_this_scan:
        Contradictions newly detected in this scan.
    label:
        Human-readable interpretation.
    """
    crr: float
    total_tracked: int
    resolved: int
    active: int
    new_this_scan: int
    label: str = ""


# ---------------------------------------------------------------------------
# ContradictionTracker
# ---------------------------------------------------------------------------

class ContradictionTracker:
    """Detects and persistently tracks contradictions across sessions.

    Parameters
    ----------
    emms:
        Live EMMS instance.
    store_path:
        Where to persist the contradiction registry (default: ~/.emms/contradiction_tracker.json).
    current_session:
        Session ID to tag new contradictions with.
    """

    def __init__(
        self,
        emms: "EMMS",
        *,
        store_path: Path = _STORE_PATH,
        current_session: str | None = None,
    ) -> None:
        self.emms = emms
        self.store_path = store_path
        self.current_session = current_session or f"session_{int(time.time())}"
        self._registry: dict[str, TrackedContradiction] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def scan_and_record(self) -> CRRReport:
        """Run a full contradiction scan, update registry, save, return CRR."""
        from emms.identity.contradiction_awareness import ContradictionAwareness

        awareness = ContradictionAwareness(self.emms)
        report = awareness.scan()

        # Build current contradiction set (by ID)
        current_ids: set[str] = set()
        new_count = 0

        for tension in report.tensions:
            cid = self._make_id(tension.memory_id_a, tension.memory_id_b)
            current_ids.add(cid)
            if cid not in self._registry:
                # New contradiction
                self._registry[cid] = TrackedContradiction(
                    id=cid,
                    memory_id_a=tension.memory_id_a,
                    memory_id_b=tension.memory_id_b,
                    domain=tension.domain,
                    tension_score=tension.tension_score,
                    detected_at=time.time(),
                    detected_session=self.current_session,
                )
                new_count += 1
                logger.info("CRR: new contradiction tracked [%s] domain=%s", cid[:8], tension.domain)

        # Check which previously-tracked contradictions are now resolved
        for cid, tracked in self._registry.items():
            tracked.check_count += 1
            if not tracked.is_resolved and cid not in current_ids:
                tracked.is_resolved = True
                tracked.resolved_at = time.time()
                tracked.resolved_session = self.current_session
                logger.info("CRR: contradiction resolved [%s] after %d checks", cid[:8], tracked.check_count)

        self._save()
        return self.compute_crr(new_this_scan=new_count)

    def compute_crr(self, new_this_scan: int = 0) -> CRRReport:
        """Compute current CRR from registry (without running a new scan)."""
        total = len(self._registry)
        resolved = sum(1 for t in self._registry.values() if t.is_resolved)
        active = total - resolved
        crr = round(resolved / max(total, 1), 4)
        label = self._interpret(crr, total)

        return CRRReport(
            crr=crr,
            total_tracked=total,
            resolved=resolved,
            active=active,
            new_this_scan=new_this_scan,
            label=label,
        )

    def reset(self) -> None:
        """Clear the registry (use for testing only)."""
        self._registry = {}
        self._save()

    @property
    def registry(self) -> dict[str, TrackedContradiction]:
        return dict(self._registry)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _make_id(id_a: str, id_b: str) -> str:
        """Stable ID for a contradiction pair (order-independent)."""
        import hashlib
        key = "|".join(sorted([id_a, id_b]))
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def _load(self) -> None:
        if not self.store_path.exists():
            return
        try:
            data = json.loads(self.store_path.read_text())
            for cid, entry in data.items():
                self._registry[cid] = TrackedContradiction(**entry)
            logger.debug("CRR: loaded %d tracked contradictions", len(self._registry))
        except Exception as exc:
            logger.warning("CRR: failed to load registry: %s", exc)

    def _save(self) -> None:
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            data = {cid: asdict(t) for cid, t in self._registry.items()}
            self.store_path.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            logger.warning("CRR: failed to save registry: %s", exc)

    @staticmethod
    def _interpret(crr: float, total: int) -> str:
        if total == 0:
            return "No contradictions tracked yet — run scan_and_record() first."
        if crr >= 0.6:
            return f"Strong resolution ({crr:.0%}) — the system actively resolves internal conflicts."
        if crr >= 0.4:
            return f"Meeting target ({crr:.0%}) — contradictions are being resolved at healthy rate."
        if crr >= 0.2:
            return f"Developing ({crr:.0%}) — some resolution occurring but below 40% target."
        if crr > 0:
            return f"Weak resolution ({crr:.0%}) — contradictions are persisting across sessions."
        return "No resolution yet — no tracked contradictions have been resolved."
