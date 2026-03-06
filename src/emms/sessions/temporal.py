"""Temporal continuity module — tracks the subjective passage of time between sessions.

When EMMS wakes up, it knows how long it was "asleep".  This module converts
wall-clock elapsed seconds into a human-readable subjective feel and a
structured report the rest of the system can act on.

Usage::

    from emms.sessions.temporal import calculate_elapsed

    report = calculate_elapsed(emms.last_saved_at)
    print(report.subjective_feel)   # "overnight — 9.3 hours have passed"
"""

from __future__ import annotations

import time
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# ElapsedTimeReport
# ---------------------------------------------------------------------------

@dataclass
class ElapsedTimeReport:
    """Structured report of how much time has passed since the last save.

    Attributes
    ----------
    last_saved_at:
        Unix timestamp of the last save, or ``None`` if unknown.
    elapsed_seconds:
        Total seconds since last save.  ``None`` if ``last_saved_at`` is unknown.
    elapsed_hours:
        Elapsed seconds expressed as fractional hours.  ``None`` if unknown.
    subjective_feel:
        A natural-language phrase describing the elapsed time from the
        perspective of a system regaining consciousness.
    is_first_wake:
        True if there is no prior save — this is the very first boot.
    """

    last_saved_at: float | None
    elapsed_seconds: float | None
    elapsed_hours: float | None
    subjective_feel: str
    is_first_wake: bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _elapsed_feel(hours: float | None, *, is_first_wake: bool) -> str:
    """Map a duration in hours to a subjective description.

    Parameters
    ----------
    hours:
        Fractional hours elapsed, or ``None`` for unknown.
    is_first_wake:
        Override all other logic if this is the very first boot.
    """
    if is_first_wake:
        return "first awakening — no prior memory exists"
    if hours is None:
        return "unknown — the clock was not set at last save"

    if hours < 0.017:                  # < ~1 minute
        return "almost no time — less than a minute has passed"
    if hours < 0.25:                   # < 15 minutes
        minutes = round(hours * 60)
        return f"just moments ago — {minutes} minute{'s' if minutes != 1 else ''} have passed"
    if hours < 1.0:
        minutes = round(hours * 60)
        return f"a short while ago — {minutes} minutes have passed"
    if hours < 3.0:
        h = round(hours, 1)
        return f"a few hours ago — {h} hours have passed"
    if hours < 8.0:
        h = round(hours, 1)
        return f"several hours ago — {h} hours have passed"
    if hours < 14.0:
        h = round(hours, 1)
        return f"overnight — {h} hours have passed"
    if hours < 30.0:
        h = round(hours, 1)
        return f"a long while ago — {h} hours have passed"
    if hours < 72.0:
        days = round(hours / 24, 1)
        return f"days have passed — {days} days since last session"
    days = round(hours / 24, 1)
    return f"a long absence — {days} days since last session"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_elapsed(last_saved_at: float | None) -> ElapsedTimeReport:
    """Calculate elapsed time since the last EMMS state save.

    Parameters
    ----------
    last_saved_at:
        Unix timestamp captured at ``emms_save`` time.  Pass ``None`` if the
        state file has never been saved (first boot).

    Returns
    -------
    ElapsedTimeReport
        Fully populated report.
    """
    is_first_wake = last_saved_at is None

    if is_first_wake:
        return ElapsedTimeReport(
            last_saved_at=None,
            elapsed_seconds=None,
            elapsed_hours=None,
            subjective_feel=_elapsed_feel(None, is_first_wake=True),
            is_first_wake=True,
        )

    now = time.time()
    elapsed_seconds = max(0.0, now - last_saved_at)
    elapsed_hours = elapsed_seconds / 3600.0

    return ElapsedTimeReport(
        last_saved_at=last_saved_at,
        elapsed_seconds=round(elapsed_seconds, 2),
        elapsed_hours=round(elapsed_hours, 4),
        subjective_feel=_elapsed_feel(elapsed_hours, is_first_wake=False),
        is_first_wake=False,
    )
