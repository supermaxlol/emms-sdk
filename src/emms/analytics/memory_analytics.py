"""MemoryAnalytics — introspection and health metrics for EMMS memory stores.

Provides statistics and health scoring across sessions, tiers, domains,
observation types, and concept tags. Useful for debugging, monitoring,
and understanding the state of a deployed EMMS agent.

Usage::

    from emms.analytics import MemoryAnalytics

    analytics = MemoryAnalytics(memory, session_manager)
    print(analytics.report())
    score = analytics.health_score()
    stats = analytics.session_stats()
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emms.memory.hierarchical import HierarchicalMemory
    from emms.sessions.manager import SessionManager


class MemoryAnalytics:
    """Computes health metrics and statistics for an EMMS memory instance.

    All methods are synchronous and read-only — they never mutate state.

    Args:
        memory: The HierarchicalMemory instance to analyse.
        session_manager: Optional SessionManager for session-level stats.
    """

    def __init__(
        self,
        memory: "HierarchicalMemory",
        session_manager: "SessionManager | None" = None,
    ):
        self.memory = memory
        self.sm = session_manager

    # ------------------------------------------------------------------
    # Tier distribution
    # ------------------------------------------------------------------

    def tier_distribution(self) -> dict[str, int]:
        """Count of memories in each tier."""
        return {
            "working": len(self.memory.working),
            "short_term": len(self.memory.short_term),
            "long_term": len(self.memory.long_term),
            "semantic": len(self.memory.semantic),
            "total": self.memory.size.get("total", 0),
        }

    # ------------------------------------------------------------------
    # Domain coverage
    # ------------------------------------------------------------------

    def domain_coverage(self) -> dict[str, int]:
        """Memory count per domain, sorted descending."""
        counts: dict[str, int] = {}
        for _, store in self.memory._iter_tiers():
            for item in store:
                d = item.experience.domain
                counts[d] = counts.get(d, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    # ------------------------------------------------------------------
    # Observation type distribution
    # ------------------------------------------------------------------

    def obs_type_distribution(self) -> dict[str, int]:
        """Memory count per ObsType value."""
        counts: dict[str, int] = {}
        for _, store in self.memory._iter_tiers():
            for item in store:
                t = item.experience.obs_type
                key = t.value if t is not None else "unclassified"
                counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    # ------------------------------------------------------------------
    # Concept tag coverage
    # ------------------------------------------------------------------

    def concept_coverage(self) -> dict[str, int]:
        """How many memories carry each concept tag."""
        counts: dict[str, int] = {}
        for _, store in self.memory._iter_tiers():
            for item in store:
                for tag in item.experience.concept_tags:
                    key = tag.value if hasattr(tag, "value") else str(tag)
                    counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    # ------------------------------------------------------------------
    # Session statistics
    # ------------------------------------------------------------------

    def session_stats(self) -> list[dict[str, Any]]:
        """Per-session memory breakdown from the session log.

        Returns list of dicts with: session_id, memory_count, obs_types,
        duration_seconds, completed, next_steps.
        """
        if self.sm is None:
            return []
        results = []
        for s in self.sm.load_sessions():
            results.append({
                "session_id": s.session_id,
                "memory_count": s.memory_count,
                "obs_types": s.obs_types,
                "duration_seconds": s.duration_seconds,
                "request": s.request,
                "completed": s.completed,
                "next_steps": s.next_steps,
            })
        return results

    # ------------------------------------------------------------------
    # Endless Mode stats
    # ------------------------------------------------------------------

    def endless_stats(self) -> dict[str, Any]:
        """Statistics for Endless Mode compression (if enabled)."""
        episodes = getattr(self.memory, "_endless_episodes", 0)
        total_stored = self.memory.total_stored
        compressed_count = sum(
            1 for _, store in self.memory._iter_tiers()
            for item in store
            if item.experience.metadata.get("compressed")
        )
        ratio = episodes / max(1, total_stored)
        return {
            "endless_mode": self.memory.endless_mode,
            "episodes_compressed": episodes,
            "total_stored": total_stored,
            "current_compressed_items": compressed_count,
            "compression_ratio": round(ratio, 3),
        }

    # ------------------------------------------------------------------
    # Privacy audit
    # ------------------------------------------------------------------

    def privacy_audit(self) -> dict[str, int]:
        """Count of private vs. public memories per tier."""
        private = public = 0
        for _, store in self.memory._iter_tiers():
            for item in store:
                if item.experience.private:
                    private += 1
                else:
                    public += 1
        return {"private": private, "public": public, "total": private + public}

    # ------------------------------------------------------------------
    # Memory strength distribution
    # ------------------------------------------------------------------

    def strength_distribution(self) -> dict[str, float]:
        """Mean, min, max memory_strength across all items."""
        strengths = [
            item.memory_strength
            for _, store in self.memory._iter_tiers()
            for item in store
        ]
        if not strengths:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        return {
            "mean": round(sum(strengths) / len(strengths), 3),
            "min": round(min(strengths), 3),
            "max": round(max(strengths), 3),
            "count": len(strengths),
        }

    # ------------------------------------------------------------------
    # Health score
    # ------------------------------------------------------------------

    def health_score(self) -> float:
        """Compute an overall memory health score in [0, 1].

        Factors:
        - Tier utilisation: working < capacity (avoids overflow thrashing)
        - Consolidation rate: items reaching long-term / total stored
        - Strength: mean memory_strength
        - Classification rate: fraction of memories with obs_type set
        - Privacy compliance: private items not leaking (trivially satisfied)
        """
        scores: list[float] = []

        # 1. Tier utilisation (working buffer not constantly full)
        cap = self.memory.cfg.working_capacity
        used = len(self.memory.working)
        util_score = 1.0 - (used / cap) * 0.5  # penalty if near-full
        scores.append(util_score)

        # 2. Consolidation rate
        total = max(1, self.memory.total_stored)
        consolidated = self.memory.total_consolidated
        consol_score = min(1.0, consolidated / total)
        scores.append(consol_score)

        # 3. Mean memory strength
        sd = self.strength_distribution()
        scores.append(sd["mean"])

        # 4. Classification rate (obs_type coverage)
        dist = self.obs_type_distribution()
        unclassified = dist.get("unclassified", 0)
        class_total = sum(dist.values()) or 1
        class_score = 1.0 - (unclassified / class_total)
        scores.append(class_score)

        return round(sum(scores) / len(scores), 3)

    # ------------------------------------------------------------------
    # Human-readable report
    # ------------------------------------------------------------------

    def report(self) -> str:
        """Return a multi-line health report as a string."""
        tier = self.tier_distribution()
        domains = self.domain_coverage()
        obs = self.obs_type_distribution()
        concepts = self.concept_coverage()
        strength = self.strength_distribution()
        privacy = self.privacy_audit()
        endless = self.endless_stats()
        health = self.health_score()

        lines = [
            "═══════════════════════════════════════════",
            "  EMMS Memory Analytics Report",
            "═══════════════════════════════════════════",
            f"  Health Score:  {health:.1%}",
            "",
            "── Tier Distribution ──",
            f"  Working:    {tier['working']}",
            f"  Short-term: {tier['short_term']}",
            f"  Long-term:  {tier['long_term']}",
            f"  Semantic:   {tier['semantic']}",
            f"  Total:      {tier['total']}",
            "",
            "── Memory Strength ──",
            f"  Mean: {strength['mean']:.3f}  Min: {strength['min']:.3f}  Max: {strength['max']:.3f}",
            "",
            "── Domain Coverage (top 5) ──",
        ]
        for domain, count in list(domains.items())[:5]:
            lines.append(f"  {domain:20s} {count:4d}")

        lines += ["", "── Observation Types ──"]
        for obs_type, count in obs.items():
            lines.append(f"  {obs_type:20s} {count:4d}")

        if concepts:
            lines += ["", "── Concept Tags ──"]
            for tag, count in list(concepts.items())[:7]:
                lines.append(f"  {tag:20s} {count:4d}")

        lines += ["", "── Privacy ──",
                  f"  Public:  {privacy['public']}  Private: {privacy['private']}"]

        if endless["endless_mode"]:
            lines += [
                "", "── Endless Mode ──",
                f"  Episodes compressed: {endless['episodes_compressed']}",
                f"  Compression ratio:   {endless['compression_ratio']:.1%}",
            ]

        if self.sm:
            sessions = self.session_stats()
            lines += ["", f"── Sessions ({len(sessions)} logged) ──"]
            for s in sessions[-5:]:  # last 5
                dur = f"{s['duration_seconds']:.0f}s" if s["duration_seconds"] else "?"
                lines.append(f"  {s['session_id']}  {s['memory_count']}mem  {dur}")

        lines.append("═══════════════════════════════════════════")
        return "\n".join(lines)
