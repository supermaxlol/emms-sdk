"""Cross-modal memory binding.

Binds experiences across modalities (text, visual, audio, temporal, spatial,
emotional) so that retrieving in one modality can surface memories stored
via another.  Uses a lightweight feature-vector approach with cosine similarity.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Sequence

import numpy as np

from emms.core.models import Experience, MemoryItem, Modality

logger = logging.getLogger(__name__)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, safe against zero vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Feature extractors  (one per modality, all produce fixed-dim vectors)
# ---------------------------------------------------------------------------

_FEATURE_DIM = 16


def _text_features(exp: Experience) -> np.ndarray:
    """Simple bag-of-characters + length features for text."""
    words = exp.content.lower().split()
    vec = np.zeros(_FEATURE_DIM, dtype=np.float32)
    vec[0] = min(1.0, len(words) / 100)           # word count (normalised)
    vec[1] = min(1.0, len(exp.content) / 500)      # char count
    vec[2] = np.mean([len(w) for w in words]) / 10 if words else 0  # avg word len
    vec[3] = len(set(words)) / max(len(words), 1)  # lexical diversity
    # Character frequency fingerprint (slots 4-15)
    for ch in exp.content.lower():
        idx = ord(ch) % 12 + 4
        vec[idx] += 1
    total = vec[4:].sum()
    if total > 0:
        vec[4:] /= total
    return vec


def _emotional_features(exp: Experience) -> np.ndarray:
    vec = np.zeros(_FEATURE_DIM, dtype=np.float32)
    vec[0] = (exp.emotional_valence + 1) / 2   # normalise to 0..1
    vec[1] = exp.emotional_intensity
    vec[2] = exp.importance
    vec[3] = exp.novelty
    return vec


def _temporal_features(exp: Experience) -> np.ndarray:
    vec = np.zeros(_FEATURE_DIM, dtype=np.float32)
    t = time.localtime(exp.timestamp)
    vec[0] = t.tm_hour / 24.0
    vec[1] = t.tm_wday / 7.0
    vec[2] = t.tm_yday / 366.0
    return vec


def _default_features(exp: Experience) -> np.ndarray:
    """Fallback: hash-based fingerprint."""
    vec = np.zeros(_FEATURE_DIM, dtype=np.float32)
    for i, ch in enumerate(exp.content.encode()):
        vec[i % _FEATURE_DIM] += ch
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


_EXTRACTORS: dict[Modality, callable] = {
    Modality.TEXT: _text_features,
    Modality.EMOTIONAL: _emotional_features,
    Modality.TEMPORAL: _temporal_features,
    Modality.VISUAL: _default_features,
    Modality.AUDIO: _default_features,
    Modality.SPATIAL: _default_features,
}


# ---------------------------------------------------------------------------
# CrossModalMemory
# ---------------------------------------------------------------------------

class CrossModalMemory:
    """Binds experiences across modalities via feature similarity."""

    def __init__(self, modalities: Sequence[Modality] | None = None):
        self.modalities = list(modalities or Modality)
        # modality → {experience_id → feature_vector}
        self._indices: dict[Modality, dict[str, np.ndarray]] = {
            m: {} for m in self.modalities
        }
        # experience_id → Experience (for retrieval)
        self._experiences: dict[str, Experience] = {}
        # Stats
        self.total_stored = 0

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store(self, experience: Experience) -> dict[str, int]:
        """Extract features per modality and index the experience."""
        self._experiences[experience.id] = experience
        stored: dict[str, int] = {}

        for modality in self.modalities:
            # Prefer explicit modality_features if provided
            if modality in experience.modality_features:
                vec = np.array(experience.modality_features[modality], dtype=np.float32)
            else:
                extractor = _EXTRACTORS.get(modality, _default_features)
                vec = extractor(experience)
            self._indices[modality][experience.id] = vec
            stored[modality.value] = len(vec)

        self.total_stored += 1
        return stored

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: Experience,
        target_modalities: Sequence[Modality] | None = None,
        max_results: int = 10,
        threshold: float = 0.2,
    ) -> list[dict]:
        """Find experiences similar to *query* across modalities.

        Returns dicts with keys: experience_id, experience, score,
        modality_scores.
        """
        targets = list(target_modalities or self.modalities)

        # Extract query features
        query_vecs: dict[Modality, np.ndarray] = {}
        for mod in targets:
            if mod in query.modality_features:
                query_vecs[mod] = np.array(query.modality_features[mod], dtype=np.float32)
            else:
                query_vecs[mod] = _EXTRACTORS.get(mod, _default_features)(query)

        # Score every stored experience
        scores: dict[str, dict] = defaultdict(lambda: {"total": 0.0, "count": 0, "per_mod": {}})

        for mod, qvec in query_vecs.items():
            for exp_id, stored_vec in self._indices[mod].items():
                sim = _cosine(qvec, stored_vec)
                scores[exp_id]["total"] += sim
                scores[exp_id]["count"] += 1
                scores[exp_id]["per_mod"][mod.value] = sim

        # Average across modalities
        results = []
        for exp_id, info in scores.items():
            avg = info["total"] / info["count"] if info["count"] else 0
            if avg >= threshold:
                results.append({
                    "experience_id": exp_id,
                    "experience": self._experiences.get(exp_id),
                    "score": avg,
                    "modality_scores": info["per_mod"],
                })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:max_results]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def size(self) -> dict[str, int]:
        return {m.value: len(idx) for m, idx in self._indices.items()}
