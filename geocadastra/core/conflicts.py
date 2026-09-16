"""Disagreement detection and conflict records (Stage 5).

Kept separate from `fusion.py`'s actual combining math: this module only
answers "do these sources disagree too much to trust an average", and
carries the record of that disagreement when they do. Nothing here is
persisted to the store yet -- Stage 5's own Done-when doesn't require it,
and Stage 3 set the precedent of returning conflicts as plain in-memory
data for the caller to act on; giving conflict records a real API/store
home is Stage 8's concern once there's an actual endpoint for them.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from geocadastra.core.crs import Geom


@dataclass(frozen=True)
class ConflictRecord:
    """doc: "emit a conflict record with parcel IDs, sources, disagreement
    magnitude, and geometry." `node_id` is None for a conflict detected
    before any graph node exists yet (a brand-new boundary candidate)."""

    node_id: int | None
    face_ids: tuple
    sources: tuple
    disagreement_m: float
    geometry: Geom


def max_pairwise_disagreement(estimates: list) -> float:
    """The largest distance between any two of these SourceEstimates.
    0.0 for zero or one estimate -- nothing to disagree with."""
    if len(estimates) < 2:
        return 0.0
    return max(((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5 for a, b in combinations(estimates, 2))
