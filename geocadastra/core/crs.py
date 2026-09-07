"""CRS-aware geometry wrapper (Stage 1).

Every geometry that crosses a module boundary in this project should be a
`Geom`, not a bare shapely object, so its CRS travels with it. Operations
between two `Geom`s in different CRSs raise `CRSMismatchError` instead of
silently producing garbage coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pyproj
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform


class CRSMismatchError(ValueError):
    pass


# Matches core/planarize.py's own GRID (not imported from there -- planarize.py
# already imports FROM this module, so importing back would be circular).
# Without an explicit precision grid, a raw GEOS overlay op can return a
# genuinely WRONG (not just imprecise) answer, or crash with GEOSException,
# for thin/near-degenerate inputs at this project's real UTM coordinate
# scale -- see synth/generator.py's GRID comment for the original,
# real (seed=660) reproduction of a 209 m^2 wrong intersection area. Found
# by review: this project's own canonical "every geometry crossing a module
# boundary should be a Geom" wrapper had skipped the fix it applies
# everywhere else.
_GRID = 1e-3


@dataclass(frozen=True)
class Geom:
    geom: BaseGeometry
    crs: str  # e.g. "EPSG:32643"

    def _require_same_crs(self, other: "Geom") -> None:
        if self.crs != other.crs:
            raise CRSMismatchError(f"CRS mismatch: {self.crs!r} vs {other.crs!r}")

    def intersection(self, other: "Geom") -> "Geom":
        self._require_same_crs(other)
        return Geom(self.geom.intersection(other.geom, grid_size=_GRID), self.crs)

    def union(self, other: "Geom") -> "Geom":
        self._require_same_crs(other)
        return Geom(self.geom.union(other.geom, grid_size=_GRID), self.crs)

    def difference(self, other: "Geom") -> "Geom":
        self._require_same_crs(other)
        return Geom(self.geom.difference(other.geom, grid_size=_GRID), self.crs)

    def distance(self, other: "Geom") -> float:
        self._require_same_crs(other)
        return self.geom.distance(other.geom)

    @property
    def area(self) -> float:
        return self.geom.area

    @property
    def length(self) -> float:
        return self.geom.length

    def transform_to(self, target_crs: str) -> "Geom":
        if target_crs == self.crs:
            return self
        transformer = get_transformer(self.crs, target_crs)
        return Geom(shapely_transform(transformer, self.geom), target_crs)


@lru_cache(maxsize=None)
def get_transformer(src_crs: str, dst_crs: str):
    """A cached, always-xy (lon/east, lat/north order) pyproj transform callable."""
    return pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True).transform


def everest_kalianpur_to_wgs84_utm(source_epsg: str, utm_epsg: str = "EPSG:32643"):
    """The transform path from a historical Indian cadastral survey CRS to
    modern WGS84 UTM -- stubbed, not faked.

    Colonial and early post-independence Indian cadastral surveys were run on
    the Everest 1830 ellipsoid under the Kalianpur datum, in one of several
    zone-specific projected CRSs (India zones I-IV; e.g. EPSG:24379 is
    Kalianpur 1937 / India zone IIb). `pyproj` can do this transform once the
    *specific* source zone is known -- it is survey-specific and not a single
    fixed EPSG code, so this raises rather than guessing one.

    Heights are a separate, unresolved problem even once the horizontal zone
    is known: Everest-datum elevations in these surveys are typically
    orthometric (height above the Indian geoid, i.e. roughly above local mean
    sea level), while WGS84/UTM heights from a plain pyproj transform are
    ellipsoidal (height above the WGS84 ellipsoid, which differs from the
    geoid by tens of metres in India). Converting between them needs a geoid
    model (e.g. EGM2008, or India's own geoid grid) applied as a separate
    vertical shift -- no such grid is wired in here, so any Z values must not
    be trusted until one is.
    """
    raise NotImplementedError(
        f"Everest/Kalianpur ({source_epsg}) -> WGS84 UTM ({utm_epsg}) is not wired up: "
        "needs a specific India survey zone EPSG code, chosen per source, and a geoid "
        "model for ellipsoidal-vs-orthometric height correction. See this function's "
        "docstring."
    )
