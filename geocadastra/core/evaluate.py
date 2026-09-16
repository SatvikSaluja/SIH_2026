"""Evaluation harness metrics (build alongside every stage, not at the
end -- see the build plan's own section 6): the geometry-comparison
numbers Stage 8's analytics endpoint reports, always stratified by
settlement type. "A mean number across formal and informal blocks hides
the only failure mode that matters" -- every function here returns a
plain number for ONE already-filtered population; the caller (the
analytics endpoint) computes one call per stratum, never one call pooling
every style together, so pooling can't happen by accident.

Deliberately NOT: mean IoU (the doc names it explicitly as the wrong
headline -- "dominated by large simple parcels and near-blind to exactly
the irregular and encroached parcels the product exists to handle").
"""
from __future__ import annotations

import numpy as np
from shapely.errors import GEOSException
from shapely.geometry import Point
from geocadastra.core.planarize import GRID

from geocadastra.core.graph import PlanarGraph


def boundary_position_error(gt_points: list, graph: PlanarGraph) -> dict:
    """For each GT point, distance to the NEAREST EDGE (not nearest
    vertex -- a GT corner can fall partway along a longer final edge if
    the graph's own vertices don't happen to land exactly on it) in
    `graph`. Returns `{"p50": ..., "p90": ..., "n": ...}` in the graph's
    own CRS units (metres, this project's working CRS).

    Empty `gt_points` returns `{"p50": None, "p90": None, "n": 0}` --
    "no GT to measure against" is a real, distinct state from "measured
    and found perfect (error 0)", so this is never silently reported as
    a suspiciously-good zero.
    """
    return summarize_errors(boundary_position_residuals(gt_points, graph))


def boundary_position_residuals(gt_points: list, graph: PlanarGraph) -> list[float]:
    """Keep residuals available so callers can aggregate within strata."""
    edges = [graph.edge_linestring(eid) for eid in graph.edges]
    if not edges:
        return []
    return [min(edge.distance(Point((gt.x, gt.y) if hasattr(gt, "x") else gt))
                for edge in edges) for gt in gt_points]


def summarize_errors(errors: list[float]) -> dict:
    if not errors:
        return {"p50": None, "p90": None, "n": 0}
    return {"p50": float(np.percentile(errors, 50)),
            "p90": float(np.percentile(errors, 90)), "n": len(errors)}


def topology_validity_rate(graph: PlanarGraph) -> dict:
    """Fraction of `graph`'s own faces that are individually valid simple
    polygons (`shapely`'s own `.is_valid`) AND collectively non-
    overlapping (no two faces' polygons share positive area) -- the two
    concrete ways a "certified parcel layer" can be topologically broken.
    Reported together as one rate, plus the two counts that made it up,
    since "which of the two failed" is exactly what a caller debugging a
    validity drop needs next.
    """
    polys = graph.faces_to_polygons()
    n = len(polys)
    if n == 0:
        return {"rate": None, "n_faces": 0, "n_invalid": 0, "n_overlapping_pairs": 0, "n_unchecked_pairs": 0}
    items = list(polys.values())
    invalid_idx = {i for i, g in enumerate(items) if not g.geom.is_valid}
    overlapping_pairs = 0
    faces_in_an_overlap: set = set()
    unchecked_pairs = 0
    indeterminate_faces = set()
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if i in invalid_idx or j in invalid_idx:
                # An invalid face cannot be safely overlaid. Count the pair
                # explicitly, without claiming its overlap is known.
                unchecked_pairs += 1
                if items[i].geom.envelope.intersects(items[j].geom.envelope):
                    indeterminate_faces.update((i, j))
                continue
            try:
                overlap = items[i].geom.intersection(items[j].geom, grid_size=GRID).area
            except GEOSException:
                unchecked_pairs += 1
                indeterminate_faces.update((i, j))
                continue
            if overlap > 1e-6:
                overlapping_pairs += 1
                faces_in_an_overlap.update((i, j))
    invalid = len(invalid_idx)
    n_bad = len(invalid_idx | faces_in_an_overlap | indeterminate_faces)
    return {
        "rate": (n - n_bad) / n,
        "n_faces": n,
        "n_invalid": invalid,
        "n_overlapping_pairs": overlapping_pairs,
        "n_unchecked_pairs": unchecked_pairs,
    }


def parcel_count_error(predicted_count: int, recorded_count: int) -> dict:
    """Signed decomposition, not just `abs(predicted - recorded)`: over-
    segmentation (too many parcels -- a real boundary split into pieces)
    and under-segmentation (too few -- two parcels merged) are different
    failure modes with different causes, and the doc asks for them
    "separately", not netted against each other."""
    diff = predicted_count - recorded_count
    return {
        "predicted": predicted_count,
        "recorded": recorded_count,
        "over_segmentation": max(diff, 0),
        "under_segmentation": max(-diff, 0),
    }


def area_error_distribution(predicted_areas: list, recorded_areas: list) -> dict:
    """Per-parcel `|predicted - recorded| / recorded` (relative error,
    comparable across parcels of very different sizes -- an absolute
    error in m^2 would be dominated by the largest parcels the same way
    mean IoU is), summarized as P50/P90. `predicted_areas[i]` must
    correspond to `recorded_areas[i]` -- the caller's responsibility to
    pair them by parcel id before calling this.
    """
    if not recorded_areas:
        return {"p50": None, "p90": None, "n": 0}
    predicted = np.asarray(predicted_areas, dtype=float)
    recorded = np.asarray(recorded_areas, dtype=float)
    rel_error = np.abs(predicted - recorded) / recorded
    return {"p50": float(np.percentile(rel_error, 50)), "p90": float(np.percentile(rel_error, 90)), "n": len(rel_error)}
