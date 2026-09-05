"""Turn a bag of linework into clean, gap/overlap-free polygonal faces (Stage 1).

Two-stage, per the build plan: (a) adaptively pre-snap near-coincident
vertices using a tolerance derived from local linework density -- handles
real-world regional variation (dense urban blocks vs sparse rural plots) --
then (b) snap-round to a fixed, fine precision grid and polygonize, which
guarantees a numerically valid planar result regardless of what stage (a)
left behind. Do not skip (b): GEOS's overlay ops are only *robustness*-
guaranteed when given an explicit precision grid -- see the note on GRID
below, learned the hard way in Stage 0.
"""
from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import LineString
from shapely.ops import polygonize

from geocadastra.core.crs import CRSMismatchError, Geom

GRID = 1e-3  # metres. Cadastral-meaningful precision (~1mm): fine enough to
# never merge two genuinely distinct survey points, coarse enough to treat
# two representations of "the same" point (float noise, independent
# construction paths) as identical downstream, and to give GEOS's overlay
# engine an explicit precision model -- without one, chained boolean ops on
# many near-degenerate inputs can return a *wrong* answer, not just an
# imprecise one (see geocadastra/synth/generator.py's GRID for the
# reproduction).


def planarize(linework: list[Geom], grid: float = GRID, fixed: list[Geom] = ()) -> list[Geom]:
    """Bag of LineString/MultiLineString `Geom`s (all one CRS) -> list of
    Polygon `Geom` faces. Faces tile their combined domain exactly, up to
    `grid`.

    `fixed` is optional authoritative linework (e.g. a ward boundary) that
    participates in pre-snapping as an attractor -- ordinary linework can
    snap onto it -- but never moves itself, so it comes back out of the
    pipeline bit-for-bit the coordinates it went in with (mod the final grid
    snap). Without this, adaptive pre-snapping is free to drag an
    authoritative boundary off its recorded position toward a nearby but
    unrelated vertex.
    """
    all_geoms = list(linework) + list(fixed)
    if not all_geoms:
        return []
    crs = all_geoms[0].crs
    for g in all_geoms:
        if g.crs != crs:
            raise CRSMismatchError(f"planarize() got mixed CRS: {crs!r} vs {g.crs!r}")

    lines = _flatten_to_linestrings(g.geom for g in linework)
    fixed_lines = _flatten_to_linestrings(g.geom for g in fixed)
    snapped = _adaptive_presnap(lines, fixed_lines=fixed_lines)

    noded = shapely.union_all([*snapped, *fixed_lines], grid_size=grid)
    faces = shapely.set_precision(np.array(list(polygonize(noded))), grid)  # one batched GEOS call, not one per face
    return [Geom(f, crs) for f in faces]


def _flatten_to_linestrings(geoms) -> list[LineString]:
    out = []
    for g in geoms:
        if g.is_empty:
            continue  # e.g. a road clipped to nothing -- contributes no linework, not an error
        if g.geom_type == "MultiLineString":
            out.extend(part for part in g.geoms if not part.is_empty)
        elif g.geom_type == "LineString":
            out.append(g)
        else:
            raise ValueError(f"planarize() expects LineString/MultiLineString linework, got {g.geom_type}")
    return out


def _adaptive_presnap(
    lines: list[LineString],
    fixed_lines: list[LineString] = (),
    min_tol: float = 0.01,
    max_tol: float = 2.0,
    factor: float = 0.25,
) -> list[LineString]:
    """Cluster near-coincident vertices and replace each with its cluster's
    representative point. Tolerance per vertex is `factor` times *that
    vertex's own line's* average segment length, clipped to [min_tol,
    max_tol]: dense linework (short segments -- a detailed urban survey)
    gets a tight tolerance so distinct nearby features aren't merged; sparse
    linework (long segments -- a large rural plot) gets a looser one.

    Deliberately not the vertex's raw nearest-neighbour distance: the near-
    miss vertices this function exists to close are often *each other's*
    nearest neighbour, which would make the naive estimate small exactly
    where a larger tolerance is needed.

    Clustering is leader/representative-based, not transitive: each cluster
    has one fixed anchor point (a `fixed_lines` vertex if one is in range,
    else the first vertex to reach that spot in a deterministic scan order),
    and a later vertex joins only if it is within tolerance of that anchor
    specifically. This is deliberate -- naive transitive clustering (union
    A-B and B-C into one cluster because each pair is close, even if A and C
    are not) can chain a whole run of vertices onto one point many multiples
    of any single tolerance away; confirmed in review to move a vertex over
    20m on realistic input. An anchor's position never moves once chosen, so
    a cluster's total spread is always bounded by its own anchor's
    tolerance, however many points join it.
    """
    pts, tol_per_pt, owner_lens, fixed_flags = [], [], [], []
    for is_fixed, ln in [(False, l) for l in lines] + [(True, l) for l in fixed_lines]:
        coords = [tuple(c) for c in ln.coords]
        owner_lens.append(len(coords))
        if not coords:
            continue
        coords_arr = np.asarray(coords, dtype=float)
        seg_lens = np.linalg.norm(np.diff(coords_arr, axis=0), axis=1) if len(coords) > 1 else np.array([0.0])
        local_scale = float(seg_lens.mean()) if seg_lens.size else min_tol / factor
        t = min(max(local_scale * factor, min_tol), max_tol)
        pts.extend(coords)
        tol_per_pt.extend([t] * len(coords))
        fixed_flags.extend([is_fixed] * len(coords))

    n_movable_lines = len(lines)
    if len(pts) < 2:
        return lines

    pts = np.asarray(pts, dtype=float)
    tol = np.asarray(tol_per_pt, dtype=float)
    is_fixed = np.asarray(fixed_flags, dtype=bool)
    n = len(pts)

    # processing order: fixed points first (so they always become anchors,
    # never get absorbed into a movable cluster), then movable points in a
    # coordinate-sorted order so the result depends only on the point set,
    # not on which line happened to list a point first
    movable_idx = np.nonzero(~is_fixed)[0]
    movable_sorted = movable_idx[np.lexsort((pts[movable_idx, 1], pts[movable_idx, 0]))]
    order = np.concatenate([np.nonzero(is_fixed)[0], movable_sorted])

    anchor_pts: list = []
    anchor_tol: list = []
    labels = np.full(n, -1, dtype=int)
    for i in order:
        p = pts[i]
        if anchor_pts:
            d = np.linalg.norm(np.asarray(anchor_pts) - p, axis=1)
            j = int(np.argmin(d))
            if d[j] <= min(tol[i], anchor_tol[j]):
                labels[i] = j
                continue
        labels[i] = len(anchor_pts)
        anchor_pts.append(p.copy())
        anchor_tol.append(tol[i])
    # ponytail: O(n * clusters-so-far) representative search -- fine while
    # clusters stay well under a few thousand; rebuild-free KD-tree over
    # anchors if a much larger ward makes this a hot path.
    anchors = np.asarray(anchor_pts)

    out = []
    idx = 0
    for length in owner_lens[:n_movable_lines]:
        if length == 0:
            out.append(LineString())
            continue
        snapped = anchors[labels[idx : idx + length]]
        idx += length
        coords = [tuple(snapped[0])]
        for pt in snapped[1:]:
            if not (abs(pt[0] - coords[-1][0]) <= min_tol / 10 and abs(pt[1] - coords[-1][1]) <= min_tol / 10):
                coords.append(tuple(pt))
        out.append(LineString(coords) if len(coords) >= 2 else LineString([coords[0], coords[0]]))
    return out
