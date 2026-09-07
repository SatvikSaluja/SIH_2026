"""Capacity-constrained parcel assignment (Stage 3) -- the differentiator.

Given one block, a list of recorded parcel areas, an optional seed point per
parcel (a legacy GIS centroid), and a boundary-evidence raster (high value =
"a boundary probably runs here"), divide the block among the known parcels
so each gets exactly its recorded area, while preferring not to cross
boundaries the evidence can see. Where evidence is strong it dominates;
where a boundary is invisible, the area constraint alone places it -- at the
only position consistent with the record.

Pipeline: oversegment the block into SLIC superpixels on the evidence field,
build a superpixel adjacency graph whose edge weights rise with local
evidence strength, compute geodesic (Dijkstra) distance from each parcel's
seed through that graph, solve the resulting optimal-transport problem
(supply = superpixel areas, demand = recorded parcel areas) with POT, round
the transport plan to a hard per-superpixel labelling, and extract label
boundaries as polygons via planarize.py.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import ot
from rasterio.features import rasterize, shapes as raster_shapes
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import Polygon, shape as shapely_shape
from shapely.ops import unary_union
from skimage.segmentation import slic

from geocadastra.core.crs import CRSMismatchError, Geom
from geocadastra.core.graph import PlanarGraph, build_graph
from geocadastra.core.planarize import GRID, planarize

# Exact LP solving (ot.lp.emd) stays fast (single-digit seconds) even at
# supply*demand ~100M -- far past any block this project will actually see
# (a few thousand superpixels x at most a few hundred parcels). Set high
# enough that the Sinkhorn fallback below is a true last resort, not
# something ordinary use silently degrades into: the Sinkhorn plan is dense
# rather than near-binary, and this module's hard-rounding scheme, which
# assumes near-binary, quietly degenerates toward the no-constraint
# nearest-seed baseline when it isn't (found by review).
EXACT_SOLVE_MAX_SIZE = 5_000_000
EVIDENCE_STRENGTH_SCALE = 20.0  # how much full (1.0) evidence multiplies a step's base distance cost
AREA_MISMATCH_TOL = 1e-3  # relative demand/supply mismatch below which no conflict is recorded
RECORDED_AREA_TOL_SUPERPIXELS = 3.0  # post-solve check: how many superpixels' worth of drift is tolerated


@dataclass
class Conflict:
    kind: str
    detail: dict = field(default_factory=dict)


@dataclass
class AssignmentResult:
    labels: np.ndarray  # (H, W) int32, superpixel id per pixel, -1 outside the block mask
    superpixel_to_parcel: dict  # superpixel id -> parcel index (int); unassigned superpixels are absent keys
    transform: object  # affine, shared by `labels` and the evidence field it was built from
    parcel_polygons: dict  # parcel index -> Geom (Polygon or MultiPolygon)
    conflicts: list


def assign_parcels(
    block: Geom,
    parcel_areas: list,
    seed_points: list,
    evidence_field: np.ndarray,
    transform,
    n_segments: int = 2000,
    compactness: float = 5.0,
) -> AssignmentResult:
    """`parcel_areas[i]`/`seed_points[i]` may be `None` (missing record / no
    legacy seed). A parcel needs both to participate in the capacity-
    constrained solve; the rest is covered by the fallbacks in the module
    docstring (see `_assign_leftover` / `_watershed_fallback`)."""
    block_polygon, crs = block.geom, block.crs
    conflicts: list[Conflict] = []
    evidence_field = _validate_evidence_field(evidence_field)
    out_shape = evidence_field.shape
    mask = rasterize([(block_polygon, 1)], out_shape, transform=transform, fill=0, dtype="uint8").astype(bool)
    gsd = abs(transform.a)

    usable = [i for i in range(len(parcel_areas)) if parcel_areas[i] is not None and seed_points[i] is not None]
    if not usable:
        return _watershed_fallback(block_polygon, crs, seed_points, evidence_field, mask, transform, conflicts)

    labels = slic(evidence_field, n_segments=n_segments, compactness=compactness, mask=mask, channel_axis=None, start_label=0)
    sp_ids, sp_areas = _superpixel_areas(labels, gsd)
    if len(sp_ids) == 0:
        return AssignmentResult(labels, {}, transform, {}, conflicts)

    adjacency = _superpixel_adjacency(labels, evidence_field)
    centroids = _superpixel_centroids(labels, sp_ids)
    seed_sp = {i: _seed_to_superpixel(seed_points[i], transform, labels) for i in usable}
    graph = _build_graph(sp_ids, adjacency, centroids, gsd)

    demand_raw = np.array([parcel_areas[i] for i in usable], dtype=float)
    supply_total = sp_areas.sum()
    demand_total = demand_raw.sum()
    if demand_total <= 0:
        return _watershed_fallback(block_polygon, crs, seed_points, evidence_field, mask, transform, conflicts)

    seed_indices = [np.searchsorted(sp_ids, seed_sp[i]) for i in usable]
    dist = dijkstra(graph, indices=seed_indices, directed=False)  # (len(usable), n_superpixels)
    finite = dist[np.isfinite(dist)]
    fill_value = float(finite.max() * 10 + 1) if finite.size else 1.0
    cost = np.where(np.isfinite(dist), dist, fill_value)  # (len(usable), n_superpixels)

    unhandled = [i for i in range(len(parcel_areas)) if i not in usable]
    if unhandled and demand_total < supply_total:
        # other parcels are waiting for whatever's left over -- don't let
        # rescaling-to-match-supply silently hand them the whole block.
        # Instead add a dummy "reservoir" demand column, costed at the
        # ceiling of every real cost so real parcels always win a superpixel
        # they're genuinely cheaper for; only once their own capacity is
        # exhausted does the remainder flow to the reservoir for
        # _assign_leftover to hand out below.
        leftover = supply_total - demand_total
        cost_full = np.vstack([cost, np.full((1, cost.shape[1]), cost.max())])
        demand_full = np.concatenate([demand_raw, [leftover]])
    else:
        rel_mismatch = abs(supply_total - demand_total) / supply_total
        if rel_mismatch > AREA_MISMATCH_TOL:
            conflicts.append(Conflict("area_sum_mismatch", {
                "recorded_total": float(demand_total), "block_total": float(supply_total), "relative_mismatch": float(rel_mismatch),
            }))
        demand_full = demand_raw * (supply_total / demand_total)  # exact rescale: POT requires equal sums
        cost_full = cost

    plan = _solve_transport(sp_areas, demand_full, cost_full.T)  # POT wants (supply_size, demand_size)
    hard = _round_plan(plan, cost_full.T)  # superpixel row index -> position in `usable` (or len(usable) == the reservoir)

    sp_to_parcel = {int(sp_ids[row]): usable[col] for row, col in enumerate(hard) if col < len(usable)}

    if unhandled:
        _assign_leftover(unhandled, parcel_areas, seed_points, sp_ids, sp_to_parcel, centroids, transform, conflicts)

    # a single, general safety net instead of point-fixing each way the
    # solve above can quietly shortchange a parcel (Sinkhorn's dense plan
    # degenerating the tiebreak, the reservoir mechanism, a starved leftover
    # parcel...): verify what was actually built against what was recorded,
    # for every parcel that has a recorded area, and say so if it's off by
    # more than a few superpixels -- found necessary by review.
    _verify_recorded_areas(parcel_areas, sp_to_parcel, sp_ids, sp_areas, conflicts)

    polys = _extract_parcel_polygons(labels, sp_to_parcel, transform, crs)
    return AssignmentResult(labels, sp_to_parcel, transform, polys, conflicts)


def assign_parcels_evidence_only(
    block: Geom,
    seed_points: list,
    evidence_field: np.ndarray,
    transform,
    n_segments: int = 2000,
    compactness: float = 5.0,
) -> AssignmentResult:
    """The baseline the headline metric is measured against: nearest-seed
    assignment through the same evidence-weighted geodesic graph, with NO
    area constraint at all. Isolates exactly what the capacity constraint in
    `assign_parcels` contributes -- same evidence, same graph, same seeds,
    the only difference is whether recorded area is allowed to matter."""
    block_polygon, crs = block.geom, block.crs
    conflicts: list[Conflict] = []
    evidence_field = _validate_evidence_field(evidence_field)
    out_shape = evidence_field.shape
    mask = rasterize([(block_polygon, 1)], out_shape, transform=transform, fill=0, dtype="uint8").astype(bool)
    gsd = abs(transform.a)

    usable = [i for i in range(len(seed_points)) if seed_points[i] is not None]
    if not usable:
        return _watershed_fallback(block_polygon, crs, seed_points, evidence_field, mask, transform, conflicts)

    labels = slic(evidence_field, n_segments=n_segments, compactness=compactness, mask=mask, channel_axis=None, start_label=0)
    sp_ids, _ = _superpixel_areas(labels, gsd)
    if len(sp_ids) == 0:
        return AssignmentResult(labels, {}, transform, {}, conflicts)

    adjacency = _superpixel_adjacency(labels, evidence_field)
    centroids = _superpixel_centroids(labels, sp_ids)
    seed_sp = {i: _seed_to_superpixel(seed_points[i], transform, labels) for i in usable}
    graph = _build_graph(sp_ids, adjacency, centroids, gsd)

    seed_indices = [np.searchsorted(sp_ids, seed_sp[i]) for i in usable]
    dist = dijkstra(graph, indices=seed_indices, directed=False)
    nearest = np.argmin(dist, axis=0)  # per superpixel, index into `usable`
    sp_to_parcel = {int(sp_ids[col]): usable[nearest[col]] for col in range(len(sp_ids))}

    polys = _extract_parcel_polygons(labels, sp_to_parcel, transform, crs)
    return AssignmentResult(labels, sp_to_parcel, transform, polys, conflicts)


def _validate_evidence_field(evidence_field: np.ndarray) -> np.ndarray:
    """Clip to [0,1] and reject non-finite values: an unclipped out-of-range
    value (a raw, un-normalized model logit, say) turns into a *negative*
    graph edge weight in `_build_graph` -- and since the graph is
    undirected, one negative edge is already a negative cycle, which sends
    `dijkstra` into unbounded memory growth instead of raising (found, and
    reproduced, by review)."""
    arr = np.asarray(evidence_field, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError("evidence_field contains NaN/Inf -- clip or clean it before calling assign_parcels")
    return np.clip(arr, 0.0, 1.0)


# --------------------------------------------------------------------------
# pixel/world coordinate conversion
# --------------------------------------------------------------------------

def _world_to_pixel(xy, transform, shape=None):
    """(x, y) world coords -> (row, col) pixel indices, rounded. If `shape`
    is given, clamps into [0, shape) on BOTH ends -- found by review: two of
    three call sites here used to clamp only the lower bound (or neither),
    which crashed (not just gave an imprecise answer) for a seed point a
    few pixels north/west of its raster."""
    col, row = ~transform @ (xy[0], xy[1])
    row, col = int(round(row)), int(round(col))
    if shape is not None:
        h, w = shape
        row = min(max(row, 0), h - 1)
        col = min(max(col, 0), w - 1)
    return row, col


# --------------------------------------------------------------------------
# superpixels
# --------------------------------------------------------------------------

def _superpixel_areas(labels: np.ndarray, gsd: float):
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    return ids, counts.astype(float) * (gsd * gsd)


def _superpixel_centroids(labels: np.ndarray, sp_ids: np.ndarray) -> dict:
    centroids = ndimage.center_of_mass(np.ones_like(labels, dtype=float), labels, sp_ids)
    return dict(zip(sp_ids.tolist(), centroids))  # id -> (row, col) in pixel space


def _superpixel_adjacency(labels: np.ndarray, evidence_field: np.ndarray) -> dict:
    """Undirected superpixel pair -> mean evidence value along their shared
    boundary. Vectorized (no per-pixel Python loop): every axis-aligned
    adjacent pixel pair with differing, both-valid labels contributes one
    evidence sample, grouped by pair via sorting."""
    pair_a, pair_b, ev = [], [], []
    for a_arr, b_arr, e_arr in (
        (labels[:, :-1], labels[:, 1:], 0.5 * (evidence_field[:, :-1] + evidence_field[:, 1:])),
        (labels[:-1, :], labels[1:, :], 0.5 * (evidence_field[:-1, :] + evidence_field[1:, :])),
    ):
        m = (a_arr != b_arr) & (a_arr >= 0) & (b_arr >= 0)
        pair_a.append(a_arr[m])
        pair_b.append(b_arr[m])
        ev.append(e_arr[m])
    if not pair_a or sum(a.size for a in pair_a) == 0:
        return {}
    a = np.concatenate(pair_a)
    b = np.concatenate(pair_b)
    e = np.concatenate(ev)
    lo, hi = np.minimum(a, b).astype(np.int64), np.maximum(a, b).astype(np.int64)
    scale = int(max(hi.max(), lo.max()) + 1)
    keys = lo * scale + hi
    order = np.argsort(keys, kind="stable")
    keys_sorted, e_sorted = keys[order], e[order]
    unique_keys, start_idx = np.unique(keys_sorted, return_index=True)
    sums = np.add.reduceat(e_sorted, start_idx)
    counts = np.diff(np.append(start_idx, len(keys_sorted)))
    means = sums / counts
    return {(int(k // scale), int(k % scale)): float(m) for k, m in zip(unique_keys, means)}


def _build_graph(sp_ids: np.ndarray, adjacency: dict, centroids: dict, gsd: float):
    n = len(sp_ids)
    id_to_idx = {int(sid): i for i, sid in enumerate(sp_ids)}
    rows, cols, weights = [], [], []
    for (a, b), strength in adjacency.items():
        ra, ca = centroids[a]
        rb, cb = centroids[b]
        base_dist = ((ra - rb) ** 2 + (ca - cb) ** 2) ** 0.5 * gsd
        w = base_dist * (1.0 + EVIDENCE_STRENGTH_SCALE * strength)
        ia, ib = id_to_idx[a], id_to_idx[b]
        rows += [ia, ib]
        cols += [ib, ia]
        weights += [w, w]
    return coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()


def _seed_to_superpixel(seed_xy, transform, labels: np.ndarray) -> int:
    row, col = _world_to_pixel(seed_xy, transform)  # deliberately unclamped here: an out-of-range
    h, w = labels.shape                              # seed still needs its true offset to search outward from
    for radius in range(0, max(h, w)):
        # both ends of each slice bound need floor- AND ceiling-clamping: a
        # row/col a little negative (a seed just north/west of the raster)
        # made the *upper* bound go negative too before max(..., 0) below,
        # which plain slicing silently reinterpreted as "count from the
        # end" instead of "empty" -- crashed a few lines down instead of
        # falling through to this function's own clean error (found by
        # review, on realistic single-digit-pixel seed drift)
        r0, r1 = max(row - radius, 0), max(min(row + radius + 1, h), 0)
        c0, c1 = max(col - radius, 0), max(min(col + radius + 1, w), 0)
        if r0 >= r1 or c0 >= c1:
            continue
        window = labels[r0:r1, c0:c1]
        valid = window[window >= 0]
        if valid.size:
            # nearest valid pixel within this growing window, not just any
            rr, cc = np.mgrid[r0:r1, c0:c1]
            d2 = (rr - row) ** 2 + (cc - col) ** 2
            d2 = np.where(window >= 0, d2, np.inf)
            return int(window.flat[np.argmin(d2)])
    raise ValueError("seed point has no reachable superpixel in this block (empty mask?)")


# --------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------

def _solve_transport(supply: np.ndarray, demand: np.ndarray, cost: np.ndarray) -> np.ndarray:
    if supply.size * demand.size <= EXACT_SOLVE_MAX_SIZE:
        return ot.lp.emd(supply, demand, cost)
    reg = float(cost[np.isfinite(cost)].mean()) * 0.01 or 1.0
    return ot.sinkhorn(supply, demand, cost, reg=reg)


def _round_plan(plan: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Hard-label each supply row (superpixel) to one demand column (parcel):
    the plan's own argmax by default, but for a superpixel split across
    several parcels (a handful, at the transport polytope's margin -- true
    for the exact LP solver; a dense Sinkhorn plan instead degenerates this
    toward the no-constraint nearest-cost baseline, which is why
    `EXACT_SOLVE_MAX_SIZE` is set high enough that ordinary use never
    reaches Sinkhorn -- see its definition), resolve by lowest cost among
    that superpixel's own nonzero candidates -- not a globally cost-blind
    pick, which would ignore the area constraint."""
    hard = np.argmax(plan, axis=1)
    nnz_per_row = (plan > 0).sum(axis=1)
    for row in np.nonzero(nnz_per_row > 1)[0]:
        candidates = np.nonzero(plan[row] > 0)[0]
        hard[row] = candidates[np.argmin(cost[row, candidates])]
    return hard


def _verify_recorded_areas(parcel_areas, sp_to_parcel, sp_ids, sp_areas, conflicts) -> None:
    """Post-solve safety net: for every parcel with a recorded area, does
    what actually got assigned match it within a few superpixels? A single
    general check here catches several different ways the solve above can
    quietly shortchange a parcel, instead of needing a point-fix for each
    (found necessary by review, which reproduced more than one such case)."""
    if not len(sp_ids):
        return
    avg_sp_area = float(sp_areas.mean())
    area_by_sp = dict(zip(sp_ids.tolist(), sp_areas.tolist()))
    assigned = defaultdict(float)
    for sid, pid in sp_to_parcel.items():
        assigned[pid] += area_by_sp.get(sid, 0.0)
    mismatched = []
    for i, recorded in enumerate(parcel_areas):
        if recorded is None:
            continue
        got = assigned.get(i, 0.0)
        tol = max(RECORDED_AREA_TOL_SUPERPIXELS * avg_sp_area, recorded * 0.05)
        if abs(got - recorded) > tol:
            mismatched.append({"parcel_index": i, "recorded": float(recorded), "assigned": float(got)})
    if mismatched:
        conflicts.append(Conflict("recorded_area_not_matched", {"parcels": mismatched}))


# --------------------------------------------------------------------------
# leftover / uncapacitated parcels and the no-record fallback
# --------------------------------------------------------------------------

def _assign_leftover(unhandled, parcel_areas, seed_points, sp_ids, sp_to_parcel, centroids, transform, conflicts):
    """Parcels missing an area or a seed can't enter the transport solve (no
    demand, or no anchor to measure distance from) -- assign superpixels
    NOT already claimed by a capacitated parcel to whichever unhandled
    parcel (with a seed) is nearest in plain pixel distance. With no seed
    either, a parcel simply gets nothing; record why."""
    have_seed = [i for i in unhandled if seed_points[i] is not None]
    if not have_seed:
        if unhandled:
            conflicts.append(Conflict("no_record_for_parcels", {"parcel_indices": unhandled}))
        return
    unclaimed_sp = [sid for sid in sp_ids if sid not in sp_to_parcel]
    if not unclaimed_sp:
        if unhandled:
            conflicts.append(Conflict("no_leftover_area_for_parcels", {"parcel_indices": unhandled}))
        return
    seed_rc = {i: _world_to_pixel(seed_points[i], transform) for i in have_seed}
    got_any = {i: False for i in have_seed}
    for sid in unclaimed_sp:
        r, c = centroids[sid]
        best_i, best_d = None, None
        for i in have_seed:
            sr, sc = seed_rc[i]
            d = (r - sr) ** 2 + (c - sc) ** 2
            if best_d is None or d < best_d:
                best_d, best_i = d, i
        sp_to_parcel[sid] = best_i
        got_any[best_i] = True
    starved = [i for i in have_seed if not got_any[i]]
    if starved:
        conflicts.append(Conflict("leftover_parcel_got_no_area", {"parcel_indices": starved}))
    missing_no_seed = [i for i in unhandled if i not in have_seed]
    if missing_no_seed:
        conflicts.append(Conflict("no_record_for_parcels", {"parcel_indices": missing_no_seed}))


def _watershed_fallback(block_polygon, crs, seed_points, evidence_field, mask, transform, conflicts):
    """No usable (area, seed) records at all: pure evidence-driven watershed.
    Treats the evidence field as a landscape (a ridge = a likely boundary)
    and floods from whatever seeds exist; with no seeds either, the whole
    block is one unresolved region and that's recorded as a conflict, not
    guessed at (and left as an empty `parcel_polygons`, not a fabricated
    single "parcel" -- there's no parcel index to attach it to)."""
    from skimage.segmentation import watershed

    conflicts.append(Conflict("no_legacy_record_for_block", {}))
    seeded = [(i, p) for i, p in enumerate(seed_points) if p is not None]
    if not seeded:
        labels = np.where(mask, 0, -1).astype(np.int32)
        return AssignmentResult(labels, {}, transform, {}, conflicts)

    markers = np.zeros(mask.shape, dtype=np.int32)
    for i, p in seeded:
        row, col = _world_to_pixel(p, transform, shape=mask.shape)
        markers[row, col] = i + 1
    labels = watershed(evidence_field, markers=markers, mask=mask) - 1
    labels[~mask] = -1
    sp_to_parcel = {int(v): int(v) for v in np.unique(labels) if v >= 0}
    polys = _extract_parcel_polygons(labels, sp_to_parcel, transform, crs)
    return AssignmentResult(labels, sp_to_parcel, transform, polys, conflicts)


# --------------------------------------------------------------------------
# boundary extraction
# --------------------------------------------------------------------------

def _extract_parcel_polygons(labels: np.ndarray, sp_to_parcel: dict, transform, crs: str) -> dict:
    parcel_label = np.full(labels.shape, -1, dtype=np.int64)
    valid = labels >= 0
    if sp_to_parcel:
        max_sp = int(labels.max())
        lut = np.full(max_sp + 1, -1, dtype=np.int64)
        for sid, pid in sp_to_parcel.items():
            if 0 <= sid <= max_sp:
                lut[sid] = pid
        safe_labels = np.where(valid, labels, 0)
        parcel_label[valid] = lut[safe_labels][valid]

    polys_by_parcel: dict[int, list] = {}
    for geom, value in raster_shapes(parcel_label.astype(np.int32), mask=parcel_label >= 0, transform=transform):
        value = int(value)
        if value < 0:
            continue
        polys_by_parcel.setdefault(value, []).append(shapely_shape(geom))

    result = {}
    for pid, polys in polys_by_parcel.items():
        merged = unary_union(polys)
        result[pid] = Geom(merged, crs)
    return result


def parcels_to_graph(parcel_polygons: dict, block: Geom, grid: float = GRID) -> PlanarGraph:
    """The rest of the pipeline the module docstring promises: extracted
    parcel boundaries -> planarize() (grid-snapped, exact by construction
    against each other since they all come from one label raster) ->
    build_graph(). The block's own boundary is passed as `fixed` linework,
    same pattern as Stage 1's blocks.py, so it can't be dragged off its
    recorded position by a nearby parcel vertex.

    Can raise NotImplementedError (from build_graph's hole guard) if the
    leftover/reservoir mechanism in assign_parcels produced an "island"
    parcel fully enclosed by one neighbor -- full multi-ring face support is
    a real feature, not a one-line fix, and is out of Stage 3's scope (see
    Stage 1's own documented limitation); this re-raises with the parcel ids
    involved named, rather than build_graph's generic message, so the cause
    is diagnosable instead of just "some face has a hole."
    """
    block_polygon, crs = block.geom, block.crs
    linework = []
    for pid, geom in parcel_polygons.items():
        if geom.crs != crs:
            # found by review: this used to silently unwrap geom.geom
            # without ever checking geom's own declared CRS matched the
            # block's -- exactly the "silently coerce" pattern Geom/
            # CRSMismatchError exist to prevent everywhere else in this
            # project.
            raise CRSMismatchError(f"parcel {pid} is in {geom.crs!r}, expected {crs!r}")
        g = geom.geom
        parts = g.geoms if g.geom_type == "MultiPolygon" else [g]
        for part in parts:
            linework.append(Geom(part.boundary, crs))
    fixed = [Geom(block_polygon.boundary, crs)]
    faces = planarize(linework, grid=grid, fixed=fixed)
    ward_buffered = block_polygon.buffer(grid)
    inside = [f for f in faces if f.geom.area > grid * grid and ward_buffered.contains(f.geom.representative_point())]
    try:
        return build_graph(inside, crs, collinear_tol=grid)
    except NotImplementedError as e:
        raise NotImplementedError(
            f"{e} -- likely an 'island' parcel fully enclosed by one neighbor, probably from the "
            "leftover/reservoir mechanism in assign_parcels() scattering an uncapacitated parcel's "
            "superpixels; full multi-ring face support does not exist yet (see STAGE_1_NOTES.md)"
        ) from e
