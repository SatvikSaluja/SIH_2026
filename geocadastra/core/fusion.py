"""Fusion (Stage 5): combine what legacy records, the model, and GT survey
points each believe about where a boundary runs, and displace existing
graph nodes to match -- never build independent polygon copies per source.

Correspondence-free by construction, per the build plan's explicit warning
("Correspondence between differently-vertexed polygons is a trap; do not
try to solve it directly"): every source is queried locally, at an
existing node's current position, rather than matched globally vertex-to-
vertex against that node --

- legacy: nearest POINT on the legacy layer's boundary linework (never the
  nearest vertex, so an independently re-vertexed or merged legacy polygon
  is still usable).
- model: nearest low-SDF pixel in a small window -- literally a local,
  windowed version of "fuse in distance-field space and re-extract the
  zero set", since the model's own output already IS a distance field.
- gt: nearest GT survey point, but only within its capture radius --
  "near-certain within their capture radius and carry no information
  outside it" is modelled as *no estimate at all* beyond that radius,
  not as an estimate with a very large sigma.

Each source becomes a `SourceEstimate` (a position + its sigma); combining
them is then a plain inverse-variance-weighted fuse -- the closed-form
combination of independent Gaussian estimates, and exactly what makes "a
confident source dominates a vague one" and "two agreeing sources are more
confident than either alone" fall out for free rather than needing
separate rules for each. Where sources disagree beyond tolerance, this
does NOT fuse -- see `conflicts.py`.

Deliberately does not touch the database or the changeset: `fuse_node`/
`fuse_block` compute WHAT should move, in plain geometry/numpy, exactly
like `transport.py` and `planarize.py` before it. Applying the result is
the caller's job via `store.changeset.apply_fusion()`, which persists
each move in its own changeset -- topology, provenance, and (crucially)
one topologically-unsafe candidate never blocking every other, safe one
in the same block.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point
from shapely.ops import nearest_points

from geocadastra.core.conflicts import ConflictRecord, max_pairwise_disagreement
from geocadastra.core.crs import Geom
from geocadastra.core.graph import OUTER, PlanarGraph
from geocadastra.core.transport import _world_to_pixel

# doc: "Legacy GIS reliability is conditioned on the block's land-use
# classification" -- roughly tracking synth.generator's own legacy
# jitter+shift magnitude per style (informal moves ~4m total, formal ~1m).
DEFAULT_SIGMA_LEGACY_BY_STYLE = {"formal": 0.5, "informal": 3.0, "institutional": 1.0}  # metres
DEFAULT_SIGMA_GT = 0.05  # metres -- GT survey points are treated as near-exact
DEFAULT_CAPTURE_RADIUS = 3.0  # metres -- beyond this, a GT point "carries no information"
DEFAULT_TOLERANCE = 6.0  # metres -- disagreement beyond this is a conflict, not an average
DEFAULT_MODEL_SEARCH_RADIUS_PX = 8
DEFAULT_MODEL_MAX_SDF_PX = 4.0


@dataclass(frozen=True)
class SourceEstimate:
    name: str
    x: float
    y: float
    sigma: float  # metres, > 0


@dataclass(frozen=True)
class FusedPosition:
    x: float
    y: float
    sigma: float
    sources: tuple  # names of the estimates that went into this


@dataclass(frozen=True)
class FuseBlockResult:
    moved: dict  # node_id -> FusedPosition
    conflicts: list  # ConflictRecord
    unchanged: list  # node_id -- no source had any information here, left alone


def legacy_estimate(node_xy: tuple, legacy_boundary, style: str, sigma_by_style: dict = DEFAULT_SIGMA_LEGACY_BY_STYLE):
    """Nearest point on `legacy_boundary` (any shapely geometry -- a line,
    a union of many) to `node_xy`. None if there's no legacy linework at
    all to query."""
    if legacy_boundary is None or legacy_boundary.is_empty:
        return None
    _, nearest = nearest_points(Point(node_xy), legacy_boundary)
    return SourceEstimate("legacy", nearest.x, nearest.y, sigma_by_style[style])


def gt_estimate(node_xy: tuple, gt_points, capture_radius: float = DEFAULT_CAPTURE_RADIUS, sigma: float = DEFAULT_SIGMA_GT):
    """Nearest of `gt_points` (an iterable of (x, y)) to `node_xy`, but only
    if within `capture_radius` -- otherwise None, not a low-confidence
    estimate: a GT point genuinely has nothing to say that far away."""
    if not gt_points:
        return None
    pts = np.asarray(gt_points, dtype=float)
    d = np.hypot(pts[:, 0] - node_xy[0], pts[:, 1] - node_xy[1])
    i = int(np.argmin(d))
    if d[i] > capture_radius:
        return None
    return SourceEstimate("gt", float(pts[i, 0]), float(pts[i, 1]), sigma)


def model_estimate(
    node_xy: tuple,
    sdf: np.ndarray,
    log_var: np.ndarray,
    transform,
    search_radius_px: int = DEFAULT_MODEL_SEARCH_RADIUS_PX,
    max_sdf_px: float = DEFAULT_MODEL_MAX_SDF_PX,
):
    """The model's local belief near `node_xy`: within a small window,
    the pixel closest to a predicted boundary (min |sdf|), at that pixel's
    own predicted variance. None if even that closest pixel is farther
    from a boundary than `max_sdf_px` -- the model has no real opinion
    that near this node, which must not silently become "the boundary is
    right here"."""
    h, w = sdf.shape
    center_row, center_col = _world_to_pixel(node_xy, transform)
    row0 = max(center_row - search_radius_px, 0)
    row1 = min(center_row + search_radius_px + 1, h)
    col0 = max(center_col - search_radius_px, 0)
    col1 = min(center_col + search_radius_px + 1, w)
    window = sdf[row0:row1, col0:col1]
    if window.size == 0:
        return None
    local_idx = np.unravel_index(np.argmin(np.abs(window)), window.shape)
    row, col = row0 + local_idx[0], col0 + local_idx[1]
    if abs(sdf[row, col]) > max_sdf_px:
        return None
    x, y = transform @ (col + 0.5, row + 0.5)  # pixel centre, world coords
    sigma = float(np.sqrt(np.exp(log_var[row, col])))
    return SourceEstimate("model", x, y, sigma)


def fuse_estimates(estimates: list) -> FusedPosition:
    """Inverse-variance-weighted combination -- the closed-form fuse of
    independent Gaussian position estimates. A single estimate passes
    through unchanged (nothing to combine it with)."""
    if not estimates:
        raise ValueError("fuse_estimates() needs at least one SourceEstimate")
    weights = np.array([1.0 / e.sigma**2 for e in estimates])
    total_w = weights.sum()
    x = sum(w * e.x for w, e in zip(weights, estimates)) / total_w
    y = sum(w * e.y for w, e in zip(weights, estimates)) / total_w
    sigma = float(np.sqrt(1.0 / total_w))
    return FusedPosition(x=x, y=y, sigma=sigma, sources=tuple(e.name for e in estimates))


def fuse_node(
    node_xy: tuple,
    *,
    face_ids: tuple = (),
    node_id: int | None = None,
    legacy_boundary=None,
    style: str | None = None,
    sdf: np.ndarray | None = None,
    log_var: np.ndarray | None = None,
    transform=None,
    gt_points=None,
    tolerance: float = DEFAULT_TOLERANCE,
    capture_radius: float = DEFAULT_CAPTURE_RADIUS,
    sigma_legacy_by_style: dict = DEFAULT_SIGMA_LEGACY_BY_STYLE,
    sigma_gt: float = DEFAULT_SIGMA_GT,
    crs: str | None = None,
):
    """One node's whole fusion pipeline: gather whichever source estimates
    apply, and either fuse them (>=1 estimate, all agreeing) or flag a
    conflict (>=2 estimates, disagreeing beyond `tolerance`).

    Returns `None` if no source had any information here (a true no-op --
    not the same as "fuse to the current position", since there was
    nothing to fuse), a `FusedPosition` if safe to move the node to, or a
    `ConflictRecord` if sources disagree too much to trust an average.
    """
    estimates = []
    if legacy_boundary is not None and style is not None:
        est = legacy_estimate(node_xy, legacy_boundary, style, sigma_legacy_by_style)
        if est is not None:
            estimates.append(est)
    if sdf is not None and log_var is not None and transform is not None:
        est = model_estimate(node_xy, sdf, log_var, transform)
        if est is not None:
            estimates.append(est)
    if gt_points:
        est = gt_estimate(node_xy, gt_points, capture_radius, sigma_gt)
        if est is not None:
            estimates.append(est)

    if not estimates:
        return None

    disagreement = max_pairwise_disagreement(estimates)
    if len(estimates) >= 2 and disagreement > tolerance:
        if crs is None:
            # invariant: never let a geometry cross a module boundary
            # without a declared CRS -- a conflict record's `geometry`
            # would do exactly that, so this is a caller error, not
            # something to silently paper over with a fake CRS string.
            raise ValueError("fuse_node(): a conflict was detected but no crs was given for its ConflictRecord.geometry")
        return ConflictRecord(
            node_id=node_id,
            face_ids=tuple(face_ids),
            sources=tuple(e.name for e in estimates),
            disagreement_m=disagreement,
            geometry=Geom(Point(node_xy), crs),
        )
    return fuse_estimates(estimates)


def _incident_face_ids(graph: PlanarGraph, node_id: int) -> tuple:
    faces: set = set()
    for edge in graph.edges.values():
        if edge.n0 == node_id or edge.n1 == node_id:
            faces.update(fid for fid in graph.faces_of_edge(edge.id) if fid != OUTER)
    return tuple(sorted(faces))


def _is_exterior_node(graph: PlanarGraph, node_id: int) -> bool:
    """True if this node sits on the block's own OUTER boundary (a road
    edge) at all, as opposed to a purely interior parcel-to-parcel
    boundary.

    Deliberately all-or-nothing -- an earlier version tried to let a
    T-junction (where an interior cut meets an otherwise-straight run of
    the perimeter) slide along its own fixed line, since this project's
    own recursive/organic ("informal") subdivision generator makes almost
    every interior edge's endpoints land back on the exterior ring too
    (blanket exclusion made fusion a near no-op for the majority block
    style). That needed a tolerance separating "T-junction with ordinary
    legacy-scale noise on its current position" from "a genuine corner
    where the perimeter changes direction" -- and on this project's own
    data, those two things overlap in magnitude (informal blocks have
    real corners as gentle as ~2.5m of perpendicular deviation, well
    inside the ~9m noise a 3-sigma legacy tolerance has to tolerate), so
    no single threshold safely tells them apart. Found by review via the
    Stage 5 topology-invariant acceptance test: a too-generous tolerance
    let a genuine corner slide, drifting total block area by ~100m^2
    without tripping the overlap or self-intersection guards at all.
    Per this project's own rule ("prefer deleting a feature to shipping
    it untested"), reverted to the simple, provably-safe version: an
    exterior node never moves. Fusion's practical reach on a block whose
    every corner touches its own perimeter is a real, honest scope limit,
    not a bug -- see STAGE_5_NOTES.md.
    """
    return any(
        OUTER in graph.faces_of_edge(edge.id)
        for edge in graph.edges.values()
        if edge.n0 == node_id or edge.n1 == node_id
    )


def fuse_block(
    graph: PlanarGraph,
    node_ids,
    *,
    style: str,
    legacy_boundary=None,
    sdf: np.ndarray | None = None,
    log_var: np.ndarray | None = None,
    transform=None,
    gt_points=None,
    tolerance: float = DEFAULT_TOLERANCE,
    capture_radius: float = DEFAULT_CAPTURE_RADIUS,
    sigma_legacy_by_style: dict = DEFAULT_SIGMA_LEGACY_BY_STYLE,
    sigma_gt: float = DEFAULT_SIGMA_GT,
) -> FuseBlockResult:
    """Fuse every node in `node_ids` (one block's worth -- "blocks are the
    unit of work"), against a single shared set of sources. Doesn't move
    anything itself; the caller applies `.moved` via
    `store.changeset.apply_fusion()`, which persists each move in its own
    changeset (so one topologically-unsafe candidate can't block every
    other, safe one in the same block) with correct provenance.

    A node on the block's own OUTER (road) boundary is never fused, no
    matter what's in `node_ids` -- silently routed to `.unchanged`, same
    bucket as "no source had any information here". A block's exterior
    comes from the road network, a fixed, more authoritative reference
    than any parcel-level evidence source here; `transport.py`'s own
    `parcels_to_graph()` sets the same precedent (the block boundary is
    passed to `planarize()` as authoritative `fixed` linework). See
    `_is_exterior_node()` for why this is all-or-nothing rather than
    letting a boundary-adjacent node slide along its own fixed edge.
    """
    moved: dict = {}
    conflicts: list = []
    unchanged: list = []
    for node_id in node_ids:
        if _is_exterior_node(graph, node_id):
            unchanged.append(node_id)
            continue
        node = graph.nodes[node_id]
        result = fuse_node(
            (node.x, node.y),
            face_ids=_incident_face_ids(graph, node_id),
            node_id=node_id,
            legacy_boundary=legacy_boundary,
            style=style,
            sdf=sdf,
            log_var=log_var,
            transform=transform,
            gt_points=gt_points,
            tolerance=tolerance,
            capture_radius=capture_radius,
            sigma_legacy_by_style=sigma_legacy_by_style,
            sigma_gt=sigma_gt,
            crs=graph.crs,
        )
        if result is None:
            unchanged.append(node_id)
        elif isinstance(result, ConflictRecord):
            conflicts.append(result)
        else:
            moved[node_id] = result
    return FuseBlockResult(moved=moved, conflicts=conflicts, unchanged=unchanged)
