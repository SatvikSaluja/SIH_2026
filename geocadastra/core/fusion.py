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

A node on the block's own OUTER boundary needs its own handling
(`_is_exterior_node`, `_project_onto_block_boundary`,
`_ring_neighbor_bounds`): fusing it toward a raw evidence-based candidate
could silently change the block's own enclosed area, which none of the
INTERNAL-boundary evidence sources above are entitled to do. Given
`block_boundary` (the block's own TRUE, authoritative exterior, e.g. from
`build_blocks()`), an exterior candidate is projected exactly onto it
instead -- correct and tested (see test_fusion.py), though real synthetic
data can still trip a separate, pre-existing Stage 0 precision issue this
module works around but does not (and cannot) fully fix -- see
STAGE_5_NOTES.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from shapely.geometry import LinearRing, Point
from shapely.ops import nearest_points

from geocadastra.core.conflicts import ConflictRecord, max_pairwise_disagreement
from geocadastra.core.crs import CRSMismatchError, Geom
from geocadastra.core.graph import OUTER, PlanarGraph
from geocadastra.core.transport import _world_to_pixel

# doc: "Legacy GIS reliability is conditioned on the block's land-use
# classification" -- roughly tracking synth.generator's own legacy
# jitter+shift magnitude per style (informal moves ~4m total, formal ~1m).
#
# MappingProxyType (read-only), not a plain dict: this is bound directly as
# a default argument on three functions below (the classic mutable-default
# pitfall -- found by review). A caller who obtains a reference to it (e.g.
# `sigmas = sigma_legacy_by_style or DEFAULT_SIGMA_LEGACY_BY_STYLE` then
# mutates `sigmas`) would otherwise silently corrupt every later call's
# default for the rest of the process. The proxy makes that raise
# immediately instead.
DEFAULT_SIGMA_LEGACY_BY_STYLE = MappingProxyType({"formal": 0.5, "informal": 3.0, "institutional": 1.0})  # metres
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
    # len(), not `not gt_points` -- plain truthiness on a numpy array of
    # 2+ points raises ValueError ("truth value... is ambiguous"), crashing
    # this documented-as-"an iterable of (x, y)" parameter for a caller who
    # passes an array instead of a list (found by review, reproduced).
    if gt_points is None or len(gt_points) == 0:
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
    # `shape=(h, w)` clamps center_row/center_col into [0, h-1]/[0, w-1] --
    # without it, a node far enough outside the raster gives a small
    # NEGATIVE row1/col1 below (center + radius + 1, ceiling-clamped to h/w
    # but never floor-clamped), and Python's negative-index slicing then
    # silently wraps sdf[row0:row1] to the array's own tail instead of
    # yielding an empty window -- found by review, reproduced directly: a
    # node 10px above a raster's top edge returned a confidently wrong
    # SourceEstimate from an unrelated row instead of the documented None.
    # transport.py's own _world_to_pixel docstring already documents this
    # exact bug class being found and fixed elsewhere by clamping both
    # ends; this call site just never adopted that fix.
    center_row, center_col = _world_to_pixel(node_xy, transform, shape=(h, w))
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
    if gt_points is not None and len(gt_points) > 0:
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

    An exterior node is never moved to a raw fused position: doing so
    could change the block's own enclosed area, since a node here can be
    either a T-junction on an otherwise-straight run of the perimeter
    (safe to slide ALONG it) or a genuine corner where the perimeter
    changes direction (must not move at all) -- and telling those apart
    from the node's own (possibly noisy) current position alone doesn't
    work on this project's real data: informal blocks have genuine
    corners as gentle as ~2.5m of perpendicular deviation, overlapping
    the ~9m of ordinary legacy-scale noise a 3-sigma tolerance has to
    accept, so no fixed threshold safely separates them (confirmed: a
    too-generous tolerance let a real corner drift, silently changing
    total block area by ~100m^2 with neither the overlap nor
    self-intersection guard tripping).

    `fuse_block`'s `block_boundary` parameter is the real fix, not a
    local inference: given the block's own TRUE, authoritative exterior
    (from `build_blocks()` -- the road network, not derived from noisy
    parcel-level data), `_project_onto_block_boundary()` can tell a
    T-junction from a corner exactly (nearest point on a KNOWN ring,
    compared against that ring's own KNOWN vertices) rather than
    statistically. Without `block_boundary`, this function's boolean
    answer is used directly and an exterior node never moves at all --
    the simple, provably-safe fallback per this project's own rule
    ("prefer deleting a feature to shipping it untested"), still correct,
    just less capable.
    """
    return any(
        OUTER in graph.faces_of_edge(edge.id)
        for edge in graph.edges.values()
        if edge.n0 == node_id or edge.n1 == node_id
    )


def _boundary_ring(block_boundary, crs: str) -> LinearRing:
    """Accept a `Geom`, a shapely Polygon, or an already-built
    LinearRing/LineString for the block's own true exterior, and return a
    LinearRing -- one normalized shape callers don't each have to handle.

    If `block_boundary` is a `Geom`, its own declared `.crs` MUST match
    `crs` (the graph's own CRS) -- found by review: this used to silently
    unwrap a `Geom` without ever comparing its CRS, exactly the "silently
    coerce" pattern `Geom`/`CRSMismatchError` exist to rule out elsewhere
    in this project. Reproduced directly: a `block_boundary` Geom tagged
    EPSG:4326 (lon/lat degrees) against a graph in EPSG:32643 (UTM
    metres) was silently accepted with no error, producing a ring with
    numerically-plausible-looking but wrong-coordinate-system numbers.
    """
    if isinstance(block_boundary, Geom):
        if block_boundary.crs != crs:
            raise CRSMismatchError(f"block_boundary is in {block_boundary.crs!r}, expected {crs!r}")
        geom = block_boundary.geom
    else:
        geom = block_boundary
    if geom.geom_type == "Polygon":
        return LinearRing(geom.exterior.coords)
    if geom.geom_type == "LinearRing":
        return geom
    return LinearRing(geom.coords)  # a LineString ring (e.g. .boundary of a Polygon)


def _project_onto_block_boundary(
    ring: LinearRing,
    x: float,
    y: float,
    dist_lo: float | None = None,
    dist_hi: float | None = None,
    vertex_snap_tol: float = 1.0,
) -> tuple[float, float]:
    """Project `(x, y)` onto `ring` -- shapely's own `project()`/
    `interpolate()`, not hand-rolled point-to-segment math (this
    project's own rule: "do not hand-roll floating-point node snapping").

    If the projected point falls within `vertex_snap_tol` of one of the
    ring's own vertices, snaps EXACTLY to that vertex instead of the
    literal projection -- a true corner, known exactly from the
    authoritative boundary, so this is a plain coordinate comparison
    against known points, not a statistical inference from noisy data
    (contrast `_is_exterior_node`'s docstring on why that doesn't work).

    `dist_lo`/`dist_hi` (from `_ring_neighbor_bounds` -- NOT necessarily
    within `ring`'s own native `[0, ring.length)` range; see there for
    why) clamp the result so it can't slide past a given bound -- one
    node's fused candidate must not be able to jump past an adjacent
    one, folding the ring's own walk order without ever tripping the
    overlap or self-intersection guards (found by review: an earlier,
    unclamped version of a similar projection let exactly this happen,
    silently changing total block area).
    """
    length = ring.length
    dist = ring.project(Point(x, y))
    clamped = dist_lo is not None and dist_hi is not None
    if clamped:
        # `dist` is shapely's own [0, length) parametrization, but the
        # valid [dist_lo, dist_hi] range may already have been shifted
        # outside that (see _ring_neighbor_bounds) to correctly represent
        # an arc that wraps the ring's own coordinate-list seam. Re-express
        # `dist` as whichever of its (infinitely many, length-periodic)
        # equivalent values sits nearest the middle of that range, clamp
        # in that shared numbering, then wrap the final answer back into
        # [0, length) for interpolate().
        center = (dist_lo + dist_hi) / 2
        dist += length * round((center - dist) / length)
        dist = min(max(dist, dist_lo), dist_hi)
        dist %= length
    proj = ring.interpolate(dist)
    vertices = list(ring.coords)
    if clamped:
        # Only snap to a vertex that's actually reachable WITHIN the
        # clamped arc -- found by review: without this, a vertex on a
        # topologically-distant part of the ring that happens to sit
        # close in plain 2D space (plausible on a concave/complex
        # boundary, e.g. a narrow near-self-touching feature) could still
        # be picked, silently jumping this node past its neighbor even
        # though `dist` itself was correctly clamped above -- reopening
        # the exact "node jumps past its neighbor, folding the ring's
        # walk order" failure mode the clamp exists to prevent.
        candidates = []
        for vx, vy in vertices:
            vdist = ring.project(Point(vx, vy))
            vdist += length * round((center - vdist) / length)
            if dist_lo <= vdist <= dist_hi:
                candidates.append((vx, vy))
    else:
        candidates = vertices
    if not candidates:
        return proj.x, proj.y
    nearest_vertex = min(candidates, key=lambda c: (c[0] - proj.x) ** 2 + (c[1] - proj.y) ** 2)
    vertex_dist = ((nearest_vertex[0] - proj.x) ** 2 + (nearest_vertex[1] - proj.y) ** 2) ** 0.5
    if vertex_dist <= vertex_snap_tol:
        return nearest_vertex[0], nearest_vertex[1]
    return proj.x, proj.y


def _ring_neighbor_bounds(graph: PlanarGraph, node_id: int, ring: LinearRing):
    """This node's two OUTER-adjacent ring-neighbors' own current
    positions, projected onto `ring` -- the `dist_lo, dist_hi` bound
    `_project_onto_block_boundary` needs. `None, None` if this node
    doesn't have exactly two OUTER-adjacent neighbors (not a simple ring
    point; caller should treat that as "don't move, don't risk it").

    Correctly handles the ring's own coordinate-list seam (the arbitrary
    point where its `[0, ring.length)` parametrization wraps from the
    last vertex back to the first) -- a plain `sorted(dist_a, dist_b)`
    gives the WRONG (and silently, not safely, wrong: the result still
    lands validly on the ring, so no downstream guard catches it) arc
    whenever a neighbor happens to be near that seam, which is common,
    not rare (found by review reproducing it in the very first hand-built
    test case tried). Fixed with the standard trick for clamping on a
    circle: re-express each neighbor's distance as a SIGNED offset from
    this node's own current position, choosing the shorter of the two
    ways around the ring -- which is unambiguous as long as the node's
    own two immediate neighbors are (as they always are here) much less
    than half the ring's total length apart, true for any block with more
    than a couple of exterior points.
    """
    outer_neighbors = []
    for e in graph.edges.values():
        if OUTER not in graph.faces_of_edge(e.id):
            continue
        other = e.n1 if e.n0 == node_id else (e.n0 if e.n1 == node_id else None)
        if other is not None:
            outer_neighbors.append(other)
    if len(outer_neighbors) != 2:
        return None, None
    node = graph.nodes[node_id]
    a, b = graph.nodes[outer_neighbors[0]], graph.nodes[outer_neighbors[1]]
    length = ring.length
    dist_node = ring.project(Point(node.x, node.y))

    def _signed_offset(dist_other: float) -> float:
        return (dist_other - dist_node + length / 2) % length - length / 2

    offset_a = _signed_offset(ring.project(Point(a.x, a.y)))
    offset_b = _signed_offset(ring.project(Point(b.x, b.y)))
    lo, hi = min(offset_a, offset_b), max(offset_a, offset_b)
    return dist_node + lo, dist_node + hi


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
    block_boundary=None,
    vertex_snap_tol: float = 1.0,
    max_boundary_distance: float = 5.0,
) -> FuseBlockResult:
    """Fuse every node in `node_ids` (one block's worth -- "blocks are the
    unit of work"), against a single shared set of sources. Doesn't move
    anything itself; the caller applies `.moved` via
    `store.changeset.apply_fusion()`, which persists each move in its own
    changeset (so one topologically-unsafe candidate can't block every
    other, safe one in the same block) with correct provenance.

    A node on the block's own OUTER (road) boundary is never moved to a
    raw fused position, since that could change the block's own enclosed
    area (a fixed, more authoritative reference than any parcel-level
    evidence source here -- `transport.py`'s own `parcels_to_graph()`
    sets the same precedent, passing the block boundary to `planarize()`
    as authoritative `fixed` linework).

    `block_boundary` (a `Geom`/Polygon/ring -- e.g. straight from
    `build_blocks()`'s own output for this block) is what makes an
    exterior node's fusion actually work correctly rather than just
    safely: its candidate is projected exactly onto that TRUE boundary
    (`_project_onto_block_boundary`) -- sliding along a straight run if
    it's a T-junction, or snapping to the exact known position if it's a
    real corner -- so area is preserved by construction, not by refusing
    to move the node at all. Without `block_boundary`, an exterior node
    is simply left unchanged (routed to `.unchanged`), the earlier,
    simpler, still-correct-but-more-limited behavior (see
    `_is_exterior_node`'s docstring for why inferring corner-vs-T-junction
    from the node's own current position doesn't work on this project's
    data, which is why this needs the true boundary rather than a
    tolerance).

    `max_boundary_distance` is a sanity check on `block_boundary` itself:
    an exterior node farther than this from the given boundary is left
    unchanged rather than projected -- found by review on REAL synthetic
    data (not a hand-built test): this project's recursive-split parcel
    generator can leave a near-zero-area but real-extent internal gap
    between two parcels that should be exactly adjacent (a GEOS precision
    artifact, not a Stage 5 concern -- see STAGE_5_NOTES.md), which
    `build_graph()` has no way to distinguish from the block's own true
    exterior: both look identical (an edge with `OUTER` as one face-side)
    from the graph alone. A node on such a gap is nowhere near the real
    perimeter (measured up to ~16m away on one real block), so blindly
    projecting it onto `block_boundary` produces a confidently wrong
    answer -- silently valid-looking (the point IS genuinely on the ring)
    but geometrically nonsensical. This check is what makes that fail
    safe (left alone) instead of failing wrong.
    """
    ring = _boundary_ring(block_boundary, graph.crs) if block_boundary is not None else None
    moved: dict = {}
    conflicts: list = []
    unchanged: list = []
    for node_id in node_ids:
        is_exterior = _is_exterior_node(graph, node_id)
        node = graph.nodes[node_id]
        if is_exterior:
            if ring is None or ring.distance(Point(node.x, node.y)) > max_boundary_distance:
                unchanged.append(node_id)
                continue
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
        elif is_exterior:
            dist_lo, dist_hi = _ring_neighbor_bounds(graph, node_id, ring)
            if dist_lo is None:
                unchanged.append(node_id)  # not a simple ring point -- don't risk it
                continue
            px, py = _project_onto_block_boundary(
                ring, result.x, result.y, dist_lo, dist_hi, vertex_snap_tol=vertex_snap_tol
            )
            moved[node_id] = FusedPosition(x=px, y=py, sigma=result.sigma, sources=result.sources)
        else:
            moved[node_id] = result
    return FuseBlockResult(moved=moved, conflicts=conflicts, unchanged=unchanged)
