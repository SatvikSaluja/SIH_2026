"""Procedural synthetic ward generator with perfect ground truth (Stage 0).

roads -> blocks -> parcels -> buildings, plus a rendered fake ortho/DSM/DTM,
a distorted "legacy GIS" layer, and a sparse set of exact GT survey points.

Every random draw goes through one `numpy.random.Generator` created from
`seed`, threaded through the call tree in a fixed order, so a given
(params, seed) pair reproduces byte-identical output. Do not use the
`random` module or any other source of randomness in here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import shapely
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter
from shapely import affinity
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, box
from shapely.ops import split, unary_union, polygonize

CRS = "EPSG:32643"  # synthetic local UTM zone; a label only, no real datum work in Stage 0
EPS = 1e-6  # area/length tolerance for "exact" tiling assertions
GRID = 1e-9  # precision grid (metres) for shapely.set_precision after chained boolean ops.
# Without an explicit grid, GEOS's overlay engine can give a *wrong* answer --
# not just imprecise -- for a pair of thin/near-degenerate polygons several
# split() generations deep: found via property testing, a real (seed=660)
# recursively-split parcel pair whose true overlap is ~0 came back from
# .intersection() as 209 m^2 (100% of the smaller parcel), verified wrong by
# hand (ray-casting). A grid this fine is well below any GEOS/float64 rounding
# already present in these coordinates, so it fixes the overlay robustness
# bug (checked down to 1e-11 relative area error over 200 wards) without
# adding measurable quantization of its own.


@dataclass
class WardParams:
    width: float = 240.0  # ward extent, metres
    height: float = 180.0
    gsd: float = 0.5  # ortho/DSM/DTM pixel size, metres

    n_arterial_h: int = 1
    n_arterial_v: int = 1
    arterial_width: float = 12.0
    minor_spacing: float = 55.0
    minor_spacing_jitter: float = 12.0
    minor_width: float = 6.0

    style_weights: dict = field(
        default_factory=lambda: {"formal": 0.5, "informal": 0.35, "institutional": 0.15}
    )

    strip_count_range: tuple = (3, 8)  # formal: parcels per block
    informal_min_area: float = 90.0  # informal: stop recursing below this
    informal_max_depth: int = 5
    institutional_single_prob: float = 0.6  # institutional: chance of one plot, else one cut

    setback: float = 2.0
    building_coverage: tuple = (0.25, 0.55)  # fraction of buildable envelope area
    min_building_area: float = 12.0
    crossing_prob_informal: float = 0.15  # chance a building deliberately crosses a parcel line

    visible_edge_fraction: float = 0.5
    n_veg_blobs: int = 25
    veg_radius_range: tuple = (2.0, 8.0)
    building_height_range: tuple = (3.0, 12.0)
    veg_height_range: tuple = (2.0, 6.0)
    dtm_base_slope: float = 0.02  # metres of rise per metre, gentle grade
    dtm_relief_amplitude: float = 1.5  # metres of smooth terrain noise
    wall_render_width: float = 0.3  # metres, thickness of a drawn boundary wall

    # legacy GIS distortion, keyed by block style
    legacy_jitter_std: dict = field(
        default_factory=lambda: {"formal": 0.3, "informal": 1.5, "institutional": 0.5}
    )
    legacy_shift_range: dict = field(
        default_factory=lambda: {"formal": 0.5, "informal": 2.5, "institutional": 1.0}
    )
    legacy_p_missing: dict = field(
        default_factory=lambda: {"formal": 0.02, "informal": 0.1, "institutional": 0.0}
    )
    legacy_p_merge: dict = field(
        default_factory=lambda: {"formal": 0.03, "informal": 0.15, "institutional": 0.0}
    )

    gt_point_fraction: float = 0.08  # fraction of parcel corners sampled as exact GT points

    def __post_init__(self):
        lo, hi = self.strip_count_range
        if lo > hi:
            raise ValueError(f"strip_count_range must be (low, high) with low <= high, got {self.strip_count_range}")
        if sum(self.style_weights.values()) <= 0:
            raise ValueError(f"style_weights must sum to a positive number, got {self.style_weights}")
        for name in ("legacy_jitter_std", "legacy_shift_range", "legacy_p_missing", "legacy_p_merge"):
            missing = set(self.style_weights) - set(getattr(self, name))
            if missing:
                raise ValueError(f"{name} has no entry for style(s) {missing} present in style_weights")


@dataclass
class Block:
    id: int
    polygon: Polygon
    style: str


@dataclass
class Parcel:
    id: int
    block_id: int
    polygon: Polygon
    area: float
    style: str


@dataclass
class Building:
    id: int
    polygon: Polygon
    parcel_ids: tuple
    height: float


@dataclass
class LegacyParcel:
    id: int
    polygon: Polygon | MultiPolygon  # a merge of two independently-shifted neighbors can miss touching
    source_parcel_ids: tuple


@dataclass
class GTPoint:
    x: float
    y: float
    parcel_id: int


@dataclass
class SyntheticWard:
    params: WardParams
    seed: int
    crs: str
    ward_polygon: Polygon
    blocks: list
    parcels: list
    edges_geom: dict  # edge_key -> LineString; key is (pid_a, pid_b) sorted, or (pid, -1) for exterior
    edges_visible: dict  # edge_key -> bool
    roads_centerline: list
    buildings: list
    legacy_parcels: list
    gt_points: list
    ortho: np.ndarray  # (H, W, 3) uint8
    dsm: np.ndarray  # (H, W) float32
    dtm: np.ndarray  # (H, W) float32
    transform: object  # affine.Affine shared by ortho/dsm/dtm


def generate_ward(params: WardParams | None = None, seed: int = 0) -> SyntheticWard:
    params = params or WardParams()
    rng = np.random.default_rng(seed)

    ward_poly = box(0, 0, params.width, params.height)
    road_lines, road_render_polys = _make_roads(ward_poly, params, rng)
    block_polys = _polygonize_blocks(ward_poly, road_lines)
    blocks = [Block(i, poly, str(_pick_style(rng, params))) for i, poly in enumerate(block_polys)]

    parcels = []
    next_pid = 0
    for blk in blocks:
        for piece in _subdivide_block(blk.polygon, blk.style, params, rng):
            parcels.append(Parcel(next_pid, blk.id, piece, piece.area, blk.style))
            next_pid += 1

    edges_geom, edges_visible = _boundary_edges(blocks, parcels, params, rng)
    buildings = _place_buildings(blocks, parcels, params, rng)

    adjacency_pairs = [k for k in edges_geom if k[1] != -1]
    legacy_parcels = _make_legacy_layer(parcels, adjacency_pairs, params, rng)
    gt_points = _make_gt_points(parcels, params, rng)

    road_render_union = unary_union(road_render_polys) if road_render_polys else Polygon()
    ortho, dsm, dtm, transform = _render_rasters(
        ward_poly, parcels, road_render_union, buildings, edges_geom, edges_visible, params, rng
    )

    return SyntheticWard(
        params=params,
        seed=seed,
        crs=CRS,
        ward_polygon=ward_poly,
        blocks=blocks,
        parcels=parcels,
        edges_geom=edges_geom,
        edges_visible=edges_visible,
        roads_centerline=road_lines,
        buildings=buildings,
        legacy_parcels=legacy_parcels,
        gt_points=gt_points,
        ortho=ortho,
        dsm=dsm,
        dtm=dtm,
        transform=transform,
    )


# --------------------------------------------------------------------------
# roads and blocks
# --------------------------------------------------------------------------

def _spaced_positions(extent, n, rng):
    if n <= 0:
        return []
    base = extent / (n + 1)
    positions = []
    for i in range(1, n + 1):
        jitter = rng.uniform(-base * 0.15, base * 0.15)
        positions.append(min(max(base * i + jitter, base * 0.3), extent - base * 0.3))
    return sorted(positions)


def _minor_grid(width, height, params, rng, exclude_y, exclude_x):
    def gen_axis(extent, exclude):
        positions = []
        spacing = params.minor_spacing
        pos = spacing * rng.uniform(0.5, 1.0)
        while pos < extent - spacing * 0.3:
            p = pos + rng.uniform(-params.minor_spacing_jitter, params.minor_spacing_jitter)
            if 1.0 < p < extent - 1.0 and all(abs(p - e) > 3.0 for e in exclude):
                positions.append(p)
            pos += spacing
        return positions

    lines = []
    for y in gen_axis(height, exclude_y):
        lines.append(LineString([(0, y), (width, y)]))
    for x in gen_axis(width, exclude_x):
        lines.append(LineString([(x, 0), (x, height)]))
    return lines


def _make_roads(ward_poly, params, rng):
    width, height = params.width, params.height
    h_pos = _spaced_positions(height, params.n_arterial_h, rng)
    v_pos = _spaced_positions(width, params.n_arterial_v, rng)

    lines = [LineString([(0, y), (width, y)]) for y in h_pos]
    lines += [LineString([(x, 0), (x, height)]) for x in v_pos]
    render_polys = [ln.buffer(params.arterial_width / 2, cap_style="flat") for ln in lines]

    minor_lines = _minor_grid(width, height, params, rng, h_pos, v_pos)
    render_polys += [ln.buffer(params.minor_width / 2, cap_style="flat") for ln in minor_lines]
    lines += minor_lines

    render_polys = [p.intersection(ward_poly) for p in render_polys]
    return lines, render_polys


def _polygonize_blocks(ward_poly, road_lines):
    network = unary_union(list(road_lines) + [ward_poly.boundary])
    faces = list(polygonize(network))
    # area filter matters: a degenerate/collinear face makes minimum_rotated_rectangle
    # return a LineString (no .exterior) and would crash _strip_split downstream
    ward_buffered = ward_poly.buffer(EPS)
    return [f for f in faces if f.area > EPS and ward_buffered.contains(f.representative_point())]


def _pick_style(rng, params):
    styles = list(params.style_weights.keys())
    weights = np.array([params.style_weights[s] for s in styles], dtype=float)
    weights /= weights.sum()
    return rng.choice(styles, p=weights)


# --------------------------------------------------------------------------
# parcel subdivision
# --------------------------------------------------------------------------

def _to_polygonal(geom):
    return geom.geom_type in ("Polygon", "MultiPolygon") and geom.area > EPS


def _strip_split(poly, n, rng):
    """Cut `poly` into n strips perpendicular to its longest oriented-bbox edge.

    Exact by construction: strips are poly intersected with a partition of
    the rotated bounding-box x-range into n contiguous, non-overlapping
    slabs, so their union is poly and pairwise overlap is zero-area.
    """
    if n <= 1:
        return [poly]
    mrr = poly.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)  # 5 pts, closed ring
    edge_lens = [Point(coords[i]).distance(Point(coords[i + 1])) for i in range(4)]
    long_edge = int(np.argmax(edge_lens))
    p0, p1 = coords[long_edge], coords[long_edge + 1]
    angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    center = poly.centroid

    poly_r = affinity.rotate(poly, -angle, origin=center)
    minx, miny, maxx, maxy = poly_r.bounds
    span = max(maxx - minx, EPS)

    raw = np.sort(rng.uniform(0.08, 0.92, size=n - 1))
    fracs = np.clip(np.diff(np.concatenate(([0.0], raw, [1.0]))), 0.05, None)
    fracs = fracs / fracs.sum()
    cuts = minx + np.cumsum(fracs) * span
    xs = [minx - 1.0] + list(cuts[:-1]) + [maxx + 1.0]

    pad = max(maxy - miny, 1.0)
    strips = []
    for i in range(len(xs) - 1):
        slab = box(xs[i], miny - pad, xs[i + 1], maxy + pad)
        piece = shapely.set_precision(poly_r.intersection(slab), GRID)
        if _to_polygonal(piece):
            strips.append(affinity.rotate(piece, angle, origin=center))
    return strips


def _recursive_split(poly, rng, min_area, max_depth, depth=0, budget=None):
    """Random guillotine cut, recursed. Exact by construction: shapely's
    split() always partitions a polygon into pieces whose union is the
    input, so this tiles exactly at every depth.

    `budget` caps total split operations for one top-level call, independent
    of max_depth/min_area: those two are user knobs and a small min_area
    paired with a large max_depth is an easy way to ask for up to 2**max_depth
    pieces by accident. Past the cap, remaining pieces are returned unsplit
    (still an exact, just coarser, tiling) instead of continuing to recurse.
    """
    if budget is None:
        budget = [4096]
    if depth >= max_depth or poly.area <= min_area * 1.6 or budget[0] <= 0:
        return [poly]
    minx, miny, maxx, maxy = poly.bounds
    diag = ((maxx - minx) ** 2 + (maxy - miny) ** 2) ** 0.5
    cx = rng.uniform(minx + (maxx - minx) * 0.3, minx + (maxx - minx) * 0.7)
    cy = rng.uniform(miny + (maxy - miny) * 0.3, miny + (maxy - miny) * 0.7)
    angle = rng.uniform(0, np.pi)
    dx, dy = np.cos(angle) * diag, np.sin(angle) * diag
    cut = LineString([(cx - dx, cy - dy), (cx + dx, cy + dy)])
    try:
        pieces = [shapely.set_precision(g, GRID) for g in split(poly, cut).geoms]
        pieces = [g for g in pieces if _to_polygonal(g)]
    except Exception:
        pieces = []
    if len(pieces) < 2:
        return [poly]
    budget[0] -= 1
    out = []
    for piece in pieces:
        out.extend(_recursive_split(piece, rng, min_area, max_depth, depth + 1, budget))
    return out


def _subdivide_block(poly, style, params, rng):
    if style == "formal":
        n = int(rng.integers(params.strip_count_range[0], params.strip_count_range[1] + 1))
        return _strip_split(poly, n, rng)
    if style == "institutional":
        if rng.random() < params.institutional_single_prob:
            return [poly]
        return _recursive_split(poly, rng, min_area=poly.area / 2.2, max_depth=1)
    return _recursive_split(poly, rng, min_area=params.informal_min_area, max_depth=params.informal_max_depth)


# --------------------------------------------------------------------------
# boundary edges (which parcel lines are "visible" in the rendered image)
# --------------------------------------------------------------------------

def _group_by_block(parcels):
    by_block = {}
    for p in parcels:
        by_block.setdefault(p.block_id, []).append(p)
    return by_block


def _boundary_edges(blocks, parcels, params, rng):
    edges_geom, edges_visible = {}, {}
    by_block = _group_by_block(parcels)

    for blk in blocks:
        members = by_block.get(blk.id, [])
        shared_segments = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                shared = a.polygon.boundary.intersection(b.polygon.boundary)
                if shared.length > EPS:
                    key = tuple(sorted((a.id, b.id)))
                    edges_geom[key] = shared
                    edges_visible[key] = bool(rng.random() < params.visible_edge_fraction)
                    shared_segments.append(shared)
        shared_union = unary_union(shared_segments) if shared_segments else None
        for p in members:
            ext = p.polygon.boundary
            if shared_union is not None:
                ext = ext.difference(shared_union)
            if ext.length > EPS:
                key = (p.id, -1)
                edges_geom[key] = ext
                edges_visible[key] = bool(rng.random() < params.visible_edge_fraction)
    return edges_geom, edges_visible


# --------------------------------------------------------------------------
# buildings
# --------------------------------------------------------------------------

def _random_rect_in(envelope, coverage_range, rng):
    minx, miny, maxx, maxy = envelope.bounds
    w, h = maxx - minx, maxy - miny
    scale = rng.uniform(*coverage_range) ** 0.5
    rw, rh = max(w * scale, 1.0), max(h * scale, 1.0)
    cx = rng.uniform(minx + rw / 2, max(maxx - rw / 2, minx + rw / 2 + 1e-6))
    cy = rng.uniform(miny + rh / 2, max(maxy - rh / 2, miny + rh / 2 + 1e-6))
    return box(cx - rw / 2, cy - rh / 2, cx + rw / 2, cy + rh / 2)


def _place_buildings(blocks, parcels, params, rng):
    by_block = _group_by_block(parcels)

    buildings, bid = [], 0
    for blk in blocks:
        members = by_block.get(blk.id, [])
        for p in members:
            envelope = p.polygon.buffer(-params.setback)
            if envelope.is_empty or envelope.area < params.min_building_area:
                continue
            rect = _random_rect_in(envelope, params.building_coverage, rng)
            cross = blk.style == "informal" and rng.random() < params.crossing_prob_informal
            if cross:
                # deliberately spill the footprint past the parcel line: push it
                # toward a random direction and clip against the block, not the
                # parcel, so it can straddle the (invisible) internal boundary
                angle = rng.uniform(0, 2 * np.pi)
                push = params.setback + rng.uniform(1.0, 3.0)
                rect = affinity.translate(rect, np.cos(angle) * push, np.sin(angle) * push)
                footprint = rect.intersection(blk.polygon.buffer(-0.2))
            else:
                footprint = rect.intersection(envelope)
            if footprint.is_empty or footprint.area < params.min_building_area * 0.3:
                continue
            owners = tuple(sorted(
                q.id for q in members
                if footprint.intersects(q.polygon) and footprint.intersection(q.polygon).area > EPS
            ))
            buildings.append(Building(bid, footprint, owners, float(rng.uniform(*params.building_height_range))))
            bid += 1
    return buildings


# --------------------------------------------------------------------------
# legacy GIS layer: jittered, shifted, some missing, some merged
# --------------------------------------------------------------------------

def _make_legacy_layer(parcels, adjacency_pairs, params, rng):
    neighbors = {}
    for a, b in adjacency_pairs:
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)

    style_of = {p.id: p.style for p in parcels}
    distorted = {}
    for p in parcels:
        shift = rng.normal(0, params.legacy_shift_range[p.style], size=2)
        # drop the duplicate closing vertex before jittering: jittering it
        # independently from the (identical) first vertex breaks ring closure
        # and leaves a stray near-duplicate point that can self-intersect the
        # ring almost every time. Shapely re-closes the ring on construction,
        # using this same jittered first point for both ends.
        coords = np.array(p.polygon.exterior.coords)[:-1]
        jitter = rng.normal(0, params.legacy_jitter_std[p.style], size=coords.shape)
        poly = Polygon(coords + jitter + shift)
        if not poly.is_valid:
            poly = poly.buffer(0)
        distorted[p.id] = poly

    present = {pid for pid in distorted if rng.random() >= params.legacy_p_missing[style_of[pid]]}

    # `settled` tracks every pid that already has a merged.append() entry --
    # both a merge's primary and its absorbed partner. Filtering candidates
    # against only `absorbed` (partners) missed the case where a pid already
    # emitted as its own standalone entry gets picked as a *later* pid's
    # merge partner, duplicating that parcel's geometry into two entries.
    merged, settled = [], set()
    for pid in sorted(present):
        if pid in settled:
            continue
        candidates = [n for n in neighbors.get(pid, []) if n in present and n not in settled]
        if candidates and rng.random() < params.legacy_p_merge[style_of[pid]]:
            other = int(rng.choice(candidates))
            geom = unary_union([distorted[pid], distorted[other]])
            merged.append(LegacyParcel(pid, geom, (pid, other)))
            settled.add(other)
        else:
            merged.append(LegacyParcel(pid, distorted[pid], (pid,)))
        settled.add(pid)
    return merged


# --------------------------------------------------------------------------
# GT survey points: exact positions of a sparse subset of parcel corners
# --------------------------------------------------------------------------

def _make_gt_points(parcels, params, rng):
    # dedupe corners *before* drawing: a corner shared by k parcels must get
    # exactly one Bernoulli(gt_point_fraction) trial, not up to k independent
    # ones (which biased selection toward high-parcel-density corners).
    corners = {}
    for p in parcels:
        for (x, y) in list(p.polygon.exterior.coords)[:-1]:
            corners.setdefault((round(x, 6), round(y, 6)), (x, y, p.id))

    pts = []
    for x, y, pid in corners.values():
        if rng.random() < params.gt_point_fraction:
            pts.append(GTPoint(x, y, pid))
    return pts


# --------------------------------------------------------------------------
# rendering: fake ortho + DSM + DTM
# --------------------------------------------------------------------------

def _noisy_color(base_rgb, shape, noise_std, rng):
    return np.array(base_rgb, dtype=np.float32).reshape(1, 1, 3) + rng.normal(0, noise_std, size=(*shape, 3))


def _scatter_veg(ward_poly, params, rng):
    minx, miny, maxx, maxy = ward_poly.bounds
    blobs, tries = [], 0
    while len(blobs) < params.n_veg_blobs and tries < params.n_veg_blobs * 8:
        tries += 1
        blob = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy)).buffer(rng.uniform(*params.veg_radius_range))
        if ward_poly.contains(blob):
            blobs.append(blob)
    return blobs


def _render_rasters(ward_poly, parcels, road_render_union, buildings, edges_geom, edges_visible, params, rng):
    gsd = params.gsd
    out_w = int(np.ceil(params.width / gsd))
    out_h = int(np.ceil(params.height / gsd))
    out_shape = (out_h, out_w)
    transform = from_origin(0, params.height, gsd, gsd)

    _, xx = np.mgrid[0:out_h, 0:out_w]
    dtm = 100.0 + params.dtm_base_slope * (xx * gsd)
    relief = rng.normal(0, 1, size=out_shape).astype(np.float32)
    relief = gaussian_filter(relief, sigma=max(out_w, out_h) * 0.03 + 2)
    relief = relief / (relief.std() + 1e-9) * params.dtm_relief_amplitude
    dtm = (dtm + relief).astype(np.float32)

    building_shapes = [(b.polygon, b.height) for b in buildings if not b.polygon.is_empty]
    building_height = (
        rasterize(building_shapes, out_shape, transform=transform, fill=0.0, dtype="float32")
        if building_shapes else np.zeros(out_shape, dtype=np.float32)
    )

    veg_polys = _scatter_veg(ward_poly, params, rng)
    veg_shapes = [(g, float(rng.uniform(*params.veg_height_range))) for g in veg_polys]
    veg_height = (
        rasterize(veg_shapes, out_shape, transform=transform, fill=0.0, dtype="float32")
        if veg_shapes else np.zeros(out_shape, dtype=np.float32)
    )
    veg_mask = veg_height > 0

    dsm = dtm + building_height + veg_height

    road_mask = (
        rasterize([(road_render_union, 1)], out_shape, transform=transform, fill=0, dtype="uint8").astype(bool)
        if not road_render_union.is_empty else np.zeros(out_shape, dtype=bool)
    )

    ortho = _noisy_color((150, 130, 100), out_shape, 12, rng)  # bare soil, painted first
    ortho[veg_mask] = _noisy_color((60, 110, 55), out_shape, 20, rng)[veg_mask]
    ortho[road_mask] = _noisy_color((90, 90, 95), out_shape, 8, rng)[road_mask]

    # one rasterize call tagging every building by (index+1) beats one call
    # per building; later shapes still win on overlap, same as painting in a loop
    live_buildings = [b for b in buildings if not b.polygon.is_empty]
    if live_buildings:
        building_id = rasterize(
            [(b.polygon, i + 1) for i, b in enumerate(live_buildings)],
            out_shape, transform=transform, fill=0, dtype="int32",
        )
        building_mask = building_id > 0
        bases = rng.uniform((120, 60, 60), (200, 120, 120), size=(len(live_buildings), 3))
        colors = bases[np.clip(building_id - 1, 0, len(live_buildings) - 1)] + rng.normal(0, 15, size=(*out_shape, 3))
        ortho[building_mask] = colors[building_mask]

    wall_shapes = [
        (edges_geom[key].buffer(params.wall_render_width / 2), 1)
        for key, visible in edges_visible.items()
        if visible and edges_geom[key].length > EPS
    ]
    if wall_shapes:
        wall_mask = rasterize(wall_shapes, out_shape, transform=transform, fill=0, dtype="uint8").astype(bool)
        ortho[wall_mask] = (40, 35, 30)

    ortho = np.clip(ortho, 0, 255).astype(np.uint8)
    return ortho, dsm.astype(np.float32), dtm.astype(np.float32), transform
