"""Node/edge/face planar graph model (Stage 1) -- the source of truth.

A face is an ordered walk of edge references, never its own coordinate list.
`edge_linestring()`/`face_polygon()`/`faces_to_polygons()` derive geometry
from current node positions on every call, so `move_node()` is the only
mutation needed for an edit to be reflected in every incident face.

Building a graph from a list of already-planarized faces (see planarize.py)
means matching up shared boundaries by exact coordinate equality -- safe
because planarize() grid-snaps everything first -- then dissolving vertices
that aren't real topological nodes (a point with exactly two incident
segments that border the same pair of faces and are collinear -- i.e. a
redundant shape point, not a corner or a junction) into the interior of one
longer Edge.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import math
from shapely import set_precision
from shapely.geometry import LineString, Point, Polygon
from geocadastra.core.planarize import GRID

from geocadastra.core.crs import CRSMismatchError, Geom

OUTER = -1  # sentinel face id: the unbounded "outside everything" face --
# every boundary edge's second face-side, so "every edge has exactly two
# face-sides" holds with no exception for ward-boundary edges.


@dataclass
class Node:
    id: int
    x: float
    y: float


@dataclass
class Edge:
    id: int
    n0: int
    n1: int
    coords: tuple  # interior vertices strictly between n0 and n1, in n0->n1 order


@dataclass
class Face:
    id: int
    boundary: tuple  # ordered ((edge_id, forward), ...); forward means traverse n0->n1
    holes: tuple = ()  # ordered edge walks for interior rings

    @property
    def rings(self) -> tuple:
        return (self.boundary, *self.holes)

    @property
    def all_edges(self) -> tuple:
        return tuple(entry for ring in self.rings for entry in ring)


class PlanarGraph:
    def __init__(self, crs: str):
        self.crs = crs
        self.nodes: dict[int, Node] = {}
        self.edges: dict[int, Edge] = {}
        self.faces: dict[int, Face] = {}
        self.face_parcel_ids: dict[int, int] = {}
        self.identity_conflicts: list[dict] = []
        self._edge_faces: dict[int, list[int]] = {}
        self._next_node = 0
        self._next_edge = 0
        self._point_to_node: dict[tuple, int] = {}

    def _get_or_add_node(self, pt: tuple[float, float]) -> int:
        key = _round_pt(pt)
        nid = self._point_to_node.get(key)
        if nid is None:
            nid = self._next_node
            self._next_node += 1
            self.nodes[nid] = Node(nid, pt[0], pt[1])
            self._point_to_node[key] = nid
        return nid

    def _add_edge(self, n0: int, n1: int, coords: list) -> int:
        eid = self._next_edge
        self._next_edge += 1
        self.edges[eid] = Edge(eid, n0, n1, tuple(coords))
        return eid

    def move_node(self, node_id: int, x: float, y: float) -> None:
        if node_id not in self.nodes:
            raise KeyError(f"move_node(): no node {node_id!r} in this graph")
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("node coordinates must be finite")
        point = set_precision(Point(x, y), GRID)
        x, y = point.x, point.y
        for other_id, node in self.nodes.items():
            if other_id != node_id and set_precision(Point(node.x, node.y), GRID).equals(point):
                raise ValueError(f"move would merge node {node_id} with node {other_id} on the precision grid")
        old_key = _round_pt((self.nodes[node_id].x, self.nodes[node_id].y))
        if self._point_to_node.get(old_key) == node_id:
            del self._point_to_node[old_key]
        self.nodes[node_id] = Node(node_id, x, y)
        self._point_to_node[_round_pt((x, y))] = node_id

    def _edge_coords(self, edge_id: int) -> list:
        e = self.edges[edge_id]
        n0, n1 = self.nodes[e.n0], self.nodes[e.n1]
        return [(n0.x, n0.y), *e.coords, (n1.x, n1.y)]

    def edge_linestring(self, edge_id: int) -> LineString:
        return LineString(self._edge_coords(edge_id))

    def face_polygon(self, face_id: int, _coords_cache: dict | None = None) -> Polygon:
        face = self.faces[face_id]
        cache = {} if _coords_cache is None else _coords_cache
        rings = []
        for walk in face.rings:
            ring: list = []
            for edge_id, forward in walk:
                coords = cache.get(edge_id)
                if coords is None:
                    coords = self._edge_coords(edge_id)
                    cache[edge_id] = coords
                if not forward:
                    coords = coords[::-1]
                if ring and ring[-1] == coords[0]:
                    coords = coords[1:]
                ring.extend(coords)
            rings.append(ring)
        return Polygon(rings[0], rings[1:])

    def faces_to_polygons(self) -> dict[int, Geom]:
        # share one edge-coordinate cache across every face this call derives:
        # almost every interior edge borders exactly two faces, so without
        # this each such edge's coordinates get rebuilt from scratch twice
        cache: dict = {}
        return {fid: Geom(self.face_polygon(fid, _coords_cache=cache), self.crs) for fid in self.faces}

    def faces_of_edge(self, edge_id: int) -> tuple:
        sides = list(self._edge_faces.get(edge_id, ()))
        while len(sides) < 2:
            sides.append(OUTER)
        return tuple(sides)


def _undirected_key(a: tuple, b: tuple) -> tuple:
    return (a, b) if a < b else (b, a)


def _round_pt(pt) -> tuple[float, float]:
    # 6 decimals (micron-level), not 9: float64 ULP noise at UTM-scale
    # magnitudes (~1e6-1e7) is already ~1e-10-1e-9, which is *not* negligible
    # against a 9-decimal round's 5e-10 tie threshold -- two bit-identical
    # real-world points could round to different grid cells. 6 decimals keeps
    # a >1000x margin above that noise at any coordinate magnitude this
    # project uses, while staying far finer than planarize.py's own 1e-3 m
    # snap grid (the actually-meaningful precision level).
    return (round(pt[0], 6), round(pt[1], 6))


def _is_collinear(p_prev, p, p_next, tol: float) -> bool:
    """Is `p` within `tol` of the straight line through `p_prev`/`p_next`?

    Perpendicular distance, not a normalized angle: an angle-ratio test gets
    *tighter* as the segments get longer for a fixed real-world wobble (a
    grid-snap's worth of perpendicular jitter on a long segment is a tiny
    angle), which made a fixed epsilon systematically fail to dissolve
    ordinary-length redundant shape points at realistic survey precision.
    Tying the threshold to `tol` (typically the same precision grid
    planarize() already snapped to) means "no straighter than our own
    measurement precision" rather than an arbitrary angle.
    """
    v = (p_next[0] - p_prev[0], p_next[1] - p_prev[1])
    length = (v[0] ** 2 + v[1] ** 2) ** 0.5
    if length < 1e-12:
        return True
    cross = (p[0] - p_prev[0]) * v[1] - (p[1] - p_prev[1]) * v[0]
    return abs(cross) / length <= tol


def build_graph(face_geoms: list[Geom], crs: str, collinear_tol: float = 1e-6) -> PlanarGraph:
    """Faces (e.g. planarize()'s output, or any list of Polygon Geoms whose
    shared boundaries are exact-coordinate matches) -> a PlanarGraph.

    `collinear_tol` gates the degree-2 dissolve (see `_is_collinear`) -- pass
    the same precision grid the faces were snapped to (e.g. planarize.py's
    `GRID`) so "redundant shape point" means "no straighter than our own
    measurement precision," not an arbitrary angle.
    """
    for g in face_geoms:
        if g.crs != crs:
            raise CRSMismatchError(f"build_graph() got a face in {g.crs!r}, expected {crs!r}")

    rings = []
    seen_rings: set = set()
    for i, g in enumerate(face_geoms):
        face_rings = [[_round_pt(c) for c in ring.coords]
                      for ring in (g.geom.exterior, *g.geom.interiors)]
        # a duplicate face (same ring, e.g. a caller accidentally including a
        # block/parcel twice) would double-count area and confuse every
        # face-side/edge-sharing invariant below -- catch it here, not later.
        # Keyed on the ring's undirected edge set (not just its vertex set)
        # so it's rotation/direction-invariant without false-colliding on a
        # different shape that happens to share the same vertices.
        ring_key = frozenset(frozenset(_undirected_key(a, b) for a, b in zip(coords[:-1], coords[1:]))
                             for coords in face_rings)
        if ring_key in seen_rings:
            raise ValueError(f"build_graph() got the same face polygon more than once (index {i})")
        seen_rings.add(ring_key)
        rings.extend((i, coords) for coords in face_rings)

    # undirected segment key -> the face indices whose ring walks it
    segment_faces: dict[tuple, list[int]] = defaultdict(list)
    for fi, ring in rings:
        for a, b in zip(ring[:-1], ring[1:]):
            if a == b:
                continue
            segment_faces[_undirected_key(a, b)].append(fi)

    point_segments: dict[tuple, set] = defaultdict(set)
    for key in segment_faces:
        point_segments[key[0]].add(key)
        point_segments[key[1]].add(key)

    def other_end(key, p):
        a, b = key
        return b if a == p else a

    def collapsible(p) -> bool:
        segs = point_segments[p]
        if len(segs) != 2:
            return False
        s1, s2 = tuple(segs)
        if set(segment_faces[s1]) != set(segment_faces[s2]):
            return False
        return _is_collinear(other_end(s1, p), p, other_end(s2, p), collinear_tol)

    node_points = {p for p in point_segments if not collapsible(p)}

    graph = PlanarGraph(crs)
    for p in node_points:
        graph._get_or_add_node(p)

    # dissolve: walk each chain of collapsible points between two real nodes
    # into one Edge, recording which raw (a, b) direction each traversed
    # segment corresponds to, so face boundaries can be mapped onto edges.
    visited: set = set()
    segkey_dir: dict[tuple, tuple[int, tuple, tuple]] = {}
    for start in node_points:
        for seg in list(point_segments[start]):
            if seg in visited:
                continue
            visited.add(seg)
            chain = [start, other_end(seg, start)]
            keys = [seg]
            cur = chain[-1]
            while cur not in node_points:
                nxt_key = next(s for s in point_segments[cur] if s != keys[-1])
                visited.add(nxt_key)
                cur = other_end(nxt_key, cur)
                chain.append(cur)
                keys.append(nxt_key)
            n0 = graph._get_or_add_node(start)
            n1 = graph._get_or_add_node(cur)
            eid = graph._add_edge(n0, n1, chain[1:-1])
            for k, a, b in zip(keys, chain[:-1], chain[1:]):
                segkey_dir[k] = (eid, a, b)  # walked a->b while building eid

    # each face's boundary: rotate its ring to start at a node (so a chain
    # never wraps across the ring's arbitrary start point), then collapse
    # consecutive raw segments that belong to the same edge into one entry.
    edge_faces: dict[int, list[int]] = defaultdict(list)
    face_walks = defaultdict(list)
    for fi, ring in rings:
        open_ring = ring[:-1]
        start_idx = next((i for i, p in enumerate(open_ring) if p in node_points), 0)
        rotated = open_ring[start_idx:] + open_ring[:start_idx]
        rotated.append(rotated[0])

        boundary = []
        last = None
        for a, b in zip(rotated[:-1], rotated[1:]):
            key = _undirected_key(a, b)
            eid, walked_a, walked_b = segkey_dir[key]
            forward = (walked_a, walked_b) == (a, b)
            entry = (eid, forward)
            if entry != last:
                boundary.append(entry)
                edge_faces[eid].append(fi)
                last = entry
        face_walks[fi].append(tuple(boundary))

    for fi, walks in face_walks.items():
        graph.faces[fi] = Face(fi, walks[0], tuple(walks[1:]))

    graph._edge_faces = dict(edge_faces)
    return graph
