"""Stage 5 Done-when, per the build plan:

"injecting GT points into a synthetic ward improves boundary accuracy near
those points and does not degrade it elsewhere; and after fusion, every
Stage 1 topology invariant still holds. That second assertion is the one
that catches the classic bug."

Built around a deterministic, hand-built 3x3 grid of parcels rather than a
generated synthetic ward's own blocks: found by review that this project's
actual block-subdivision styles (strip/institutional/single-cut/recursive-
organic) never produce a genuinely interior node -- every interior cut's
own endpoints land back on the block's true perimeter too (confirmed by
reading `_recursive_split`/`_strip_split` in synth/generator.py, and
empirically across 40 real synthetic seeds: zero interior nodes, every
time). Since fusion correctly never moves an exterior node (see
core/fusion.py's `_is_exterior_node`, needed to keep the block's own total
area invariant), a real ward's blocks exercise only the "nothing moves"
half of fusion -- covered separately below -- not the actual "GT/legacy
evidence corrects an imprecise position" mechanism this Done-when
criterion is actually about. A hand-built grid with a real interior point
(matching the precedent already set for the deterministic overlap
regression in test_changeset.py) tests the real mechanism directly and
deterministically, instead of hoping some future seed produces one.
"""
import numpy as np
from shapely.affinity import translate
from shapely.geometry import box
from shapely.ops import unary_union
from sqlalchemy.orm import Session

from geocadastra.core.crs import Geom
from geocadastra.core.fusion import _is_exterior_node, fuse_block
from geocadastra.core.graph import build_graph
from geocadastra.store.changeset import apply_fusion, load_block_graph, seed_block_graph
from geocadastra.synth.generator import generate_ward
from geocadastra.tests.test_stage1_roundtrip import _check_block_graph_invariants

CRS = "EPSG:32643"


def _grid_graph(n=3, cell=4.0):
    """An n x n grid of `cell`-sized square parcels -- (n-1)**2 genuinely
    interior nodes where 4 parcels meet, none of them on the block's own
    perimeter. n=3 gives 4 interior nodes at (4,4), (8,4), (4,8), (8,8)."""
    polys = [box(x, y, x + cell, y + cell) for x in np.arange(n) * cell for y in np.arange(n) * cell]
    return build_graph([Geom(p, CRS) for p in polys], CRS), sum(p.area for p in polys)


def _legacy_cross(n=3, cell=4.0, shift=(0.3, -0.2)):
    """A jittered copy of the grid's own internal lines -- an independent
    (if noisy) source for fusion, standing in for a real legacy layer."""
    interior = [box(x, y, x + cell, y + cell).boundary for x in np.arange(n) * cell for y in np.arange(n) * cell]
    return translate(unary_union(interior), *shift)


def _dist(true_xy, nid, x, y):
    tx, ty = true_xy[nid]
    return ((x - tx) ** 2 + (y - ty) ** 2) ** 0.5


def test_gt_points_improve_accuracy_near_them_without_degrading_elsewhere():
    graph, _domain_area = _grid_graph()
    true_xy = {nid: (n.x, n.y) for nid, n in graph.nodes.items()}
    interior_nodes = [nid for nid in graph.nodes if not _is_exterior_node(graph, nid)]
    assert len(interior_nodes) == 4  # (4,4), (8,4), (4,8), (8,8)

    rng = np.random.default_rng(11)
    for nid in interior_nodes:
        x, y = true_xy[nid]
        dx, dy = rng.normal(0, 1.5, size=2)
        graph.move_node(nid, x + dx, y + dy)

    gt_node = interior_nodes[0]
    gt_xy = [true_xy[gt_node]]

    result = fuse_block(graph, list(graph.nodes), style="formal", legacy_boundary=_legacy_cross(), gt_points=gt_xy)

    assert set(result.moved) == set(interior_nodes), "expected exactly the 4 interior nodes to move, no others"
    assert not result.conflicts

    gt_node_pos = graph.nodes[gt_node]
    near_before = _dist(true_xy, gt_node, gt_node_pos.x, gt_node_pos.y)
    near_after = _dist(true_xy, gt_node, result.moved[gt_node].x, result.moved[gt_node].y)
    assert near_after < near_before, f"GT-adjacent node did not improve: before={near_before:.3f}m after={near_after:.3f}m"
    assert near_after < 0.1, f"GT-adjacent node should land almost exactly on the true position, got {near_after:.3f}m off"

    far_regressions = 0
    for nid in interior_nodes[1:]:
        node = graph.nodes[nid]
        before = _dist(true_xy, nid, node.x, node.y)
        after = _dist(true_xy, nid, result.moved[nid].x, result.moved[nid].y)
        if after > before * 1.1:  # small slack -- legacy alone is noisy but should not make things meaningfully worse
            far_regressions += 1
    assert far_regressions == 0, "fusion degraded a node with no nearby GT point"


def test_exterior_nodes_are_never_moved_by_fusion(db_session: Session):
    """Real synthetic wards' own blocks never have a genuinely interior
    node (see module docstring) -- confirm fusion's response to that is
    exactly "move nothing", not silently something else, on a real ward's
    block. The meaningful "does fusion actually correct a position" case
    is exercised deterministically above."""
    ward = generate_ward(seed=300)
    by_block: dict = {}
    for p in ward.parcels:
        by_block.setdefault(p.block_id, []).append(p)
    block_id, parcels = max(by_block.items(), key=lambda kv: len(kv[1]))
    block = next(b for b in ward.blocks if b.id == block_id)

    local_graph = build_graph([Geom(p.polygon, CRS) for p in parcels], CRS)
    seed_block_graph(db_session, block_id=block.id, graph=local_graph, description="initial load")
    db_session.commit()
    graph = load_block_graph(db_session, block.id)

    gt_xy = [(g.x, g.y) for g in ward.gt_points if g.parcel_id in {p.id for p in parcels}]
    result = fuse_block(graph, list(graph.nodes), style=block.style, gt_points=gt_xy, tolerance=8.0)
    assert not result.moved
    assert set(result.unchanged) == set(graph.nodes)


def test_topology_invariants_still_hold_after_fusion(db_session: Session):
    """"after fusion, every Stage 1 topology invariant still holds. That
    second assertion is the one that catches the classic bug" -- run
    fusion through a real ChangesetContext (`apply_fusion()`, not one big
    all-or-nothing changeset -- see its own docstring) against the actual
    store, then re-check every Stage 1 invariant on the graph reloaded
    fresh afterward."""
    local_graph, domain_area = _grid_graph()
    true_xy = {nid: (n.x, n.y) for nid, n in local_graph.nodes.items()}
    interior_nodes = [nid for nid in local_graph.nodes if not _is_exterior_node(local_graph, nid)]

    rng = np.random.default_rng(23)
    for nid in interior_nodes:
        x, y = true_xy[nid]
        dx, dy = rng.normal(0, 1.5, size=2)
        local_graph.move_node(nid, x + dx, y + dy)

    seed_block_graph(db_session, block_id=900, graph=local_graph, description="initial load")
    db_session.commit()
    graph = load_block_graph(db_session, 900)

    gt_xy = [true_xy[interior_nodes[0]]]
    result = fuse_block(graph, list(graph.nodes), style="formal", legacy_boundary=_legacy_cross(), gt_points=gt_xy)
    assert result.moved, "fusion moved nothing -- test setup gives it no real work to do"

    apply_result = apply_fusion(db_session, 900, result)
    assert apply_result.applied, "every fused move was refused -- test setup gives fusion no safe work to do"

    final_graph = load_block_graph(db_session, 900)
    _check_block_graph_invariants(final_graph, domain_area)
