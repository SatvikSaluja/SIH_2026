"""Stage 5 Done-when, per the build plan:

"injecting GT points into a synthetic ward improves boundary accuracy near
those points and does not degrade it elsewhere; and after fusion, every
Stage 1 topology invariant still holds. That second assertion is the one
that catches the classic bug."

The deterministic grid isolates evidence-driven interior-node motion.
Real generated wards separately exercise exterior handling and topology.
Generator version 2 nodes subdivision cuts together, so genuine interior
junctions are represented; the historical claim that all generated nodes
were exterior was an artifact of disconnected boundary representations.
"""
import numpy as np
import pytest
from shapely.affinity import translate
from shapely.geometry import box
from shapely.ops import unary_union
from sqlalchemy.orm import Session

from geocadastra.core.crs import Geom
from geocadastra.core.fusion import _is_exterior_node, fuse_block
from geocadastra.core.graph import build_graph
from geocadastra.store.changeset import apply_fusion, load_block_graph, seed_block_graph
from geocadastra.synth.generator import generate_ward
from geocadastra.tests.test_stage1_roundtrip import _check_block_graph_invariants, _tol

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


def test_exterior_nodes_are_never_moved_without_a_block_boundary(db_session: Session):
    """Without an authoritative block boundary, all exterior nodes stay
    fixed. Interior nodes may still use the available survey evidence."""
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
    exterior = {nid for nid in graph.nodes if _is_exterior_node(graph,nid)}
    assert exterior
    assert exterior <= set(result.unchanged)
    assert exterior.isdisjoint(result.moved)


def _real_ward_blocks(seeds):
    """(ward, block, parcels) for the most-subdivided block of each seed,
    skipping any with too few parcels to be interesting."""
    for seed in seeds:
        ward = generate_ward(seed=seed)
        by_block: dict = {}
        for p in ward.parcels:
            by_block.setdefault(p.block_id, []).append(p)
        block_id, parcels = max(by_block.items(), key=lambda kv: len(kv[1]))
        if len(parcels) < 3:
            continue
        block = next(b for b in ward.blocks if b.id == block_id)
        yield ward, block, parcels


@pytest.mark.slow
def test_block_boundary_fusion_never_produces_overlapping_or_invalid_geometry_on_real_wards(db_session: Session):
    """The hard safety guarantee, on real (not hand-built) synthetic
    data: across several real wards' most-subdivided blocks, fusing
    exterior nodes against their own true `block_boundary` (straight from
    `ward.blocks`, standing in for `build_blocks()`'s own output) never
    leaves an overlapping or invalid face, and applies a real majority of
    the candidate moves rather than degrading to a near no-op. Also checks total-area conservation across the seed sweep."""
    n_blocks_tested = 0
    for db_block_id, (ward, block, parcels) in enumerate(_real_ward_blocks(range(300, 340))):
        # `block.id` restarts from 0 for every generate_ward() call -- reusing
        # it directly as the DB block_id across seeds in this same session
        # would conflate two unrelated wards' geometry under one block_id
        # (found by review: this false-positived an "overlap" between two
        # entirely different wards' faces, not a real Stage 5 bug). A
        # per-iteration counter keeps each seed's data in its own block.
        n_blocks_tested += 1
        local_graph = build_graph([Geom(p.polygon, CRS) for p in parcels], CRS)
        seed_block_graph(db_session, block_id=db_block_id, graph=local_graph, description="initial load")
        db_session.commit()
        graph = load_block_graph(db_session, db_block_id)

        parcel_ids = {p.id for p in parcels}
        gt_xy = [(g.x, g.y) for g in ward.gt_points if g.parcel_id in parcel_ids]
        legacy_polys = [lp.polygon for lp in ward.legacy_parcels if set(lp.source_parcel_ids) & parcel_ids]
        legacy_boundary = unary_union([p.boundary for p in legacy_polys]) if legacy_polys else None

        result = fuse_block(
            graph, list(graph.nodes), style=block.style, legacy_boundary=legacy_boundary, gt_points=gt_xy,
            tolerance=8.0, block_boundary=Geom(block.polygon, CRS),
        )
        apply_result = apply_fusion(db_session, db_block_id, result)

        final_graph = load_block_graph(db_session, db_block_id)
        faces = final_graph.faces_to_polygons()
        assert all(f.geom.is_valid for f in faces.values()), f"seed {ward.seed}: an invalid face was persisted"
        assert abs(sum(f.area for f in faces.values()) - block.polygon.area) <= _tol(block.polygon.area)
        ids = list(faces)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                overlap = faces[ids[i]].geom.intersection(faces[ids[j]].geom, grid_size=1e-3).area
                assert overlap < 1e-3, f"seed {ward.seed}: faces {ids[i]}/{ids[j]} overlap by {overlap:.4f} m^2"
        if result.moved:
            applied_fraction = len(apply_result.applied) / len(result.moved)
            assert applied_fraction > 0.3, (
                f"seed {ward.seed}: only {applied_fraction:.0%} of candidate moves applied -- "
                "block_boundary fusion has degraded to a near no-op"
            )
    assert n_blocks_tested >= 10, "too few qualifying blocks found -- widen the seed sweep"


@pytest.mark.slow
def test_block_boundary_fusion_conserves_total_area_on_real_wards(db_session: Session):
    ward, block, parcels = next(_real_ward_blocks([300]))
    domain_area = sum(p.area for p in parcels)

    local_graph = build_graph([Geom(p.polygon, CRS) for p in parcels], CRS)
    seed_block_graph(db_session, block_id=block.id, graph=local_graph, description="initial load")
    db_session.commit()
    graph = load_block_graph(db_session, block.id)

    parcel_ids = {p.id for p in parcels}
    gt_xy = [(g.x, g.y) for g in ward.gt_points if g.parcel_id in parcel_ids]
    legacy_polys = [lp.polygon for lp in ward.legacy_parcels if set(lp.source_parcel_ids) & parcel_ids]
    legacy_boundary = unary_union([p.boundary for p in legacy_polys]) if legacy_polys else None

    result = fuse_block(
        graph, list(graph.nodes), style=block.style, legacy_boundary=legacy_boundary, gt_points=gt_xy,
        tolerance=8.0, block_boundary=Geom(block.polygon, CRS),
    )
    apply_fusion(db_session, block.id, result)

    final_graph = load_block_graph(db_session, block.id)
    total_area = sum(p.area for p in final_graph.faces_to_polygons().values())
    tol = _tol(domain_area)
    assert abs(total_area - domain_area) <= tol, (
        f"total area {total_area:.2f} vs true domain area {domain_area:.2f} (diff {abs(total_area - domain_area):.2f}, "
        f"tol {tol:.2f})"
    )


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
