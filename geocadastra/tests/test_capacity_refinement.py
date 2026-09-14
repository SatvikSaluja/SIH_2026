import pytest
from shapely.geometry import box
from geocadastra.core.crs import Geom
from geocadastra.core.transport import parcels_to_graph
from geocadastra.core.capacity import refine_recorded_areas

CRS="EPSG:32643"


def test_shared_boundary_moves_to_recorded_areas_without_changing_identity():
    block=Geom(box(0,0,10,10),CRS)
    graph=parcels_to_graph({10:Geom(box(0,0,4,10),CRS),20:Geom(box(4,0,10,10),CRS)},block)
    original={nid:(n.x,n.y) for nid,n in graph.nodes.items()}
    result=refine_recorded_areas(graph,block,{10:50.,20:50.},{10:.01,20:.01})
    assert result.converged,result.detail
    assert result.graph.face_parcel_ids==graph.face_parcel_ids
    assert {nid:(n.x,n.y) for nid,n in graph.nodes.items()}==original
    for fid,polygon in result.graph.faces_to_polygons().items():
        assert polygon.area==pytest.approx(50,abs=.01)
    assert sorted(result.graph.edges)==sorted(graph.edges)


def test_infeasible_totals_are_declined_without_rescaling_records():
    block=Geom(box(0,0,10,10),CRS)
    graph=parcels_to_graph({10:block},block)
    result=refine_recorded_areas(graph,block,{10:200.},{10:.01})
    assert not result.converged
    assert result.graph is graph
    assert result.detail["reason"]=="infeasible_recorded_total"


def test_worker_refines_then_persists_recorded_capacities(committed_session,monkeypatch):
    from types import SimpleNamespace
    from sqlalchemy import select
    from geoalchemy2.shape import from_shape
    from shapely.geometry import Point
    from geocadastra.jobs import orchestrator as jobs
    from geocadastra.store.changeset import allocate_ids,load_block_graph
    from geocadastra.store.schema import RecordedParcel,recorded_parcel_id_seq,SRID,Provenance
    from geocadastra.store.constraints import parcel_area_report
    from geocadastra.tests.test_review_regressions import ingest,process
    job,bid,pid=ingest(committed_session)
    committed_session.get(RecordedParcel,pid).area=200
    other=allocate_ids(committed_session,recorded_parcel_id_seq,1)[0]
    committed_session.add(RecordedParcel(id=other,ward_job_id=job.id,block_id=bid,area=200,
                                        style="institutional",seed_point=from_shape(Point(15,10),srid=SRID)))
    committed_session.commit()
    monkeypatch.setattr(jobs,"assign_parcels",lambda *a,**kw:SimpleNamespace(
        parcel_polygons={0:Geom(box(0,0,8,20),CRS),1:Geom(box(8,0,20,20),CRS)},conflicts=[]))
    assert process(job,bid)=="done"
    report=parcel_area_report(committed_session,bid,load_block_graph(committed_session,bid))
    assert report["constraints_satisfied"],report
    assert all(abs(r["predicted_area_m2"]-200)<=.01 for r in report["parcels"])
    provenance=committed_session.scalars(select(Provenance).where(Provenance.evidence_type=="initial_load")).all()
    assert provenance[0].payload["evidence"]["area_refinement"]["reason"]=="refined"


def test_fusion_pins_known_corners_even_when_evidence_projects_along_an_edge(monkeypatch):
    from geocadastra.core import fusion
    graph=parcels_to_graph({10:Geom(box(0,0,10,10),CRS)},Geom(box(0,0,10,10),CRS))
    node=next(n for n in graph.nodes.values() if (n.x,n.y)==(0,0))
    monkeypatch.setattr(fusion,"fuse_node",lambda *a,**kw:fusion.FusedPosition(0,3,1,("legacy",)))
    result=fusion.fuse_block(graph,[node.id],style="formal",block_boundary=Geom(box(0,0,10,10),CRS))
    assert (result.moved[node.id].x,result.moved[node.id].y)==(0,0)


def test_three_record_capacities_are_solved_together():
    block=Geom(box(0,0,10,10),CRS)
    graph=parcels_to_graph({1:Geom(box(0,0,2.5,10),CRS),2:Geom(box(2.5,0,5,10),CRS),3:Geom(box(5,0,10,10),CRS)},block)
    targets={1:20.,2:30.,3:50.}
    result=refine_recorded_areas(graph,block,targets,{p:.01 for p in targets})
    assert result.converged,result.detail
    for fid,g in result.graph.faces_to_polygons().items():
        assert abs(g.area-targets[result.graph.face_parcel_ids[fid]])<=.01


def _graph(start):
    block=Geom(box(0,0,10,10),CRS)
    return block,parcels_to_graph({p:Geom(box(*b),CRS) for p,b in start.items()},block)


UNEVEN={1:(0,0,2,10),2:(2,0,4,10),3:(4,0,10,10)}


def test_tolerance_is_spent_before_the_solve_not_only_checked_after():
    """A recorded total the block cannot hold exactly is still satisfiable.

    34/34/34 sums to 102 in a 100 m2 block, but +/-1 each admits 33.33 each.
    Solving for exact recorded areas refused this; the tolerance interval is
    what the record actually permits.
    """
    block,graph=_graph(UNEVEN)
    result=refine_recorded_areas(graph,block,{1:34.,2:34.,3:34.},{1:1.,2:1.,3:1.})
    assert result.converged,result.detail
    for fid,polygon in result.graph.faces_to_polygons().items():
        assert abs(polygon.area-34.)<=1.
    assert result.graph.face_parcel_ids==graph.face_parcel_ids


def test_slack_goes_to_the_parcel_whose_record_permits_it():
    block,graph=_graph(UNEVEN)
    result=refine_recorded_areas(graph,block,{1:34.,2:34.,3:34.},{1:.001,2:.001,3:3.})
    assert result.converged,result.detail
    areas={result.graph.face_parcel_ids[fid]:g.area for fid,g in result.graph.faces_to_polygons().items()}
    assert abs(areas[1]-34.)<=.001 and abs(areas[2]-34.)<=.001
    assert abs(areas[3]-32.)<=.01  # absorbs the whole 2 m2 deficit


def test_total_outside_every_tolerance_is_still_refused():
    block,graph=_graph(UNEVEN)
    result=refine_recorded_areas(graph,block,{1:50.,2:50.,3:50.},{1:1.,2:1.,3:1.})
    assert not result.converged
    assert result.detail["reason"]=="infeasible_recorded_total"
