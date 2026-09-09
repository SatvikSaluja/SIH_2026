import sys, uuid, threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unittest.mock import patch
import numpy as np
from shapely.geometry import box, Point, Polygon
from shapely.ops import unary_union
from sqlalchemy import create_engine, text, select, func
from sqlalchemy.orm import Session
from geoalchemy2.shape import to_shape
from geocadastra.store.schema import Base, Face, Provenance, IngestedBlock, RecordedParcel, LegacyRecord, BlockJob, WardJob, PersistedConflict, SurveyPoint
from geocadastra.store.changeset import seed_block_graph, load_block_graph, ChangesetContext, apply_fusion
from geocadastra.core.crs import Geom
from geocadastra.core.graph import build_graph
from geocadastra.core.fusion import legacy_estimate, FuseBlockResult, FusedPosition
from geocadastra.core.conformal import calibrate
from geocadastra.core.priority import face_uncertainty
from geocadastra.jobs import orchestrator as o
from geocadastra.api.main import coregister, CoRegisterRequest, field_verification, FieldVerificationRequest, status
from geocadastra.synth.generator import WardParams, generate_ward

def test_review_probes():
 url='postgresql+psycopg://geocadastra:geocadastra@localhost:5432/geocadastra'
 schema='review_'+uuid.uuid4().hex[:10]
 admin=create_engine(url)
 with admin.begin() as c:c.execute(text(f'CREATE SCHEMA {schema}'))
 engine=create_engine(url, connect_args={'options':f'-csearch_path={schema},public'})
 Base.metadata.create_all(engine)
 try:
  with Session(engine) as s:
   g=build_graph([Geom(box(0,0,10,10),'EPSG:32643')],'EPSG:32643')
   seed_block_graph(s,999,g);s.commit()
   print('seed provenance count:',s.scalar(select(func.count()).select_from(Provenance)))
   g=load_block_graph(s,999); nid=next(iter(g.nodes));n=g.nodes[nid]
   with ChangesetContext(s,999) as cs:cs.move_node(nid,n.x+0.000123,n.y)
   ng=load_block_graph(s,999)
   print('off-grid persisted x:',ng.nodes[nid].x)
   # Force both callers past optimistic checks before either writes.
   barrier=threading.Barrier(2);orig=ChangesetContext._check_no_concurrent_modification
   def checked(cs):orig(cs);barrier.wait(timeout=10)
   errors=[]
   ids=list(ng.nodes)[:2]
   def edit(i):
    try:
     with Session(engine) as t:
      with ChangesetContext(t,999) as cs:
       n=cs.graph.nodes[ids[i]];cs.move_node(n.id,n.x+0.1,n.y+0.1)
    except Exception as e:errors.append(type(e).__name__+':'+str(e))
   with patch.object(ChangesetContext,'_check_no_concurrent_modification',checked):
    ts=[threading.Thread(target=edit,args=(i,)) for i in range(2)]
    for t in ts:t.start()
    for t in ts:t.join(timeout=20)
   s.expire_all();gg=load_block_graph(s,999);f=s.get(Face,next(iter(gg.faces)))
   print('concurrent errors:',errors,'cache vs graph symmetric difference:',to_shape(f.geom).symmetric_difference(gg.face_polygon(f.id)).area)
   p=WardParams(width=20,height=20,gsd=1,n_arterial_h=0,n_arterial_v=0,minor_spacing=100,style_weights={'institutional':1},institutional_single_prob=1)
   ward=generate_ward(p,1);job=o.ingest_synthetic_ward(s,ward,1);jid=job.id;source=job.source;params=job.params
   bid=s.scalar(select(IngestedBlock.block_id).where(IngestedBlock.ward_job_id==jid))
   before=s.scalar(select(LegacyRecord).where(LegacyRecord.ward_job_id==jid));wkb=bytes(to_shape(before.geom).wkb)
   cr=coregister(jid,CoRegisterRequest(control_points=[(0,0,10,20),(1,0,11,20),(0,1,10,21)]),s)
   s.expire_all();after=s.scalar(select(LegacyRecord).where(LegacyRecord.ward_job_id==jid))
   print('coreg translation:',cr,'geometry unchanged:',bytes(to_shape(after.geom).wkb)==wkb)
   # Crash after fusion's first real committing changeset.
   def crash(session,block_id,result,author=None):
    graph=load_block_graph(session,block_id);n=next(iter(graph.nodes.values()))
    with ChangesetContext(session,block_id) as cs:cs.move_node(n.id,n.x,n.y)
    raise RuntimeError('injected failure after first fusion commit')
   with patch.object(o,'apply_fusion',crash):
    try:o.process_block(url,schema,jid,bid,source,params)
    except RuntimeError:pass
   s.expire_all();count=lambda:s.scalar(select(func.count()).select_from(Face).where(Face.block_id==bid))
   print('failed block persisted faces:',count())
   try:o.process_block(url,schema,jid,bid,source,params)
   except Exception as e:print("retry raised:",type(e).__name__,str(e))
   s.expire_all()
   print('retry persisted faces:',count(),'block status:',s.get(BlockJob,(jid,bid)).status)
   # A known area conflict from the real solver is silently dropped.
   job3=o.ingest_synthetic_ward(s,ward,1);j3=job3.id;b3=s.scalar(select(IngestedBlock.block_id).where(IngestedBlock.ward_job_id==j3))
   rp=s.scalar(select(RecordedParcel).where(RecordedParcel.ward_job_id==j3));rp.area=800;s.commit()
   observed=[];orig_assign=o.assign_parcels
   def track_assignment(*a,**kw):
    result=orig_assign(*a,**kw);observed.extend(c.kind for c in result.conflicts);return result
   with patch.object(o,'assign_parcels',track_assignment):o.process_block(url,schema,j3,b3,job3.source,job3.params)
   print('assignment conflicts:',observed,'persisted:',s.scalar(select(func.count()).select_from(PersistedConflict).where(PersistedConflict.ward_job_id==j3)))
   wrong=build_graph([Geom(box(70,20,70.01,20.01),'EPSG:4326')],'EPSG:4326')
   seed_block_graph(s,998,wrong);s.commit()
   print('store CRS silently relabeled:',wrong.crs,'->',load_block_graph(s,998).crs)
   # Simulate async dispatch without executing at dispatch time.
   job2=o.ingest_synthetic_ward(s,ward,1);j2=job2.id;b2=s.scalar(select(IngestedBlock.block_id).where(IngestedBlock.ward_job_id==j2))
   src2=job2.source;par2=job2.params
   with patch.object(o.process_block,'delay',lambda *a,**kw:None):o.run_ward(s,url,schema,j2)
   o.process_block(url,schema,j2,b2,src2,par2);s.expire_all()
   print('async final status:',status(j2,s))
   graph=load_block_graph(s,b2);n=next(iter(graph.nodes.values()))
   field_verification(j2,FieldVerificationRequest(block_id=b2,node_id=n.id,parcel_id=987654321,x=n.x,y=n.y),s)
   print('orphan survey accepted:',s.scalar(select(func.count()).select_from(SurveyPoint).where(SurveyPoint.parcel_id==987654321)))
   # Fail the second commit, when attaching the survey evidence.
   graph=load_block_graph(s,b2);n=next(iter(graph.nodes.values()));newx=n.x+.01
   original_commit=s.commit;commits=[0]
   def fail_attachment():
    commits[0]+=1
    if commits[0]==2:raise RuntimeError('injected survey insert failure')
    return original_commit()
   real_pid=s.scalar(select(RecordedParcel.id).where(RecordedParcel.ward_job_id==j2))
   with patch.object(s,'commit',fail_attachment):
    try:field_verification(j2,FieldVerificationRequest(block_id=b2,node_id=n.id,parcel_id=real_pid,x=newx,y=n.y),s)
    except RuntimeError:pass
   s.rollback();s.expire_all()
   print('failed verification retained geometry change:',load_block_graph(s,b2).nodes[n.id].x==newx)

  # Pool a locked connection, hold it elsewhere, then let Session use another.
  lock_engine=create_engine(url,pool_size=2,max_overflow=0)
  with Session(lock_engine) as ls:
   pid1=ls.scalar(text('select pg_backend_pid()'))
   ls.execute(text('select pg_advisory_lock(77123,88456)'));ls.commit()
   with lock_engine.connect() as held:
    pid2=ls.scalar(text('select pg_backend_pid()'))
    unlocked=ls.scalar(text('select pg_advisory_unlock(77123,88456)'))
    print('lock backend switched:',pid1!=pid2,'unlock succeeded:',unlocked)
    held.execute(text('select pg_advisory_unlock(77123,88456)'))
  lock_engine.dispose()
 finally:
  engine.dispose()
  cache=o._engine_cache.pop((url,schema),None)
  if cache:cache.dispose()
  with admin.begin() as c:c.execute(text(f'DROP SCHEMA {schema} CASCADE'))
  admin.dispose()
