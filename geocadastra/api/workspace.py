"""Local operator workspace: real assets, persisted jobs/reviews and opt-in vision QA.

Run one API worker. This is a local research console, not an authenticated public service.
Only manifest-listed datasets and checkpoints under runs/ can be requested.
"""
from __future__ import annotations
import base64
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time
import uuid

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, Field
from scipy.ndimage import sobel, maximum_filter, binary_dilation, label

router = APIRouter(prefix='/workspace', tags=['research workspace'])
ROOT = Path(__file__).resolve().parents[2]
POOL = ThreadPoolExecutor(max_workers=1)
VISION_LOCK = threading.Lock()
BOOT = uuid.uuid4().hex


def state_root():
    p = Path(os.environ.get('GEOCADASTRA_WORKSPACE_STATE', str(ROOT/'.local_workspace')))
    p.mkdir(parents=True, exist_ok=True)
    return p


@contextmanager
def connect():
    db = sqlite3.connect(state_root()/'workspace.sqlite', timeout=20)
    db.row_factory = sqlite3.Row
    db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, data TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS reviews (id TEXT PRIMARY KEY, data TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS vision_calls (id TEXT PRIMARY KEY, day TEXT, status TEXT)')
    try:
        with db:
            yield db
    finally:
        db.close()


def write_record(table, key, data):
    if table not in ('jobs','reviews'): raise ValueError('Unknown table')
    with connect() as db:
        db.execute(f'INSERT OR REPLACE INTO {table} VALUES (?,?)', (key,json.dumps(data,allow_nan=False)))


def records(table):
    if table not in ('jobs','reviews'): raise ValueError('Unknown table')
    with connect() as db:
        return [json.loads(r['data']) for r in db.execute(f'SELECT data FROM {table} ORDER BY rowid DESC')]


def safe_child(root, relative):
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()): raise HTTPException(400,'Path outside dataset')
    return path


def read_json(path, default=None):
    try: return json.loads(path.read_text())
    except (OSError, ValueError): return default


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def datasets():
    result={}
    for p in sorted((ROOT/'data').glob('*/manifest.json')):
        m=read_json(p,{})
        if not isinstance(m,dict): continue
        tiles=m.get('tiles',[])
        usable=[t for t in tiles if isinstance(t,dict) and t.get('tile') and t.get('path','').endswith('.npz')]
        if usable:
            result[p.parent.name]=(p,m,usable)
    return result


def tile_entry(dataset,tile):
    found=datasets().get(dataset)
    if not found: raise HTTPException(404,'Dataset not found')
    path,m,tiles=found
    entry=next((t for t in tiles if t['tile']==tile),None)
    if not entry: raise HTTPException(404,'Tile not found')
    asset=safe_child(path.parent,entry['path'])
    if not asset.is_file(): raise HTTPException(404,'Tile file unavailable')
    return asset,m,entry


@lru_cache(maxsize=3)
def _arrays(path, modified):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files if k in ('rgb','ndsm','distance','boundary','valid')}


def arrays(dataset,tile):
    path,m,e=tile_entry(dataset,tile)
    return _arrays(str(path),path.stat().st_mtime_ns),m,e,path


def checkpoint_map():
    result={}
    for p in (ROOT/'runs').glob('*/*'):
        if p.is_file() and p.name in ('best.pt','last.pt','last.pt.best','model_a.pt'):
            result[p.relative_to(ROOT/'runs').as_posix()]=p
    return result


def png(array):
    output=io.BytesIO();Image.fromarray(array).save(output,format='PNG');return output.getvalue()


def normalized(a):
    finite=a[np.isfinite(a)]
    if not finite.size: return np.zeros_like(a,dtype=float)
    lo,hi=np.percentile(finite,[5,95])
    return np.clip((np.nan_to_num(a)-lo)/max(hi-lo,1e-6),0,1)


def reference(a,gsd):
    if 'distance' in a: return a['distance']<=gsd
    return a.get('boundary',np.zeros(a['rgb'].shape[1:],bool)).astype(bool)


def evidence_arrays(a,gsd):
    gray=a['rgb'].astype('float32').mean(axis=0)/255
    edge=np.hypot(sobel(gray,axis=0),sobel(gray,axis=1))
    rgb_support=maximum_filter(normalized(edge),size=5)
    height_support=np.zeros_like(gray)
    if 'ndsm' in a:
        h=np.nan_to_num(a['ndsm']);height_support=maximum_filter(normalized(np.hypot(sobel(h,axis=0),sobel(h,axis=1))),size=5)
    score=np.maximum(rgb_support, height_support*.75)
    ref=reference(a,gsd)&a.get('valid',np.ones(gray.shape,bool)).astype(bool)
    # Relative edge evidence only. Neither absence nor presence proves visibility.
    classes=np.zeros(gray.shape,np.uint8)
    classes[ref]=1;classes[ref&(score>=.25)]=2;classes[ref&(score>=.65)]=3
    return score,classes


@router.get('/catalog')
def catalog():
    return {'datasets':[{'id':k,'count':len(t),'status':m.get('status','snapshot'),
                         'crs':m.get('crs'),'splits':{s:sum(e.get('split')==s for e in t) for s in ('train','val','test')},
                         'source':m.get('parcel_source'),'limitations':m.get('limitations',[])} for k,(p,m,t) in datasets().items()],
            'checkpoints':[{'id':k,'bytes':p.stat().st_size,'modified':p.stat().st_mtime,
                            'scope':'Research checkpoint; verify training scope before use'} for k,p in checkpoint_map().items()],
            'vision':vision_config(), 'certification':'not_calibrated'}


@router.get('/tiles')
def tiles(dataset:str, search:str='', split:str='', offset:int=Query(0,ge=0),limit:int=Query(48,ge=1,le=100)):
    found=datasets().get(dataset)
    if not found: raise HTTPException(404,'Dataset not found')
    _,m,entries=found
    selected=[e for e in entries if search.lower() in e['tile'].lower() and (not split or e.get('split')==split)]
    return {'total':len(selected),'tiles':[{'id':e['tile'],'split':e.get('split','unknown'),'gsd_m':e.get('gsd_m',m.get('gsd_m',.3)),
                                         'parcels':e.get('parcels'),'valid_fraction':e.get('valid_fraction')} for e in selected[offset:offset+limit]]}


@router.get('/tiles/{dataset}/{tile}')
def tile_info(dataset:str,tile:str):
    a,m,e,p=arrays(dataset,tile)
    metadata=read_json(ROOT/'data/nz_height_100/metadata'/f'{tile}.json',{})
    gsd=e.get('gsd_m',m.get('gsd_m',.3))
    height=a.get('ndsm');vals=height[np.isfinite(height)] if height is not None else []
    return {'id':tile,'dataset':dataset,'split':e.get('split'),'width':a['rgb'].shape[2],'height':a['rgb'].shape[1],
            'gsd_m':gsd,'crs':m.get('crs'),'layers':['rgb','reference','evidence']+(['height'] if height is not None else []),
            'height_range_m':[float(np.percentile(vals,5)),float(np.percentile(vals,95))] if len(vals) else None,
            'transform':metadata.get('png_pixel_to_map_transform'),'bbox':e.get('bbox_wgs84'),
            'source':e.get('image_url',m.get('parcel_source')),'sha256':e.get('sha256',e.get('training_sha256')),
            'limitations':m.get('limitations',[]),'capture':metadata.get('rgb_stac',{}).get('properties',{}).get('start_datetime'),
            'height_note':'Measured nDSM where available; native height resolution may be coarser than imagery.',
            'has_height':'ndsm' in a}


@router.get('/tiles/{dataset}/{tile}/image')
def tile_image(dataset:str,tile:str,layer:str='rgb',thumb:bool=False):
    a,m,e,p=arrays(dataset,tile);gsd=e.get('gsd_m',m.get('gsd_m',.3))
    rgb=a['rgb'].transpose(1,2,0).astype('uint8').copy()
    if layer=='reference': rgb[binary_dilation(reference(a,gsd),iterations=1)]=[109,255,174]
    elif layer=='height':
        if 'ndsm' not in a: raise HTTPException(404,'No height data')
        n=normalized(a['ndsm']);rgb=np.stack([30+190*n,45+170*n,95+100*(1-n)],axis=-1).astype('uint8')
    elif layer=='evidence':
        _,c=evidence_arrays(a,gsd)
        for k,color in [(1,[239,112,105]),(2,[246,192,87]),(3,[79,215,171])]: rgb[c==k]=color
    elif layer!='rgb': raise HTTPException(400,'Unknown image layer')
    if thumb:
        im=Image.fromarray(rgb);im.thumbnail((320,240));rgb=np.array(im)
    return Response(png(rgb),media_type='image/png',headers={'Cache-Control':'private, max-age=60'})


@router.get('/tiles/{dataset}/{tile}/evidence')
def evidence(dataset:str,tile:str):
    a,m,e,p=arrays(dataset,tile);gsd=e.get('gsd_m',m.get('gsd_m',.3));score,c=evidence_arrays(a,gsd)
    counts={name:int((c==k).sum()) for k,name in [(1,'weak'),(2,'uncertain'),(3,'supported')]}
    candidates=[];h,w=c.shape
    for y in range(0,h,192):
        for x in range(0,w,192):
            region=c[y:y+192,x:x+192];total=int((region>0).sum())
            if total<10: continue
            weak=float(((region==1)|(region==2)).sum()/total)
            candidates.append({'id':f'{x}-{y}','x':x,'y':y,'width':min(192,w-x),'height':min(192,h-y),
                               'review_fraction':weak,'reference_pixels':total})
    candidates.sort(key=lambda r:r['review_fraction'],reverse=True)
    return {'counts':counts,'candidates':candidates[:30],
            'method':'RGB Sobel + optional nDSM gradient, 5-pixel neighbourhood, tile-relative scaling; thresholds 0.25 / 0.65.',
            'warning':'Heuristic image support, not verified visibility or legal boundary certainty. Shadows can appear supported. Weak evidence does not prove absence.'}


class InferenceRequest(BaseModel):
    dataset:str
    tile:str
    checkpoint:str
    threshold_m:float=Field(.3,gt=0,le=5)


@router.post('/inference',status_code=202)
def inference(body:InferenceRequest):
    a,m,e,p=arrays(body.dataset,body.tile)
    if 'ndsm' not in a: raise HTTPException(422,'This checkpoint requires measured nDSM. Choose a height-matched tile.')
    cp=checkpoint_map().get(body.checkpoint)
    if cp is None: raise HTTPException(404,'Checkpoint not found')
    if sum(j['status'] in ('queued','running') and j.get('boot')==BOOT for j in records('jobs'))>=3:
        raise HTTPException(429,'Queue is full; wait for a job to complete')
    job={'id':uuid.uuid4().hex,'kind':'inference','status':'queued','created':time.time(),'boot':BOOT,**body.model_dump()}
    write_record('jobs',job['id'],job)
    POOL.submit(run_inference,job,cp)
    return job


def run_inference(job,cp):
    start=time.monotonic();folder=state_root()/job['id'];folder.mkdir(exist_ok=True)
    try:
        import torch
        from geocadastra.models.backbone import MultiTaskNet
        from geocadastra.models.infer import run_tiled_inference
        job.update(status='running',started=time.time());write_record('jobs',job['id'],job)
        a,m,e,p=arrays(job['dataset'],job['tile']);source_hash=digest(p);checkpoint_hash=digest(cp)
        expected=e.get('sha256',e.get('training_sha256'))
        if expected and expected!=source_hash: raise ValueError('Tile checksum differs from manifest')
        saved=torch.load(cp,map_location='cpu',weights_only=True)
        net=MultiTaskNet();net.load_state_dict(saved.get('model',saved),strict=True)
        torch.set_num_threads(2)
        output=run_tiled_inference(net,a['rgb'].astype('float32')/255,np.nan_to_num(a['ndsm']).astype('float32'),tile_size=128,overlap=32)
        mask=a.get('valid',np.ones_like(output['sdf'],bool)).astype(bool)&np.isfinite(a['ndsm'])
        pred=(abs(output['sdf'])<=job['threshold_m'])&mask
        rgb=a['rgb'].transpose(1,2,0).copy();rgb[binary_dilation(pred)&mask]=[255,192,90]
        (folder/'prediction.png').write_bytes(png(rgb))
        np.savez_compressed(folder/'prediction.npz',sdf=output['sdf'],log_var=output['log_var'],valid=mask)
        metrics=None
        if 'distance' in a or 'boundary' in a:
            truth=reference(a,e.get('gsd_m',m.get('gsd_m',.3)))&mask
            tp=int((pred&truth).sum());fp=int((pred&~truth).sum());fn=int((~pred&truth).sum())
            metrics={'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'f1':2*tp/max(2*tp+fp+fn,1),
                     'iou':tp/max(tp+fp+fn,1),'true_positive':tp,'false_positive':fp,'false_negative':fn,
                     'definition':'Stitched valid-pixel overlap; reference band is one image pixel. Not independent survey accuracy.'}
        job.update(status='completed',elapsed_seconds=time.monotonic()-start,metrics=metrics,
                   checkpoint_sha256=checkpoint_hash,input_sha256=source_hash,device='cpu',
                   predicted_fraction=float(pred[mask].mean()) if mask.any() else 0,
                   evaluation_scope='Reference agreement; training overlap unknown. No generalization claim.',
                   certification='not_calibrated',finished=time.time())
    except Exception as exc: job.update(status='failed',error=f'{type(exc).__name__}: {exc}',finished=time.time())
    write_record('jobs',job['id'],job)


@router.get('/jobs')
def jobs():
    output=records('jobs')
    for j in output:
        if j.get('boot')!=BOOT and j['status'] in ('queued','running'):
            j['status']='interrupted';j['error']='API restarted; rerun explicitly.'
    return output


@router.get('/jobs/{job_id}/artifact')
def artifact(job_id:str,kind:str='image'):
    job=next((j for j in records('jobs') if j['id']==job_id),None)
    if not job or job['status']!='completed': raise HTTPException(404,'Completed job not found')
    if kind=='report':return Response(json.dumps(job,indent=2),media_type='application/json',headers={'Content-Disposition':'attachment; filename="inference-report.json"'})
    name='prediction.png' if kind=='image' else 'prediction.npz' if kind=='arrays' else None
    if not name:raise HTTPException(400,'Unknown artifact')
    path=state_root()/job_id/name
    if not path.exists():raise HTTPException(404,'Artifact unavailable')
    return Response(path.read_bytes(),media_type='image/png' if kind=='image' else 'application/octet-stream')


class ReviewRequest(BaseModel):
    dataset:str
    tile:str
    region:str=Field(max_length=100)
    decision:str=Field(pattern='^(accept|reject|needs_survey)$')
    note:str=Field('',max_length=2000)


@router.get('/reviews')
def reviews():return records('reviews')


@router.post('/reviews')
def review(body:ReviewRequest):
    tile_entry(body.dataset,body.tile)
    record={'id':uuid.uuid4().hex,'created':time.time(),**body.model_dump()}
    write_record('reviews',record['id'],record);return record


def vision_config():
    return {'configured':bool(os.environ.get('GEOCADASTRA_VISION_KEY') and os.environ.get('GEOCADASTRA_VISION_MODEL')),
            'provider':'Anthropic Messages','model':os.environ.get('GEOCADASTRA_VISION_MODEL'),
            'daily_limit':int(os.environ.get('GEOCADASTRA_VISION_DAILY_LIMIT','10')),
            'max_output_tokens':512,'automatic_calls':False}


class VisionRequest(BaseModel):
    dataset:str
    tile:str
    x:int=Field(ge=0)
    y:int=Field(ge=0)


@router.post('/vision')
def vision(body:VisionRequest):
    cfg=vision_config()
    if not cfg['configured']:raise HTTPException(503,'Configure GEOCADASTRA_VISION_KEY and GEOCADASTRA_VISION_MODEL on the server.')
    a,m,e,p=arrays(body.dataset,body.tile);rgb=a['rgb'].transpose(1,2,0).copy();h,w=rgb.shape[:2]
    if body.x>=w or body.y>=h:raise HTTPException(422,'Crop lies outside image')
    rgb=rgb[body.y:body.y+192,body.x:body.x+192];marked=rgb.copy()
    ref=reference(a,e.get('gsd_m',m.get('gsd_m',.3)))[body.y:body.y+192,body.x:body.x+192]
    marked[binary_dilation(ref)]=[255,40,180]
    picture=png(np.concatenate([rgb,marked],axis=1));day=datetime.now(timezone.utc).date().isoformat();call_id=uuid.uuid4().hex
    with VISION_LOCK,connect() as db:
        used=db.execute('SELECT count(*) FROM vision_calls WHERE day=?',(day,)).fetchone()[0]
        if used>=cfg['daily_limit']:raise HTTPException(429,'Daily vision request cap reached')
        db.execute('INSERT INTO vision_calls VALUES (?,?,?)',(call_id,day,'reserved'))
    import httpx
    prompt=('The left image is raw aerial RGB; the right repeats it with a MAGENTA reference parcel line. '
            'Do not treat the drawn line as physical evidence. Describe visible fences, walls, hedges, kerbs or shadows near the line. '
            'Assess possible misalignment and say uncertain when the image cannot resolve it. '
            'Do not infer ownership, exact legal boundaries or calibrated confidence. Give a short advisory assessment for human review.')
    try:
        response=httpx.post('https://api.anthropic.com/v1/messages',headers={'x-api-key':os.environ['GEOCADASTRA_VISION_KEY'],
                            'anthropic-version':'2023-06-01'},json={'model':cfg['model'],'max_tokens':512,
                            'messages':[{'role':'user','content':[{'type':'image','source':{'type':'base64','media_type':'image/png','data':base64.b64encode(picture).decode()}},
                                                               {'type':'text','text':prompt}]}]},timeout=45)
        response.raise_for_status();result=response.json()
        answer='\n'.join(c.get('text','') for c in result.get('content',[]) if c.get('type')=='text')
    except Exception:
        raise HTTPException(502,'Vision provider request failed. Attempt counts toward daily cap; check server configuration.')
    record={'id':call_id,'kind':'vision_advice','created':time.time(),**body.model_dump(),'model':cfg['model'],
            'advice':answer,'usage':result.get('usage',{}),'decision':'advisory_only'}
    write_record('reviews',call_id,record);return record


@router.get('/training')
def training():
    output=[]
    for folder in sorted((ROOT/'runs').iterdir()):
        if not folder.is_dir():continue
        history=read_json(folder/'history.json',[]);result=read_json(folder/'result.json')
        if not history and not result:continue
        output.append({'id':folder.name,'history':history,'result':result,
                       'epochs':len(history),'status':'Recorded results; live execution not inferred',
                       'checkpoints':[k for k in checkpoint_map() if k.startswith(folder.name+'/')],
                       'scope':'Same-tile fit diagnostic' if 'overfit' in folder.name else 'See run configuration and held-out metrics'})
    return output


class TrainRequest(BaseModel):
    dataset:str
    epochs:int=Field(1,ge=1,le=100)
    batch_size:int=Field(4,ge=1,le=16)


@router.post('/training',status_code=202)
def start_training(body:TrainRequest):
    ds=datasets().get(body.dataset)
    if not ds:raise HTTPException(404,'Dataset not found')
    if ds[1].get('status')=='in_progress':raise HTTPException(409,'Freeze a dataset snapshot before training')
    # Check every selected file schema/checksum before scheduling an expensive run.
    for e in ds[2]:
        p=safe_child(ds[0].parent,e['path'])
        if 'sha256' not in e:raise HTTPException(422,'Normalize dataset to real trainer manifest format first')
        with np.load(p,allow_pickle=False) as z:
            if not {'rgb','ndsm','distance','valid'}.issubset(z.files):raise HTTPException(422,'Dataset has missing training inputs')
    if any(j['status'] in ('queued','running') and j.get('boot')==BOOT for j in records('jobs')):
        raise HTTPException(409,'Wait for the active workspace job before starting training')
    job={'id':uuid.uuid4().hex,'kind':'training','status':'queued','created':time.time(),'boot':BOOT,**body.model_dump()}
    write_record('jobs',job['id'],job);POOL.submit(run_training,job,ds[0]);return job


def run_training(job,manifest):
    out=ROOT/'runs'/('workspace_'+job['id']);out.mkdir()
    job.update(status='running',run=out.name,started=time.time());write_record('jobs',job['id'],job)
    try:
        cmd=[sys.executable,str(ROOT/'scripts/colab/train_real.py'),'--data',str(manifest.parent),'--out',str(out),
             '--epochs',str(job['epochs']),'--batch-size',str(job['batch_size'])]
        with (out/'console.log').open('w') as stream:
            proc=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=21600)
        job.update(status='completed' if proc.returncode==0 else 'failed',finished=time.time(),
                   error=None if proc.returncode==0 else 'Trainer failed; inspect '+str(out/'console.log'))
    except Exception as exc:job.update(status='failed',error=str(exc),finished=time.time())
    write_record('jobs',job['id'],job)
