"""Research workspace contracts: real inference, local provenance and advisory-only review."""
import hashlib
import json
import numpy as np
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from geocadastra.api import workspace as ws


@pytest.fixture
def local(tmp_path,monkeypatch):
    monkeypatch.setattr(ws,'ROOT',tmp_path)
    monkeypatch.setenv('GEOCADASTRA_WORKSPACE_STATE',str(tmp_path/'state'))
    monkeypatch.delenv('GEOCADASTRA_VISION_KEY',raising=False)
    monkeypatch.delenv('GEOCADASTRA_VISION_MODEL',raising=False)
    data=tmp_path/'data'/'example';data.mkdir(parents=True)
    (tmp_path/'runs').mkdir()
    rgb=np.zeros((3,64,64),dtype='uint8');rgb[:,:,32:]=255
    boundary=np.zeros((64,64),bool);boundary[:,32]=True
    valid=np.ones((64,64),bool)
    np.savez_compressed(data/'tile.npz',rgb=rgb,ndsm=np.zeros((64,64),dtype='float32'),boundary=boundary,valid=valid)
    sha=hashlib.sha256((data/'tile.npz').read_bytes()).hexdigest()
    (data/'manifest.json').write_text(json.dumps({'crs':'EPSG:2193','gsd_m':.3,'tiles':[{'tile':'tile','path':'tile.npz','split':'val','sha256':sha}]}))
    app=FastAPI();app.include_router(ws.router)
    ws._arrays.cache_clear()
    return TestClient(app),tmp_path


def test_catalog_assets_and_actual_evidence(local):
    c,_=local
    bad=local[1]/'data'/'unrelated';bad.mkdir();(bad/'manifest.json').write_text('[]')
    cat=c.get('/workspace/catalog').json()
    assert cat['datasets'][0]['count']==1
    assert cat['vision']['configured'] is False
    assert c.get('/workspace/tiles/example/tile').json()['has_height']
    image=c.get('/workspace/tiles/example/tile/image?layer=evidence')
    assert image.status_code==200 and image.content.startswith(b'\x89PNG')
    evidence=c.get('/workspace/tiles/example/tile/evidence').json()
    assert evidence['counts']['supported']==64
    assert sum(evidence['counts'].values())==64
    assert c.get('/workspace/tiles/example/missing/image').status_code==404
    assert c.get('/workspace/tiles/example/tile/image?layer=madeup').status_code==400


def test_path_escape_and_review_persistence(local):
    c,root=local
    with pytest.raises(HTTPException):ws.safe_child(root/'data','../../etc/passwd')
    result=c.post('/workspace/reviews',json={'dataset':'example','tile':'tile','region':'0-0','decision':'needs_survey','note':'No fence visible'})
    assert result.status_code==200
    assert ws.reviews()[0]['note']=='No fence visible'
    assert c.post('/workspace/reviews',json={'dataset':'example','tile':'tile','region':'0-0','decision':'invent'}).status_code==422
    assert c.post('/workspace/vision',json={'dataset':'example','tile':'tile','x':0,'y':0}).status_code==503


def test_real_inference_records_weights_and_pixel_metrics(local):
    import torch
    from geocadastra.models.backbone import MultiTaskNet
    c,root=local
    torch.set_num_threads(2)
    folder=root/'runs'/'tiny';folder.mkdir()
    path=folder/'best.pt';torch.save({'model':MultiTaskNet().state_dict()},path)
    job={'id':'run1','kind':'inference','dataset':'example','tile':'tile','checkpoint':'tiny/best.pt','threshold_m':.3,'status':'queued','boot':ws.BOOT}
    ws.write_record('jobs','run1',job)
    ws.run_inference(job,path)
    assert job['status']=='completed',job
    assert job['checkpoint_sha256']==ws.digest(path)
    assert job['elapsed_seconds']>0
    metrics=job['metrics']
    assert metrics['true_positive']+metrics['false_negative']==64
    assert 0<=metrics['f1']<=1
    assert c.get('/workspace/jobs/run1/artifact').content.startswith(b'\x89PNG')
    assert c.get('/workspace/jobs/run1/artifact?kind=report').json()['certification']=='not_calibrated'


def test_provider_is_advisory_and_daily_cap_is_durable(local,monkeypatch):
    c,_=local
    monkeypatch.setenv('GEOCADASTRA_VISION_KEY','test-secret')
    monkeypatch.setenv('GEOCADASTRA_VISION_MODEL','test-model')
    monkeypatch.setenv('GEOCADASTRA_VISION_DAILY_LIMIT','1')
    import httpx
    def reply(url,**kw):
        assert url=='https://api.anthropic.com/v1/messages'
        assert kw['json']['max_tokens']==512
        return httpx.Response(200,json={'content':[{'type':'text','text':'Uncertain: possible shadow.'}],'usage':{'input_tokens':20,'output_tokens':5}},request=httpx.Request('POST',url))
    monkeypatch.setattr(httpx,'post',reply)
    body={'dataset':'example','tile':'tile','x':0,'y':0}
    result=c.post('/workspace/vision',json=body)
    assert result.status_code==200
    assert result.json()['decision']=='advisory_only'
    assert 'test-secret' not in result.text
    assert c.post('/workspace/vision',json=body).status_code==429


def test_interrupted_job_and_incompatible_training_refused(local):
    c,_=local
    ws.write_record('jobs','stale',{'id':'stale','status':'running','boot':'old'})
    assert c.get('/workspace/jobs').json()[0]['status']=='interrupted'
    # This tile has boundary labels, but not the distance field needed by trainer.
    assert c.post('/workspace/training',json={'dataset':'example','epochs':1,'batch_size':1}).status_code==422
