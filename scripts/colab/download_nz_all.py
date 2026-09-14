"""Download every native RGB GeoTIFF in the Christchurch 2025 LINZ collection.
Run again to resume. Existing files are checksum-verified, partial files restart.
Only this imagery collection is downloaded; parcel/height datasets are separate.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
from urllib.request import Request, urlopen
from urllib.parse import urljoin

COLLECTION='https://nz-imagery.s3.ap-southeast-2.amazonaws.com/canterbury/christchurch_2025_0.075m/rgb/2193/collection.json'

def checksum(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def expected(value):
    if not value or not value.startswith('1220') or len(value)!=68:
        raise ValueError('Missing or unsupported SHA256 checksum')
    return value[4:]

def fetch(url):
    return urlopen(Request(url,headers={'User-Agent':'GeoCadastra-public-imagery-download/1.0'}),timeout=120)

def atomic_json(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));os.replace(temp,path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='data/nz_christchurch_all')
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--reserve-gb',type=float,default=15)
    p.add_argument('--limit',type=int,default=0,help='Testing only: 0 means all tiles')
    args=p.parse_args()
    if not 1<=args.workers<=16 or args.reserve_gb<1 or args.limit<0: p.error('Invalid limits')
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    # Prevent simultaneous runs writing the same files on Linux/Colab.
    import fcntl
    lock=(root/'download.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (root/'images').mkdir(exist_ok=True);(root/'metadata').mkdir(exist_ok=True)
    with fetch(COLLECTION) as r: raw=r.read()
    c=json.loads(raw);(root/'collection.json').write_bytes(raw)
    links=[x for x in c['links'] if x['rel']=='item']
    if args.limit: links=links[:args.limit]
    state={'collection':COLLECTION,'title':c['title'],'license':c['license'],'providers':c['providers'],
           'started_utc':datetime.now(timezone.utc).isoformat(),'total':len(links),'verified':0,
           'verified_bytes':0,'failed':0,'status':'running','errors':[]}
    stop=threading.Event();reserve=int(args.reserve_gb*1024**3)
    def task(link):
        url=urljoin(COLLECTION,link['href']); name=Path(link['href']).stem
        if Path(name).name!=name: raise ValueError('Invalid tile name')
        metadata=root/'metadata'/(name+'.json');meta_hash=expected(link['file:checksum'])
        last=None
        for attempt in range(5):
            if stop.is_set(): raise RuntimeError('Download paused: disk reserve reached')
            try:
                if not metadata.exists() or checksum(metadata)!=meta_hash:
                    with fetch(url) as r: payload=r.read()
                    if hashlib.sha256(payload).hexdigest()!=meta_hash: raise ValueError('Metadata checksum mismatch')
                    metadata.write_bytes(payload)
                item=json.loads(metadata.read_bytes());asset=item['assets']['visual'];sha=expected(asset['file:checksum'])
                target=root/'images'/(name+'.tiff')
                if target.exists() and checksum(target)==sha: return target.stat().st_size
                pilot=Path('data/nz_pilot')/name/'rgb_native.tif'
                if not target.exists() and pilot.exists() and checksum(pilot)==sha:
                    try: os.link(pilot,target)
                    except OSError: shutil.copyfile(pilot,target)
                    return target.stat().st_size
                temporary=target.with_suffix('.tiff.part');h=hashlib.sha256()
                with fetch(urljoin(url,asset['href'])) as response, temporary.open('wb') as f:
                    while chunk:=response.read(4*1024*1024):
                        if shutil.disk_usage(root).free<len(chunk)+reserve:
                            stop.set();raise RuntimeError('Disk reserve reached')
                        f.write(chunk);h.update(chunk)
                if h.hexdigest()!=sha: raise ValueError('Image checksum mismatch: '+name)
                os.replace(temporary,target)
                return target.stat().st_size
            except Exception as exc:
                last=exc
                if not stop.is_set() and attempt<4: time.sleep(min(2**attempt,16))
        raise RuntimeError(str(last))
    atomic_json(root/'status.json',state)
    print('Downloading',len(links),'tiles to',root,flush=True)
    start=time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(task,link):link for link in links}
        for future in as_completed(futures):
            try:
                state['verified_bytes']+=future.result();state['verified']+=1
            except Exception as exc:
                state['failed']+=1;state['errors'].append({'tile':futures[future]['href'],'error':str(exc)})
            state['updated_utc']=datetime.now(timezone.utc).isoformat()
            state['elapsed_seconds']=round(time.monotonic()-start)
            atomic_json(root/'status.json',state)
            if (state['verified']+state['failed'])%25==0:
                print(f"Verified {state['verified']}/{len(links)}; {state['verified_bytes']/1024**3:.2f} GiB; failures {state['failed']}",flush=True)
    state['status']='complete' if state['verified']==len(links) else 'incomplete'
    atomic_json(root/'status.json',state)
    print(state['status'],state['verified'],'verified;',state['failed'],'failed',flush=True)
    if state['failed']: raise SystemExit(1)

if __name__=='__main__':main()
