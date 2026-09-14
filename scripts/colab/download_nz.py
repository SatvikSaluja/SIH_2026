"""Small public Christchurch RGB/parcel pilot. No credentials or paid services.
Parcel labels are from a public LINZ-derived ArcGIS mirror, not independent surveys.
"""
import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import urljoin, urlencode

COLLECTION = 'https://nz-imagery.s3.ap-southeast-2.amazonaws.com/canterbury/christchurch_2025_0.075m/rgb/2193/collection.json'
PARCELS = 'https://services.arcgis.com/xdsHIIxuCWByZiCB/arcgis/rest/services/LINZ_NZ_Primary_Parcels/FeatureServer/0'
# Separate source tiles, not random crops from the same source image.
SITES = [('BX24_500_015044','train'), ('BX24_500_025044','train'),
         ('BX24_500_035044','train'), ('BX24_500_020044','train'),
         ('BX24_500_025024','val'), ('BX24_500_015024','test')]

def get_json(url):
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={'User-Agent':'GeoCadastra-research-pilot/1.0'}),timeout=90) as r:
                value=json.load(r)
            if 'error' in value: raise RuntimeError(value['error'])
            return value
        except Exception:
            if attempt == 2: raise
            time.sleep(2)

def save_json(path, value):
    path.write_text(json.dumps(value,indent=2))

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def download(url,path,checksum=None):
    expected=checksum[4:] if checksum and checksum.startswith('1220') else None
    if path.exists() and expected and digest(path)==expected: return
    temp=path.with_suffix(path.suffix+'.part')
    with urlopen(url,timeout=120) as src, open(temp,'wb') as dst:
        while chunk:=src.read(1024*1024): dst.write(chunk)
    if expected and digest(temp)!=expected:
        raise ValueError('Download checksum mismatch: '+str(path))
    temp.replace(path)

def main():
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import rasterize
    from rasterio.transform import array_bounds
    from shapely.geometry import shape, box
    from shapely.ops import transform as transform_geometry
    from pyproj import Transformer
    from scipy.ndimage import binary_erosion
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='data/nz_pilot')
    p.add_argument('--gsd',type=float,default=0.30)
    args=p.parse_args()
    if not 0.075<=args.gsd<=1: raise ValueError('gsd must be between 0.075 and 1 metre')
    root=Path(args.out); root.mkdir(parents=True,exist_ok=True)
    c=get_json(COLLECTION); save_json(root/'collection.json',c)
    metadata=get_json(PARCELS+'?f=pjson'); save_json(root/'parcel_service.json',metadata)
    available={Path(x['href']).stem:x for x in c['links'] if x['rel']=='item'}
    entries=[]; ids_by_split={}; footprints=[]
    project=Transformer.from_crs(4326,2193,always_xy=True).transform
    for tile,split in SITES:
        if tile not in available: raise ValueError('Missing source tile '+tile)
        folder=root/tile; folder.mkdir(exist_ok=True)
        item_url=urljoin(COLLECTION,available[tile]['href'])
        item=get_json(item_url); save_json(folder/'item.json',item)
        asset=item['assets']['visual']; image_url=urljoin(item_url,asset['href'])
        raw=folder/'rgb_native.tif'
        print('Downloading',tile,split,flush=True)
        download(image_url,raw,asset.get('file:checksum'))
        with rasterio.open(raw) as src:
            if src.crs.to_epsg()!=2193 or src.count<3 or src.dtypes[0]!='uint8':
                raise ValueError('Unexpected source raster format')
            w=round(src.width*abs(src.transform.a)/args.gsd)
            h=round(src.height*abs(src.transform.e)/args.gsd)
            affine=src.transform*src.transform.scale(src.width/w,src.height/h)
            rgb=src.read([1,2,3],out_shape=(3,h,w),resampling=Resampling.average)
            image_valid=src.dataset_mask(out_shape=(h,w),resampling=Resampling.nearest)>0
            profile=dict(driver='GTiff',width=w,height=h,count=3,dtype='uint8',crs=src.crs,transform=affine,compress='deflate')
        with rasterio.open(folder/'rgb.tif','w',**profile) as dst:
            dst.write(rgb); dst.write_mask(image_valid.astype('uint8')*255)
        query={'f':'geojson','where':'1=1','geometry':','.join(map(str,item['bbox'])),
               'geometryType':'esriGeometryEnvelope','inSR':4326,'outSR':4326,
               'spatialRel':'esriSpatialRelIntersects','outFields':'id,parcel_intent,survey_area,calc_area',
               'returnGeometry':'true','resultRecordCount':1000,'orderByFields':'OBJECTID'}
        features=[]; offset=0
        while True:
            response=get_json(PARCELS+'/query?'+urlencode(dict(query,resultOffset=offset)))
            batch=response['features']; features.extend(batch)
            if not response.get('exceededTransferLimit') and len(batch)<1000: break
            if not batch: raise RuntimeError('Parcel pagination stalled')
            offset+=len(batch)
        if not features: raise ValueError('No parcels returned for '+tile)
        save_json(folder/'parcels.geojson',{'type':'FeatureCollection','features':features})
        bounds=box(*array_bounds(h,w,affine)); footprints.append((split,bounds))
        polygons=[]; lines=[]; split_ids=set(); records=[]
        for feature in features:
            geom=transform_geometry(project,shape(feature['geometry']))
            if not geom.is_valid: raise ValueError('Invalid parcel geometry; review before training')
            if not geom.intersects(bounds): continue
            props=feature['properties']; split_ids.add(str(props['id']))
            polygons.append((geom,1)); lines.append((geom.boundary,1))
            records.append({**props,'fully_within_tile':bounds.contains(geom)})
        ids_by_split.setdefault(split,set()).update(split_ids)
        coverage=rasterize(polygons,out_shape=(h,w),transform=affine,dtype='uint8')>0
        boundary=rasterize(lines,out_shape=(h,w),transform=affine,all_touched=True,dtype='uint8')
        valid=binary_erosion(coverage & image_valid,iterations=2,border_value=0)
        if valid.mean()<0.25 or boundary[valid].sum()<20:
            raise ValueError('Insufficient labelled imagery for '+tile)
        with open(folder/'records.csv','w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['id','parcel_intent','survey_area','calc_area','fully_within_tile'])
            writer.writeheader(); writer.writerows(records)
        np.savez_compressed(folder/'training.npz',rgb=rgb,boundary=boundary,valid=valid)
        entries.append({'tile':tile,'split':split,'path':tile+'/training.npz','gsd_m':abs(affine.a),
                        'parcels':len(records),'valid_fraction':float(valid.mean()),'image_url':image_url,
                        'native_sha256':digest(raw),'training_sha256':digest(folder/'training.npz'),
                        'parcel_sha256':digest(folder/'parcels.geojson'),'bbox_wgs84':item['bbox']})
        print('Prepared',tile,len(records),'parcels',rgb.shape,flush=True)
    for a in ids_by_split:
        for b in ids_by_split:
            if a!=b and ids_by_split[a]&ids_by_split[b]:
                raise ValueError('Parcel identity leakage across splits; choose different sites')
    for a,ga in footprints:
        for b,gb in footprints:
            if a!=b and ga.distance(gb)<100: raise ValueError('Split sites must be separated by at least 100m')
    save_json(root/'manifest.json',{'format_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),
        'collection_url':COLLECTION,'parcel_source':PARCELS,'parcel_source_type':'public LINZ-derived mirror',
        'imagery_license':c['license'],'providers':c['providers'],'crs':'EPSG:2193',
        'limitations':['Current parcel snapshot versus March 2025 imagery: temporal alignment unverified',
                      'Parcel maps are reference labels, not independently surveyed ground truth',
                      'No height data, historic input map, or independent survey checks in this pilot',
                      'survey_area retained separately; never replaced by geometry-derived area',
                      'Tiny spatial holdout is an engineering pilot, not national validation'],
        'tiles':entries})
    print('Ready:',root/'manifest.json',flush=True)

if __name__=='__main__': main()
