"""Create 100 PNGs paired with actual LINZ DSM/DEM rasters and metadata.
This is image/height data, not new parcel-boundary labels or building-specific heights.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import html
import json
import math
from pathlib import Path
from urllib.parse import urljoin
from datetime import datetime,timezone
import numpy as np
from PIL import Image,ImageDraw
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from shapely.geometry import shape
from download_nz import get_json,download,digest

BASE='https://nz-elevation.s3.ap-southeast-2.amazonaws.com/canterbury/christchurch_2024-2025/'

def write_json(path,value):path.write_text(json.dumps(value,indent=2))

def stats(a):
 return {k:float(v) for k,v in zip(['min','p05','median','p95','max'],np.percentile(a,[0,5,50,95,100]))}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--count',type=int,default=100);a=p.parse_args()
 if not 1<=a.count<=1000:raise ValueError('Invalid count')
 root=Path('data/nz_height_100');root.mkdir(exist_ok=True)
 for d in ['png','height','metadata','source_elevation']: (root/d).mkdir(exist_ok=True)
 collections={};items={};urls={}
 for kind in ['dem','dsm']:
  url=BASE+kind+'_1m/2193/collection.json';urls[kind]=url
  c=get_json(url);collections[kind]=c;write_json(root/(kind+'_collection.json'),c)
  links=[l for l in c['links'] if l['rel']=='item']
  def load(link):
   local=root/'source_elevation'/(kind+'_'+Path(link['href']).name)
   download(urljoin(url,link['href']),local,link['file:checksum'])
   return json.loads(local.read_bytes())
  with ThreadPoolExecutor(max_workers=6) as pool: records=list(pool.map(load,links))
  items[kind]={x['id']:x for x in records}
  print('Indexed',kind,len(records),'elevation tiles',flush=True)
 imagery=Path('data/nz_christchurch_all');pilot=Path('data/nz_pilot')
 collection=json.loads((imagery/'collection.json').read_text());write_json(root/'imagery_collection.json',collection)
 # Prefer the existing pilot locations, then the other completed downloads.
 candidates=[];seen=set()
 for t in json.loads((pilot/'manifest.json').read_text())['tiles']:
  candidates.append((t['tile'],pilot/t['tile']/'rgb_native.tif',pilot/t['tile']/'item.json'));seen.add(t['tile'])
 for image in sorted((imagery/'images').glob('*.tiff')):
  if image.stem not in seen:candidates.append((image.stem,image,imagery/'metadata'/(image.stem+'.json')))
 entries=[];rejections=[];cache={}
 def elevation(kind,key):
  pair=(kind,key)
  if pair not in cache:
   item=items[kind][key];asset=item['assets']['visual'];path=root/'source_elevation'/(kind+'_'+key+'.tiff')
   print('Downloading elevation',kind,key,flush=True)
   download(urljoin(urls[kind],asset['href']),path,asset['file:checksum'])
   cache[pair]=path
  return cache[pair]
 for name,image,meta_path in candidates:
  if len(entries)==a.count:break
  if not meta_path.exists():continue
  item=json.loads(meta_path.read_bytes());footprint=shape(item['geometry'])
  matches=[k for k,d in items['dem'].items() if k in items['dsm'] and shape(d['geometry']).buffer(1e-7).covers(footprint) and shape(items['dsm'][k]['geometry']).buffer(1e-7).covers(footprint)]
  if not matches:rejections.append([name,'No single matching DEM+DSM tile covering full image']);continue
  key=matches[0]
  with rasterio.open(image) as src:
   if src.crs.to_epsg()!=2193 or src.dtypes[0]!='uint8':raise ValueError('Unexpected imagery format')
   w=round(src.width*abs(src.transform.a)/0.3);h=round(src.height*abs(src.transform.e)/0.3)
   transform=src.transform*src.transform.scale(src.width/w,src.height/h)
   rgb=src.read([1,2,3],out_shape=(3,h,w),resampling=Resampling.average)
   rgb_valid=src.dataset_mask(out_shape=(h,w),resampling=Resampling.nearest)>0
   if rgb_valid.mean()<0.995 or (rgb.max(axis=0)>5).mean()<0.99:
    rejections.append([name,'Incomplete/blank RGB coverage']);continue
   original={'path':str(image),'sha256':digest(image),'width':src.width,'height':src.height,'transform':list(src.transform)[:6],'crs':str(src.crs),'gsd_m':abs(src.transform.a)}
  surfaces={};sources={};valid=rgb_valid.copy(); coverage_ok=True
  for kind in ['dem','dsm']:
   path=elevation(kind,key)
   with rasterio.open(path) as src:
    if src.crs.to_epsg()!=2193 or not math.isclose(abs(src.transform.a),1):raise ValueError('Unexpected elevation CRS/resolution')
    if not (src.bounds.left<=transform.c and src.bounds.right>=transform.c+w*transform.a and src.bounds.top>=transform.f and src.bounds.bottom<=transform.f+h*transform.e):
     coverage_ok=False;break
    arr=np.full((h,w),np.nan,dtype='float32')
    reproject(rasterio.band(src,1),arr,src_transform=src.transform,src_crs=src.crs,src_nodata=src.nodata,
      dst_transform=transform,dst_crs='EPSG:2193',dst_nodata=np.nan,resampling=Resampling.bilinear)
    surfaces[kind]=arr;valid &= np.isfinite(arr)
    sources[kind]={'item':items[kind][key],'raster_crs':str(src.crs),'native_gsd_m':abs(src.transform.a),
     'band_units':list(src.units),'tags':src.tags(),'band_tags':src.tags(1),'source_file':str(path),
     'source_checksum':items[kind][key]['assets']['visual']['file:checksum']}
  if not coverage_ok:
   rejections.append([name,'Elevation raster bounds do not fully cover image']);continue
  if valid.mean()<0.995:
   rejections.append([name,'Less than 99.5 percent joint RGB/DSM/DEM coverage']);continue
  ndsm=surfaces['dsm']-surfaces['dem']
  # Preserve negative residuals for quality review; never silently clamp them away.
  for arr in [*surfaces.values(),ndsm]:arr[~valid]=np.nan
  Image.fromarray(np.moveaxis(rgb,0,-1)).save(root/'png'/(name+'.png'))
  profile=dict(driver='GTiff',height=h,width=w,count=3,dtype='float32',crs='EPSG:2193',transform=transform,nodata=np.nan,compress='deflate',predictor=3)
  with rasterio.open(root/'height'/(name+'.tif'),'w',**profile) as dst:
   for i,(label,arr) in enumerate([('DEM_ground_elevation_m',surfaces['dem']),('DSM_surface_elevation_m',surfaces['dsm']),('nDSM_height_above_ground_m',ndsm)],1):
    dst.write(arr,i);dst.set_band_description(i,label);dst.set_band_unit(i,'m')
   dst.write_mask(valid.astype('uint8')*255)
  metadata={'tile':name,'rgb_png':'png/'+name+'.png','height_geotiff':'height/'+name+'.tif',
    'crs':'EPSG:2193','png_pixel_to_map_transform':list(transform)[:6],'png_width':w,'png_height':h,'png_gsd_m':abs(transform.a),
    'height_units':'metres','native_height_gsd_m':1,'height_resampling':'bilinear to RGB PNG grid; does not create 30 cm height detail',
    'vertical_datum':'See source metadata; not yet independently verified',
    'rgb_source':original,'rgb_stac':item,'height_sources':sources,'valid_fraction':float(valid.mean()),
    'ground_elevation_m':stats(surfaces['dem'][valid]),'surface_elevation_m':stats(surfaces['dsm'][valid]),
    'height_above_ground_m':stats(ndsm[valid]),'negative_height_fraction':float((ndsm[valid]<0).mean()),
    'limitations':['Height above ground includes trees and buildings; no building-instance height labels',
      'Elevation captured Dec 2024-Jan 2025; imagery March 2025, not simultaneous',
      'No new parcel labels, recorded areas or independent survey checks added by this export',
      'Negative DSM-DEM residuals preserved; all height estimates require quality review'],
    'png_sha256':digest(root/'png'/(name+'.png')),'height_sha256':digest(root/'height'/(name+'.tif'))}
  write_json(root/'metadata'/(name+'.json'),metadata)
  entries.append({'tile':name,'png':metadata['rgb_png'],'metadata':'metadata/'+name+'.json','height':metadata['height_geotiff'],
    'valid_fraction':metadata['valid_fraction'],'height_p95_m':metadata['height_above_ground_m']['p95']})
  write_json(root/'manifest.json',{'status':'preparing','count':len(entries),'requested':a.count,'tiles':entries,'rejected':rejections})
  print('Prepared',len(entries),'/',a.count,name,flush=True)
 write_json(root/'manifest.json',{'status':'complete' if len(entries)==a.count else 'incomplete','count':len(entries),'requested':a.count,'created_utc':datetime.now(timezone.utc).isoformat(),'tiles':entries,'rejected':rejections})
 cards=''.join(f'<article><a href="{e["png"]}"><img loading="lazy" src="{e["png"]}"></a><p>{html.escape(e["tile"])}<br>95th percentile height: {e["height_p95_m"]:.2f} m<br><a href="{e["metadata"]}">Metadata</a> · <a href="{e["height"]}">Height raster</a></p></article>' for e in entries)
 (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>New Zealand imagery with height data</title><style>body{font:16px system-ui;background:#eef2f6;margin:24px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}article{background:white;padding:10px}img{width:100%;height:300px;object-fit:contain}</style><h1>'+str(len(entries))+' images with matched ground and surface elevation</h1><p>Height is DSM minus DEM, including trees and buildings. It is not a separate building-height label. Click any image for its PNG.</p><div class="grid">'+cards+'</div>')
 for page in range(math.ceil(len(entries)/20)):
  canvas=Image.new('RGB',(1200,1440),'#eef2f6');draw=ImageDraw.Draw(canvas)
  for j,e in enumerate(entries[page*20:(page+1)*20]):
   im=Image.open(root/e['png']);im.thumbnail((230,315));x=j%5*240;y=j//5*360
   canvas.paste(im,(x,y));draw.text((x+3,y+320),e['tile'],fill='black');draw.text((x+3,y+340),f"Height p95 {e['height_p95_m']:.2f} m",fill='black')
  canvas.save(root/f'grid_{page+1:02d}.png')
 print('Finished',len(entries),'height-paired PNGs',flush=True)
 if len(entries)!=a.count:raise SystemExit('Not enough valid candidates; download more images and rerun')

if __name__=='__main__':main()
