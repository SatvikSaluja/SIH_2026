"""Create PNG viewing copies of 20 actual downloaded New Zealand image tiles."""
import html
import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from PIL import Image, ImageDraw

out=Path('data/nz_png_preview');out.mkdir(parents=True,exist_ok=True)
pilot=Path('data/nz_pilot')
manifest=json.loads((pilot/'manifest.json').read_text())
candidates=[(pilot/t['tile']/'rgb_native.tif',t['tile'],t['split']) for t in manifest['tiles']]
known={t['tile'] for t in manifest['tiles']}
candidates.extend((p,p.stem,'additional candidate') for p in sorted(Path('data/nz_christchurch_all/images').glob('*.tiff')) if p.stem not in known)
entries=[]
for path,name,split in candidates:
    with rasterio.open(path) as src:
        factor=min(1,1000/max(src.width,src.height))
        h=max(1,round(src.height*factor));w=max(1,round(src.width*factor))
        rgb=src.read([1,2,3],out_shape=(3,h,w),resampling=Resampling.average)
        valid=src.dataset_mask(out_shape=(h,w),resampling=Resampling.nearest)>0
        if valid.mean()<0.85 or (rgb.max(axis=0)>5).mean()<0.8:continue
        if src.dtypes[0]!='uint8':raise ValueError('Expected 8-bit RGB')
        im=Image.fromarray(np.moveaxis(rgb,0,-1))
        im.save(out/(name+'.png'))
        entries.append({'tile':name,'split':split,'source':str(path),'png':name+'.png',
                        'native_size':[src.width,src.height],'preview_size':[w,h],
                        'crs':str(src.crs),'transform':list(src.transform)[:6]})
    if len(entries)==20:break
if len(entries)<20:raise RuntimeError('Need more nonblank downloaded images')
cell_w,cell_h=320,470
canvas=Image.new('RGB',(cell_w*5,cell_h*4),'#f1f5f9');draw=ImageDraw.Draw(canvas)
for i,e in enumerate(entries):
    im=Image.open(out/e['png']);im.thumbnail((300,410))
    x=(i%5)*cell_w;y=(i//5)*cell_h
    canvas.paste(im,(x+(cell_w-im.width)//2,y+10))
    draw.text((x+10,y+425),f"{i+1:02d}  {e['tile']}",fill='#111827')
    draw.text((x+10,y+445),e['split'],fill='#334155')
canvas.save(out/'all_20_images.png')
(out/'manifest.json').write_text(json.dumps(entries,indent=2))
cards=''.join(f'<a href="{e["png"]}"><img loading="lazy" src="{e["png"]}"><span>{html.escape(e["tile"])} — {html.escape(e["split"])}</span></a>' for e in entries)
(out/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>20 New Zealand image previews</title>
<style>body{font:16px system-ui;background:#f1f5f9;margin:24px;color:#172033}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:20px}a{background:white;padding:12px;border-radius:10px;color:inherit;text-decoration:none}img{width:100%;height:320px;object-fit:contain}span{display:block;font-size:13px}p{max-width:900px;line-height:1.5}</style>
<h1>20 actual New Zealand images</h1><p>Click an image for its PNG. These are reduced-size viewing copies of downloaded GeoTIFFs, with no generated content. Six are the current pilot split; fourteen are additional imagery candidates whose parcel labels and training splits have not yet been prepared. Original GeoTIFFs retain map coordinates.</p><div class="grid">'''+cards+'</div>')
print('Created',len(entries),'PNGs, contact sheet and gallery:',out)
