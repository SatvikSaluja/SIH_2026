"""Create the uploadable Colab ZIP from the downloaded pilot (no native rasters)."""
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
from PIL import Image,ImageDraw
import rasterio
from rasterio.features import rasterize
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform

root=Path('data/nz_pilot'); manifest=json.loads((root/'manifest.json').read_text())
canvas=Image.new('RGB',(1200,900),'white'); draw=ImageDraw.Draw(canvas)
project=Transformer.from_crs(4326,2193,always_xy=True).transform
for i,entry in enumerate(manifest['tiles']):
    folder=root/entry['tile']
    with rasterio.open(folder/'rgb.tif') as src:
        rgb=src.read().transpose(1,2,0); affine=src.transform
    features=json.loads((folder/'parcels.geojson').read_text())['features']
    edges=[(transform(project,shape(f['geometry'])).boundary,1) for f in features]
    boundary=rasterize(edges,out_shape=rgb.shape[:2],transform=affine,all_touched=True)>0
    # Overlay parcel-map boundaries in red for visual alignment review.
    rgb[boundary]=[255,30,30]
    im=Image.fromarray(rgb); im.thumbnail((390,410)); x=i%3*400; y=i//3*450
    canvas.paste(im,(x,y)); draw.text((x,y+415),entry['tile']+' / '+entry['split'],fill='black')
canvas.save(root/'overview.jpg',quality=90)
out=Path('artifacts/nz_colab_bundle.zip'); out.parent.mkdir(exist_ok=True)
paths=list(Path('scripts/colab').glob('*.py'))+list(Path('scripts/colab').glob('*.txt'))+[Path('notebooks/GeoCadastra_NZ_Colab.ipynb'),Path('notebooks/NZ_COLAB_README.md')]
paths += [root/p for p in ['manifest.json','collection.json','parcel_service.json','overview.jpg']]
for entry in manifest['tiles']:
    paths += [p for p in (root/entry['tile']).iterdir() if p.is_file() and p.name!='rgb_native.tif']
with ZipFile(out,'w',ZIP_DEFLATED) as z:
    for path in paths: z.write(path,str(path))
print(out,round(out.stat().st_size/1024**2,1),'MiB')
