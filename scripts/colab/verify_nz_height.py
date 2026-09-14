"""Validate the complete 100-image PNG and elevation bundle before packaging."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
from PIL import Image
import rasterio

root=Path('data/nz_height_100');manifest=json.loads((root/'manifest.json').read_text())
assert manifest['status']=='complete' and manifest['count']==100
assert len({e['tile'] for e in manifest['tiles']})==100
for e in manifest['tiles']:
 m=json.loads((root/e['metadata']).read_text())
 with Image.open(root/e['png']) as im:
  assert im.format=='PNG' and im.mode=='RGB';w,h=im.size
 with rasterio.open(root/e['height']) as src:
  assert (src.width,src.height)==(w,h)
  assert list(src.transform)[:6]==m['png_pixel_to_map_transform']
  assert src.crs.to_epsg()==2193 and src.count==3 and src.units==('m','m','m')
  arrays=src.read();valid=src.dataset_mask()>0
  assert valid.mean()>=0.995 and abs(valid.mean()-m['valid_fraction'])<1e-10
  assert np.isfinite(arrays[:,valid]).all()
  np.testing.assert_allclose(arrays[2,valid],arrays[1,valid]-arrays[0,valid],rtol=1e-6,atol=1e-6)
 for key,filekey in [('png_sha256','rgb_png'),('height_sha256','height_geotiff')]:
  assert hashlib.sha256((root/m[filekey]).read_bytes()).hexdigest()==m[key]
summary={'verified_images':100,'checks':['PNG format and dimensions','height band units and CRS','pixel-to-map transform agreement','at least 99.5% valid height coverage','finite valid heights','height equals DSM minus DEM','output SHA256 checksums'],'scope':'File integrity and numerical alignment, not independent survey accuracy'}
(root/'verification.json').write_text(json.dumps(summary,indent=2))
paths=[p for p in root.rglob('*') if p.is_file() and 'source_elevation' not in p.parts]
archive=Path('artifacts/nz_height_100.zip')
with ZipFile(archive,'w',ZIP_DEFLATED) as z:
 for path in paths:z.write(path,str(path.relative_to(root)))
with ZipFile(archive) as z:assert z.testzip() is None
print('Verified 100 PNG/height pairs. ZIP:',archive,round(archive.stat().st_size/1024**2,1),'MiB')
