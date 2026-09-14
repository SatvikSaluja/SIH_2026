"""Join verified NZ parcel references and measured height on exactly matching grids."""
import hashlib
import json
from pathlib import Path
import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt


def compute_distance_and_mask(boundary, valid, ndsm, gsd_m):
    """`(distance, mask)` for one tile: distance to the nearest labeled
    boundary in metres, and the pixels where that distance is provably
    correct.

    A pixel's distance is only trustworthy when the nearest UNKNOWN
    (unlabeled) region is farther away than the nearest labeled boundary --
    otherwise there could be a closer, unmapped boundary hiding just past
    the edge of known coverage, and the computed distance would understate
    how close the true nearest boundary actually is.
    """
    distance = distance_transform_edt(~boundary.astype(bool)) * gsd_m
    known = valid.astype(bool) & np.isfinite(ndsm)
    unknown_distance = distance_transform_edt(np.pad(known, 1))[1:-1, 1:-1] * gsd_m
    mask = known & (unknown_distance > distance)
    return distance, mask


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', default='data/nz_pilot', help='Parcel-labelled pilot root (e.g. data/nz_pilot_v2)')
    p.add_argument('--heights', default='data/nz_height_100')
    p.add_argument('--out', default='data/nz_multitask_real')
    args = p.parse_args()
    pilot, heights, out = Path(args.pilot), Path(args.heights), Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    manifest = json.loads((pilot/'manifest.json').read_text())
    result = dict(crs=manifest['crs'],gsd_m=.3,tiles=[],skipped=[],
        parcel_source=manifest['parcel_source'],imagery_license=manifest['imagery_license'],
        limitations=['Small spatial holdout; parcel mirror accuracy unverified',
                    'Imagery and height capture dates differ; vertical datum unverified',
                    'Only parcel distance supervised; other heads have no labels'])
    for entry in manifest['tiles']:
        tile = entry['tile']; heightpath = heights/'height'/f'{tile}.tif'
        if not heightpath.exists():
            result['skipped'].append(dict(tile=tile,reason='No matched height')); continue
        src = pilot/entry['path']
        assert hashlib.sha256(src.read_bytes()).hexdigest()==entry['training_sha256']
        metadata = json.loads((heights/'metadata'/f'{tile}.json').read_text())
        assert hashlib.sha256(heightpath.read_bytes()).hexdigest()==metadata['height_sha256']
        with np.load(src) as z:
            rgb,boundary,valid = (z[k].copy() for k in ('rgb','boundary','valid'))
        with rasterio.open(pilot/tile/'rgb.tif') as image, rasterio.open(heightpath) as height:
            assert image.crs==height.crs and image.transform==height.transform and image.shape==height.shape
            assert height.shape==boundary.shape and np.array_equal(image.read(),rgb)
            assert height.descriptions[2]=='nDSM_height_above_ground_m'
            # compute_distance_and_mask() scales an index-space distance transform
            # by one scalar gsd_m -- only correct for square, axis-aligned pixels.
            # A rotated or non-square grid would silently mis-scale every distance.
            a,b,d,e = height.transform.a,height.transform.b,height.transform.d,height.transform.e
            assert b==0 and d==0, 'rotated raster: distance scaling assumes axis-aligned pixels'
            assert abs(abs(a)-.3)<1e-6 and abs(abs(e)-.3)<1e-6, f'expected 0.3m square pixels, got {a},{e}'
            ndsm = height.read(3)
        # Distances are computed before cropping. Unknown areas and the tile exterior
        # cannot provide a closer unseen boundary than the supervised reference.
        distance, valid = compute_distance_and_mask(boundary, valid, ndsm, .3)
        dest = out/f'{tile}.npz'
        np.savez_compressed(dest,rgb=rgb,ndsm=np.nan_to_num(ndsm).astype('float32'),
                            distance=distance.astype('float32'),valid=valid)
        result['tiles'].append(dict(tile=tile,split=entry['split'],path=dest.name,
            sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
            image_training_sha256=entry['training_sha256'],height_sha256=metadata['height_sha256'],
            valid_fraction=float(valid.mean())))
    (out/'manifest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
