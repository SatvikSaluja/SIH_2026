"""Extend the parcel-labelled NZ pilot with more real tiles, reusing
download_nz.py's own per-tile fetch/rasterize logic rather than
duplicating it.

Differences from download_nz.py's original 6-site run, all required at
this scale:
  - A bad tile logs and is skipped, not aborted into losing every tile
    already fetched in the same run (download_nz.py's SITES list was
    small and hand-vetted; a screened batch of real candidates is not).
  - Split is an explicit per-tile choice, not inferred -- see the
    module docstring in screen_nz_candidates.py for why a geographically
    clustered batch should not be scattered across train/val/test.
  - The original six tiles' files are copied, not re-downloaded or
    re-processed, into a new versioned root -- data/nz_pilot stays an
    untouched, checksummed snapshot; data/nz_pilot_v2 is the first
    self-contained expanded dataset prepare_real.py can point at.
  - Footprint-separation and parcel-ID-overlap checks run across the
    COMBINED old+new set, not just within the new batch.
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import download_nz as base  # noqa: E402


def process_one_tile(tile, split, args, collection, available):
    """download_nz.py's main() body, for exactly one tile, returning its
    manifest entry. Raises on any failure -- the caller decides whether
    that aborts the batch or just skips this tile."""
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import rasterize
    from rasterio.transform import array_bounds
    from shapely.geometry import shape, box
    from shapely.ops import transform as transform_geometry
    from pyproj import Transformer
    from scipy.ndimage import binary_erosion
    from urllib.parse import urljoin, urlencode

    if tile not in available:
        raise ValueError('Missing source tile ' + tile)
    folder = args.out / tile
    folder.mkdir(parents=True, exist_ok=True)
    item_url = urljoin(base.COLLECTION, available[tile]['href'])
    item = base.get_json(item_url)
    base.save_json(folder / 'item.json', item)
    asset = item['assets']['visual']
    image_url = urljoin(item_url, asset['href'])
    raw = folder / 'rgb_native.tif'
    print('Downloading', tile, split, flush=True)
    base.download(image_url, raw, asset.get('file:checksum'))
    with rasterio.open(raw) as src:
        if src.crs.to_epsg() != 2193 or src.count < 3 or src.dtypes[0] != 'uint8':
            raise ValueError('Unexpected source raster format')
        w = round(src.width * abs(src.transform.a) / args.gsd)
        h = round(src.height * abs(src.transform.e) / args.gsd)
        affine = src.transform * src.transform.scale(src.width / w, src.height / h)
        rgb = src.read([1, 2, 3], out_shape=(3, h, w), resampling=Resampling.average)
        image_valid = src.dataset_mask(out_shape=(h, w), resampling=Resampling.nearest) > 0
        profile = dict(driver='GTiff', width=w, height=h, count=3, dtype='uint8',
                       crs=src.crs, transform=affine, compress='deflate')
    with rasterio.open(folder / 'rgb.tif', 'w', **profile) as dst:
        dst.write(rgb)
        dst.write_mask(image_valid.astype('uint8') * 255)

    project = Transformer.from_crs(4326, 2193, always_xy=True).transform
    query = {'f': 'geojson', 'where': '1=1', 'geometry': ','.join(map(str, item['bbox'])),
             'geometryType': 'esriGeometryEnvelope', 'inSR': 4326, 'outSR': 4326,
             'spatialRel': 'esriSpatialRelIntersects', 'outFields': 'id,parcel_intent,survey_area,calc_area',
             'returnGeometry': 'true', 'resultRecordCount': 1000, 'orderByFields': 'OBJECTID'}
    features = []
    offset = 0
    while True:
        response = base.get_json(base.PARCELS + '/query?' + urlencode(dict(query, resultOffset=offset)))
        batch = response['features']
        features.extend(batch)
        if not response.get('exceededTransferLimit') and len(batch) < 1000:
            break
        if not batch:
            raise RuntimeError('Parcel pagination stalled')
        offset += len(batch)
    if not features:
        raise ValueError('No parcels returned for ' + tile)
    base.save_json(folder / 'parcels.geojson', {'type': 'FeatureCollection', 'features': features})

    bounds = box(*array_bounds(h, w, affine))
    polygons, lines, split_ids, records = [], [], set(), []
    for feature in features:
        # Original, unclipped geometry rasterized against the tile grid --
        # rasterize() only draws what falls inside the output array; it
        # never invents a boundary at the tile edge the way clipping the
        # polygon to `bounds` first would.
        geom = transform_geometry(project, shape(feature['geometry']))
        if not geom.is_valid:
            raise ValueError('Invalid parcel geometry; review before training')
        if not geom.intersects(bounds):
            continue
        props = feature['properties']
        split_ids.add(str(props['id']))
        polygons.append((geom, 1))
        lines.append((geom.boundary, 1))
        records.append({**props, 'fully_within_tile': bounds.contains(geom)})

    coverage = rasterize(polygons, out_shape=(h, w), transform=affine, dtype='uint8') > 0
    boundary = rasterize(lines, out_shape=(h, w), transform=affine, all_touched=True, dtype='uint8')
    valid = binary_erosion(coverage & image_valid, iterations=2, border_value=0)
    if valid.mean() < 0.25 or boundary[valid].sum() < 20:
        raise ValueError(f'Insufficient labelled imagery for {tile} '
                         f'(valid={valid.mean():.3f}, boundary_px={int(boundary[valid].sum())})')

    import csv
    with open(folder / 'records.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'parcel_intent', 'survey_area', 'calc_area', 'fully_within_tile'])
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(folder / 'training.npz', rgb=rgb, boundary=boundary, valid=valid)
    entry = {'tile': tile, 'split': split, 'path': tile + '/training.npz', 'gsd_m': abs(affine.a),
            'parcels': len(records), 'valid_fraction': float(valid.mean()), 'image_url': image_url,
            'native_sha256': base.digest(raw), 'training_sha256': base.digest(folder / 'training.npz'),
            'parcel_sha256': base.digest(folder / 'parcels.geojson'), 'bbox_wgs84': item['bbox']}
    print('Prepared', tile, len(records), 'parcels', rgb.shape, flush=True)
    return entry, split_ids, bounds


def main():
    from shapely.geometry import box as shapely_box

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates', required=True, help='JSON: [{"tile":..., "split":...}, ...]')
    p.add_argument('--original-pilot', type=Path, default=Path('data/nz_pilot'))
    p.add_argument('--out', type=Path, default=Path('data/nz_pilot_v2'))
    p.add_argument('--gsd', type=float, default=0.30)
    args = p.parse_args()
    if not 0.075 <= args.gsd <= 1:
        raise ValueError('gsd must be between 0.075 and 1 metre')
    args.out.mkdir(parents=True, exist_ok=True)

    original = json.loads((args.original_pilot / 'manifest.json').read_text())
    # Copy the original six, unmodified -- never re-download what we already
    # have and have already checksum-verified once.
    kept_entries = []
    for entry in original['tiles']:
        src_folder = args.original_pilot / entry['tile']
        dst_folder = args.out / entry['tile']
        if not dst_folder.exists():
            shutil.copytree(src_folder, dst_folder)
        actual = base.digest(dst_folder / 'training.npz')
        if actual != entry['training_sha256']:
            raise ValueError(f'Copied original tile {entry["tile"]} does not match its recorded checksum')
        kept_entries.append(entry)
    print(f'Carried over {len(kept_entries)} original tiles unchanged (their split assignments untouched)', flush=True)

    requested = json.loads(Path(args.candidates).read_text())
    collection = base.get_json(base.COLLECTION)
    base.save_json(args.out / 'collection.json', collection)
    base.save_json(args.out / 'parcel_service.json', base.get_json(base.PARCELS + '?f=pjson'))
    available = {Path(x['href']).stem: x for x in collection['links'] if x['rel'] == 'item'}

    new_entries, failures = [], []
    footprints = [(e['split'], shapely_box(*e['bbox_wgs84'])) for e in kept_entries]
    ids_by_split = {}
    for e in kept_entries:
        # Original six's own parcel ids: re-derive from their parcels.geojson
        # so the overlap check below covers the FULL combined set, not just
        # the new batch.
        geojson = json.loads((args.out / e['tile'] / 'parcels.geojson').read_text())
        ids_by_split.setdefault(e['split'], set()).update(
            str(f['properties']['id']) for f in geojson['features'])

    for item in requested:
        tile, split = item['tile'], item['split']
        try:
            entry, split_ids, bounds = process_one_tile(tile, split, args, collection, available)
        except Exception as e:
            print(f'SKIPPED {tile}: {e}', flush=True)
            failures.append({'tile': tile, 'reason': str(e)})
            continue
        for other_split, other_bounds in footprints:
            if other_split != split and bounds.distance(other_bounds) < 100:
                print(f'SKIPPED {tile}: within 100m of an existing {other_split} tile footprint', flush=True)
                failures.append({'tile': tile, 'reason': f'too close to an existing {other_split} tile'})
                (args.out / tile).rename(args.out / (tile + '.rejected'))
                break
        else:
            overlap_split = next((s for s, ids in ids_by_split.items() if s != split and ids & split_ids), None)
            if overlap_split:
                print(f'SKIPPED {tile}: shares parcel ids with existing {overlap_split} split', flush=True)
                failures.append({'tile': tile, 'reason': f'parcel id overlap with {overlap_split}'})
                (args.out / tile).rename(args.out / (tile + '.rejected'))
                continue
            new_entries.append(entry)
            footprints.append((split, bounds))
            ids_by_split.setdefault(split, set()).update(split_ids)

    manifest = {
        'format_version': 2, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'extends': str(args.original_pilot / 'manifest.json'),
        'collection_url': base.COLLECTION, 'parcel_source': base.PARCELS,
        'parcel_source_type': 'public LINZ-derived mirror', 'imagery_license': collection['license'],
        'providers': collection['providers'], 'crs': 'EPSG:2193',
        'limitations': original['limitations'] + [
            'Expanded batch selected by parcel-density screen (screen_nz_candidates.py), not random or exhaustive',
            'New tiles are geographically clustered around the same neighbourhood and were all assigned to train; '
            'the original single val/test tile is the only held-out measurement so far',
        ],
        'tiles': kept_entries + new_entries,
        'expansion_failures': failures,
    }
    base.save_json(args.out / 'manifest.json', manifest)
    print(f'\n{len(new_entries)}/{len(requested)} new tiles accepted, {len(failures)} skipped.')
    print(f'Manifest: {args.out / "manifest.json"} ({len(kept_entries)} original + {len(new_entries)} new = '
          f'{len(kept_entries) + len(new_entries)} total tiles)')


if __name__ == '__main__':
    main()
