"""Produce a training-ready npz directly for any catalogue tile: native RGB
+ parcels (download_nz.py's logic) AND matched DEM/DSM height
(prepare_nz_height.py's logic), fused in one place.

Why this exists rather than reusing the two-stage nz_pilot -> nz_height_100
-> prepare_real.py join: that flow depended on two EARLIER, separately-run
scripts having already prepared overlapping tile pools by coincidence (5 of
6 original tiles happened to have both). At the scale of thousands of new
candidate tiles there's no such coincidence to depend on -- every tile
needs both pieces fetched together, deliberately.

DEM/DSM is cheap regardless of how many target tiles are processed: only
27 DEM + 27 DSM SOURCE tiles cover the entire imagery catalogue's extent,
each large enough to serve many targets, and downloads are cached by
source key -- confirmed by reading dem_collection.json/dsm_collection.json
before writing this, not assumed.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).parent))
import download_nz as base  # noqa: E402

DEM_BASE = 'https://nz-elevation.s3.ap-southeast-2.amazonaws.com/canterbury/christchurch_2024-2025/'


def get_json(url, timeout=60):
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'GeoCadastra-research-pilot/1.0'}), timeout=timeout) as r:
                value = json.load(r)
            if 'error' in value:
                raise RuntimeError(value['error'])
            return value
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


class ElevationIndex:
    """DEM/DSM source-tile lookup, shared across every target tile in one
    run -- each source raster is downloaded at most once regardless of how
    many target tiles it ends up serving."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.items = {}
        self.urls = {}
        self._raster_cache = {}
        from shapely.geometry import shape
        self._shape = shape
        for kind in ('dem', 'dsm'):
            url = DEM_BASE + kind + '_1m/2193/collection.json'
            self.urls[kind] = url
            collection = get_json(url)
            (cache_dir / f'{kind}_collection.json').write_text(json.dumps(collection, indent=2))
            links = [x for x in collection['links'] if x['rel'] == 'item']
            records = {}
            for link in links:
                local = cache_dir / (kind + '_' + Path(link['href']).name)
                base.download(urljoin(url, link['href']), local, link.get('file:checksum'))
                item = json.loads(local.read_bytes())
                records[item['id']] = item
            self.items[kind] = records
            print(f'Indexed {kind}: {len(records)} elevation source tiles', flush=True)

    def match(self, footprint):
        """DEM+DSM key covering `footprint` fully, or None."""
        matches = [k for k, d in self.items['dem'].items()
                  if k in self.items['dsm']
                  and self._shape(d['geometry']).buffer(1e-7).covers(footprint)
                  and self._shape(self.items['dsm'][k]['geometry']).buffer(1e-7).covers(footprint)]
        return matches[0] if matches else None

    def raster_path(self, kind, key):
        pair = (kind, key)
        if pair not in self._raster_cache:
            item = self.items[kind][key]
            asset = item['assets']['visual']
            path = self.cache_dir / f'{kind}_{key}.tiff'
            print(f'Downloading elevation source {kind} {key}', flush=True)
            base.download(urljoin(self.urls[kind], asset['href']), path, asset.get('file:checksum'))
            self._raster_cache[pair] = path
        return self._raster_cache[pair]


def process_one_tile(tile, split, elevation_index, args, collection, available):
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import rasterize
    from rasterio.transform import array_bounds
    from rasterio.warp import reproject
    from scipy.ndimage import binary_erosion
    from shapely.geometry import shape, box
    from shapely.ops import transform as transform_geometry
    from pyproj import Transformer

    sys.path.insert(0, str(Path(__file__).parent))
    from prepare_real import compute_distance_and_mask

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

    footprint = shape(item['geometry'])
    key = elevation_index.match(footprint)
    if key is None:
        raise ValueError(f'No DEM+DSM source tile covers {tile}')
    surfaces, joint_valid = {}, image_valid.copy()
    for kind in ('dem', 'dsm'):
        path = elevation_index.raster_path(kind, key)
        with rasterio.open(path) as esrc:
            if esrc.crs.to_epsg() != 2193 or not math.isclose(abs(esrc.transform.a), 1):
                raise ValueError('Unexpected elevation CRS/resolution')
            if not (esrc.bounds.left <= affine.c and esrc.bounds.right >= affine.c + w * affine.a
                   and esrc.bounds.top >= affine.f and esrc.bounds.bottom <= affine.f + h * affine.e):
                raise ValueError(f'{kind} raster does not fully cover {tile}')
            arr = np.full((h, w), np.nan, dtype='float32')
            reproject(rasterio.band(esrc, 1), arr, src_transform=esrc.transform, src_crs=esrc.crs,
                     src_nodata=esrc.nodata, dst_transform=affine, dst_crs='EPSG:2193',
                     dst_nodata=np.nan, resampling=Resampling.bilinear)
            surfaces[kind] = arr
            joint_valid &= np.isfinite(arr)
    ndsm = surfaces['dsm'] - surfaces['dem']

    project = Transformer.from_crs(4326, 2193, always_xy=True).transform
    query = {'f': 'geojson', 'where': '1=1', 'geometry': ','.join(map(str, item['bbox'])),
             'geometryType': 'esriGeometryEnvelope', 'inSR': 4326, 'outSR': 4326,
             'spatialRel': 'esriSpatialRelIntersects', 'outFields': 'id,parcel_intent,survey_area,calc_area',
             'returnGeometry': 'true', 'resultRecordCount': 1000, 'orderByFields': 'OBJECTID'}
    features, offset = [], 0
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

    bounds = box(*array_bounds(h, w, affine))
    polygons, lines, split_ids, records = [], [], set(), []
    for feature in features:
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
    boundary_raster = rasterize(lines, out_shape=(h, w), transform=affine, all_touched=True, dtype='uint8')
    parcel_valid = binary_erosion(coverage & image_valid, iterations=2, border_value=0)
    if parcel_valid.mean() < 0.25 or boundary_raster[parcel_valid].sum() < 20:
        raise ValueError(f'Insufficient labelled imagery for {tile} '
                         f'(valid={parcel_valid.mean():.3f}, boundary_px={int(boundary_raster[parcel_valid].sum())})')

    distance, distance_mask = compute_distance_and_mask(boundary_raster, parcel_valid & joint_valid, ndsm, args.gsd)
    if distance_mask.mean() < 0.10:
        raise ValueError(f'Insufficient joint RGB/parcel/height coverage for {tile} ({distance_mask.mean():.3f})')

    import numpy as np
    np.savez_compressed(folder / 'training.npz', rgb=rgb, ndsm=np.nan_to_num(ndsm).astype('float32'),
                        distance=distance.astype('float32'), valid=distance_mask)
    (folder / 'parcels.geojson').write_text(json.dumps({'type': 'FeatureCollection', 'features': features}))
    entry = {'tile': tile, 'split': split, 'path': tile + '/training.npz', 'gsd_m': abs(affine.a),
            'parcels': len(records), 'valid_fraction': float(distance_mask.mean()),
            'elevation_source_key': key, 'image_url': image_url,
            'native_sha256': base.digest(raw), 'training_sha256': base.digest(folder / 'training.npz'),
            'parcel_sha256': base.digest(folder / 'parcels.geojson'), 'bbox_wgs84': item['bbox']}
    print(f'Prepared {tile}: {len(records)} parcels, {distance_mask.mean():.3f} valid, '
         f'elevation from {key}', flush=True)
    return entry, split_ids, bounds


def main():
    from shapely.geometry import box as shapely_box

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates', required=True, help='JSON: [{"tile":..., "split":...}, ...]')
    p.add_argument('--existing-manifest', type=Path, default=None,
                   help='A prior expand_full.py/expand_nz_labels.py manifest to extend '
                        '(its tiles are carried over unchanged, never re-fetched)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gsd', type=float, default=0.30)
    p.add_argument('--elevation-cache', type=Path, default=Path('data/nz_elevation_cache'))
    args = p.parse_args()
    if not 0.075 <= args.gsd <= 1:
        raise ValueError('gsd must be between 0.075 and 1 metre')
    args.out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    EXPECTED_KEYS = {'rgb', 'ndsm', 'distance', 'valid'}
    kept_entries, footprints, ids_by_split, incompatible = [], [], {}, []
    if args.existing_manifest and args.existing_manifest.exists():
        existing = json.loads(args.existing_manifest.read_text())
        for entry in existing['tiles']:
            src = args.existing_manifest.parent / entry['tile']
            dst = args.out / entry['tile']
            if not dst.exists():
                import shutil
                shutil.copytree(src, dst)
            actual = base.digest(dst / 'training.npz')
            if actual != entry['training_sha256']:
                raise ValueError(f'Copied tile {entry["tile"]} does not match its recorded checksum')
            # A byte-identical copy is not the same claim as "this file has
            # the schema this pipeline's own readers expect". Confirmed
            # concretely: every one of the 16 tiles carried over from the
            # older expand_nz_labels.py pipeline has {'rgb','boundary',
            # 'valid'} -- that pipeline stored ndsm/distance separately
            # (prepare_real.py joined them from a different height file at
            # a later step), not inside training.npz at all. Silently
            # keeping these would crash the FIRST time anything actually
            # trained on them, not now while it's cheap to catch.
            with np.load(dst / 'training.npz') as npz:
                keys = set(npz.files)
            if keys != EXPECTED_KEYS:
                import shutil as _shutil
                _shutil.rmtree(dst)
                incompatible.append({'tile': entry['tile'], 'found_keys': sorted(keys)})
                print(f'DROPPED (incompatible schema) {entry["tile"]}: has {sorted(keys)}, '
                     f'need {sorted(EXPECTED_KEYS)} -- re-request it via --candidates to refetch properly', flush=True)
                continue
            kept_entries.append(entry)
            footprints.append((entry['split'], shapely_box(*entry['bbox_wgs84'])))
            geojson_path = dst / 'parcels.geojson'
            if geojson_path.exists():
                geojson = json.loads(geojson_path.read_text())
                ids_by_split.setdefault(entry['split'], set()).update(
                    str(f['properties']['id']) for f in geojson['features'])
        print(f'Carried over {len(kept_entries)} tiles from {args.existing_manifest} '
             f'({len(incompatible)} dropped for schema mismatch)', flush=True)

    requested = json.loads(Path(args.candidates).read_text())
    already = {e['tile'] for e in kept_entries}
    requested = [r for r in requested if r['tile'] not in already]
    print(f'{len(requested)} new candidates to process ({len(already)} already present, skipped)', flush=True)

    collection = base.get_json(base.COLLECTION)
    base.save_json(args.out / 'collection.json', collection)
    base.save_json(args.out / 'parcel_service.json', base.get_json(base.PARCELS + '?f=pjson'))
    available = {Path(x['href']).stem: x for x in collection['links'] if x['rel'] == 'item'}
    elevation_index = ElevationIndex(args.elevation_cache)

    new_entries, failures = [], []
    for i, req in enumerate(requested, 1):
        tile, split = req['tile'], req['split']
        try:
            entry, split_ids, bounds = process_one_tile(tile, split, elevation_index, args, collection, available)
        except Exception as e:
            print(f'[{i}/{len(requested)}] SKIPPED {tile}: {e}', flush=True)
            failures.append({'tile': tile, 'reason': str(e)})
            continue
        blocked = False
        for other_split, other_bounds in footprints:
            if other_split != split and bounds.distance(other_bounds) < 100:
                print(f'[{i}/{len(requested)}] SKIPPED {tile}: within 100m of an existing {other_split} tile', flush=True)
                failures.append({'tile': tile, 'reason': f'too close to an existing {other_split} tile'})
                (args.out / tile).rename(args.out / (tile + '.rejected'))
                blocked = True
                break
        if blocked:
            continue
        overlap_split = next((s for s, ids in ids_by_split.items() if s != split and ids & split_ids), None)
        if overlap_split:
            print(f'[{i}/{len(requested)}] SKIPPED {tile}: parcel id overlap with {overlap_split}', flush=True)
            failures.append({'tile': tile, 'reason': f'parcel id overlap with {overlap_split}'})
            (args.out / tile).rename(args.out / (tile + '.rejected'))
            continue
        new_entries.append(entry)
        footprints.append((split, bounds))
        ids_by_split.setdefault(split, set()).update(split_ids)
        print(f'[{i}/{len(requested)}] accepted ({len(new_entries)} total new so far)', flush=True)
        if i % 25 == 0:
            base.save_json(args.out / 'manifest.json', {
                'format_version': 3, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'status': 'in_progress', 'crs': 'EPSG:2193', 'gsd_m': args.gsd,
                'tiles': kept_entries + new_entries, 'failures': failures})

    manifest = {
        'format_version': 3, 'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'complete',
        'crs': 'EPSG:2193', 'gsd_m': args.gsd, 'parcel_source': base.PARCELS,
        'tiles': kept_entries + new_entries, 'failures': failures,
        'dropped_incompatible_carryover': incompatible,
    }
    base.save_json(args.out / 'manifest.json', manifest)
    print(f'\n{len(new_entries)}/{len(requested)} new tiles accepted, {len(failures)} skipped.')
    print(f'Total: {len(kept_entries)} carried over + {len(new_entries)} new = '
         f'{len(kept_entries) + len(new_entries)} tiles')


if __name__ == '__main__':
    main()
