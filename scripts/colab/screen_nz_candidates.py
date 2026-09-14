"""Rank the 100 already-downloaded image/height tiles by likely value for
parcel-boundary training, before spending any bandwidth on full imagery or
parcel-geometry downloads.

Two signals, cheapest first:
  1. height_p95_m -- already computed when nz_height_100 was built, free.
     Near-zero P95 height means little structure above ground (bare
     ground, farmland, water): weak boundary signal for an urban parcel
     task, and the problem statement is explicitly about dense URBAN
     settlements, not rural coverage.
  2. A returnCountOnly parcel-service query per tile footprint -- one
     small JSON response, no geometry transferred. Screens out tiles
     whose bbox happens to hold few/no parcels (edge of coverage, a
     single large rural block) before download_nz.py's own quality gate
     (valid.mean()<0.25 or too few boundary pixels) would reject them
     anyway, after a full-resolution GeoTIFF download.

This produces a ranked report, not a decision -- it does not itself
download imagery or parcels. It only tells you which of the 95 candidate
tiles are worth spending that cost on.
"""
import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

PARCELS = 'https://services.arcgis.com/xdsHIIxuCWByZiCB/arcgis/rest/services/LINZ_NZ_Primary_Parcels/FeatureServer/0'


def get_json(url):
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'GeoCadastra-research-pilot/1.0'}), timeout=30) as r:
                value = json.load(r)
            if 'error' in value:
                raise RuntimeError(value['error'])
            return value
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def tile_bbox_wgs84(metadata_path):
    """(minx, miny, maxx, maxy) in EPSG:4326, from the tile's own recorded
    pixel-to-map transform -- no raster needs to be opened for this."""
    from pyproj import Transformer
    meta = json.loads(metadata_path.read_text())
    a, b, c, d, e, f = meta['png_pixel_to_map_transform']
    w, h = meta['png_width'], meta['png_height']
    corners_2193 = [(c, f), (c + a * w, f), (c, f + e * h), (c + a * w, f + e * h)]
    to_wgs84 = Transformer.from_crs(meta['crs'], 4326, always_xy=True).transform
    xs, ys = zip(*(to_wgs84(x, y) for x, y in corners_2193))
    return min(xs), min(ys), max(xs), max(ys)


def parcel_count(bbox_wgs84):
    query = {'f': 'json', 'where': '1=1', 'returnCountOnly': 'true',
             'geometry': ','.join(map(str, bbox_wgs84)), 'geometryType': 'esriGeometryEnvelope',
             'inSR': 4326, 'spatialRel': 'esriSpatialRelIntersects'}
    return get_json(PARCELS + '/query?' + urlencode(query))['count']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--height-root', default='data/nz_height_100')
    p.add_argument('--pilot-manifest', default='data/nz_pilot/manifest.json')
    p.add_argument('--out', default='data/nz_height_100/candidate_screen.json')
    p.add_argument('--min-height-p95', type=float, default=1.0,
                   help='Below this, skip the parcel-count query entirely -- too flat to be worth it')
    args = p.parse_args()
    height_root = Path(args.height_root)

    tiles = json.loads((height_root / 'manifest.json').read_text())['tiles']
    pilot_tiles = {t['tile'] for t in json.loads(Path(args.pilot_manifest).read_text())['tiles']}
    candidates = [t for t in tiles if t['tile'] not in pilot_tiles]
    print(f'{len(candidates)} candidates ({len(tiles)} total, {len(pilot_tiles)} already piloted)', flush=True)

    results = []
    for i, entry in enumerate(candidates, 1):
        tile = entry['tile']
        row = {'tile': tile, 'height_p95_m': entry['height_p95_m'], 'valid_fraction': entry['valid_fraction']}
        if entry['height_p95_m'] < args.min_height_p95:
            row['parcel_count'] = None
            row['skipped_reason'] = f'height_p95_m {entry["height_p95_m"]:.2f} < {args.min_height_p95} (too flat)'
        else:
            bbox = tile_bbox_wgs84(height_root / entry['metadata'])
            row['bbox_wgs84'] = bbox
            row['parcel_count'] = parcel_count(bbox)
        results.append(row)
        print(f'  [{i}/{len(candidates)}] {tile}: {row.get("parcel_count", row.get("skipped_reason"))}', flush=True)

    results.sort(key=lambda r: (r['parcel_count'] is not None, r.get('parcel_count') or 0, r['height_p95_m']), reverse=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    scored = [r for r in results if r['parcel_count'] is not None]
    print(f'\n{len(scored)}/{len(candidates)} screened (rest skipped as too flat); report at {args.out}')


if __name__ == '__main__':
    main()
