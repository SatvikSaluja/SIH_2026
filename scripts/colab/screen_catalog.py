"""Screen the FULL Christchurch LINZ catalogue (8,182 tiles) for parcel
density, without downloading a single image.

Two network calls per tile, both small JSON, no imagery or height:
  1. The tile's own STAC item.json, for its WGS84 bbox (the bulk
     collection.json lists hrefs only, not bbox -- has to be fetched
     per item).
  2. A returnCountOnly parcel-service query against that bbox.

This is the same cost-before-commitment principle as
screen_nz_candidates.py, just applied to the full catalogue instead of
the 100 tiles someone had already pre-selected. It produces a ranked
report; it does not itself decide anything or download imagery/height
for any tile -- that stays a separate, deliberate step (expand_nz_labels.py
for parcels+imagery, prepare_nz_height.py for elevation), same as before.

Run with --limit to test on a slice before committing to all 8,182 --
that's the whole reason --limit defaults to 1000, not 0/unlimited.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin, urlencode
from urllib.request import Request, urlopen

COLLECTION = 'https://nz-imagery.s3.ap-southeast-2.amazonaws.com/canterbury/christchurch_2025_0.075m/rgb/2193/collection.json'
PARCELS = 'https://services.arcgis.com/xdsHIIxuCWByZiCB/arcgis/rest/services/LINZ_NZ_Primary_Parcels/FeatureServer/0'


def get_json(url, timeout=30):
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


def screen_one(tile, href, already_have):
    """(row, error) -- error is None on success. Never raises: a batch of
    thousands must survive individual tile failures (a bad checksum entry,
    a transient timeout) without losing everything already screened."""
    if tile in already_have:
        return None, None  # not re-screened -- already have parcel labels for it
    try:
        item = get_json(urljoin(COLLECTION, href))
        bbox = item['bbox']  # STAC bbox is already WGS84 (EPSG:4326)
        query = {'f': 'json', 'where': '1=1', 'returnCountOnly': 'true',
                 'geometry': ','.join(map(str, bbox)), 'geometryType': 'esriGeometryEnvelope',
                 'inSR': 4326, 'spatialRel': 'esriSpatialRelIntersects'}
        count = get_json(PARCELS + '/query?' + urlencode(query))['count']
        return {'tile': tile, 'bbox_wgs84': bbox, 'parcel_count': count}, None
    except Exception as e:
        return None, {'tile': tile, 'error': str(e)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--limit', type=int, default=1000, help='0 = the whole catalogue')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--already-have', default='data/nz_pilot_v2/manifest.json',
                   help='Tiles already parcel-labelled -- skipped, not re-screened')
    p.add_argument('--out', default='data/christchurch_catalog_screen.json')
    args = p.parse_args()
    if not 1 <= args.workers <= 16:
        p.error('workers must be 1-16')

    collection = get_json(COLLECTION)
    items = [(Path(x['href']).stem, x['href']) for x in collection['links'] if x['rel'] == 'item']
    total_catalog = len(items)
    if args.limit:
        items = items[:args.limit]

    already_have = set()
    have_path = Path(args.already_have)
    if have_path.exists():
        already_have = {t['tile'] for t in json.loads(have_path.read_text())['tiles']}

    print(f'Screening {len(items)}/{total_catalog} catalogue tiles '
          f'({len(already_have)} already labelled, skipped) with {args.workers} workers', flush=True)

    results, errors = [], []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(screen_one, tile, href, already_have): tile for tile, href in items}
        for future in as_completed(futures):
            row, error = future.result()
            done += 1
            if row:
                results.append(row)
            elif error:
                errors.append(error)
                print(f'  [{done}/{len(items)}] ERROR {error["tile"]}: {error["error"]}', flush=True)
            if done % 100 == 0:
                print(f'  [{done}/{len(items)}] {len(results)} scored, {len(errors)} errors so far', flush=True)

    results.sort(key=lambda r: r['parcel_count'], reverse=True)
    Path(args.out).write_text(json.dumps(
        {'catalog_total': total_catalog, 'screened': len(items), 'already_labelled_skipped': len(already_have),
         'scored': len(results), 'errors': errors, 'tiles': results}, indent=2))
    print(f'\n{len(results)} scored, {len(errors)} errors, out of {len(items)} attempted '
          f'(catalogue has {total_catalog} total). Report: {args.out}')
    if results:
        print('Top 10 by parcel count:')
        for r in results[:10]:
            print(f'  {r["tile"]:20s} {r["parcel_count"]:4d} parcels')


if __name__ == '__main__':
    main()
