"""Freeze a fixed, verified snapshot from whatever expand_full.py has
downloaded SO FAR -- not the full 2,455-tile target, and not the live,
still-growing directory (which a background job may still be writing to).

Every tile is independently re-verified here, not trusted from the
manifest a possibly-still-running process wrote:
  - training.npz's real sha256 matches what the manifest claims (catches
    a mid-write read, or any corruption).
  - The schema is exactly {rgb, ndsm, distance, valid} -- the same check
    that caught 16 carried-over tiles from the older, incompatible
    pipeline still sitting in this exact directory (this run started
    before that fix landed, so its effect is not retroactive).
  - No cross-split tile pair sits within 100m of another (recomputed
    independently, not assumed from expand_full.py's own claim to have
    enforced it).
  - Each tile's valid_fraction and boundary-pixel density fall in a
    plausible range -- flags statistical outliers for a manual look
    rather than trusting "more parcels" as a quality proxy on its own.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

EXPECTED_KEYS = {'rgb', 'ndsm', 'distance', 'valid'}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='data/nz_pilot_v3')
    p.add_argument('--out', default='data/nz_pilot_snapshot')
    args = p.parse_args()
    source = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((source / 'manifest.json').read_text())
    verified, dropped, outliers = [], [], []

    for entry in manifest['tiles']:
        tile_dir = source / entry['tile']
        npz_path = tile_dir / 'training.npz'
        if not npz_path.exists():
            dropped.append({'tile': entry['tile'], 'reason': 'training.npz missing'})
            continue
        actual_sha = digest(npz_path)
        if actual_sha != entry.get('training_sha256'):
            dropped.append({'tile': entry['tile'], 'reason': 'checksum mismatch vs manifest (mid-write or stale?)'})
            continue
        with np.load(npz_path) as z:
            keys = set(z.files)
            if keys != EXPECTED_KEYS:
                dropped.append({'tile': entry['tile'], 'reason': f'schema mismatch: {sorted(keys)}'})
                continue
            valid_frac = float(z['valid'].mean())
            boundary_frac = float((z['distance'][z['valid']] <= 0.3).mean()) if z['valid'].any() else 0.0
        if not (0.10 <= valid_frac <= 1.0):
            outliers.append({'tile': entry['tile'], 'reason': f'valid_fraction={valid_frac:.3f} outside [0.10, 1.0]'})
        if not (0.005 <= boundary_frac <= 0.40):
            outliers.append({'tile': entry['tile'], 'reason': f'boundary_fraction={boundary_frac:.4f} outside [0.005, 0.40]'})
        # Actually copy the verified bytes -- a manifest that only points
        # back at `source` is not a snapshot, it's a reference to whatever
        # `source` happens to contain when something later reads it, which
        # for a still-running expansion job is a moving target.
        dest_dir = out / entry['tile']
        dest_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(npz_path, dest_dir / 'training.npz')
        geojson_path = tile_dir / 'parcels.geojson'
        if geojson_path.exists():
            shutil.copy2(geojson_path, dest_dir / 'parcels.geojson')
        # train_real.py's Patches() -- the actual production reader -- was
        # built against prepare_real.py's manifest field name ('sha256'),
        # not expand_full.py's ('training_sha256'). Confirmed by running
        # Patches() against this exact snapshot: KeyError: 'sha256'. Fixed
        # HERE rather than in expand_full.py, since the live 2,455-tile job
        # already has that field name loaded into memory and restarting it
        # would lose real progress; the snapshot is the actual training
        # input, so this is where it needs to be correct.
        entry_out = {**entry, 'sha256': actual_sha, 'valid_fraction_recomputed': valid_frac,
                    'boundary_fraction': boundary_frac}
        entry_out.pop('training_sha256', None)
        verified.append(entry_out)

    # Independent geographic-separation re-check across the whole verified
    # set -- not trusted from expand_full.py's own claim to have enforced
    # it at fetch time.
    from shapely.geometry import box as shapely_box
    by_split = {}
    for e in verified:
        by_split.setdefault(e['split'], []).append((e['tile'], shapely_box(*e['bbox_wgs84'])))
    separation_violations = []
    splits = list(by_split)
    for i, split_a in enumerate(splits):
        for split_b in splits[i + 1:]:
            for tile_a, box_a in by_split[split_a]:
                for tile_b, box_b in by_split[split_b]:
                    # ~1e-3 deg is roughly 100m at this latitude -- consistent
                    # with expand_full.py's own 100m rule, checked in degrees
                    # since these are WGS84 boxes.
                    if box_a.distance(box_b) < 0.0009:
                        separation_violations.append((tile_a, split_a, tile_b, split_b))

    # Parcel-id overlap across splits, independently recomputed.
    id_overlap = []
    ids_by_split = {}
    for e in verified:
        geojson_path = source / e['tile'] / 'parcels.geojson'
        if geojson_path.exists():
            ids = {str(f['properties']['id']) for f in json.loads(geojson_path.read_text())['features']}
            for other_split, other_ids in ids_by_split.items():
                if other_split != e['split'] and ids & other_ids:
                    id_overlap.append((e['tile'], e['split'], other_split))
            ids_by_split.setdefault(e['split'], set()).update(ids)

    from collections import Counter
    split_counts = Counter(e['split'] for e in verified)
    snapshot = {
        'format_version': 4, 'source': str(source), 'crs': 'EPSG:2193', 'gsd_m': manifest.get('gsd_m', 0.3),
        'n_source_manifest_tiles': len(manifest['tiles']), 'n_verified': len(verified),
        'n_dropped': len(dropped), 'dropped': dropped,
        'n_outliers_flagged_not_dropped': len(outliers), 'outliers': outliers,
        'split_counts': dict(split_counts),
        'cross_split_separation_violations': separation_violations,
        'cross_split_parcel_id_overlap': id_overlap,
        'tiles': verified,
    }
    (out / 'manifest.json').write_text(json.dumps(snapshot, indent=2))
    print(f"Verified {len(verified)}/{len(manifest['tiles'])} tiles from {source}")
    print(f"Dropped: {len(dropped)}  |  Flagged outliers (kept): {len(outliers)}")
    print(f"Split counts: {dict(split_counts)}")
    print(f"Cross-split separation violations: {len(separation_violations)}")
    print(f"Cross-split parcel-id overlap: {len(id_overlap)}")
    if dropped:
        print("Dropped detail:", json.dumps(dropped[:10], indent=2), "..." if len(dropped) > 10 else "")
    if separation_violations or id_overlap:
        print("SPLIT INTEGRITY PROBLEM -- see manifest for detail")

    # A snapshot missing val or test entirely still writes a manifest and
    # exits 0 -- Patches(root,'val')/'test' then silently iterate zero
    # tiles rather than erroring, so training would either crash deep in
    # evaluate() (division by zero) or, worse, "succeed" with meaningless
    # validation metrics. Confirmed concretely: a real run of this script
    # produced exactly this (all val/test carryover tiles had the old,
    # incompatible schema and were correctly dropped, leaving zero of
    # either split) and exited 0 anyway.
    missing = [s for s in ('val', 'test') if split_counts.get(s, 0) == 0]
    if missing:
        raise ValueError(f"Snapshot has zero tiles in split(s) {missing} -- unusable for training/evaluation. "
                         f"Split counts: {dict(split_counts)}. Check `dropped` in {out / 'manifest.json'}.")


if __name__ == '__main__':
    main()
