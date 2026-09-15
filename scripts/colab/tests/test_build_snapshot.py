"""Focused checks for build_snapshot.py's split-integrity guarantee.

Real bug this covers: a live run of this script (against nz_pilot_v3)
correctly DROPPED every val/test tile for schema mismatch -- they were
carried over by a since-fixed version of expand_full.py still running with
its old code loaded in memory -- but then wrote a train-only manifest and
exited 0 anyway. Patches(root,'val')/'test' would have silently iterated
zero tiles rather than raising, so training would have crashed deep inside
evaluate() (division by zero) instead of failing clearly at snapshot time,
when it's cheap to catch and fix.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('build_snapshot', Path(__file__).parents[1] / 'build_snapshot.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _write_tile(root: Path, tile: str, split: str, keys=('rgb', 'ndsm', 'distance', 'valid'), h=8, w=8):
    folder = root / tile
    folder.mkdir(parents=True)
    arrays = {'rgb': np.zeros((3, h, w), dtype='uint8'), 'ndsm': np.zeros((h, w), dtype='float32'),
             'distance': np.full((h, w), 5.0, dtype='float32'), 'valid': np.ones((h, w), dtype=bool),
             'boundary': np.zeros((h, w), dtype='uint8')}  # only used if 'boundary' is in keys
    np.savez_compressed(folder / 'training.npz', **{k: arrays[k] for k in keys})
    digest = hashlib.sha256((folder / 'training.npz').read_bytes()).hexdigest()
    return {'tile': tile, 'split': split, 'training_sha256': digest, 'bbox_wgs84': [0, 0, 1, 1]}


def _write_manifest(root: Path, entries):
    root.mkdir(parents=True, exist_ok=True)
    (root / 'manifest.json').write_text(json.dumps({'tiles': entries, 'gsd_m': 0.3}))


def test_missing_val_or_test_split_fails_loudly(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    entries = [
        _write_tile(source, 'TRAIN1', 'train'),
        _write_tile(source, 'TRAIN2', 'train'),
        # val/test tiles present in the manifest but with the old,
        # incompatible schema -- exactly what a stale-code carryover produces.
        _write_tile(source, 'VAL_BAD', 'val', keys=('rgb', 'boundary', 'valid')),
        _write_tile(source, 'TEST_BAD', 'test', keys=('rgb', 'boundary', 'valid')),
    ]
    _write_manifest(source, entries)

    monkeypatch.setattr('sys.argv', ['build_snapshot.py', '--source', str(source), '--out', str(tmp_path / 'out')])
    with pytest.raises(ValueError, match="zero tiles in split"):
        module.main()


def test_verified_snapshot_keeps_good_tiles_and_copies_files(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    entries = [
        _write_tile(source, 'TRAIN1', 'train'),
        _write_tile(source, 'VAL1', 'val'),
        _write_tile(source, 'TEST1', 'test'),
    ]
    _write_manifest(source, entries)
    out = tmp_path / 'out'

    monkeypatch.setattr('sys.argv', ['build_snapshot.py', '--source', str(source), '--out', str(out)])
    module.main()

    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['split_counts'] == {'train': 1, 'val': 1, 'test': 1}
    assert (out / 'TRAIN1' / 'training.npz').exists()  # actually copied, not just referenced
    for entry in manifest['tiles']:
        assert 'sha256' in entry and 'training_sha256' not in entry  # renamed for Patches()' reader
