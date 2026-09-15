"""Focused checks for expand_full.py's carry-over schema validation."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('expand_full', Path(__file__).parents[1] / 'expand_full.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _write_old_schema_tile(root: Path, tile: str):
    """The expand_nz_labels.py / download_nz.py schema -- rgb/boundary/valid,
    no ndsm or distance (those were joined from a SEPARATE height file by
    prepare_real.py in the old two-stage pipeline, never stored here)."""
    folder = root / tile
    folder.mkdir(parents=True)
    np.savez_compressed(folder / 'training.npz',
                        rgb=np.zeros((3, 8, 8), dtype='uint8'),
                        boundary=np.zeros((8, 8), dtype='uint8'),
                        valid=np.ones((8, 8), dtype=bool))
    import hashlib
    digest = hashlib.sha256((folder / 'training.npz').read_bytes()).hexdigest()
    manifest = {'tiles': [{'tile': tile, 'split': 'train', 'training_sha256': digest,
                           'bbox_wgs84': [0, 0, 1, 1]}]}
    (root / 'manifest.json').write_text(json.dumps(manifest))
    return root / 'manifest.json'


def test_schema_incompatible_carryover_is_dropped_not_silently_kept(tmp_path, monkeypatch, capsys):
    """Regression: a byte-identical copy of an old-schema tile used to be
    silently accepted into the new manifest -- it would crash the first
    time anything actually tried to train on it (KeyError: 'ndsm' is not
    a file in the archive), confirmed by hitting this exact error against
    16 real carried-over tiles."""
    old_root = tmp_path / 'old'
    manifest_path = _write_old_schema_tile(old_root, 'BAD_TILE')

    monkeypatch.setattr('sys.argv', [
        'expand_full.py', '--candidates', str(tmp_path / 'empty_candidates.json'),
        '--existing-manifest', str(manifest_path), '--out', str(tmp_path / 'out'),
    ])
    (tmp_path / 'empty_candidates.json').write_text('[]')
    monkeypatch.setattr(module.base, 'get_json', lambda *a, **kw: {'links': [], 'license': '', 'providers': []})

    class DummyElevationIndex:
        def __init__(self, *a, **kw): pass
    monkeypatch.setattr(module, 'ElevationIndex', DummyElevationIndex)

    module.main()

    manifest = json.loads((tmp_path / 'out' / 'manifest.json').read_text())
    assert manifest['tiles'] == []  # the bad tile must NOT appear in the usable set
    assert manifest['dropped_incompatible_carryover'] == [
        {'tile': 'BAD_TILE', 'found_keys': ['boundary', 'rgb', 'valid']}]
    assert not (tmp_path / 'out' / 'BAD_TILE').exists()  # cleaned up, not left half-copied
    assert 'DROPPED (incompatible schema) BAD_TILE' in capsys.readouterr().out


def test_schema_compatible_carryover_is_kept(tmp_path, monkeypatch):
    old_root = tmp_path / 'old'
    folder = old_root / 'GOOD_TILE'
    folder.mkdir(parents=True)
    np.savez_compressed(folder / 'training.npz',
                        rgb=np.zeros((3, 8, 8), dtype='uint8'), ndsm=np.zeros((8, 8), dtype='float32'),
                        distance=np.zeros((8, 8), dtype='float32'), valid=np.ones((8, 8), dtype=bool))
    import hashlib
    digest = hashlib.sha256((folder / 'training.npz').read_bytes()).hexdigest()
    manifest_path = old_root / 'manifest.json'
    manifest_path.write_text(json.dumps({'tiles': [
        {'tile': 'GOOD_TILE', 'split': 'train', 'training_sha256': digest, 'bbox_wgs84': [0, 0, 1, 1]}]}))

    monkeypatch.setattr('sys.argv', [
        'expand_full.py', '--candidates', str(tmp_path / 'empty_candidates.json'),
        '--existing-manifest', str(manifest_path), '--out', str(tmp_path / 'out'),
    ])
    (tmp_path / 'empty_candidates.json').write_text('[]')
    monkeypatch.setattr(module.base, 'get_json', lambda *a, **kw: {'links': [], 'license': '', 'providers': []})

    class DummyElevationIndex:
        def __init__(self, *a, **kw): pass
    monkeypatch.setattr(module, 'ElevationIndex', DummyElevationIndex)

    module.main()

    manifest = json.loads((tmp_path / 'out' / 'manifest.json').read_text())
    assert [t['tile'] for t in manifest['tiles']] == ['GOOD_TILE']
    assert manifest['dropped_incompatible_carryover'] == []


def _write_new_schema_tile(root: Path, tile: str, split: str = 'train'):
    folder = root / tile
    folder.mkdir(parents=True)
    np.savez_compressed(folder / 'training.npz',
                        rgb=np.zeros((3, 8, 8), dtype='uint8'), ndsm=np.zeros((8, 8), dtype='float32'),
                        distance=np.full((8, 8), 5.0, dtype='float32'), valid=np.ones((8, 8), dtype=bool))
    import hashlib
    digest = hashlib.sha256((folder / 'training.npz').read_bytes()).hexdigest()
    return {'tile': tile, 'split': split, 'path': f'{tile}/training.npz',
           'training_sha256': digest, 'bbox_wgs84': [0, 0, 1, 1], 'parcels': 1}


def test_build_snapshot_actually_copies_files_and_uses_patches_field_names(tmp_path):
    """Regression, both real bugs found by actually running the pipeline:
    (1) the snapshot only verified tiles in place and never copied them --
    a manifest pointing back at a live, still-growing source directory is
    not a snapshot; (2) it wrote 'training_sha256', but Patches() (the
    actual production reader in train_real.py) reads 'sha256' -- confirmed
    by running Patches() against the output and hitting KeyError: 'sha256'.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location('build_snapshot', Path(__file__).parents[1] / 'build_snapshot.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    source = tmp_path / 'source'
    entry = _write_new_schema_tile(source, 'TILE_A')
    # build_snapshot.py now refuses to write a snapshot missing val/test
    # entirely (see test_build_snapshot.py) -- a real one always has both,
    # so this fixture needs them too even though this test only asserts
    # on TILE_A.
    val_entry = _write_new_schema_tile(source, 'TILE_VAL', split='val')
    test_entry = _write_new_schema_tile(source, 'TILE_TEST', split='test')
    (source / 'manifest.json').write_text(json.dumps({'tiles': [entry, val_entry, test_entry]}))

    out = tmp_path / 'snapshot'
    import sys
    old_argv = sys.argv
    sys.argv = ['build_snapshot.py', '--source', str(source), '--out', str(out)]
    try:
        module.main()
    finally:
        sys.argv = old_argv

    # (1) the file must actually exist in the snapshot, independent of source
    assert (out / 'TILE_A' / 'training.npz').exists()
    import shutil
    shutil.rmtree(source)  # if this were a reference, not a copy, the snapshot would now be broken
    assert (out / 'TILE_A' / 'training.npz').exists()

    # (2) Patches() must be able to read the snapshot's own manifest directly
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parents[1]))
    from train_real import Patches
    patches = Patches(out, 'train', size=8)
    assert len(patches) == 1
