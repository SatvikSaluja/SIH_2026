"""Focused checks for the full-catalogue screener: the parts that matter
at 8,166-tile scale -- one bad tile can't kill the batch, and an already-
labelled tile is never re-screened (it costs a real network call for
nothing, and the point of screening is to be cheap)."""
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('screen_catalog', Path(__file__).parents[1] / 'screen_catalog.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_already_labelled_tiles_are_never_screened(monkeypatch):
    calls = []

    def fake_get_json(url, timeout=30):
        calls.append(url)
        return {'bbox': [0, 0, 1, 1]} if 'query' not in url else {'count': 5}

    monkeypatch.setattr(module, 'get_json', fake_get_json)
    row, error = module.screen_one('ALREADY_HAVE', 'ALREADY_HAVE.json', already_have={'ALREADY_HAVE'})
    assert row is None and error is None
    assert calls == []  # not one network call for a tile we already have


def test_a_failing_tile_reports_an_error_without_raising(monkeypatch):
    def fake_get_json(url, timeout=30):
        raise TimeoutError('simulated network failure')

    monkeypatch.setattr(module, 'get_json', fake_get_json)
    row, error = module.screen_one('BAD_TILE', 'BAD_TILE.json', already_have=set())
    assert row is None
    assert error == {'tile': 'BAD_TILE', 'error': 'simulated network failure'}


def test_a_succeeding_tile_returns_bbox_and_count(monkeypatch):
    def fake_get_json(url, timeout=30):
        if 'query' in url:
            return {'count': 42}
        return {'bbox': [172.0, -43.0, 172.1, -42.9]}

    monkeypatch.setattr(module, 'get_json', fake_get_json)
    row, error = module.screen_one('GOOD_TILE', 'GOOD_TILE.json', already_have=set())
    assert error is None
    assert row == {'tile': 'GOOD_TILE', 'bbox_wgs84': [172.0, -43.0, 172.1, -42.9], 'parcel_count': 42}


def test_main_sorts_by_parcel_count_descending_and_skips_already_labelled(tmp_path, monkeypatch, capsys):
    counts = {'A': 5, 'B': 50, 'C': 20}

    # main() only needs get_json for the bulk collection listing (item bbox
    # and parcel-count lookups happen inside screen_one, patched below), so
    # faking every possible query URL's tile identity here isn't necessary.
    monkeypatch.setattr(module, 'get_json', lambda url, timeout=30: (
        {'links': [{'href': f'./{t}.json', 'rel': 'item'} for t in ('A', 'B', 'C', 'D')]}))

    def fake_screen_one(tile, href, already_have):
        if tile in already_have:
            return None, None
        if tile == 'D':
            return None, {'tile': 'D', 'error': 'boom'}
        return {'tile': tile, 'bbox_wgs84': [0, 0, 1, 1], 'parcel_count': counts[tile]}, None

    monkeypatch.setattr(module, 'screen_one', fake_screen_one)

    already = tmp_path / 'already.json'
    already.write_text(json.dumps({'tiles': [{'tile': 'A'}]}))
    out = tmp_path / 'out.json'
    monkeypatch.setattr('sys.argv', ['screen_catalog.py', '--limit', '0', '--workers', '2',
                                     '--already-have', str(already), '--out', str(out)])
    module.main()

    result = json.loads(out.read_text())
    assert result['already_labelled_skipped'] == 1
    assert [t['tile'] for t in result['tiles']] == ['B', 'C']  # A skipped, D errored, sorted desc by count
    assert result['errors'] == [{'tile': 'D', 'error': 'boom'}]
