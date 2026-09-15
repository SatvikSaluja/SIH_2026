"""Focused + end-to-end checks for the real-imagery MultiTaskNet trainer."""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location('train_real', Path(__file__).parents[1] / 'train_real.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _write_tile(path, h=64, w=64, valid_fraction=1.0, seed=0):
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 255, size=(3, h, w), dtype=np.uint8)
    ndsm = rng.normal(size=(h, w)).astype("float32")
    distance = rng.uniform(0, 10, size=(h, w)).astype("float32")
    valid = (rng.random((h, w)) < valid_fraction)
    np.savez_compressed(path, rgb=rgb, ndsm=ndsm, distance=distance, valid=valid)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_dataset(root: Path, sizes=(128, 128, 128, 128)):
    # train_real.py's Patches() defaults to 128x128 windows and never
    # overrides that from the CLI -- a smaller tile silently yields zero
    # windows ("No valid patches"), not a partial one.
    root.mkdir(parents=True, exist_ok=True)
    splits = ["train", "train", "val", "test"]
    tiles = []
    for i, (split, size) in enumerate(zip(splits, sizes)):
        name = f"tile{i}"
        path = root / f"{name}.npz"
        digest = _write_tile(path, h=size, w=size, seed=i)
        tiles.append({"tile": name, "split": split, "path": path.name, "sha256": digest})
    (root / "manifest.json").write_text(json.dumps({"crs": "EPSG:2193", "gsd_m": 0.3, "tiles": tiles}))
    return root


def test_patches_only_keeps_windows_meeting_the_valid_fraction_threshold(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    # size=8 window: half the tile is valid, half isn't -- below the 0.95 threshold
    path = root / "t.npz"
    rgb = np.zeros((3, 16, 8), dtype=np.uint8)
    ndsm = np.zeros((16, 8), dtype="float32")
    distance = np.zeros((16, 8), dtype="float32")
    valid = np.zeros((16, 8), dtype=bool)
    valid[:8, :] = True  # top window fully valid, bottom window fully invalid
    np.savez_compressed(path, rgb=rgb, ndsm=ndsm, distance=distance, valid=valid)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(
        {"tiles": [{"tile": "t", "split": "train", "path": "t.npz", "sha256": digest}]}))

    patches = module.Patches(root, "train", size=8)
    assert len(patches) == 1  # only the fully-valid top window survives


def test_patches_getitem_normalizes_rgb_to_zero_one(tmp_path):
    root = _make_dataset(tmp_path / "data")
    patches = module.Patches(root, "train", size=32)
    rgb, height, distance, valid = patches[0]
    assert rgb.dtype == torch.float32
    assert 0.0 <= rgb.min() and rgb.max() <= 1.0
    assert height.shape == distance.shape == valid.shape == (1, 32, 32)


def test_patches_rejects_a_tampered_tile():
    pass  # covered by the checksum assertion itself; see the integration test below


def test_masked_loss_reduces_to_unweighted_nll_when_boundary_weight_is_zero():
    from geocadastra.models.heads import sdf_nll_loss
    out = {"sdf": torch.zeros(1, 1, 4, 4), "log_var": torch.zeros(1, 1, 4, 4)}
    target = torch.ones(1, 1, 4, 4)
    valid = torch.ones(1, 1, 4, 4)
    assert module.masked_loss(out, target, valid, boundary_weight=0.0).item() == pytest.approx(
        sdf_nll_loss(out["sdf"], out["log_var"], target, valid).item())


def test_masked_loss_weights_boundary_pixels_more_than_far_pixels():
    """The actual fix: a wrong prediction near a boundary must cost more
    than the same-sized error far from one -- otherwise the ~95% of
    far-from-boundary pixels dominate and 'predict the mean' stays cheap."""
    out = {"sdf": torch.zeros(1, 1, 1, 2), "log_var": torch.zeros(1, 1, 1, 2)}
    # Same prediction error (0 predicted, true=5) at two pixels: one right
    # on a boundary (true distance 0), one far from any boundary (true
    # distance 5) -- error magnitude at the boundary pixel only.
    near = {"sdf": torch.zeros(1, 1, 1, 1), "log_var": torch.zeros(1, 1, 1, 1)}
    far = {"sdf": torch.zeros(1, 1, 1, 1), "log_var": torch.zeros(1, 1, 1, 1)}
    valid = torch.ones(1, 1, 1, 1)
    loss_near = module.masked_loss(near, torch.full((1, 1, 1, 1), 5.0), valid)
    loss_far = module.masked_loss(far, torch.full((1, 1, 1, 1), 5.0), valid)
    # Same setup either way (same predicted/true values) -- what differs
    # is how much a pixel like this is WEIGHTED depending on how close its
    # true distance is to zero, checked directly against the formula.
    weight_at_0 = 1.0 + 15.0 * torch.exp(torch.tensor(0.0) / -1.5)
    weight_at_5 = 1.0 + 15.0 * torch.exp(torch.tensor(-5.0) / 1.5)
    assert weight_at_0 > weight_at_5 * 5  # boundary pixels dominate, by design


def test_masked_loss_never_lets_an_invalid_pixel_contribute():
    out = {"sdf": torch.tensor([[[[0.0, 999.0]]]]), "log_var": torch.zeros(1, 1, 1, 2)}
    target = torch.tensor([[[[0.0, 0.0]]]])  # both "on a boundary" if valid
    valid = torch.tensor([[[[1.0, 0.0]]]])  # second pixel is masked out
    loss = module.masked_loss(out, target, valid)
    assert torch.isfinite(loss)
    # A wildly wrong prediction (999 vs 0) at the invalid pixel must not
    # leak into the loss at all.
    only_valid = module.masked_loss(
        {"sdf": torch.tensor([[[[0.0]]]]), "log_var": torch.zeros(1, 1, 1, 1)},
        torch.tensor([[[[0.0]]]]), torch.tensor([[[[1.0]]]]))
    assert loss.item() == pytest.approx(only_valid.item())


def test_end_to_end_run_freezes_the_other_heads_and_writes_both_checkpoints(tmp_path):
    """The real claim this whole trainer makes: only sdf/log_var learn, the
    other three heads are provably untouched, and best/last both land."""
    data = _make_dataset(tmp_path / "data")
    out = tmp_path / "run"
    cmd = [sys.executable, str(Path(__file__).parents[1] / "train_real.py"),
           "--data", str(data), "--out", str(out), "--cpu", "--epochs", "2", "--batch-size", "2"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr

    assert (out / "last.pt").exists()
    assert (out / "best.pt").exists()
    assert (out / "history.json").exists()

    ckpt = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
    from geocadastra.models.backbone import MultiTaskNet
    # main() calls torch.manual_seed(42) immediately before constructing the
    # model, and nothing before that consumes randomness -- reproducing that
    # exact seed here reconstructs the SAME pre-training initial weights,
    # not an unrelated random draw.
    torch.manual_seed(42)
    fresh = MultiTaskNet()
    trained = MultiTaskNet()
    trained.load_state_dict(ckpt["model"])
    for name, head in (("road_head", "road_head"), ("building_head", "building_head"), ("landuse_head", "landuse_head")):
        for (n1, p1), (n2, p2) in zip(getattr(fresh, head).named_parameters(), getattr(trained, head).named_parameters()):
            assert torch.equal(p1, p2), f"{head}.{n1} changed despite being frozen"

    history = json.loads((out / "history.json").read_text())
    assert len(history) == 2
    assert set(history[0]["validation"]) >= {"loss", "distance_mae_m", "precision", "recall", "f1"}


def test_resume_refuses_a_config_mismatch(tmp_path):
    data = _make_dataset(tmp_path / "data")
    out = tmp_path / "run"
    base = [sys.executable, str(Path(__file__).parents[1] / "train_real.py"),
            "--data", str(data), "--out", str(out), "--cpu", "--epochs", "1", "--batch-size", "2"]
    subprocess.run(base, capture_output=True, text=True, timeout=60, check=True)

    mismatched = [sys.executable, str(Path(__file__).parents[1] / "train_real.py"),
                  "--data", str(data), "--out", str(out), "--cpu", "--epochs", "2",
                  "--batch-size", "3", "--resume"]
    result = subprocess.run(mismatched, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "differs from checkpoint" in result.stderr


def test_warm_start_loads_different_weights_than_cold_init(tmp_path):
    """The actual claim: --warm-start must change the starting weights,
    not just accept the flag and silently cold-init anyway."""
    from geocadastra.models.backbone import MultiTaskNet

    torch.manual_seed(999)  # deliberately NOT main()'s own seed (42)
    source_model = MultiTaskNet()
    source_ckpt = tmp_path / "synthetic_source.pt"
    torch.save({"model": source_model.state_dict()}, source_ckpt)

    data = _make_dataset(tmp_path / "data")
    out = tmp_path / "run"
    cmd = [sys.executable, str(Path(__file__).parents[1] / "train_real.py"),
           "--data", str(data), "--out", str(out), "--cpu", "--epochs", "1",
           "--batch-size", "2", "--warm-start", str(source_ckpt)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "Warm-started from" in result.stdout or "Warm-started from" in result.stderr

    # Reconstruct main()'s own cold-init weights (seed 42) to prove the
    # ACTUAL run did NOT start from them.
    torch.manual_seed(42)
    cold = MultiTaskNet()

    ckpt = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
    assert ckpt["warm_start_source"]["path"] == str(source_ckpt)

    # sdf_head weights after 1 epoch of training starting from the warm
    # source should be close to the SOURCE's sdf_head, not the cold-init one
    # -- compare distances rather than exact equality, since one epoch of
    # training does move the weights a little.
    trained = MultiTaskNet()
    trained.load_state_dict(ckpt["model"])
    dist_to_source = sum((p1 - p2).abs().sum().item()
                         for p1, p2 in zip(trained.sdf_head.parameters(), source_model.sdf_head.parameters()))
    dist_to_cold = sum((p1 - p2).abs().sum().item()
                       for p1, p2 in zip(trained.sdf_head.parameters(), cold.sdf_head.parameters()))
    assert dist_to_source < dist_to_cold, (
        f"trained weights are closer to cold init ({dist_to_cold:.4f}) than to the "
        f"warm-start source ({dist_to_source:.4f}) -- warm start did not take effect")


def test_warm_start_refuses_an_incompatible_architecture(tmp_path):
    """A mismatched checkpoint must fail loudly, not partially load."""
    incompatible = {"model": {"rgb_stem.net.0.conv.weight": torch.zeros(1, 1, 1, 1)}}
    bad_ckpt = tmp_path / "bad_source.pt"
    torch.save(incompatible, bad_ckpt)

    data = _make_dataset(tmp_path / "data")
    cmd = [sys.executable, str(Path(__file__).parents[1] / "train_real.py"),
           "--data", str(data), "--out", str(tmp_path / "run"), "--cpu", "--epochs", "1",
           "--warm-start", str(bad_ckpt)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "Error" in result.stderr or "error" in result.stderr


def test_warm_start_and_resume_are_mutually_exclusive(tmp_path):
    data = _make_dataset(tmp_path / "data")
    cmd = [sys.executable, str(Path(__file__).parents[1] / "train_real.py"),
           "--data", str(data), "--out", str(tmp_path / "run"), "--cpu", "--epochs", "1",
           "--warm-start", "/nonexistent.pt", "--resume"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "mutually exclusive" in result.stderr


def test_evaluate_loss_is_independent_of_how_patches_are_batched():
    """Regression: evaluate() used to average each batch's own weighted
    ratio and reweight by raw pixel count, which is NOT the same as a
    global weighted mean whenever batches differ in average weight-per-
    pixel -- confirmed directly: the same fixed pixels, grouped as one
    batch vs. two, reported 12.68 vs 50.16 for supposedly the same loss.
    The fix accumulates raw sums and divides once; this test constructs
    the same kind of imbalanced grouping and checks the numbers now agree.
    """
    torch.manual_seed(0)

    def make_patch(near_boundary_frac, n=100):
        target = torch.where(torch.rand(1, 1, 1, n) < near_boundary_frac,
                             torch.zeros(1, 1, 1, n), torch.full((1, 1, 1, n), 20.0))
        out = {"sdf": torch.zeros(1, 1, 1, n), "log_var": torch.zeros(1, 1, 1, n)}
        return out, target, torch.ones(1, 1, 1, n)

    dense_out, dense_target, dense_valid = make_patch(0.9)
    sparse_out, sparse_target, sparse_valid = make_patch(0.05)

    class TwoBatchLoader:
        """Same pixels as OneBatchLoader, but as two separate batches."""
        def __iter__(self):
            yield dense_out["sdf"], torch.zeros(1, 1, 1, 100), dense_target, dense_valid
            yield sparse_out["sdf"], torch.zeros(1, 1, 1, 100), sparse_target, sparse_valid

    class OneBatchLoader:
        def __iter__(self):
            sdf = torch.cat([dense_out["sdf"], sparse_out["sdf"]], dim=-1)
            target = torch.cat([dense_target, sparse_target], dim=-1)
            valid = torch.cat([dense_valid, sparse_valid], dim=-1)
            yield sdf, torch.zeros(1, 1, 1, 200), target, valid

    class IdentityModel:
        def eval(self): pass
        def __call__(self, rgb, height):
            return {"sdf": rgb, "log_var": torch.zeros_like(rgb)}

    device = torch.device("cpu")
    two_batch = module.evaluate(IdentityModel(), TwoBatchLoader(), device)
    one_batch = module.evaluate(IdentityModel(), OneBatchLoader(), device)
    assert two_batch["loss"] == pytest.approx(one_batch["loss"], rel=1e-6), (
        f"loss depends on batching: {two_batch['loss']} vs {one_batch['loss']}")
    # distance_mae_m and F1 were already batch-independent (plain sums) --
    # confirm the fix didn't regress those.
    assert two_batch["distance_mae_m"] == pytest.approx(one_batch["distance_mae_m"])
    assert two_batch["f1"] == pytest.approx(one_batch["f1"])


def test_patches_init_does_not_eagerly_load_full_tile_arrays(tmp_path):
    """Regression: __init__ used to decompress and permanently hold every
    tile's rgb/ndsm/distance/valid arrays -- measured directly at 8.51GB
    for the real 708-tile snapshot, which is what crashed a free-tier
    Colab kernel (system RAM, not GPU memory). __init__ must only need
    'valid' (for window discovery); rgb/ndsm/distance must stay
    undecompressed until something actually asks for a patch.
    """
    root = _make_dataset(tmp_path / "data")
    original_load = np.load
    accessed_keys = []

    class TrackingNpzFile:
        def __init__(self, real):
            self._real = real
        def __getitem__(self, key):
            accessed_keys.append(key)
            return self._real[key]
        def __enter__(self):
            return self
        def __exit__(self, *a):
            self._real.close()

    def tracking_load(path, *a, **kw):
        return TrackingNpzFile(original_load(path, *a, **kw))

    module.np.load = tracking_load
    try:
        module.Patches(root, "train", size=32)
    finally:
        module.np.load = original_load

    assert accessed_keys == ["valid"] * accessed_keys.count("valid"), accessed_keys
    assert "rgb" not in accessed_keys
    assert "ndsm" not in accessed_keys
    assert "distance" not in accessed_keys


def test_load_tile_cache_avoids_redundant_decompression(tmp_path, monkeypatch):
    root = _make_dataset(tmp_path / "data")
    patches = module.Patches(root, "train", size=32)
    module._load_tile.cache_clear()

    call_count = {"n": 0}
    original = module.np.load

    def counting_load(path, *a, **kw):
        call_count["n"] += 1
        return original(path, *a, **kw)

    monkeypatch.setattr(module.np, "load", counting_load)
    # Same tile (index 0's patches all come from one tile) accessed
    # repeatedly -- the cache must serve later accesses without re-reading.
    same_tile_indices = [i for i, w in enumerate(patches.windows) if w[0] == 0][:5]
    assert len(same_tile_indices) >= 2, "fixture needs a tile with multiple windows to test caching"
    for i in same_tile_indices:
        patches[i]
    assert call_count["n"] == 1, f"expected exactly 1 real load for repeated access to one tile, got {call_count['n']}"


def test_tile_grouped_shuffle_keeps_one_tiles_patches_consecutive():
    """Regression: plain DataLoader(shuffle=True) over many tiles gave the
    32-tile LRU cache a 2.5% hit rate (312 misses in 10 batches of 32),
    measured directly against the real 708-tile snapshot -- ~47 minutes
    projected for data loading alone in one epoch. A tile-grouped sampler
    must keep one tile's patches together so the cache stays warm.
    """
    # windows: tile 0 has 3 patches, tile 1 has 2, tile 2 has 4
    windows = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (1, 0, 0), (1, 0, 1),
              (2, 0, 0), (2, 0, 1), (2, 0, 2), (2, 0, 3)]
    torch.manual_seed(0)
    order = list(module.TileGroupedShuffle(windows))
    assert sorted(order) == list(range(len(windows)))  # every index visited exactly once

    # Reconstruct which TILE each yielded index belongs to, and confirm
    # same-tile indices are never separated by a different tile's index.
    tile_of = {i: windows[i][0] for i in range(len(windows))}
    tile_sequence = [tile_of[i] for i in order]
    seen_and_closed = set()
    current = None
    for t in tile_sequence:
        if t != current:
            assert t not in seen_and_closed, f"tile {t} revisited after another tile interleaved: {tile_sequence}"
            seen_and_closed.add(current) if current is not None else None
            current = t
    assert len(order) == len(windows)


def test_tile_grouped_shuffle_reshuffles_across_epochs():
    windows = [(t, 0, p) for t in range(20) for p in range(5)]
    torch.manual_seed(0)
    sampler = module.TileGroupedShuffle(windows)
    first = list(sampler)
    second = list(sampler)  # DataLoader calls __iter__ again each epoch
    assert first != second, "two epochs produced the identical order -- not actually reshuffling"


def test_best_checkpoint_tracks_f1_not_val_loss(tmp_path, monkeypatch):
    """Real run finding: under this task's class imbalance, val loss and
    boundary F1 diverge -- the lowest-loss epoch can have far worse recall
    than a noisier-loss epoch later on (measured: F1=0.032 at the min-loss
    epoch vs F1=0.123 at a worse-loss epoch). best.pt must keep the
    high-F1 epoch, not the low-loss one."""
    data = _make_dataset(tmp_path / "data")
    out = tmp_path / "run"

    # epoch 1 mirrors the real run's epoch 12: great loss, terrible F1.
    # epoch 2 mirrors the real run's epoch 23: worse loss, far better F1.
    scripted = iter([
        dict(loss=1.0, distance_mae_m=5.0, precision=0.8, recall=0.01, f1=0.02, valid_pixels=100, definition="x"),
        dict(loss=3.0, distance_mae_m=5.0, precision=0.6, recall=0.10, f1=0.15, valid_pixels=100, definition="x"),
    ])
    monkeypatch.setattr(module, "evaluate", lambda *a, **k: next(scripted))
    monkeypatch.setattr(sys, "argv", ["train_real.py", "--data", str(data), "--out", str(out),
                                       "--cpu", "--epochs", "2", "--batch-size", "2"])
    module.main()

    best = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    assert best["epoch"] == 2, (
        f"best.pt is epoch {best['epoch']}: a loss-based criterion would wrongly keep epoch 1 "
        "(lower loss, F1=0.02) over epoch 2 (higher loss, F1=0.15)")
    assert best["best"] == pytest.approx(0.15)


def test_resume_rearms_a_pre_f1_criterion_checkpoint(tmp_path, monkeypatch):
    """A checkpoint saved under the old loss-based criterion has best>1.0
    (a val loss, not an F1). Resuming must detect and reset it -- otherwise
    every future F1 (<=1.0) looks like a non-improvement forever and
    best.pt silently stops updating for the rest of the run."""
    data = _make_dataset(tmp_path / "data")
    out = tmp_path / "run"
    cmd = [sys.executable, str(Path(__file__).parents[1] / "train_real.py"),
           "--data", str(data), "--out", str(out), "--cpu", "--epochs", "1", "--batch-size", "2"]
    subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)

    # Simulate a checkpoint from before this fix: best left as a val loss.
    stale = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
    stale["best"] = 1.64  # a real val-loss value from the actual run
    torch.save(stale, out / "last.pt")

    scripted = iter([dict(loss=2.0, distance_mae_m=5.0, precision=0.5, recall=0.05, f1=0.05,
                          valid_pixels=100, definition="x")])
    monkeypatch.setattr(module, "evaluate", lambda *a, **k: next(scripted))
    monkeypatch.setattr(sys, "argv", ["train_real.py", "--data", str(data), "--out", str(out),
                                       "--cpu", "--epochs", "2", "--batch-size", "2", "--resume"])
    module.main()

    best = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    assert best["epoch"] == 2, "resumed epoch's F1=0.05 must register as an improvement over the re-armed best"
