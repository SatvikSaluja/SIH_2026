"""Real RGB+nDSM parcel-distance training for MultiTaskNet; other heads unsupervised."""
import argparse
import functools
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from geocadastra.models.backbone import MultiTaskNet


def weighted_nll_terms(out, target, valid, boundary_weight=15.0, decay=1.5):
    """(per_pixel_loss, weight) tensors -- the raw terms, not yet reduced.

    Split out from masked_loss() specifically so evaluate() can accumulate
    a true GLOBAL weighted mean across an entire epoch (sum of per_pixel*
    weight, sum of weight, divide once at the end) instead of averaging
    each batch's own ratio and reweighting by raw pixel count -- those are
    not the same number whenever batches differ in average weight-per-pixel,
    which they do here (some patches are boundary-dense, most are not).
    Confirmed as a real discrepancy, not a theoretical one: the same fixed
    pixels, grouped as one batch vs. two, reported 12.68 vs 50.16 for what
    was supposed to be the same "loss" (see PR notes) -- a 4x difference
    with zero change to the model or the data.
    """
    precision = torch.exp(-out['log_var'])
    per_pixel = 0.5 * precision * (out['sdf'] - target) ** 2 + 0.5 * out['log_var']
    weight = valid * (1.0 + boundary_weight * torch.exp(-target.clamp(min=0) / decay))
    return per_pixel, weight


def masked_loss(out, target, valid, boundary_weight=15.0, decay=1.5):
    """Heteroscedastic NLL (same math as sdf_nll_loss), but weighted toward
    near-boundary pixels.

    sdf_nll_loss's own mask is binary (in/out of the labelled region) --
    correct for the production model, but insufficient here: ~95% of real
    pixels are far from any boundary, so an unweighted mean lets "predict
    roughly the average distance everywhere" become the cheapest solution.
    That mechanism is confirmed independently of this loss (see the F1
    section below); this weighting is this project's attempt at a fix,
    not itself proof the fix worked.

    The weight decays smoothly with true distance (same exp(-d/decay) shape
    as geocadastra.models.infer.sdf_to_evidence, reused deliberately for a
    consistent "how much does this pixel matter" convention across the
    project) rather than a hard threshold, so the model isn't abruptly
    blind to the bulk regression signal -- every valid pixel keeps weight
    >= its own validity, boundary pixels get up to (1+boundary_weight)x.

    This per-batch ratio is correct to train on (each batch's own gradient
    only ever needs its own batch's mean). It is NOT safe to accumulate
    across batches by multiplying by pixel count and re-averaging -- see
    weighted_nll_terms()'s docstring; evaluate() does the accumulation
    that way for that exact reason.
    """
    per_pixel, weight = weighted_nll_terms(out, target, valid, boundary_weight, decay)
    return (per_pixel * weight).sum() / weight.sum().clamp(min=1.0)


@functools.lru_cache(maxsize=32)
def _load_tile(path):
    """Bounded cache, not "load everything": 708 tiles held permanently
    (the original design) measured at 8.85GB RAM for this exact snapshot --
    confirmed directly, not estimated -- which is why a free-tier Colab
    kernel restarted mid-run. 32 tiles caps memory at roughly 32x a single
    tile's size regardless of dataset size, at the cost of re-decompressing
    a tile on a cache miss; correct for a dataset that keeps growing past
    what fits in memory, wrong to "optimize away" for a dataset small
    enough that eager loading was actually fine (the sub-20-tile case this
    class was first written and tested against).
    """
    with np.load(path) as z:
        return tuple(z[k].copy() for k in ('rgb', 'ndsm', 'distance', 'valid'))


class Patches(Dataset):
    def __init__(self, root, split, size=128):
        self.paths, self.windows = [], []
        self.size = size
        manifest = json.loads((root / 'manifest.json').read_text())
        for entry in manifest['tiles']:
            if entry['split'] != split:
                continue
            path = root / entry['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], 'Changed training data'
            # Only 'valid' is needed to find windows -- an npz's per-key
            # access decompresses just that array, not rgb/ndsm/distance
            # too, so this stays cheap even at hundreds of tiles.
            with np.load(path) as z:
                valid = z['valid']
                h, w = valid.shape
                index = len(self.paths)
                for r in range(0, h-size+1, size):
                    for c in range(0, w-size+1, size):
                        if valid[r:r+size,c:c+size].mean() >= .95:
                            self.windows.append((index,r,c))
            self.paths.append(path)
        if not self.windows:
            raise ValueError('No valid patches for ' + split)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        k,r,c = self.windows[index]
        rgb, height, distance, valid = _load_tile(self.paths[k])
        s = self.size
        crop = lambda a: torch.from_numpy(a[...,r:r+s,c:c+s].astype('float32').copy())
        return crop(rgb)/255, crop(height[None]), crop(distance[None]), crop(valid[None])


class TileGroupedShuffle(Sampler):
    """Shuffle which TILE comes next, and shuffle patch order WITHIN each
    tile, but keep one tile's own patches consecutive -- restores the
    locality plain index-level shuffling destroys.

    Measured, not assumed: full torch DataLoader(shuffle=True) over 708
    tiles gave _load_tile's 32-entry LRU cache a 2.5% hit rate (312 misses
    in 10 batches of 32) -- a batch of 32 uniformly random patches almost
    never repeats a tile from the previous batch, so nearly every patch
    re-reads and re-decompresses its tile from disk. Projected ~47 minutes
    of pure data loading for one epoch, before any GPU compute.

    This is not full IID shuffling -- it's shuffled at tile granularity,
    the standard tradeoff for a dataset too large to hold in memory as a
    whole. Both the tile order and each tile's patch order are reshuffled
    every epoch (a fresh Sampler instance each epoch, matching how
    DataLoader already re-invokes __iter__ per epoch), so training still
    sees a different sequence each time, just not fully independent
    sample-by-sample.
    """
    def __init__(self, windows):
        self.by_tile = {}
        for i, (tile_idx, _r, _c) in enumerate(windows):
            self.by_tile.setdefault(tile_idx, []).append(i)
        self._len = len(windows)

    def __iter__(self):
        tile_ids = list(self.by_tile)
        for t in torch.randperm(len(tile_ids)).tolist():
            indices = self.by_tile[tile_ids[t]]
            perm = torch.randperm(len(indices)).tolist()
            for p in perm:
                yield indices[p]

    def __len__(self):
        return self._len


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    weighted_sum = weight_total = pixels = tp = fp = fn = absolute = 0.
    for batch in loader:
        rgb, height, target, valid = [x.to(device) for x in batch]
        out = model(rgb,height)
        n = valid.sum().item()
        per_pixel, weight = weighted_nll_terms(out, target, valid)
        # Accumulate the RAW sums and divide once at the end -- a true
        # global weighted mean over every pixel in the split, not a
        # per-batch ratio re-averaged by raw pixel count (which depends on
        # batch_size for the same underlying pixels; see weighted_nll_terms).
        weighted_sum += (per_pixel * weight).sum().item()
        weight_total += weight.sum().item()
        absolute += ((out['sdf']-target).abs()*valid).sum().item()
        pixels += n
        pred, truth, mask = out['sdf'].abs() <= .3, target <= .3, valid.bool()
        tp += (pred & truth & mask).sum().item()
        fp += (pred & ~truth & mask).sum().item()
        fn += (~pred & truth & mask).sum().item()
    return dict(loss=weighted_sum/max(weight_total,1.), distance_mae_m=absolute/pixels,
                precision=tp/max(tp+fp,1), recall=tp/max(tp+fn,1),
                f1=2*tp/max(2*tp+fp+fn,1), valid_pixels=int(pixels),
                definition='Pixel overlap within 0.3 m of rasterized parcel reference; not survey accuracy')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/nz_multitask_real')
    p.add_argument('--out',default='runs/nz_multitask_real')
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--evaluate',action='store_true')
    p.add_argument('--warm-start',help='Load model weights from a DIFFERENT checkpoint '
                   '(e.g. a synthetic-trained one) before training starts. Distinct from '
                   '--resume: this starts a fresh epoch 0/optimizer/history, not a continued run.')
    a = p.parse_args()
    if min(a.epochs,a.batch_size)<1: raise ValueError('Counts must be positive')
    if a.warm_start and a.resume: raise ValueError('--warm-start and --resume are mutually exclusive')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() and not a.cpu else 'cpu')
    root, outdir = Path(a.data), Path(a.out)
    outdir.mkdir(parents=True,exist_ok=True)
    signature = hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
    config = dict(batch_size=a.batch_size,patch_size=128,lr=.001,seed=42)
    model = MultiTaskNet().to(device)
    warm_start_source = None
    if a.warm_start:
        source_path = Path(a.warm_start)
        if not source_path.exists(): raise ValueError('--warm-start checkpoint does not exist: '+str(source_path))
        source = torch.load(source_path,map_location=device,weights_only=True)
        source_state = source.get('model',source)  # geocadastra.models.train's own checkpoint shape
        # Loud, exact failure on any architecture mismatch -- a partial or
        # silently-reshaped load would train a model that LOOKS warm-started
        # but is actually running on wrong or randomly-reinitialized weights.
        model.load_state_dict(source_state,strict=True)  # raises RuntimeError on any key/shape mismatch
        warm_start_source = {'path':str(source_path),'sha256':hashlib.sha256(source_path.read_bytes()).hexdigest()}
        print(f'Warm-started from {source_path} ({warm_start_source["sha256"][:12]})',flush=True)
    for head in (model.road_head,model.building_head,model.landuse_head):
        for param in head.parameters(): param.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['lr'])
    # A flat LR produced real instability late in training (epoch 24 of the
    # first expanded run: val loss spiked from ~2.6 to 3.5, MAE from ~6.4m
    # to 10.5m) -- the synthetic trainer already fixed the identical
    # symptom this same way; T_max=a.epochs so a --resume with a longer
    # --epochs value still anneals to the NEW final epoch, not the old one.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=a.epochs,eta_min=config['lr']*0.01)
    history, best, start = [], float('inf'), 0
    if a.resume or a.evaluate:
        saved = torch.load(outdir/('best.pt' if a.evaluate else 'last.pt'),map_location=device,weights_only=True)
        if saved['manifest_sha256'] != signature or saved['config'] != config:
            raise ValueError('Data/config differs from checkpoint')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        if 'scheduler' in saved: scheduler.load_state_dict(saved['scheduler'])
        history,best,start = saved['history'],saved['best'],saved['epoch']
        torch.set_rng_state(saved['rng_state'].cpu())
        if device.type=='cuda' and saved['cuda_rng_state'] is not None:
            torch.cuda.set_rng_state(saved['cuda_rng_state'].cpu())
    elif (outdir/'last.pt').exists(): raise ValueError('Use --resume or a new output folder')
    if a.evaluate:
        result = evaluate(model,DataLoader(Patches(root,'test'),batch_size=a.batch_size),device)
        result['selected_epoch'] = start
        (outdir/'test_metrics.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2)); return
    train_patches = Patches(root,'train')
    train = DataLoader(train_patches,batch_size=a.batch_size,sampler=TileGroupedShuffle(train_patches.windows))
    val = DataLoader(Patches(root,'val'),batch_size=a.batch_size)
    print(f'Device: {device}; REAL imagery; {len(train.dataset)} training patches; {len(val.dataset)} validation patches',flush=True)
    for epoch in range(start,a.epochs):
        began = time.monotonic(); model.train(); total = 0
        for batch in train:
            rgb,height,target,valid = [x.to(device) for x in batch]
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(model(rgb,height),target,valid)
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            loss.backward(); optimizer.step(); total += loss.item()*len(rgb)
        scheduler.step()
        metrics = evaluate(model,val,device)
        if not np.isfinite(metrics['loss']): raise ValueError('Nonfinite validation loss')
        improved = metrics['loss'] < best
        best = min(best,metrics['loss'])
        history.append(dict(epoch=epoch+1,train_loss=total/len(train.dataset),validation=metrics,seconds=time.monotonic()-began))
        payload = dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch+1,
                       best=best,history=history,config=config,manifest_sha256=signature,
                       architecture='MultiTaskNet',trained_heads=['sdf','log_var'],
                       untrained_heads=['road','building','landuse'],warm_start_source=warm_start_source,
                       limitations='Real NZ pilot only; other heads untrained; NOT ready for worker deployment or calibrated',
                       rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state() if device.type=='cuda' else None)
        for name in (['best.pt','last.pt'] if improved else ['last.pt']):
            temporary = outdir/(name+'.tmp'); torch.save(payload,temporary); temporary.replace(outdir/name)
        (outdir/'history.json').write_text(json.dumps(history,indent=2))
        print(json.dumps(history[-1]),flush=True)


if __name__ == '__main__': main()
