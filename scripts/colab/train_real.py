"""Real RGB+nDSM parcel-distance training for MultiTaskNet; other heads unsupervised."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from geocadastra.models.backbone import MultiTaskNet


def masked_loss(out, target, valid, boundary_weight=15.0, decay=1.5):
    """Heteroscedastic NLL (same math as sdf_nll_loss), but weighted toward
    near-boundary pixels.

    sdf_nll_loss's own mask is binary (in/out of the labelled region) --
    correct for the production model, but insufficient here: ~95% of real
    pixels are far from any boundary, so an unweighted mean lets "predict
    roughly the average distance everywhere" become the cheapest solution.
    Confirmed, not assumed: two full training runs (cold-start and
    warm-started+5x data) both converged to exactly that -- predicted SDF
    never dropped below ~1.1m anywhere in 425k validation pixels, and F1
    was 0.0 at every epoch of both runs. More data alone did not change
    this; the loss shape does not reward finding the sparse boundary class.

    The weight decays smoothly with true distance (same exp(-d/decay) shape
    as geocadastra.models.infer.sdf_to_evidence, reused deliberately for a
    consistent "how much does this pixel matter" convention across the
    project) rather than a hard threshold, so the model isn't abruptly
    blind to the bulk regression signal -- every valid pixel keeps weight
    >= its own validity, boundary pixels get up to (1+boundary_weight)x.
    """
    precision = torch.exp(-out['log_var'])
    per_pixel = 0.5 * precision * (out['sdf'] - target) ** 2 + 0.5 * out['log_var']
    weight = valid * (1.0 + boundary_weight * torch.exp(-target.clamp(min=0) / decay))
    return (per_pixel * weight).sum() / weight.sum().clamp(min=1.0)


class Patches(Dataset):
    def __init__(self, root, split, size=128):
        self.arrays, self.windows = [], []
        self.size = size
        manifest = json.loads((root / 'manifest.json').read_text())
        for entry in manifest['tiles']:
            if entry['split'] != split:
                continue
            path = root / entry['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], 'Changed training data'
            with np.load(path) as z:
                arrays = tuple(z[k].copy() for k in ('rgb', 'ndsm', 'distance', 'valid'))
            index = len(self.arrays)
            self.arrays.append(arrays)
            h, w = arrays[-1].shape
            for r in range(0, h-size+1, size):
                for c in range(0, w-size+1, size):
                    if arrays[-1][r:r+size,c:c+size].mean() >= .95:
                        self.windows.append((index,r,c))
        if not self.windows:
            raise ValueError('No valid patches for ' + split)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        k,r,c = self.windows[index]
        rgb, height, distance, valid = self.arrays[k]
        s = self.size
        crop = lambda a: torch.from_numpy(a[...,r:r+s,c:c+s].astype('float32').copy())
        return crop(rgb)/255, crop(height[None]), crop(distance[None]), crop(valid[None])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = pixels = tp = fp = fn = absolute = 0.
    for batch in loader:
        rgb, height, target, valid = [x.to(device) for x in batch]
        out = model(rgb,height)
        n = valid.sum().item()
        total += masked_loss(out,target,valid).item()*n
        absolute += ((out['sdf']-target).abs()*valid).sum().item()
        pixels += n
        pred, truth, mask = out['sdf'].abs() <= .3, target <= .3, valid.bool()
        tp += (pred & truth & mask).sum().item()
        fp += (pred & ~truth & mask).sum().item()
        fn += (~pred & truth & mask).sum().item()
    return dict(loss=total/pixels, distance_mae_m=absolute/pixels,
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
    train = DataLoader(Patches(root,'train'),batch_size=a.batch_size,shuffle=True)
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
