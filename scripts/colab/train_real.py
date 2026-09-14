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
from geocadastra.models.heads import sdf_nll_loss


def masked_loss(out, target, valid):
    # Match the production head's Gaussian NLL; missing tasks contribute no loss.
    return sdf_nll_loss(out['sdf'], out['log_var'], target, valid)


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
    a = p.parse_args()
    if min(a.epochs,a.batch_size)<1: raise ValueError('Counts must be positive')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() and not a.cpu else 'cpu')
    root, outdir = Path(a.data), Path(a.out)
    outdir.mkdir(parents=True,exist_ok=True)
    signature = hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
    config = dict(batch_size=a.batch_size,patch_size=128,lr=.001,seed=42)
    model = MultiTaskNet().to(device)
    for head in (model.road_head,model.building_head,model.landuse_head):
        for param in head.parameters(): param.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['lr'])
    history, best, start = [], float('inf'), 0
    if a.resume or a.evaluate:
        saved = torch.load(outdir/('best.pt' if a.evaluate else 'last.pt'),map_location=device,weights_only=True)
        if saved['manifest_sha256'] != signature or saved['config'] != config:
            raise ValueError('Data/config differs from checkpoint')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
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
        metrics = evaluate(model,val,device)
        if not np.isfinite(metrics['loss']): raise ValueError('Nonfinite validation loss')
        improved = metrics['loss'] < best
        best = min(best,metrics['loss'])
        history.append(dict(epoch=epoch+1,train_loss=total/len(train.dataset),validation=metrics,seconds=time.monotonic()-began))
        payload = dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,
                       best=best,history=history,config=config,manifest_sha256=signature,
                       architecture='MultiTaskNet',trained_heads=['sdf','log_var'],
                       untrained_heads=['road','building','landuse'],
                       limitations='Real NZ pilot only; other heads untrained; NOT ready for worker deployment or calibrated',
                       rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state() if device.type=='cuda' else None)
        for name in (['best.pt','last.pt'] if improved else ['last.pt']):
            temporary = outdir/(name+'.tmp'); torch.save(payload,temporary); temporary.replace(outdir/name)
        (outdir/'history.json').write_text(json.dumps(history,indent=2))
        print(json.dumps(history[-1]),flush=True)


if __name__ == '__main__': main()
