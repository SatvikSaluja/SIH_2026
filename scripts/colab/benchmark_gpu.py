"""Measure, not estimate: training time/epoch, validation time, and peak
GPU memory for a short batched run on the verified snapshot -- before
committing to any full run, and before trusting any speedup number
someone (including an earlier version of this project's own commentary)
asserted without measuring it on this specific model.

Run on whatever device is available; on CPU it still reports real
per-epoch/per-validation timing (just no memory figure), so the same
script gives a real CPU baseline to compare a GPU number against later --
not a guess, an actual paired measurement.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from torch.utils.data import DataLoader

from geocadastra.models.backbone import MultiTaskNet
from train_real import Patches, evaluate, masked_loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/nz_pilot_snapshot', help='expand_full.py/build_snapshot.py '
                   'output -- already rgb/ndsm/distance/valid, the schema Patches() reads directly.')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f'Device: {device}', flush=True)
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
        torch.cuda.reset_peak_memory_stats(device)

    root = Path(args.data)
    train_loader = DataLoader(Patches(root, 'train'), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Patches(root, 'val'), batch_size=args.batch_size)
    print(f'{len(train_loader.dataset)} training patches, {len(val_loader.dataset)} validation patches, '
         f'batch_size={args.batch_size} -> {len(train_loader)} train batches/epoch', flush=True)

    model = MultiTaskNet().to(device)
    for head in (model.road_head, model.building_head, model.landuse_head):
        for param in head.parameters():
            param.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    results = []
    for epoch in range(args.epochs):
        model.train()
        train_start = time.monotonic()
        for rgb, height, target, valid in train_loader:
            rgb, height, target, valid = rgb.to(device), height.to(device), target.to(device), valid.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(model(rgb, height), target, valid)
            loss.backward()
            optimizer.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        train_seconds = time.monotonic() - train_start

        val_start = time.monotonic()
        metrics = evaluate(model, val_loader, device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        val_seconds = time.monotonic() - val_start

        peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == 'cuda' else None
        results.append({'epoch': epoch + 1, 'train_seconds': train_seconds, 'val_seconds': val_seconds,
                        'peak_gpu_memory_mb': peak_mem_mb, 'val_loss': metrics['loss']})
        print(f'  epoch {epoch+1}: train {train_seconds:.2f}s, val {val_seconds:.2f}s, '
             f'peak_mem={peak_mem_mb}, val_loss={metrics["loss"]:.4f}', flush=True)

    import json
    mean_train = sum(r['train_seconds'] for r in results) / len(results)
    mean_val = sum(r['val_seconds'] for r in results) / len(results)
    n_train_tiles = sum(1 for _ in root.glob('*')) # rough, informational only
    summary = {'device': str(device), 'gpu_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
              'batch_size': args.batch_size, 'epochs_measured': args.epochs,
              'mean_train_seconds_per_epoch': mean_train, 'mean_val_seconds': mean_val,
              'peak_gpu_memory_mb': results[-1]['peak_gpu_memory_mb'], 'per_epoch': results}
    Path('benchmark_result.json').write_text(json.dumps(summary, indent=2))
    print('\n' + json.dumps(summary, indent=2))
    print('\nTo estimate a full run: (target_epochs / epochs_measured) * mean_train_seconds_per_epoch '
         '* (target_train_tiles / tiles_used_here) -- do not extrapolate past what was actually measured.')


if __name__ == '__main__':
    main()
