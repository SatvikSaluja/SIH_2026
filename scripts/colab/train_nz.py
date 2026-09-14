"""Cheap RGB-only parcel-boundary baseline; NOT a MultiTaskNet checkpoint.
Runs without Postgres. Validation selects checkpoints; test is evaluation-only.
"""
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import binary_dilation


def block(a,b):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.ReLU(),
                         nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(4,b),nn.ReLU())

class BoundaryNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1=block(3,16); self.e2=block(16,32); self.e3=block(32,64)
        self.d2=block(96,32); self.d1=block(48,16); self.out=nn.Conv2d(16,1,1)
    def forward(self,x):
        a=self.e1(x); b=self.e2(F.max_pool2d(a,2)); c=self.e3(F.max_pool2d(b,2))
        d=self.d2(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        d=self.d1(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
        return self.out(d)

class Patches(Dataset):
    def __init__(self,root,manifest,split,size=256):
        self.arrays=[]; self.windows=[]; self.size=size; self.augment=split=='train'
        for entry in manifest['tiles']:
            if entry['split']!=split: continue
            path=root/entry['path']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['training_sha256']:
                raise ValueError('Training data checksum differs from manifest')
            with np.load(path) as z:
                rgb=z['rgb'].copy(); y=z['boundary'].copy(); valid=z['valid'].copy()
            # Two-pixel band is a map-raster training target, not an accuracy claim.
            target=binary_dilation(y>0,iterations=1).astype('float32')
            index=len(self.arrays); self.arrays.append((rgb,target,valid))
            h,w=valid.shape
            for r in range(0,h-size+1,size):
                for c in range(0,w-size+1,size):
                    if valid[r:r+size,c:c+size].mean()>=0.75:
                        self.windows.append((index,r,c))
        if not self.windows: raise ValueError('No usable patches for '+split)
    def __len__(self): return len(self.windows)
    def __getitem__(self,i):
        k,r,c=self.windows[i]; n=self.size; rgb,y,v=self.arrays[k]
        x=rgb[:,r:r+n,c:c+n].astype('float32')/255
        y=y[None,r:r+n,c:c+n]; v=v[None,r:r+n,c:c+n].astype('float32')
        if self.augment:
            if random.random()<0.5: x,y,v=[np.flip(a,-1) for a in (x,y,v)]
            if random.random()<0.5: x,y,v=[np.flip(a,-2) for a in (x,y,v)]
        return tuple(torch.from_numpy(a.copy()) for a in (x,y,v))

def loss_fn(logits,target,valid):
    bce=F.binary_cross_entropy_with_logits(logits,target,reduction='none',pos_weight=logits.new_tensor(8.0))
    return (bce*valid).sum()/valid.sum().clamp_min(1)

@torch.no_grad()
def evaluate(model,loader,device):
    model.eval(); totals=np.zeros(5); loss_sum=0; pixels=0
    for x,y,v in loader:
        x,y,v=[a.to(device) for a in (x,y,v)]
        logits=model(x); n=v.sum().item()
        loss_sum+=loss_fn(logits,y,v).item()*n; pixels+=n
        pred=logits.sigmoid()>=0.5; truth=y>0; mask=v>0
        tp=(pred&truth&mask).sum().item(); fp=(pred&~truth&mask).sum().item(); fn=(~pred&truth&mask).sum().item()
        totals[:3]+=[tp,fp,fn]
    tp,fp,fn=map(float,totals[:3])
    return {'loss':loss_sum/max(pixels,1),'precision':tp/max(tp+fp,1),
            'recall':tp/max(tp+fn,1),'f1':2*tp/max(2*tp+fp+fn,1),
            'iou':tp/max(tp+fp+fn,1),'valid_pixels':int(pixels),
            'metric_definition':'pixel overlap of a 3-pixel reference boundary band; threshold 0.5'}

def atomic_save(value,path):
    temporary=path.with_suffix('.tmp')
    torch.save(value,temporary); os.replace(temporary,path)

@torch.no_grad()
def preview(model,dataset,device,path):
    from PIL import Image, ImageDraw
    dataset.augment=False
    x,y,v=dataset[0]; model.eval()
    prob=model(x[None].to(device)).sigmoid()[0,0].cpu().numpy()
    rgb=(x.permute(1,2,0).numpy()*255).astype('uint8')
    images=[Image.fromarray(rgb),Image.fromarray((y[0].numpy()*255).astype('uint8')).convert('RGB'),
            Image.fromarray((prob*255).astype('uint8')).convert('RGB')]
    n=rgb.shape[0]; canvas=Image.new('RGB',(3*n,n+30),'white'); draw=ImageDraw.Draw(canvas)
    for i,(im,label) in enumerate(zip(images,['Held-out RGB','Parcel-map reference','Boundary probability'])):
        canvas.paste(im,(i*n,30)); draw.text((i*n+5,5),label,fill='black')
    canvas.save(path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/nz_pilot'); p.add_argument('--out',default='runs/nz_rgb')
    p.add_argument('--epochs',type=int,default=20); p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--patch-size',type=int,default=256); p.add_argument('--lr',type=float,default=0.001)
    p.add_argument('--patience',type=int,default=5); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-minutes',type=float,default=60); p.add_argument('--resume',action='store_true')
    p.add_argument('--evaluate',action='store_true'); p.add_argument('--cpu',action='store_true')
    args=p.parse_args()
    if min(args.epochs,args.batch_size,args.patience,args.patch_size)<=0 or args.patch_size%4 or args.max_minutes<=0:
        raise ValueError('Positive counts required; patch size must be divisible by 4')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    root=Path(args.data); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    manifest_bytes=(root/'manifest.json').read_bytes(); manifest=json.loads(manifest_bytes)
    signature=hashlib.sha256(manifest_bytes).hexdigest()
    config={k:getattr(args,k) for k in ['batch_size','patch_size','lr','seed','patience']}
    model=BoundaryNet().to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    start=0; best=float('inf'); stale=0; history=[]
    def load(path):
        ckpt=torch.load(path,map_location=device,weights_only=True)
        if ckpt['manifest_sha256']!=signature or ckpt['config']!=config: raise ValueError('Checkpoint/data/config mismatch')
        model.load_state_dict(ckpt['model']); return ckpt
    if args.evaluate:
        load(out/'best.pt')
        test=Patches(root,manifest,'test',args.patch_size)
        results=evaluate(model,DataLoader(test,batch_size=args.batch_size),device)
        (out/'test_metrics.json').write_text(json.dumps(results,indent=2))
        preview(model,test,device,out/'test_preview.png'); print(json.dumps(results,indent=2)); return
    if args.resume:
        c=load(out/'last.pt'); optimizer.load_state_dict(c['optimizer'])
        if c['scaler'] and scaler.is_enabled(): scaler.load_state_dict(c['scaler'])
        start=c['epoch']; best=c['best']; stale=c['stale']; history=c['history']
        torch.set_rng_state(c['torch_rng'].cpu()); random.setstate(c['python_rng'])
        if device.type=='cuda' and c['cuda_rng'] is not None: torch.cuda.set_rng_state(c['cuda_rng'].cpu())
    elif (out/'last.pt').exists(): raise ValueError('Existing run: use --resume or a new --out directory')
    train=Patches(root,manifest,'train',args.patch_size); val=Patches(root,manifest,'val',args.patch_size)
    train_loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=0)
    val_loader=DataLoader(val,batch_size=args.batch_size,num_workers=0)
    print('Device:',device,'patches:',len(train),'train,',len(val),'validation',flush=True)
    beginning=time.monotonic()
    for epoch in range(start,args.epochs):
        if stale>=args.patience: break
        model.train(); total=0; steps=0; epoch_start=time.monotonic()
        for x,y,v in train_loader:
            x,y,v=[a.to(device) for a in (x,y,v)]; optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                loss=loss_fn(model(x),y,v)
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total+=loss.item(); steps+=1
        metrics=evaluate(model,val_loader,device); improved=metrics['loss']<best
        best=min(best,metrics['loss']); stale=0 if improved else stale+1
        row={'epoch':epoch+1,'train_loss':total/steps,'validation':metrics,'seconds':time.monotonic()-epoch_start}
        history.append(row)
        payload={'architecture':'nz_rgb_boundary_v1','model':model.state_dict(),'optimizer':optimizer.state_dict(),
                 'scaler':scaler.state_dict(),'epoch':epoch+1,'best':best,'stale':stale,'history':history,
                 'manifest_sha256':signature,'config':config,'torch_rng':torch.get_rng_state(),
                 'python_rng':random.getstate(),'cuda_rng':torch.cuda.get_rng_state() if device.type=='cuda' else None,
                 'limitations':'RGB pilot only; not compatible with GeoCadastra MultiTaskNet; not calibrated'}
        atomic_save(payload,out/'last.pt')
        if improved: atomic_save(payload,out/'best.pt')
        (out/'history.json').write_text(json.dumps(history,indent=2))
        print(json.dumps(row),flush=True)
        # Checked at epoch boundaries so the last checkpoint is always resumable.
        if time.monotonic()-beginning>=args.max_minutes*60: break
    print('Saved',out,'; run --evaluate once model choices are final.',flush=True)

if __name__=='__main__': main()
