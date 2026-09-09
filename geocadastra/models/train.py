"""Training entry point (Stage 4): train on synthetic first.

Real drone data doesn't exist yet in this repo -- everything works on
synthetic first, and the synthetic-trained result is the baseline any
later real-data model must beat.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import os
import tempfile

import torch
import torch.nn.functional as F

from geocadastra.models.backbone import MultiTaskNet
from geocadastra.models.dataset import N_LANDUSE_CLASSES, ward_to_tensors
from geocadastra.models.heads import sdf_nll_loss
from geocadastra.synth.generator import WardParams, generate_ward


@dataclass
class TrainConfig:
    n_wards: int = 8
    epochs: int = 30
    lr: float = 1e-3
    seed_offset: int = 0
    device: str = "cpu"
    ward_params: WardParams = field(
        default_factory=lambda: WardParams(width=64, height=64, gsd=1.0, n_arterial_h=0, n_arterial_v=0, minor_spacing=30)
    )
    loss_weights: dict = field(default_factory=lambda: {"sdf": 1.0, "road": 1.0, "building": 1.0, "landuse": 1.0})


def make_dataset(config: TrainConfig) -> list[dict[str, torch.Tensor]]:
    return [
        ward_to_tensors(generate_ward(params=config.ward_params, seed=config.seed_offset + i))
        for i in range(config.n_wards)
    ]


def compute_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], weights: dict) -> tuple[torch.Tensor, dict]:
    sdf_loss = sdf_nll_loss(out["sdf"], out["log_var"], batch["sdf_true"])
    road_loss = F.binary_cross_entropy_with_logits(out["road"], batch["road_true"])
    building_loss = F.binary_cross_entropy_with_logits(out["building"], batch["building_true"])
    landuse_loss = F.cross_entropy(out["landuse"], batch["landuse_true"])
    total = (
        weights["sdf"] * sdf_loss
        + weights["road"] * road_loss
        + weights["building"] * building_loss
        + weights["landuse"] * landuse_loss
    )
    return total, {"sdf": sdf_loss.item(), "road": road_loss.item(), "building": building_loss.item(), "landuse": landuse_loss.item()}


def train(config: TrainConfig | None = None, model: MultiTaskNet | None = None, *,
          checkpoint_path: str | Path | None = None, resume_from: str | Path | None = None,
          max_epochs_per_run: int | None = None) -> tuple[MultiTaskNet, list[float]]:
    """Train on the declared device, optionally resuming an epoch checkpoint.

    `epochs` is the total schedule length, including restored epochs.
    `max_epochs_per_run` allows a bounded run without changing that schedule.
    A custom architecture must be supplied again when resuming it.
    """
    config = config or TrainConfig()
    if config.n_wards < 1 or config.epochs < 1 or (max_epochs_per_run is not None and max_epochs_per_run < 1):
        raise ValueError("ward and epoch counts must be positive")
    device = torch.device(config.device)
    dataset = make_dataset(config)
    model = model or MultiTaskNet(n_landuse_classes=N_LANDUSE_CLASSES)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    # a constant LR over many epochs on a small dataset went visibly unstable
    # in practice (loss spiking after it had already dropped, not just
    # plateauing) -- cosine decay to a small fraction of the starting LR
    # keeps early training fast and late training from overshooting a
    # minimum it already found
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.lr * 0.01)

    loss_history = []
    start_epoch = 0
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device, weights_only=True)
        saved_config, requested_config = dict(checkpoint["config"]), asdict(config)
        saved_config.pop("device", None)
        requested_config.pop("device", None)
        if saved_config != requested_config:
            raise ValueError("resume configuration differs from checkpoint training schedule or dataset")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        loss_history = list(checkpoint["loss_history"])
        start_epoch = checkpoint["epoch"]
        torch.set_rng_state(checkpoint["rng_state"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu(), device=device)
    stop_epoch = config.epochs if max_epochs_per_run is None else min(config.epochs, start_epoch + max_epochs_per_run)
    for _epoch in range(start_epoch, stop_epoch):
        epoch_loss = 0.0
        for batch in dataset:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            out = model(batch["rgb"].unsqueeze(0), batch["ndsm"].unsqueeze(0))
            targets = {
                "sdf_true": batch["sdf_true"].unsqueeze(0),
                "road_true": batch["road_true"].unsqueeze(0),
                "building_true": batch["building_true"].unsqueeze(0),
                "landuse_true": batch["landuse_true"].unsqueeze(0),
            }
            loss, _parts = compute_loss(out, targets, config.loss_weights)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        loss_history.append(epoch_loss / len(dataset))
        if checkpoint_path is not None:
            path = Path(checkpoint_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"format_version": 1, "epoch": _epoch + 1, "config": asdict(config),
                       "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                       "scheduler": scheduler.state_dict(), "loss_history": loss_history,
                       "rng_state": torch.get_rng_state(),
                       "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}
            # A killed writer must leave the previous complete checkpoint usable.
            fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
            try:
                with os.fdopen(fd, "wb") as stream:
                    torch.save(payload, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
    return model, loss_history
