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
    # 1 reproduces the original per-ward loop exactly (existing checkpoints/
    # tests are unaffected when this is left at its default). A GPU run
    # should set this explicitly -- one sample per forward/backward pass
    # is why the CPU loop never needed batching, and porting it unchanged
    # to a GPU wastes most of the device's parallelism per step.
    batch_size: int = 1
    # 0 disables validation entirely (default, zero behavior change).
    # Nonzero draws that many EXTRA wards from seeds strictly after the
    # training range (seed_offset+n_wards .. +n_val_wards), so held-out
    # wards can never silently coincide with a training one.
    n_val_wards: int = 0
    ward_params: WardParams = field(
        default_factory=lambda: WardParams(width=64, height=64, gsd=1.0, n_arterial_h=0, n_arterial_v=0, minor_spacing=30)
    )
    loss_weights: dict = field(default_factory=lambda: {"sdf": 1.0, "road": 1.0, "building": 1.0, "landuse": 1.0})


def make_dataset(config: TrainConfig) -> list[dict[str, torch.Tensor]]:
    return [
        ward_to_tensors(generate_ward(params=config.ward_params, seed=config.seed_offset + i))
        for i in range(config.n_wards)
    ]


def make_val_dataset(config: TrainConfig) -> list[dict[str, torch.Tensor]]:
    start = config.seed_offset + config.n_wards
    return [
        ward_to_tensors(generate_ward(params=config.ward_params, seed=start + i))
        for i in range(config.n_val_wards)
    ]


def _stack_batch(group: list[dict[str, torch.Tensor]], device) -> dict[str, torch.Tensor]:
    keys = ("rgb", "ndsm", "sdf_true", "road_true", "building_true", "landuse_true")
    return {key: torch.stack([sample[key] for sample in group]).to(device) for key in keys}


def _forward_loss(model, batch: dict[str, torch.Tensor], loss_weights: dict):
    out = model(batch["rgb"], batch["ndsm"])
    targets = {
        "sdf_true": batch["sdf_true"], "road_true": batch["road_true"],
        "building_true": batch["building_true"], "landuse_true": batch["landuse_true"],
    }
    return compute_loss(out, targets, loss_weights)


@torch.no_grad()
def evaluate(model, val_dataset: list[dict[str, torch.Tensor]], loss_weights: dict, device) -> float | None:
    """Mean per-ward loss on held-out wards, never used for a gradient step."""
    if not val_dataset:
        return None
    was_training = model.training
    model.eval()
    total = 0.0
    for sample in val_dataset:
        batch = _stack_batch([sample], device)
        loss, _ = _forward_loss(model, batch, loss_weights)
        total += loss.item()
    if was_training:
        model.train()
    return total / len(val_dataset)


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
    if (config.n_wards < 1 or config.epochs < 1 or config.batch_size < 1 or config.n_val_wards < 0
            or (max_epochs_per_run is not None and max_epochs_per_run < 1)):
        raise ValueError("ward, epoch, batch-size and val-ward counts must be positive (val wards may be 0)")
    device = torch.device(config.device)
    dataset = make_dataset(config)
    val_dataset = make_val_dataset(config)
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
    val_loss_history = []
    best_val_loss = float("inf")
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
        val_loss_history = list(checkpoint.get("val_loss_history", []))
        best_val_loss = min(val_loss_history, default=float("inf"))
        start_epoch = checkpoint["epoch"]
        torch.set_rng_state(checkpoint["rng_state"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu(), device=device)

    def save_checkpoint(path: Path, epoch: int):
        # A killed writer must leave the previous complete checkpoint usable.
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format_version": 1, "epoch": epoch, "config": asdict(config),
                   "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "scheduler": scheduler.state_dict(), "loss_history": loss_history,
                   "val_loss_history": val_loss_history, "best_val_loss": best_val_loss,
                   "rng_state": torch.get_rng_state(),
                   "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}
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

    stop_epoch = config.epochs if max_epochs_per_run is None else min(config.epochs, start_epoch + max_epochs_per_run)
    for _epoch in range(start_epoch, stop_epoch):
        epoch_loss = 0.0
        for start in range(0, len(dataset), config.batch_size):
            group = dataset[start:start + config.batch_size]
            batch = _stack_batch(group, device)
            optimizer.zero_grad()
            loss, _parts = _forward_loss(model, batch, config.loss_weights)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(group)
        scheduler.step()
        loss_history.append(epoch_loss / len(dataset))
        val_loss = evaluate(model, val_dataset, config.loss_weights, device)
        if val_loss is not None:
            val_loss_history.append(val_loss)
        if checkpoint_path is not None:
            path = Path(checkpoint_path)
            # A "best" checkpoint only exists when there's something held out
            # to judge it by -- picking "best epoch" by TRAINING loss is not
            # a validation signal, it's just the last epoch restated.
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(path.with_name(path.name + ".best"), _epoch + 1)
            # Persist the updated minimum, so resume cannot promote a worse epoch.
            save_checkpoint(path, _epoch + 1)
    return model, loss_history
