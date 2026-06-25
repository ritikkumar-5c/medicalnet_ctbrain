"""Train the MedicalNet CT-brain classifier.

Example:
    python src/train.py --config configs/default.yaml
    python src/train.py --config configs/default.yaml --train.epochs 80 --train.lr 1e-4
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import CTSeriesDataset, build_transforms  # noqa: E402
from engine import run_epoch  # noqa: E402
from losses import build_loss  # noqa: E402
from models import build_model  # noqa: E402
from utils import (  # noqa: E402
    balanced_class_weights, compute_metrics, load_config, pick_device,
    read_manifest, sample_weights, set_seed, CLASS_NAMES,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    # a few common dotted overrides
    for key in ["train.epochs", "train.batch_size", "train.num_workers",
                "model.depth", "model.pretrained_path", "data.manifest", "output.dir",
                "train.loss", "train.monitor", "train.resume"]:
        ap.add_argument(f"--{key}")
    ap.add_argument("--train.lr", dest="train.lr", type=float)
    ap.add_argument("--train.target_sensitivity", dest="train.target_sensitivity", type=float)
    args, _ = ap.parse_known_args()
    overrides = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    # cast int-ish overrides
    for k in ["train.epochs", "train.batch_size", "train.num_workers", "model.depth"]:
        if k in overrides:
            overrides[k] = int(overrides[k])
    return args.config, overrides


def build_loaders(cfg):
    md = cfg["data"]
    spatial = tuple(md["spatial_size"])
    windows = tuple(md["windows"])
    n_cls = cfg["model"]["num_classes"]

    cache_man = md.get("cache_manifest")
    if cache_man:
        # Fast path: read pre-built uint8 volumes from the ctcache cache (already
        # resized + windowed); only spatial/intensity aug at load time.
        from ctcache import CachedVolumeDataset, read_cache_manifest
        from data.transforms import build_cached_transforms
        train_rows = read_cache_manifest(cache_man, split="train")
        val_rows = read_cache_manifest(cache_man, split="val")
        print(f"[cache] using {cache_man}: train={len(train_rows)} val={len(val_rows)} series")
        train_ds = CachedVolumeDataset(cache_man, split="train", mode="volume",
                                       transform=build_cached_transforms(True), rows=train_rows)
        val_ds = CachedVolumeDataset(cache_man, split="val", mode="volume",
                                     transform=build_cached_transforms(False), rows=val_rows)
    else:
        train_rows = read_manifest(md["manifest"], split="train")
        val_rows = read_manifest(md["manifest"], split="val")
        if not train_rows:
            raise SystemExit("No training rows in manifest. Run scripts/build_index.py first.")
        train_ds = CTSeriesDataset(
            train_rows, build_transforms(spatial, windows, train=True), md["min_slices"])
        val_ds = CTSeriesDataset(
            val_rows, build_transforms(spatial, windows, train=False), md["min_slices"])

    tc = cfg["train"]
    if tc["sampler"] == "weighted":
        w = sample_weights(train_rows, n_cls)
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        shuffle = False
    else:
        sampler, shuffle = None, True

    train_loader = DataLoader(
        train_ds, batch_size=tc["batch_size"], sampler=sampler, shuffle=shuffle,
        num_workers=tc["num_workers"], pin_memory=True, drop_last=True,
        persistent_workers=tc["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tc["batch_size"], shuffle=False,
        num_workers=tc["num_workers"], pin_memory=True,
        persistent_workers=tc["num_workers"] > 0,
    )
    return train_loader, val_loader, train_rows


def make_scheduler(optimizer, cfg, steps_per_epoch):
    tc = cfg["train"]
    name = tc["scheduler"]
    warmup = tc["warmup_epochs"]
    total_epochs = tc["epochs"]
    if name == "cosine":
        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / max(1, warmup)
            prog = (epoch - warmup) / max(1, total_epochs - warmup)
            return 0.5 * (1 + math.cos(math.pi * prog))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), "epoch"
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=4), "plateau"
    return None, None


def main():
    config_path, overrides = parse_args()
    cfg = load_config(config_path, overrides)
    set_seed(cfg["train"]["seed"])

    device = pick_device()
    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Device: {device}")

    train_loader, val_loader, train_rows = build_loaders(cfg)
    n_cls = cfg["model"]["num_classes"]
    in_ch = len(cfg["data"]["windows"])

    model = build_model(
        depth=cfg["model"]["depth"], num_classes=n_cls, in_channels=in_ch,
        dropout=cfg["model"]["dropout"], pretrained=cfg["model"]["pretrained"],
        pretrained_path=cfg["model"]["pretrained_path"],
    ).to(device)

    # Loss with optional balanced class weights + label smoothing.
    tc = cfg["train"]
    weight = None
    if tc["class_weighting"] == "balanced":
        weight = torch.tensor(balanced_class_weights(train_rows, n_cls), device=device)
        print(f"Class weights: {weight.tolist()}")
    criterion = build_loss(tc, n_cls, weight)             # weighted_ce | focal | cost_sensitive
    print(f"Loss: {tc.get('loss', 'weighted_ce')}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scheduler, sched_kind = make_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.cuda.amp.GradScaler() if (tc["amp"] and device.type == "cuda") else None

    # TensorBoard — logs to <output.dir>/tb. View with: tensorboard --logdir runs/
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))
    except Exception as e:
        print(f"[tb] TensorBoard unavailable ({e}); logging to history.json only.")

    monitor = tc["monitor"]
    best = -1.0
    best_epoch = -1
    bad_epochs = 0
    history = []
    start_epoch = 0

    def save_ckpt(path, epoch, metrics):
        """Save COMPLETE training state so a run can resume bit-for-bit."""
        import random
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch, "best": best, "best_epoch": best_epoch, "bad_epochs": bad_epochs,
            "monitor": monitor, "cfg": cfg, "metrics": metrics, "history": history,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        }, path)

    # Optional full-state resume (model+optimizer+scheduler+scaler+counters+RNG).
    resume = tc.get("resume") or cfg.get("resume")
    if resume and os.path.exists(resume):
        import random
        ck = torch.load(resume, map_location=device)
        model.load_state_dict(ck["model"])
        if ck.get("optimizer"): optimizer.load_state_dict(ck["optimizer"])
        if ck.get("scheduler") and scheduler is not None: scheduler.load_state_dict(ck["scheduler"])
        if ck.get("scaler") and scaler is not None: scaler.load_state_dict(ck["scaler"])
        best = ck.get("best", -1.0); best_epoch = ck.get("best_epoch", -1)
        bad_epochs = ck.get("bad_epochs", 0); history = ck.get("history", [])
        start_epoch = int(ck.get("epoch", -1)) + 1
        rng = ck.get("rng")
        if rng:
            try:
                random.setstate(rng["python"]); np.random.set_state(rng["numpy"])
                torch.set_rng_state(rng["torch"])
                if rng.get("cuda") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(rng["cuda"])
            except Exception as e:
                print(f"[resume] RNG restore skipped: {e}")
        print(f"[resume] {resume}: continuing from epoch {start_epoch} (best {monitor}={best:.4f})")

    # Backbone freeze state must reflect where we (re)start, not always epoch 0.
    freeze_n = tc["freeze_backbone_epochs"]
    frozen = freeze_n > 0 and start_epoch < freeze_n
    model.freeze_backbone(frozen)
    if frozen:
        print(f"Backbone frozen until epoch {freeze_n} (head warmup)")

    for epoch in range(start_epoch, tc["epochs"]):
        if frozen and epoch == tc["freeze_backbone_epochs"]:
            model.freeze_backbone(False)
            frozen = False
            print(f"[epoch {epoch}] unfroze backbone")

        tr = run_epoch(model, train_loader, device, criterion, optimizer,
                       scaler=scaler, grad_clip=tc["grad_clip"], train=True,
                       desc=f"train {epoch+1}/{tc['epochs']}")
        va = run_epoch(model, val_loader, device, criterion, train=False,
                       desc=f"val   {epoch+1}/{tc['epochs']}")
        m = compute_metrics(va["y_true"], va["y_pred"], va["y_prob"], n_cls)

        if sched_kind == "epoch":
            scheduler.step()
        elif sched_kind == "plateau":
            scheduler.step(m[monitor])

        lr_now = optimizer.param_groups[0]["lr"]
        rec = {"epoch": epoch, "lr": lr_now, "train_loss": tr["loss"],
               "val_loss": va["loss"], **m}
        history.append(rec)

        if writer is not None:
            writer.add_scalar("loss/train", tr["loss"], epoch)
            writer.add_scalar("loss/val", va["loss"], epoch)
            writer.add_scalar("lr", lr_now, epoch)
            # log every scalar metric (skip the confusion-matrix list)
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f"metrics/{k}", v, epoch)
            writer.flush()

        print(f"[{epoch+1:3d}] lr={lr_now:.2e} train_loss={tr['loss']:.4f} "
              f"val_loss={va['loss']:.4f} acc={m['accuracy']:.3f} "
              f"bal_acc={m['balanced_acc']:.3f} macro_f1={m['macro_f1']:.3f} "
              + " ".join(f"recall_{c}={m.get(f'recall_{c}',0):.2f}"
                         for c in CLASS_NAMES[:n_cls]))

        score = m[monitor]
        improved = score > best
        if improved:
            best, best_epoch, bad_epochs = score, epoch, 0
        else:
            bad_epochs += 1
        save_ckpt(os.path.join(out_dir, "last.pth"), epoch, m)   # full state (exact resume)
        if improved:
            save_ckpt(os.path.join(out_dir, "best.pth"), epoch, m)
            print(f"      ** new best {monitor}={best:.4f} -> saved best.pth")

        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        if tc["early_stop_patience"] and bad_epochs >= tc["early_stop_patience"]:
            print(f"Early stop: no {monitor} improvement for {bad_epochs} epochs.")
            break

    if writer is not None:
        writer.close()
    print(f"Done. Best {monitor}={best:.4f} at epoch {best_epoch+1}. "
          f"Checkpoints in {out_dir}/")


if __name__ == "__main__":
    main()
