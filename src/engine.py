"""Train / evaluate one epoch. Shared by train.py and evaluate.py."""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def run_epoch(model, loader, device, criterion=None, optimizer=None,
              scaler=None, grad_clip=None, train=False, desc=""):
    model.train(train)
    losses = []
    all_true, all_pred, all_prob = [], [], []

    autocast_dev = "cuda" if device.type == "cuda" else "cpu"
    use_amp = scaler is not None and device.type == "cuda"

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=autocast_dev, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y) if criterion is not None else None

            if train:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    if grad_clip:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

        prob = F.softmax(logits.detach().float(), dim=1)
        all_prob.append(prob.cpu().numpy())
        all_pred.append(prob.argmax(1).cpu().numpy())
        all_true.append(y.detach().cpu().numpy())
        if loss is not None:
            losses.append(float(loss.detach()))
            pbar.set_postfix(loss=f"{np.mean(losses):.4f}")

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "y_true": np.concatenate(all_true) if all_true else np.array([]),
        "y_pred": np.concatenate(all_pred) if all_pred else np.array([]),
        "y_prob": np.concatenate(all_prob) if all_prob else None,
    }
