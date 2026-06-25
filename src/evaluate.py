"""Evaluate a trained checkpoint on a manifest split.

    python src/evaluate.py --ckpt runs/medicalnet_r34/best.pth --split test
"""
import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import CTSeriesDataset, build_transforms  # noqa: E402
from engine import run_epoch  # noqa: E402
from models import build_model  # noqa: E402
from utils import compute_metrics, pick_device, read_manifest, CLASS_NAMES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--manifest", default=None, help="override manifest path")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    device = pick_device()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["cfg"]
    n_cls = cfg["model"]["num_classes"]
    in_ch = len(cfg["data"]["windows"])
    spatial = tuple(cfg["data"]["spatial_size"])
    windows = tuple(cfg["data"]["windows"])
    manifest = args.manifest or cfg["data"]["manifest"]

    rows = read_manifest(manifest, split=args.split)
    if not rows:
        raise SystemExit(f"No rows for split '{args.split}' in {manifest}")
    ds = CTSeriesDataset(rows, build_transforms(spatial, windows, train=False),
                         cfg["data"]["min_slices"])
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model = build_model(depth=cfg["model"]["depth"], num_classes=n_cls,
                        in_channels=in_ch, dropout=cfg["model"]["dropout"],
                        pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])

    out = run_epoch(model, loader, device, criterion=None, train=False,
                    desc=f"eval[{args.split}]")
    m = compute_metrics(out["y_true"], out["y_pred"], out["y_prob"], n_cls)

    print(f"\n=== {args.split} ({len(rows)} series) ===")
    print(f"accuracy     : {m['accuracy']:.4f}")
    print(f"balanced_acc : {m['balanced_acc']:.4f}")
    print(f"macro_f1     : {m['macro_f1']:.4f}")
    if "macro_auc" in m:
        print(f"macro_auc    : {m['macro_auc']:.4f}")
    print(f"macro_sens   : {m['macro_sensitivity']:.4f}   macro_spec: {m['macro_specificity']:.4f}")
    for c in CLASS_NAMES[:n_cls]:
        print(f"  {c:11s} P={m[f'precision_{c}']:.3f} "
              f"sens={m[f'sensitivity_{c}']:.3f} spec={m[f'specificity_{c}']:.3f} "
              f"F1={m[f'f1_{c}']:.3f}")
    print("confusion matrix (rows=true, cols=pred):")
    print("            " + " ".join(f"{c[:8]:>8s}" for c in CLASS_NAMES[:n_cls]))
    for i, row in enumerate(m["confusion_matrix"]):
        print(f"  {CLASS_NAMES[i][:10]:10s} " + " ".join(f"{v:8d}" for v in row))

    out_path = os.path.join(os.path.dirname(args.ckpt), f"metrics_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(m, f, indent=2)
    print(f"\nSaved metrics -> {out_path}")


if __name__ == "__main__":
    main()
