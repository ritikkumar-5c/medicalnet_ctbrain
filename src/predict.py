"""Run inference on a single study or series directory.

    # one series folder (.../study/series/*.dcm)
    python src/predict.py --ckpt runs/medicalnet_r34/best.pth --series /path/to/study/series

    # a whole study: predicts each series, plus a study-level vote
    python src/predict.py --ckpt runs/medicalnet_r34/best.pth --study /path/to/study
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import build_transforms  # noqa: E402
from data.dicom_utils import list_series_dirs, load_series_volume  # noqa: E402
from models import build_model  # noqa: E402
from utils import pick_device, CLASS_NAMES  # noqa: E402


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]
    model = build_model(depth=cfg["model"]["depth"], num_classes=cfg["model"]["num_classes"],
                        in_channels=len(cfg["data"]["windows"]),
                        dropout=cfg["model"]["dropout"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def predict_series(model, cfg, series_dir, device):
    vol, _ = load_series_volume(series_dir, min_slices=cfg["data"]["min_slices"])
    if vol is None:
        return None
    tfm = build_transforms(tuple(cfg["data"]["spatial_size"]),
                           tuple(cfg["data"]["windows"]), train=False)
    data = tfm({"image": vol.astype(np.float32), "label": 0})
    x = data["image"].unsqueeze(0).to(device)
    prob = F.softmax(model(x).float(), dim=1)[0].cpu().numpy()
    return prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--series", help="a single series directory")
    ap.add_argument("--study", help="a study directory (predict every series)")
    args = ap.parse_args()

    device = pick_device()
    model, cfg = load_model(args.ckpt, device)
    names = CLASS_NAMES[:cfg["model"]["num_classes"]]

    if args.series:
        targets = [args.series]
    elif args.study:
        targets = list_series_dirs(args.study)
        if not targets:
            raise SystemExit(f"No series found under {args.study}")
    else:
        raise SystemExit("Pass --series or --study")

    probs = []
    for sd in targets:
        p = predict_series(model, cfg, sd, device)
        if p is None:
            print(f"{os.path.basename(sd)}: <unreadable / too few slices>")
            continue
        probs.append(p)
        top = int(p.argmax())
        print(f"{os.path.basename(sd)}: {names[top]} ({p[top]*100:.1f}%)  "
              + " ".join(f"{n}={p[i]*100:.1f}%" for i, n in enumerate(names)))

    if args.study and probs:
        mean = np.mean(probs, axis=0)
        top = int(mean.argmax())
        print(f"\nSTUDY VERDICT: {names[top]} ({mean[top]*100:.1f}%)  "
              + " ".join(f"{n}={mean[i]*100:.1f}%" for i, n in enumerate(names)))


if __name__ == "__main__":
    main()
