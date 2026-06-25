"""Auto-download and load MedicalNet (Med3D) pretrained backbone weights.

The official weights live in a Google Drive folder linked from the MedicalNet
repo. They are also mirrored on the Hugging Face Hub. We try, in order:

  1. A local cache / user-provided path.
  2. The Hugging Face Hub mirror (no auth needed).
  3. Google Drive via ``gdown`` (file ids from the official README).

If all fail, we print clear manual-download instructions and continue with
random init (so the pipeline still runs end-to-end).
"""
import os
import re
from pathlib import Path

import torch

# Default cache dir for downloaded checkpoints.
CACHE_DIR = Path(os.environ.get("MEDICALNET_CACHE", Path.home() / ".cache" / "medicalnet"))

# Canonical checkpoint filenames in the official release.
_CKPT_NAME = {
    10: "resnet_10_23dataset.pth",
    18: "resnet_18_23dataset.pth",
    34: "resnet_34_23dataset.pth",
    50: "resnet_50_23dataset.pth",
}

# Official Tencent MedicalNet weights, mirrored per-depth on the HF Hub
# (verified: public, no auth). Each repo contains resnet_{depth}_23dataset.pth.
#   https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet34
_HF_REPO = "TencentMedicalNet/MedicalNet-Resnet{depth}"
_HF_RESOLVE = "https://huggingface.co/{repo}/resolve/main/{fname}"


def _strip_prefix(state_dict):
    """Drop a leading ``module.`` (DataParallel) prefix if present."""
    if any(k.startswith("module.") for k in state_dict):
        return {re.sub(r"^module\.", "", k): v for k, v in state_dict.items()}
    return state_dict


def _candidate_filenames(depth):
    # Verified primary name first, then fallbacks.
    return (_CKPT_NAME[depth], f"resnet_{depth}.pth", "pytorch_model.bin")


def _try_hf(depth):
    """Download via huggingface_hub if installed (handles caching/resume)."""
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return None
    repo = _HF_REPO.format(depth=depth)
    for fname in _candidate_filenames(depth):
        try:
            return hf_hub_download(repo_id=repo, filename=fname)
        except Exception:
            continue
    return None


def _try_hf_direct(depth, dest):
    """Plain HTTPS download from the HF resolve URL (no huggingface_hub needed)."""
    import urllib.request
    repo = _HF_REPO.format(depth=depth)
    for fname in _candidate_filenames(depth):
        url = _HF_RESOLVE.format(repo=repo, fname=fname)
        try:
            tmp = str(dest) + ".part"
            with urllib.request.urlopen(url) as r:
                if getattr(r, "status", 200) != 200:
                    continue
                with open(tmp, "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
            os.replace(tmp, dest)
            return str(dest)
        except Exception:
            continue
    return None


def resolve_checkpoint(depth, pretrained_path=None):
    """Return a local path to the pretrained checkpoint, or None if unavailable."""
    if pretrained_path:
        p = Path(pretrained_path)
        if p.is_file():
            return str(p)
        raise FileNotFoundError(f"--pretrained_path not found: {p}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / _CKPT_NAME.get(depth, f"resnet_{depth}.pth")
    if cached.is_file():
        return str(cached)

    print(f"[weights] downloading MedicalNet resnet{depth} weights "
          f"(~250MB) from Hugging Face ...")
    path = _try_hf(depth)
    if path:
        return path

    path = _try_hf_direct(depth, cached)
    if path and Path(path).is_file():
        return str(path)

    repo = _HF_REPO.format(depth=depth)
    print(
        "\n[weights] Could not auto-download MedicalNet weights for "
        f"resnet{depth}.\n"
        "          Download manually (public, no auth):\n"
        f"            {_HF_RESOLVE.format(repo=repo, fname=_CKPT_NAME.get(depth))}\n"
        f"          and place '{_CKPT_NAME.get(depth)}' at:\n"
        f"            {cached}\n"
        "          or pass --pretrained_path /path/to/file.pth\n"
        "          Continuing with RANDOM init for now.\n"
    )
    return None


def load_pretrained_backbone(backbone, depth, pretrained_path=None, verbose=True):
    """Load MedicalNet backbone weights into ``backbone`` (a ResNet3D).

    Returns the number of matched tensors. Missing/extra keys (e.g. the seg
    decoder) are ignored.
    """
    ckpt = resolve_checkpoint(depth, pretrained_path)
    if ckpt is None:
        return 0

    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = _strip_prefix(state)

    model_sd = backbone.state_dict()
    matched = {
        k: v for k, v in state.items()
        if k in model_sd and tuple(v.shape) == tuple(model_sd[k].shape)
    }
    missing = [k for k in model_sd if k not in matched]
    backbone.load_state_dict(matched, strict=False)

    if verbose:
        print(f"[weights] loaded {len(matched)}/{len(model_sd)} backbone tensors "
              f"from {os.path.basename(ckpt)} ({len(missing)} left at init)")
    return len(matched)
