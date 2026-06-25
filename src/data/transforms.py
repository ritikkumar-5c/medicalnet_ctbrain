"""MONAI transform pipelines for CT-brain volumes.

Input to these pipelines is a dict ``{"image": (D,H,W) float32 HU, "label": int}``.
We resize every volume to a fixed ``(D,H,W)`` and window/normalize intensities.

CT windowing: brain parenchyma lives around 0-80 HU. We support either a
single brain window or a multi-window (brain / subdural / bone) stack fed as
separate input channels — useful because many abnormalities (acute blood,
fractures) are most visible in different windows.
"""
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    Lambdad,
    RandAdjustContrastd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandShiftIntensityd,
    Resized,
    ScaleIntensityRanged,
    ToTensord,
)

# (window_center, window_width) presets.
WINDOWS = {
    "brain": (40, 80),
    "subdural": (50, 130),
    "bone": (600, 2800),
    "stroke": (35, 40),
}


def _window_to_unit(img, center, width):
    """Map HU to [0,1] for a (center,width) window (operates on np or tensor)."""
    lo, hi = center - width / 2.0, center + width / 2.0
    img = (img - lo) / (hi - lo)
    return img.clip(0, 1)


def _multi_window(img, windows):
    """Stack several windows along the channel axis. img: (1,D,H,W) -> (C,D,H,W)."""
    import numpy as np
    base = img[0]
    chans = [_window_to_unit(base, *WINDOWS[w]) for w in windows]
    return np.stack(chans, axis=0).astype("float32")


def build_transforms(spatial_size, windows=("brain",), train=True):
    """Return (train_or_val) transform Compose.

    spatial_size: (D, H, W) target.
    windows: tuple of window names -> number of input channels.
    """
    keys = ["image"]
    t = [
        EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),  # (1,D,H,W)
        Resized(keys=keys, spatial_size=spatial_size, mode="trilinear", align_corners=False),
    ]

    if len(windows) == 1:
        c, w = WINDOWS[windows[0]]
        t.append(ScaleIntensityRanged(
            keys=keys, a_min=c - w / 2, a_max=c + w / 2, b_min=0.0, b_max=1.0, clip=True,
        ))
    else:
        t.append(Lambdad(keys=keys, func=lambda x: _multi_window(x, windows)))

    if train:
        t += [
            RandAffined(
                keys=keys, prob=0.5,
                rotate_range=(0.0, 0.0, 0.26),      # ~15deg in-plane
                scale_range=(0.1, 0.1, 0.1),
                translate_range=(4, 8, 8),
                mode="bilinear", padding_mode="zeros",
            ),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=2),  # left-right
            RandShiftIntensityd(keys=keys, prob=0.4, offsets=0.05),
            RandAdjustContrastd(keys=keys, prob=0.3, gamma=(0.8, 1.2)),
            RandGaussianNoised(keys=keys, prob=0.2, std=0.02),
        ]

    t.append(ToTensord(keys=keys))
    return Compose(t)


def build_cached_transforms(train=True):
    """Transforms for inputs from the ctcache uint8 cache.

    The cache already holds resized + windowed volumes in [0,1] as (C,D,H,W),
    so we skip Resize/Window/EnsureChannelFirst and apply spatial/intensity
    augmentation only (train), then ToTensor.
    """
    keys = ["image"]
    t = []
    if train:
        t += [
            RandAffined(
                keys=keys, prob=0.5,
                rotate_range=(0.0, 0.0, 0.26), scale_range=(0.1, 0.1, 0.1),
                translate_range=(4, 8, 8), mode="bilinear", padding_mode="zeros",
            ),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
            RandShiftIntensityd(keys=keys, prob=0.4, offsets=0.05),
            RandAdjustContrastd(keys=keys, prob=0.3, gamma=(0.8, 1.2)),
            RandGaussianNoised(keys=keys, prob=0.2, std=0.02),
        ]
    t.append(ToTensord(keys=keys))
    return Compose(t)
