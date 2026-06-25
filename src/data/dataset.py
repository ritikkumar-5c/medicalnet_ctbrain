"""Dataset that turns a manifest of DICOM series into model-ready volumes.

Manifest columns (see scripts/build_index.py):
    series_dir, study_id, series_id, label, label_name, split
Each row is one series = one training sample.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from .dicom_utils import load_series_volume


class CTSeriesDataset(Dataset):
    def __init__(self, rows, transforms, min_slices=10, cache=False):
        """rows: list of dicts with at least 'series_dir' and 'label'."""
        self.rows = rows
        self.transforms = transforms
        self.min_slices = min_slices
        self.cache = cache
        self._cache = {}

    def __len__(self):
        return len(self.rows)

    def _load_volume(self, series_dir):
        if self.cache and series_dir in self._cache:
            return self._cache[series_dir]
        vol, _spacing = load_series_volume(series_dir, min_slices=self.min_slices)
        if self.cache:
            self._cache[series_dir] = vol
        return vol

    def __getitem__(self, idx):
        row = self.rows[idx]
        vol = self._load_volume(row["series_dir"])
        if vol is None:
            # Unreadable series — fall back to a neighbour so a batch never
            # crashes. (build_index already filters most of these out.)
            return self.__getitem__((idx + 1) % len(self.rows))

        data = {"image": vol.astype(np.float32), "label": int(row["label"])}
        data = self.transforms(data)
        label = torch.as_tensor(data["label"], dtype=torch.long)
        return {"image": data["image"], "label": label, "series_dir": row["series_dir"]}
