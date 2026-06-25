"""Read a DICOM series directory into a Hounsfield-unit 3D volume.

A "series" here is one ``.../{study}/{series}/`` folder of ``*.dcm`` slices.
We load all slices, sort them by anatomical position, apply the rescale
slope/intercept to get HU, and return a ``(D, H, W)`` float32 array plus the
voxel spacing ``(sz, sy, sx)`` in mm.
"""
import glob
import os

import numpy as np
import pydicom


def list_series_dirs(study_dir):
    """Return subdirectories of a study that contain at least one .dcm file."""
    out = []
    for d in sorted(glob.glob(os.path.join(study_dir, "*"))):
        if os.path.isdir(d) and glob.glob(os.path.join(d, "*.dcm")):
            out.append(d)
    # Some studies may drop dcm files directly under the study dir.
    if not out and glob.glob(os.path.join(study_dir, "*.dcm")):
        out.append(study_dir)
    return out


def _slice_sort_key(ds):
    """Sort by through-plane position; fall back to InstanceNumber."""
    ipp = getattr(ds, "ImagePositionPatient", None)
    iop = getattr(ds, "ImageOrientationPatient", None)
    if ipp is not None and iop is not None:
        # Project position onto the slice normal for a robust ordering.
        try:
            r = np.array(iop[:3], dtype=float)
            c = np.array(iop[3:], dtype=float)
            normal = np.cross(r, c)
            return float(np.dot(np.array(ipp, dtype=float), normal))
        except Exception:
            pass
    if ipp is not None:
        try:
            return float(ipp[2])
        except Exception:
            pass
    return float(getattr(ds, "InstanceNumber", 0) or 0)


def load_series_volume(series_dir, min_slices=10):
    """Load a series folder into (volume_HU, spacing_zyx).

    Returns (None, None) if the series is unreadable or too thin (e.g. a
    scout/localizer). Slices whose in-plane shape disagrees with the majority
    are dropped.
    """
    files = glob.glob(os.path.join(series_dir, "*.dcm"))
    if len(files) < min_slices:
        return None, None

    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(f)
            if "PixelData" not in ds:
                continue
            slices.append(ds)
        except Exception:
            continue
    if len(slices) < min_slices:
        return None, None

    # Keep the dominant in-plane shape (guards against mixed/secondary capture).
    shapes = [(int(s.Rows), int(s.Columns)) for s in slices]
    dom = max(set(shapes), key=shapes.count)
    slices = [s for s, sh in zip(slices, shapes) if sh == dom]
    if len(slices) < min_slices:
        return None, None

    slices.sort(key=_slice_sort_key)

    # Decode pixel data per slice; drop any slice whose pixels won't decode
    # (e.g. a compressed transfer syntax with no backend, or corruption).
    planes, kept = [], []
    for ds in slices:
        try:
            arr = ds.pixel_array.astype(np.float32)
        except Exception:
            continue
        if arr.shape != dom:
            continue
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        planes.append(arr * slope + intercept)
        kept.append(ds)
    if len(planes) < min_slices:
        return None, None
    vol = np.stack(planes, axis=0).astype(np.float32)
    slices = kept

    # Voxel spacing (z, y, x) in mm.
    ref = slices[0]
    ps = getattr(ref, "PixelSpacing", [1.0, 1.0])
    z = _estimate_slice_spacing(slices)
    spacing = (float(z), float(ps[0]), float(ps[1]))
    return vol, spacing


def _estimate_slice_spacing(slices):
    """Median inter-slice distance; fall back to SliceThickness."""
    positions = []
    for s in slices:
        ipp = getattr(s, "ImagePositionPatient", None)
        if ipp is not None:
            try:
                positions.append(float(ipp[2]))
            except Exception:
                pass
    if len(positions) >= 2:
        diffs = np.abs(np.diff(sorted(positions)))
        diffs = diffs[diffs > 1e-3]
        if diffs.size:
            return float(np.median(diffs))
    st = getattr(slices[0], "SliceThickness", 1.0)
    try:
        return float(st) or 1.0
    except Exception:
        return 1.0
