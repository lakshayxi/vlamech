"""How much a background-only corruption reaches protected pixels of the model input.

Used by: scripts/measure_preprocessing_leakage.py, tests/test_perturbations.py.

Background-only families (bg_patches, matched_bg_box) leave every protected RAW pixel unchanged (tested). OFT's
preprocessing (vlamech.models.oft.policy_view: JPEG round-trip with 4:2:0 chroma, 16x16 MCUs; Lanczos-3 resize
256->224; 90%-area centre crop resized back with bilinear interpolation) mixes neighbouring pixels, so model-input
pixels near the protected/background boundary can change. This module measures that, never changes it.

Geometry: every model-input pixel is mapped to the raw pixel its centre samples (exact coordinate transforms of
tf.image.resize with half-pixel centres and tf.image.crop_and_resize); a model-input pixel is "protected" if that
raw pixel is, and its depth = Chebyshev distance (raw px) from that raw pixel to the nearest background pixel.
Changes are binned by depth, so the report says how far into protected regions the preprocessing carries them.

exact_model_input_views(): the strict digital variant. Both the clean and the corrupted frame go through the
upstream preprocessing; model-input pixels whose source neighbourhood (MAP_MARGIN raw px) contains a protected
pixel are taken from the clean result, all others from the corrupted one. Protected model-input pixels are then
exactly clean; the price is a hard seam at the boundary (labelled as a digital condition, never as physical).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np

RAW, MODEL = 256, 224
CROP_SCALE = 0.9  # upstream center_crop_image (area)
MAP_MARGIN = 2  # raw px: max error of the nearest-pixel map (tests/test_perturbations.py, linear-ramp check)
DEPTH_BINS = ((1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 10**6))


@lru_cache(maxsize=2)
def source_pixels(center_crop: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(rows, cols) of the raw pixel sampled by the centre of each model-input pixel, each (224,)."""
    s = np.arange(MODEL, dtype=np.float64)
    if center_crop:  # crop_and_resize: in = y1*(H-1) + out*(y2-y1)*(H-1)/(crop-1), H = crop = 224
        side = np.sqrt(CROP_SCALE)
        y1 = (1 - side) / 2
        s = y1 * (MODEL - 1) + s * side * (MODEL - 1) / (MODEL - 1)
    raw = (s + 0.5) * RAW / MODEL - 0.5  # resize with half-pixel centres
    idx = np.clip(np.rint(raw), 0, RAW - 1).astype(int)
    return idx, idx  # square images: rows and columns use the same map


def depth_map(protected: np.ndarray) -> np.ndarray:
    """Chebyshev distance (raw px) of each protected pixel to the nearest background pixel; 0 on background."""
    from scipy import ndimage

    if not protected.any():
        return np.zeros(protected.shape, np.int64)
    if protected.all():
        return np.full(protected.shape, 10**6, np.int64)
    return ndimage.distance_transform_cdt(protected, metric="chessboard").astype(np.int64)


def leakage(clean_raw: np.ndarray, corrupted_raw: np.ndarray, protected_raw: np.ndarray,
            center_crop: bool = True, corrupted_view: Optional[np.ndarray] = None) -> dict:
    """Raw and model-input changes of protected vs background pixels, model-input changes binned by depth.
    corrupted_view: the network input actually used (exact-model-input variant); default policy_view(corrupted_raw).

    Returns {"raw_protected_changed": int, "model": {"protected": stats, "background": stats,
    "by_depth": {"a-b": stats}}, "max_changed_depth": int | None}; stats = {"n", "changed", "mean_abs", "max_abs"}
    over pixels (a pixel is changed if any channel differs; |.| = max over channels, uint8 levels).
    """
    from vlamech.models.oft import policy_view

    if not (clean_raw.shape == corrupted_raw.shape and protected_raw.shape == clean_raw.shape[:2]):
        raise ValueError("shape mismatch")
    raw_changed = int(np.any(clean_raw != corrupted_raw, axis=-1)[protected_raw].sum())
    a = np.asarray(policy_view(clean_raw, center_crop), np.int16)
    b = np.asarray(policy_view(corrupted_raw, center_crop) if corrupted_view is None else corrupted_view, np.int16)
    diff = np.abs(a - b).max(axis=-1)  # (224, 224)
    rows, cols = source_pixels(center_crop)
    prot = protected_raw[np.ix_(rows, cols)]
    depth = depth_map(protected_raw)[np.ix_(rows, cols)]

    def stats(sel):
        d = diff[sel]
        return {"n": int(sel.sum()), "changed": int((d > 0).sum()), "mean_abs": float(d.mean()) if d.size else 0.0,
                "max_abs": int(d.max()) if d.size else 0}

    by_depth = {f"{lo}-{hi}" if hi < 10**6 else f">{lo - 1}": stats(prot & (depth >= lo) & (depth <= hi))
                for lo, hi in DEPTH_BINS}
    changed_depths = depth[prot & (diff > 0)]
    return {"raw_protected_changed": raw_changed,
            "model": {"protected": stats(prot), "background": stats(~prot), "by_depth": by_depth},
            "max_changed_depth": int(changed_depths.max()) if changed_depths.size else None}


def model_input_protected(protected_raw: np.ndarray, center_crop: bool = True) -> np.ndarray:
    """Conservative protected mask in model-input space (224x224): protected_raw dilated by MAP_MARGIN, sampled
    at each model-input pixel's source pixel."""
    from scipy import ndimage

    grown = ndimage.binary_dilation(protected_raw, structure=np.ones((3, 3), bool), iterations=MAP_MARGIN)
    rows, cols = source_pixels(center_crop)
    return grown[np.ix_(rows, cols)]


def exact_model_input_views(clean_raw: np.ndarray, corrupted_raw: np.ndarray, protected_raw: np.ndarray,
                            center_crop: bool = True) -> np.ndarray:
    """224x224x3 uint8 network input: clean preprocessing on (conservatively) protected pixels, corrupted
    preprocessing elsewhere."""
    from vlamech.models.oft import policy_view

    a = np.asarray(policy_view(clean_raw, center_crop))
    b = np.asarray(policy_view(corrupted_raw, center_crop))
    keep = model_input_protected(protected_raw, center_crop)
    return np.where(keep[..., None], a, b).astype(np.uint8)
