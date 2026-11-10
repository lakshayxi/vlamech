"""Observation-level image corruptions.

All functions are pure (never modify their input; tested) and deterministic given their seed keys.
Images are uint8 HxWx3 *camera frames* after the 180-degree flip of the OFT pipeline and BEFORE the
policy's own preprocessing (JPEG round-trip, resize, crop), which is applied afterwards unchanged.
Corrupting the raw frame models a camera-side change.

A corruption is specified by its family and its *physical parameter value* (not an opaque index), so every
result records exactly what was applied:
  identity          value ignored                      control: must reproduce clean behavior exactly
  photometric       exposure reduction in stops        gain = 2**-stops (uniform over channels)
  noise             Gaussian pixel std (0-255)          resampled every control step (sensor noise)
  blur              Gaussian sigma in pixels            defocus
  jpeg              JPEG quality (1-100)                compression artifacts
  bg_patches        fraction of background covered     composited BEHIND the foreground via segmentation,
                                                        so never covers robot/objects (task-irrelevant by construction)
  target_occlusion  value ignored                       tight box over the instructed target (task-relevant control)
  matched_bg_box    value ignored                       gray box of the SAME size as the current target box, placed
                                                        entirely on background (area-matched irrelevant control)

Common random numbers: draws are keyed by (family, episode, camera[, step]), never by the value, so every
severity of a family sees the same underlying draw (noise = sigma * z with the same z; patch layouts are nested
across coverages because the draw sequence is identical and only the stopping point differs).

No numeric parameters live in this file. Ladders, occluder gray, margins, and patch-size distribution come
from configs/protocol.yaml; calibrated operating points from configs/frozen_levels.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

from vlamech.config import protocol
from vlamech.utils.seeding import derive_seed

FAMILIES = ("identity", "photometric", "noise", "blur", "jpeg", "bg_patches", "target_occlusion", "matched_bg_box")
LADDER_KEY = {
    "photometric": "corruption.ladders.photometric_gain_stops",
    "noise": "corruption.ladders.noise_sigma",
    "blur": "corruption.ladders.blur_sigma_px",
    "jpeg": "corruption.ladders.jpeg_quality",
    "bg_patches": "corruption.ladders.bg_patch_coverage",
}
NEEDS_BACKGROUND = {"bg_patches", "matched_bg_box"}
NEEDS_TARGET = {"target_occlusion", "matched_bg_box"}


def ladder(family: str) -> list:
    return list(protocol(LADDER_KEY[family]))


@dataclass(frozen=True)
class CorruptionSpec:
    family: str = "identity"
    value: Optional[float] = None  # physical parameter; None for identity / target_occlusion
    cameras: tuple[str, ...] = ("agentview", "wrist")
    # exact-model-input variant: composite AFTER the OFT preprocessing, so protected model-input pixels equal the
    # clean ones exactly (vlamech.perturbations.leakage.exact_model_input_views); background-only families only
    exact_model_input: bool = False

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"unknown corruption family {self.family!r}")
        if self.exact_model_input and self.family not in NEEDS_BACKGROUND:
            raise ValueError(f"exact_model_input applies only to background-only families {sorted(NEEDS_BACKGROUND)}")
        bad = set(self.cameras) - {"agentview", "wrist"}
        if bad:
            raise ValueError(f"unknown cameras {bad}")
        if self.family in LADDER_KEY:
            if self.value is None:
                raise ValueError(f"{self.family} needs a parameter value")
            if not np.isfinite(self.value):
                raise ValueError(f"{self.family} value must be finite")
            if self.family == "jpeg" and not (1 <= self.value <= 100 and float(self.value).is_integer()):
                raise ValueError("jpeg quality must be an integer in [1, 100]")
            if self.family == "bg_patches" and not 0 < self.value <= 1:
                raise ValueError("bg_patches coverage must be in (0, 1]")
            if self.family in ("noise", "blur", "photometric") and self.value < 0:
                raise ValueError(f"{self.family} value must be >= 0")
        elif self.value is not None:
            raise ValueError(f"{self.family} takes no value")

    @property
    def is_clean(self) -> bool:
        return self.family == "identity" or not self.cameras

    @property
    def tag(self) -> str:
        cams = "+".join(sorted(self.cameras)) if self.cameras else "none"
        v = "" if self.value is None else f"-{self.value:g}"
        mi = "-exactmi" if self.exact_model_input else ""
        return f"{self.family}{v}{mi}-{cams}"

    def to_dict(self) -> dict:
        d = {"family": self.family, "value": self.value, "cameras": list(self.cameras), "tag": self.tag}
        return d | ({"exact_model_input": True} if self.exact_model_input else {})


# ----------------------------------------------------------------------------- primitive operations
def _check(img: np.ndarray) -> None:
    if not (isinstance(img, np.ndarray) and img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3):
        raise TypeError(f"expected uint8 HxWx3 image, got {getattr(img, 'dtype', None)} {getattr(img, 'shape', None)}")


def photometric(img: np.ndarray, stops: float) -> np.ndarray:
    """Exposure reduction by `stops` photographic stops: every channel multiplied by 2**-stops."""
    _check(img)
    if stops == 0:
        return img.copy()
    out = img.astype(np.float64) * (2.0 ** -stops)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    _check(img)
    if sigma == 0:
        return img.copy()
    out = img.astype(np.float64) + rng.normal(0.0, sigma, size=img.shape)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    _check(img)
    if sigma == 0:
        return img.copy()
    k = int(2 * np.ceil(3 * sigma) + 1)  # +/-3 sigma support: truncation error < 0.3% of kernel mass
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)


def jpeg(img: np.ndarray, quality: float) -> np.ndarray:
    _check(img)
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(img[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return np.ascontiguousarray(cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1])


def make_bg_patch_layer(shape, background_mask: np.ndarray, coverage: float, side_lengths: Sequence[int],
                        rng: np.random.Generator):
    """Sample uniformly-colored rectangles until `coverage` of the background pixels lie under a rectangle.

    Rectangle side lengths are drawn (with replacement) from `side_lengths`: the empirical bounding-box sides of
    the task objects in clean calibration frames, so patches are object-sized. Positions are uniform over the
    image; colors uniform over RGB. Returns (layer, cover).
    """
    h, w = shape[:2]
    sides = np.asarray(side_lengths, dtype=int)
    if sides.size == 0 or sides.min() < 1:
        raise ValueError("side_lengths must be a non-empty list of positive ints")
    layer = np.zeros((h, w, 3), np.uint8)
    cover = np.zeros((h, w), bool)
    bg_total = int(background_mask.sum())
    if bg_total == 0:
        raise ValueError("empty background mask")
    # Failsafe only (raises; never changes a stimulus): 50x the number of median-sized rectangles that would
    # tile the requested area. Reaching it means the mask is too fragmented for the requested coverage.
    max_draws = 50 * int(np.ceil(coverage * bg_total / max(1, int(np.median(sides)) ** 2)) + 1)
    draws = 0
    while (cover & background_mask).sum() / bg_total < coverage:
        if draws >= max_draws:
            raise RuntimeError(f"could not reach coverage {coverage} after {draws} rectangles")
        draws += 1
        ph, pw = (int(min(s, h)) for s in rng.choice(sides, size=2))
        y, x = int(rng.integers(0, h - ph + 1)), int(rng.integers(0, w - pw + 1))
        layer[y : y + ph, x : x + pw] = rng.integers(0, 256, size=3, dtype=np.uint8)
        cover[y : y + ph, x : x + pw] = True
    return layer, cover


def composite_behind(img: np.ndarray, layer: np.ndarray, cover: np.ndarray, background_mask: np.ndarray) -> np.ndarray:
    _check(img)
    if background_mask.shape != img.shape[:2] or cover.shape != img.shape[:2]:
        raise ValueError("mask shape mismatch")
    out = img.copy()
    sel = cover & background_mask
    out[sel] = layer[sel]
    return out


def occlude_box(img: np.ndarray, target_mask: np.ndarray, margin: float, gray: int) -> np.ndarray:
    """Fill the bounding box of target_mask (expanded by `margin` of its size) with `gray`."""
    _check(img)
    if target_mask.shape != img.shape[:2]:
        raise ValueError("mask shape mismatch")
    out = img.copy()
    box = target_box(target_mask, margin)
    if box is None:  # target not visible in this camera: nothing to occlude
        return out
    y0, y1, x0, x1 = box
    out[y0:y1, x0:x1] = gray
    return out


def target_box(target_mask: np.ndarray, margin: float = 0.0) -> Optional[tuple[int, int, int, int]]:
    """(y0, y1, x0, x1) half-open bounding box of target_mask expanded by `margin` of its size; None if empty."""
    ys, xs = np.nonzero(target_mask)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    dy, dx = int(round((y1 - y0) * margin)), int(round((x1 - x0) * margin))
    h, w = target_mask.shape
    return max(0, y0 - dy), min(h, y1 + dy), max(0, x0 - dx), min(w, x1 + dx)


def valid_box_positions(background_mask: np.ndarray, bh: int, bw: int) -> np.ndarray:
    """Top-left corners (N,2) where a bh x bw box lies entirely on background pixels (integral image)."""
    h, w = background_mask.shape
    if bh > h or bw > w:
        return np.zeros((0, 2), int)
    fg = (~background_mask).astype(np.int64)
    ii = np.zeros((h + 1, w + 1), np.int64)
    ii[1:, 1:] = fg.cumsum(0).cumsum(1)
    s = ii[bh:, bw:] - ii[:-bh, bw:] - ii[bh:, :-bw] + ii[:-bh, :-bw]  # (h-bh+1, w-bw+1) foreground counts
    return np.argwhere(s == 0)


# ----------------------------------------------------------------------------- episode-level application
@dataclass
class EpisodeCorruptor:
    """Applies a CorruptionSpec consistently over one episode.

    Randomness: episode-level draws (patch layout, matched-box placement) use
    derive_seed(namespace, family, episode_key, camera, "layout"); step-level draws (noise) use the control-step
    index instead of "layout". The VALUE is deliberately not part of the key (common random numbers across
    severities). `episode_key` must identify the paired episode (e.g. "task3-init27"), so every condition sees
    the same draws for the same episode.
    `flags` records per-query events that analyses must report (e.g. no valid matched-box position).
    `bg_side_lengths` is required for bg_patches (from the calibration frames, see protocol.yaml).
    """

    spec: CorruptionSpec
    episode_key: str
    bg_side_lengths: Optional[Sequence[int]] = None
    _layers: dict = field(default_factory=dict)
    _box_pos: dict = field(default_factory=dict)
    flags: list = field(default_factory=list)

    def __post_init__(self):
        self._ns = protocol("corruption.seed_namespace")
        self._gray = int(protocol("corruption.occluder_gray"))
        self._margin = float(protocol("corruption.target_occlusion_margin"))
        if self.spec.family == "bg_patches" and not self.bg_side_lengths:
            raise ValueError("bg_patches needs bg_side_lengths")

    def _rng(self, *keys) -> np.random.Generator:
        return np.random.default_rng(derive_seed(self._ns, self.spec.family, self.episode_key, *keys))

    def __call__(self, camera: str, img: np.ndarray, step: int,
                 background_mask: Optional[np.ndarray] = None, target_mask: Optional[np.ndarray] = None) -> np.ndarray:
        _check(img)
        if camera not in ("agentview", "wrist"):
            raise ValueError(camera)
        if self.spec.is_clean or camera not in self.spec.cameras:
            return img.copy()
        fam, val = self.spec.family, self.spec.value
        if fam == "photometric":
            return photometric(img, val)
        if fam == "noise":
            return noise(img, val, self._rng(camera, step))
        if fam == "blur":
            return blur(img, val)
        if fam == "jpeg":
            return jpeg(img, val)
        if fam == "bg_patches":
            if background_mask is None:
                raise ValueError("bg_patches needs a background mask")
            if camera not in self._layers:
                self._layers[camera] = make_bg_patch_layer(img.shape, background_mask, val, self.bg_side_lengths,
                                                           self._rng(camera, "layout"))
            layer, cover = self._layers[camera]
            return composite_behind(img, layer, cover, background_mask)
        if fam == "target_occlusion":
            if target_mask is None:
                raise ValueError("target_occlusion needs a target mask")
            return occlude_box(img, target_mask, self._margin, self._gray)
        if fam == "matched_bg_box":
            if target_mask is None or background_mask is None:
                raise ValueError("matched_bg_box needs target and background masks")
            return self._matched_box(camera, img, step, target_mask, background_mask)
        raise AssertionError(fam)

    def _matched_box(self, camera, img, step, target_mask, background_mask):
        out = img.copy()
        box = target_box(target_mask, self._margin)
        if box is None:  # mirrors target_occlusion: nothing occluded when the target is not visible
            return out
        bh, bw = box[1] - box[0], box[3] - box[2]
        valid = valid_box_positions(background_mask, bh, bw)
        if len(valid) == 0:
            self.flags.append({"camera": camera, "step": step, "event": "no_valid_matched_box", "size": [bh, bw]})
            return out
        prev = self._box_pos.get(camera)
        if prev is None:
            y, x = valid[self._rng(camera, "layout").integers(len(valid))]
        else:  # closest valid position to the previous one (ties -> first in row-major order)
            y, x = valid[np.argmin(((valid - prev) ** 2).sum(1))]
        self._box_pos[camera] = np.array([y, x])
        out[y : y + bh, x : x + bw] = self._gray
        return out
