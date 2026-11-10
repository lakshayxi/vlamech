"""Unit tests for observation corruptions: purity, determinism, pairing, task-irrelevance guarantees."""
import numpy as np
import pytest

from vlamech.perturbations.image import (
    LADDER_KEY,
    CorruptionSpec,
    EpisodeCorruptor,
    composite_behind,
    jpeg,
    ladder,
    make_bg_patch_layer,
    occlude_box,
    photometric,
    target_box,
    valid_box_positions,
)

SIDES = [20, 30, 40]  # stand-in for measured object bbox sides


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)


@pytest.fixture
def masks():
    fg = np.zeros((256, 256), bool)
    fg[100:160, 80:200] = True  # "robot/objects"
    target = np.zeros((256, 256), bool)
    target[120:140, 100:130] = True
    return ~fg, target


def _all_specs():
    specs = [CorruptionSpec("target_occlusion"), CorruptionSpec("matched_bg_box")]
    for fam in LADDER_KEY:
        specs += [CorruptionSpec(fam, v) for v in ladder(fam)]
    return specs


@pytest.mark.parametrize("spec", _all_specs(), ids=lambda s: s.tag)
def test_pure_and_deterministic(img, masks, spec):
    bg, target = masks
    before = img.copy()
    kw = dict(background_mask=bg, target_mask=target)
    out1 = EpisodeCorruptor(spec, "task3-init27", SIDES)("agentview", img, step=5, **kw)
    out2 = EpisodeCorruptor(spec, "task3-init27", SIDES)("agentview", img, step=5, **kw)
    assert np.array_equal(img, before), "input image was mutated"
    assert out1.dtype == np.uint8 and out1.shape == img.shape
    assert np.array_equal(out1, out2), "same episode key must give identical output"
    assert out1 is not img


def test_identity_is_exact(img):
    out = EpisodeCorruptor(CorruptionSpec("identity"), "k")("agentview", img, 0)
    assert np.array_equal(out, img)


def test_uncorrupted_camera_untouched(img):
    spec = CorruptionSpec("noise", 64, cameras=("agentview",))
    assert np.array_equal(EpisodeCorruptor(spec, "k")("wrist", img, 0), img)


def test_noise_resampled_per_step_and_paired_across_conditions(img):
    c = EpisodeCorruptor(CorruptionSpec("noise", 16), "task0-init5")
    assert not np.array_equal(c("agentview", img, 0), c("agentview", img, 1))
    # different episodes get different draws; same episode key gets identical draws (pairing)
    d = EpisodeCorruptor(CorruptionSpec("noise", 16), "task0-init6")
    assert not np.array_equal(c("agentview", img, 0), d("agentview", img, 0))
    c2 = EpisodeCorruptor(CorruptionSpec("noise", 16), "task0-init5")
    assert np.array_equal(c("agentview", img, 3), c2("agentview", img, 3))


def test_photometric_is_exact_gain(img):
    out = photometric(img, 1.0)
    expect = np.clip(np.round(img.astype(np.float64) * 0.5), 0, 255).astype(np.uint8)
    assert np.array_equal(out, expect)
    assert np.array_equal(photometric(img, 0.0), img)


def test_jpeg_lower_quality_more_distortion(img):
    errs = [np.abs(jpeg(img, q).astype(int) - img).mean() for q in (96, 24, 3)]
    assert errs[0] < errs[1] < errs[2]


def test_bg_patches_never_touch_foreground(img, masks):
    bg, _ = masks
    for cov in ladder("bg_patches"):
        out = EpisodeCorruptor(CorruptionSpec("bg_patches", cov), "k", SIDES)("agentview", img, 0, background_mask=bg)
        assert np.array_equal(out[~bg], img[~bg]), "foreground pixels changed"
        layer, cover = make_bg_patch_layer(img.shape, bg, cov, SIDES, np.random.default_rng(1))
        assert (cover & bg).sum() / bg.sum() >= cov


def test_bg_patches_follow_moving_foreground(img, masks):
    """Layout is fixed per episode, but compositing uses the *current* mask each frame."""
    bg, _ = masks
    c = EpisodeCorruptor(CorruptionSpec("bg_patches", 0.8), "k", SIDES)
    c("agentview", img, 0, background_mask=bg)
    bg2 = np.ones_like(bg)
    bg2[0:60, 0:60] = False  # foreground moved
    out = c("agentview", img, 1, background_mask=bg2)
    assert np.array_equal(out[~bg2], img[~bg2])


def test_target_occlusion_covers_exactly_bbox(img, masks):
    _, target = masks
    out = occlude_box(img, target, margin=0.0, gray=128)
    assert (out[120:140, 100:130] == 128).all()
    outside = np.ones(target.shape, bool)
    outside[120:140, 100:130] = False
    assert np.array_equal(out[outside], img[outside])
    assert np.array_equal(occlude_box(img, np.zeros_like(target), 0.0, 128), img)


def test_spec_validation():
    with pytest.raises(ValueError):
        CorruptionSpec("nope")
    with pytest.raises(ValueError):
        CorruptionSpec("noise")  # missing value
    with pytest.raises(ValueError):
        CorruptionSpec("jpeg", 0)
    with pytest.raises(ValueError):
        CorruptionSpec("bg_patches", 1.5)
    with pytest.raises(ValueError):
        CorruptionSpec("identity", 3.0)
    with pytest.raises(ValueError):
        CorruptionSpec("noise", 8, cameras=("left",))


def test_bg_patches_requires_sides():
    with pytest.raises(ValueError):
        EpisodeCorruptor(CorruptionSpec("bg_patches", 0.1), "k")


def test_ladders_are_strictly_monotone():
    for fam in LADDER_KEY:
        v = ladder(fam)
        d = np.diff(v)
        assert (d > 0).all() or (d < 0).all(), f"{fam} ladder not strictly monotone: {v}"


def test_composite_shape_check(img):
    with pytest.raises(ValueError):
        composite_behind(img, img, np.ones((10, 10), bool), np.ones((10, 10), bool))


def test_common_random_numbers_noise_scales_one_draw():
    """Noise at 2*sigma is the same field scaled by 2 (same z), not a fresh draw."""
    flat = np.full((64, 64, 3), 128, np.uint8)  # mid-gray: no clipping at these sigmas
    r8 = EpisodeCorruptor(CorruptionSpec("noise", 8), "task1-init3")("agentview", flat, 7).astype(int) - 128
    r16 = EpisodeCorruptor(CorruptionSpec("noise", 16), "task1-init3")("agentview", flat, 7).astype(int) - 128
    assert np.abs(r16 - 2 * r8).max() <= 2  # rounding only
    assert np.corrcoef(r8.ravel(), r16.ravel())[0, 1] > 0.99


def test_common_random_numbers_bg_layouts_nested():
    """The patch layout at a higher coverage contains the layout at a lower coverage."""
    bg = np.ones((256, 256), bool)
    bg[100:160, 80:200] = False
    lo, cov_lo = make_bg_patch_layer((256, 256), bg, 0.1, SIDES, np.random.default_rng(11))
    hi, cov_hi = make_bg_patch_layer((256, 256), bg, 0.4, SIDES, np.random.default_rng(11))
    assert (cov_lo <= cov_hi).all() and cov_hi.sum() > cov_lo.sum()


def test_target_box_half_open_and_margin():
    m = np.zeros((50, 50), bool)
    m[10:20, 5:9] = True
    assert target_box(m) == (10, 20, 5, 9)
    assert target_box(m, 0.5) == (5, 25, 3, 11)
    assert target_box(np.zeros_like(m)) is None


def test_valid_box_positions_exact():
    bg = np.ones((10, 10), bool)
    bg[4, 4] = False
    pos = valid_box_positions(bg, 3, 3)
    brute = [(y, x) for y in range(8) for x in range(8) if bg[y:y + 3, x:x + 3].all()]
    assert sorted(map(tuple, pos.tolist())) == brute


def test_matched_bg_box_same_size_on_background_only(img, masks):
    bg, target = masks
    c = EpisodeCorruptor(CorruptionSpec("matched_bg_box"), "task3-init27")
    out = c("agentview", img, 0, background_mask=bg, target_mask=target)
    changed = np.any(out != img, axis=2)
    ys, xs = np.nonzero(changed)
    y0, y1, x0, x1 = target_box(target)
    # one gray box of exactly the target box size (random pixels equal to 128 by chance can only shrink 'changed')
    assert ys.max() - ys.min() + 1 <= y1 - y0 and xs.max() - xs.min() + 1 <= x1 - x0
    assert changed.sum() >= 0.95 * (y1 - y0) * (x1 - x0)
    assert bg[changed].all(), "matched box touched a protected pixel"
    assert (out[changed] == 128).all()


def test_matched_bg_box_is_stable_and_flags_impossible(img, masks):
    bg, target = masks
    c = EpisodeCorruptor(CorruptionSpec("matched_bg_box"), "task3-init27")
    a = c("agentview", img, 0, background_mask=bg, target_mask=target)
    b = c("agentview", img, 8, background_mask=bg, target_mask=target)
    assert np.array_equal(a, b), "box should stay put while its position stays valid"
    none_bg = np.zeros_like(bg)
    out = c("wrist", img, 0, background_mask=none_bg, target_mask=target)
    assert np.array_equal(out, img) and c.flags and c.flags[-1]["event"] == "no_valid_matched_box"
    # target not visible: nothing occluded (mirrors target_occlusion)
    assert np.array_equal(c("agentview", img, 16, background_mask=bg, target_mask=np.zeros_like(target)), img)


def test_spec_rejects_nonfinite_and_fractional_jpeg():
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            CorruptionSpec("noise", bad)
    with pytest.raises(ValueError):
        CorruptionSpec("jpeg", 12.5)


# ----------------------------------------------------------------------------- raw vs model-input protection
def _square_scene(lo=40, hi=216):
    img = np.full((256, 256, 3), 100, np.uint8)
    img[lo:hi, lo:hi] = 200
    prot = np.zeros((256, 256), bool)
    prot[lo:hi, lo:hi] = True
    noisy = img.copy()
    noisy[~prot] = np.random.default_rng(0).integers(0, 256, ((~prot).sum(), 3))
    return img, noisy, prot


def test_source_pixel_map_error_within_margin():
    from vlamech.models.oft import policy_view
    from vlamech.perturbations.leakage import MAP_MARGIN, source_pixels

    ramp = np.tile(np.arange(256, dtype=np.uint8)[None, :, None], (256, 1, 3))
    for cc in (False, True):
        out = np.asarray(policy_view(ramp, cc), float)[112, 4:-4, 0]  # away from the image border
        assert np.abs(out - source_pixels(cc)[1][4:-4]).max() <= MAP_MARGIN


def test_background_only_is_exact_on_raw_but_leaks_after_preprocessing():
    """Raw and model-input changes are measured separately; the raw guarantee does not carry over."""
    from vlamech.perturbations.leakage import leakage

    img, noisy, prot = _square_scene()
    r = leakage(img, noisy, prot)
    assert r["raw_protected_changed"] == 0
    assert r["model"]["protected"]["changed"] > 0  # documented leakage (results/preprocessing_leakage)
    assert r["model"]["by_depth"][">32"]["changed"] == 0  # bounded: JPEG 16x16 MCU + resize support


def test_exact_model_input_variant_keeps_protected_network_pixels_identical():
    from vlamech.models.oft import policy_view
    from vlamech.perturbations.leakage import exact_model_input_views, model_input_protected, source_pixels

    img, noisy, prot = _square_scene()
    view = exact_model_input_views(img, noisy, prot)
    clean = np.asarray(policy_view(img))
    rows, cols = source_pixels()
    nearest = prot[np.ix_(rows, cols)]
    keep = model_input_protected(prot)
    assert (nearest <= keep).all()  # conservative: superset of the nearest-pixel protected set
    assert np.array_equal(view[keep], clean[keep])
    assert not np.array_equal(view[~keep], clean[~keep])  # the background is still corrupted


def test_exact_model_input_only_for_background_families():
    from vlamech.perturbations.image import CorruptionSpec

    s = CorruptionSpec("bg_patches", 0.2, exact_model_input=True)
    assert s.tag == "bg_patches-0.2-exactmi-agentview+wrist" and s.to_dict()["exact_model_input"] is True
    assert "exact_model_input" not in CorruptionSpec("bg_patches", 0.2).to_dict()  # unchanged for existing runs
    with pytest.raises(ValueError):
        CorruptionSpec("noise", 8.0, exact_model_input=True)
