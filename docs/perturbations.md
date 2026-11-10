# Visual conditions

Every condition is applied to the camera frame (256x256, after the 180-degree flip of the OFT pipeline) and before
the policy's own preprocessing, which runs unchanged. Parameter ladders and rules are in `configs/protocol.yaml`
(`corruption.*`); the calibrated levels are frozen in `configs/frozen_levels.yaml`. Code: `src/vlamech/perturbations/image.py`.

## Families

| Family | Parameter (unit) | Ladder | Frozen levels | Task relevance |
|---|---|---|---|---|
| `photometric` | exposure reduction, stops (gain 2^-stops) | 0.5 to 4.0 in 0.5-stop steps | 3.5, 4.5 | all pixels |
| `noise` | Gaussian pixel noise, std on 0-255, redrawn every step | 2 to 128, doubling | 64, 128 | all pixels |
| `blur` | Gaussian defocus, sigma in px | 0.5 to 16, doubling | 1, 4, 8 | all pixels |
| `jpeg` | JPEG quality | 96 to 1, halving | 6, 1 | all pixels |
| `bg_patches` | fraction of background covered by object-sized patches | 0.05 to 0.8, doubling | 0.8 | background only |

Levels were chosen on the calibration split (init states 0-4) by a fixed rule: an isotonic fit of success against
severity, and for each target relative success (0.9, 0.6, 0.3) the closest level. Targets that a family cannot
reach within its ladder and the rule's one-step extension are recorded as unreachable: 0.3 for `photometric` and
`jpeg`, 0.3 and 0.6 for `bg_patches`.

## Controls

| Condition | What it does | Purpose |
|---|---|---|
| `identity` | nothing | the paired clean reference; must reproduce clean behaviour exactly |
| `target_occlusion` | mid-gray box over the tight bounding box of the instructed task's entities | task-relevant control: the policy should fail |
| `matched_bg_box` | mid-gray box of the same size as the current target box, entirely on background | separates *what* is hidden from *how much* |

The matched box is placed uniformly over valid positions at the first query and then at the valid position closest
to the previous one, so it moves as smoothly as the target box it mirrors.

## Background only: raw frame and model input

The simulator's instance segmentation gives the robot and every entity that any LIBERO-Goal task refers to; these
pixels are *protected*, dilated by 1 px because anti-aliased object boundaries decode to wrong instance ids.
Background-only conditions change no protected pixel of the camera frame (tested).

OFT's preprocessing (JPEG round-trip with 16x16 blocks, Lanczos resize, center crop) then mixes neighbouring
pixels: on the calibration frames, background patches change 3-46% of the protected model-input pixels, by 0.2-3
intensity levels on average, and no change reaches more than 19 px into an object
(`results/preprocessing_leakage/summary.md`). The `-exactmi` variant (`--exact-model-input`) composites after
preprocessing with a conservative mask, so every protected pixel the network sees is bit-identical to clean
(tested). Both variants are run; claims that "only the background changed" state which guarantee they use.

On the wrist camera, `bg_patches` are fixed in image coordinates while the camera moves, so they are a screen-space
corruption of background pixels rather than a change of the scene.

## Common random numbers

Random draws are keyed by (family, episode, camera[, step]) and never by the severity, so every level of a family
sees the same underlying draw: noise at sigma is sigma * z with the same z, and the patch layout at a higher
coverage contains the layout at a lower one. Severity effects are therefore not confounded with which draw an
episode received. Seeds derive from `corruption.seed_namespace` (`vlamech.utils.seeding.derive_seed`).
