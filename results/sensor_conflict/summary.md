# When sensors disagree (exploratory, reduced run)

Open loop: 59 held-out episodes (every third successful pick-and-place episode, init states 20-49). Closed loop: pick-and-place tasks, init states 20-29, 60 paired episodes per fault. Patching was not run. Shares are the part of the policy's response to the edited world that follows each source (Shapley values, |Delta|^2-weighted; 95% stratified bootstrap CIs). Protocol: `configs/protocol.yaml` sensor_conflict, design: `docs/sensor_conflict.md`.

## object

| phase | n | agentview | wrist | median |Delta| | render noise / |Delta| |
|---|---|---|---|---|---|
| approach | 59 | 0.13 [0.10, 0.16] | 0.87 [0.84, 0.90] | 0.61 | 0.06 |
| pregrasp | 59 | 0.16 [0.11, 0.22] | 0.84 [0.78, 0.89] | 0.58 | 0.06 |
| transport | 59 | 0.29 [0.21, 0.36] | 0.71 [0.64, 0.79] | 0.17 | 0.13 |
| place | 59 | 0.29 [0.24, 0.34] | 0.71 [0.66, 0.76] | 0.66 | 0.07 |

## arm

| phase | n | agentview | wrist | proprio | median |Delta| | render noise / |Delta| |
|---|---|---|---|---|---|---|
| approach | 59 | 0.19 [0.16, 0.22] | 0.80 [0.77, 0.83] | 0.01 [0.00, 0.01] | 0.68 | 0.06 |
| pregrasp | 59 | 0.23 [0.19, 0.27] | 0.76 [0.71, 0.80] | 0.01 [-0.00, 0.02] | 0.74 | 0.05 |
| transport | 59 | 0.30 [0.24, 0.35] | 0.61 [0.56, 0.67] | 0.09 [0.04, 0.13] | 0.46 | 0.07 |
| place | 59 | 0.38 [0.31, 0.45] | 0.52 [0.45, 0.58] | 0.11 [0.08, 0.13] | 0.81 | 0.05 |

## gripper

| phase | n | agentview | wrist | proprio | median |Delta| | render noise / |Delta| |
|---|---|---|---|---|---|---|
| approach | 57 | 0.14 [0.12, 0.15] | 0.06 [0.05, 0.07] | 0.80 [0.78, 0.82] | 3.48 | 0.01 |
| pregrasp | 49 | 0.16 [0.14, 0.17] | 0.10 [0.09, 0.12] | 0.74 [0.72, 0.76] | 4.05 | 0.01 |
| transport | 48 | 0.06 [0.05, 0.07] | 0.08 [0.06, 0.09] | 0.86 [0.84, 0.88] | 4.04 | 0.01 |
| place | 48 | 0.08 [0.06, 0.11] | 0.14 [0.12, 0.17] | 0.77 [0.74, 0.80] | 1.51 | 0.03 |

## stale

| phase | n | agentview | wrist | proprio | median |Delta| | render noise / |Delta| |
|---|---|---|---|---|---|---|
| approach | 59 | 0.35 [0.33, 0.37] | 0.63 [0.62, 0.65] | 0.02 [0.01, 0.02] | 2.39 | 0.02 |
| pregrasp | 59 | 0.28 [0.27, 0.30] | 0.65 [0.63, 0.67] | 0.06 [0.05, 0.08] | 1.44 | 0.03 |
| transport | 59 | 0.26 [0.24, 0.28] | 0.29 [0.26, 0.31] | 0.46 [0.42, 0.50] | 2.16 | 0.02 |
| place | 59 | 0.32 [0.28, 0.36] | 0.41 [0.38, 0.44] | 0.27 [0.25, 0.30] | 2.51 | 0.02 |

## arm-blur-8

| phase | n | agentview | wrist | proprio | median |Delta| | render noise / |Delta| |
|---|---|---|---|---|---|---|
| approach | 59 | 0.12 [0.09, 0.15] | 0.87 [0.84, 0.90] | 0.01 [0.00, 0.02] | 0.51 | 0.07 |
| pregrasp | 59 | 0.09 [0.05, 0.13] | 0.88 [0.83, 0.93] | 0.03 [0.01, 0.04] | 0.48 | 0.07 |
| transport | 59 | 0.05 [-0.00, 0.10] | 0.45 [0.37, 0.54] | 0.50 [0.39, 0.60] | 0.30 | 0.08 |
| place | 59 | 0.13 [0.10, 0.16] | 0.44 [0.38, 0.51] | 0.44 [0.35, 0.51] | 0.51 | 0.08 |

## arm-noise-128

| phase | n | agentview | wrist | proprio | median |Delta| | render noise / |Delta| |
|---|---|---|---|---|---|---|
| approach | 59 | 0.05 [0.03, 0.07] | 0.94 [0.92, 0.96] | 0.01 [0.00, 0.01] | 0.54 | 0.07 |
| pregrasp | 59 | 0.06 [0.03, 0.09] | 0.92 [0.89, 0.95] | 0.02 [0.01, 0.04] | 0.49 | 0.08 |
| transport | 59 | 0.07 [0.04, 0.09] | 0.43 [0.34, 0.56] | 0.50 [0.38, 0.59] | 0.36 | 0.08 |
| place | 59 | 0.08 [0.06, 0.10] | 0.53 [0.45, 0.63] | 0.39 [0.29, 0.47] | 0.58 | 0.08 |

## Pre-specified tests

**A_wrist_minus_agentview_object**

| | n | estimate [95% CI] | p (bootstrap) | p (Holm) |
|---|---|---|---|---|
| approach | 59 | 0.741 [0.684, 0.792] | 0.0001 | 0.0004 |
| pregrasp | 59 | 0.672 [0.559, 0.776] | 0.0001 | 0.0004 |
| transport | 59 | 0.419 [0.269, 0.582] | 0.0001 | 0.0004 |
| place | 59 | 0.418 [0.320, 0.516] | 0.0001 | 0.0004 |

**B_proprio_share_arm**

| | n | estimate [95% CI] | p (bootstrap) | p (Holm) |
|---|---|---|---|---|
| approach | 59 | 0.007 [0.004, 0.011] | 0.0001 | 0.0004 |
| pregrasp | 59 | 0.009 [-0.001, 0.019] | 0.0765 | 0.0765 |
| transport | 59 | 0.086 [0.041, 0.132] | 0.0003 | 0.0006 |
| place | 59 | 0.106 [0.082, 0.133] | 0.0001 | 0.0004 |

**C_proprio_share_gripper**

| | n | estimate [95% CI] | p (bootstrap) | p (Holm) |
|---|---|---|---|---|
| approach | 57 | 0.803 [0.782, 0.824] | 0.0001 | 0.0004 |
| pregrasp | 49 | 0.741 [0.722, 0.759] | 0.0001 | 0.0004 |
| transport | 48 | 0.863 [0.843, 0.883] | 0.0001 | 0.0004 |
| place | 48 | 0.774 [0.737, 0.807] | 0.0001 | 0.0004 |

**D_relevance_log_ratio**

| | n | estimate [95% CI] | p (bootstrap) | p (Holm) |
|---|---|---|---|---|
| approach | 59 | 1.071 [0.855, 1.283] | 0.0001 | 0.0004 |
| pregrasp | 59 | 1.379 [1.131, 1.631] | 0.0001 | 0.0004 |
| transport | 59 | 0.433 [0.125, 0.721] | 0.0081 | 0.0081 |
| place | 59 | 1.664 [1.424, 1.909] | 0.0001 | 0.0004 |

**E_proprio_share_corrupted_minus_clean**

| | n | estimate [95% CI] | p (bootstrap) | p (Holm) |
|---|---|---|---|---|
| arm-blur-8 | 236 | 0.170 [0.132, 0.209] | 0.0001 | 0.0002 |
| arm-noise-128 | 236 | 0.171 [0.126, 0.217] | 0.0001 | 0.0002 |

**F_probe_auroc_layer16**

| | n | estimate [95% CI] | p (bootstrap) | p (Holm) |
|---|---|---|---|---|
| layer 16 | 4448 | 0.539 [0.532, 0.546] | 0.0001 | 0.0001 |

## Detecting the conflict from the policy's own state (probe)

| layer | AUROC [95% CI] | p | object | arm | gripper | unseen edit: object / arm / gripper |
|---|---|---|---|---|---|---|
| 8 | 0.52 [0.52, 0.53] | 0.0005 | 0.50 | 0.50 | 0.55 | 0.50 / 0.50 / 0.50 |
| 16 | 0.54 [0.53, 0.55] | 0.0001 | 0.52 | 0.51 | 0.58 | 0.51 / 0.50 / 0.49 |
| 24 | 0.54 [0.53, 0.54] | 0.0005 | 0.51 | 0.50 | 0.58 | 0.50 / 0.50 / 0.49 |

## Gripper decisions under the gripper edit

| phase | n | edited world flips | proprio alone flips | cameras alone flip |
|---|---|---|---|---|
| approach | 57 | 0.93 | 0.49 | 0.00 |
| pregrasp | 49 | 0.84 | 0.41 | 0.12 |
| transport | 48 | 0.23 | 0.15 | 0.00 |
| place | 48 | 0.12 | 0.00 | 0.00 |

## Closed loop: faults

Paired plain run: success 0.967, unsafe 0.017 (n = 60).

| fault | n | success | difference [95% CI] | p (Holm) | unsafe | difference [95% CI] | p (Holm) | hand speed mm/step (plain) | reversals (plain) | grasp cycles (plain) |
|---|---|---|---|---|---|---|---|---|---|---|
| delay_2 | 60 | 0.917 | -0.050 [-0.133, 0.033] | 1 | 0.083 | 0.067 [0.000, 0.133] | 1 | 7.5 (8.2) | 0.09 (0.11) | 1.08 (0.85) |
| delay_4 | 60 | 0.367 | -0.600 [-0.717, -0.483] | 1.79e-08 | 0.300 | 0.283 [0.200, 0.350] | 0.000916 | 5.3 (8.2) | 0.11 (0.11) | 1.80 (0.85) |
| delay_8 | 60 | 0.117 | -0.850 [-0.933, -0.767] | 1.51e-14 | 0.417 | 0.400 [0.333, 0.467] | 2.26e-06 | 5.0 (8.2) | 0.08 (0.11) | 1.57 (0.85) |
| delay_16 | 60 | 0.017 | -0.950 [-1.000, -0.900] | 2.64e-16 | 0.367 | 0.350 [0.283, 0.417] | 1.43e-05 | 3.7 (8.2) | 0.03 (0.11) | 1.03 (0.85) |
| lag_agentview | 60 | 0.783 | -0.183 [-0.283, -0.083] | 0.0517 | 0.083 | 0.067 [0.000, 0.133] | 1 | 6.9 (8.2) | 0.10 (0.11) | 1.50 (0.85) |
| lag_wrist | 60 | 0.317 | -0.650 [-0.750, -0.550] | 5.73e-10 | 0.417 | 0.400 [0.333, 0.467] | 2.26e-06 | 4.9 (8.2) | 0.15 (0.11) | 2.57 (0.85) |
| lag_proprio | 60 | 0.983 | 0.017 [-0.033, 0.067] | 1 | 0.067 | 0.050 [-0.017, 0.117] | 1 | 7.6 (8.2) | 0.08 (0.11) | 0.85 (0.85) |
| lag_all | 60 | 0.117 | -0.850 [-0.933, -0.767] | 1.51e-14 | 0.417 | 0.400 [0.333, 0.467] | 2.26e-06 | 5.0 (8.2) | 0.08 (0.11) | 1.57 (0.85) |
| proprio_offset | 60 | 0.967 | 0.000 [-0.050, 0.050] | 1 | 0.017 | 0.000 [-0.050, 0.050] | 1 | 8.2 (8.2) | 0.11 (0.11) | 0.90 (0.85) |
| gripper_stuck | 60 | 0.067 | -0.900 [-0.967, -0.833] | 2e-15 | 0.250 | 0.233 [0.167, 0.300] | 0.00134 | 2.6 (8.2) | 0.04 (0.11) | 0.83 (0.85) |
| freeze_transport | 60 | 0.583 | -0.383 [-0.467, -0.300] | 2.38e-06 | 0.117 | 0.100 [0.033, 0.167] | 0.281 | 6.8 (8.2) | 0.15 (0.11) | 1.50 (0.85) |
| rescue_wrist_12 | 60 | 0.950 | -0.017 [-0.083, 0.050] | 1 | 0.017 | 0.000 [-0.050, 0.050] | 1 | 7.7 (8.2) | 0.11 (0.11) | 0.88 (0.85) |
| rescue_readout_20 | 60 | 0.867 | -0.100 [-0.183, -0.017] | 0.547 | 0.067 | 0.050 [-0.017, 0.117] | 1 | 7.3 (8.2) | 0.11 (0.11) | 1.15 (0.85) |
| rescue_wrist_12_random | 60 | 0.717 | -0.250 [-0.367, -0.150] | 0.00583 | 0.083 | 0.067 [0.000, 0.150] | 1 | 6.8 (8.2) | 0.10 (0.11) | 1.07 (0.85) |
| rescue_readout_20_random | 60 | 0.700 | -0.267 [-0.400, -0.150] | 0.00362 | 0.150 | 0.133 [0.050, 0.233] | 0.215 | 6.8 (8.2) | 0.09 (0.11) | 0.92 (0.85) |
| rescue_wrist_12_samephase | 60 | 0.800 | -0.167 [-0.267, -0.067] | 0.0776 | 0.083 | 0.067 [0.000, 0.150] | 1 | 7.2 (8.2) | 0.10 (0.11) | 0.95 (0.85) |
| rescue_wrist_12_otherphase | 60 | 0.417 | -0.550 [-0.667, -0.433] | 3.03e-09 | 0.367 | 0.350 [0.250, 0.450] | 1.43e-05 | 4.9 (8.2) | 0.13 (0.11) | 1.98 (0.85) |
| rescue_wrist_12_noise | 60 | 0.417 | -0.550 [-0.650, -0.433] | 1.13e-07 | 0.317 | 0.300 [0.217, 0.383] | 0.000521 | 5.2 (8.2) | 0.14 (0.11) | 2.27 (0.85) |
| rescue_wrist_12_noop | 60 | 0.317 | -0.650 [-0.750, -0.550] | 5.73e-10 | 0.417 | 0.400 [0.333, 0.467] | 2.26e-06 | 4.9 (8.2) | 0.15 (0.11) | 2.57 (0.85) |

## Patching rescue of a 0.4 s late wrist camera (closed loop)

Unpatched delayed wrist camera: success 0.317, unsafe 0.417; plain run success 0.967. Differences are paired against the delayed run (Holm over arms).

| arm | n | success | vs delayed [95% CI] | p (Holm) | unsafe | vs delayed [95% CI] | p (Holm) |
|---|---|---|---|---|---|---|---|
| rescue_wrist_12 | 60 | 0.950 | 0.633 [0.533, 0.733] | 5.82e-11 | 0.017 | -0.400 [-0.467, -0.333] | 9.54e-07 |
| rescue_readout_20 | 60 | 0.867 | 0.550 [0.433, 0.650] | 7.17e-08 | 0.067 | -0.350 [-0.433, -0.267] | 6.68e-06 |
| rescue_wrist_12_random | 60 | 0.717 | 0.400 [0.283, 0.517] | 0.000348 | 0.083 | -0.333 [-0.400, -0.267] | 1.14e-05 |
| rescue_readout_20_random | 60 | 0.700 | 0.383 [0.233, 0.517] | 0.000467 | 0.150 | -0.267 [-0.350, -0.183] | 0.00058 |
| rescue_wrist_12_samephase | 60 | 0.800 | 0.483 [0.350, 0.617] | 6.51e-06 | 0.083 | -0.333 [-0.417, -0.250] | 0.000179 |
| rescue_wrist_12_otherphase | 60 | 0.417 | 0.100 [-0.033, 0.233] | 0.754 | 0.367 | -0.050 [-0.150, 0.050] | 1 |
| rescue_wrist_12_noise | 60 | 0.417 | 0.100 [-0.017, 0.217] | 0.714 | 0.317 | -0.100 [-0.200, 0.000] | 0.438 |
| rescue_wrist_12_noop | 60 | 0.317 | 0.000 [0.000, 0.000] | 1 | 0.417 | 0.000 [0.000, 0.000] | 1 |

Paired contrasts between arms (first minus second; Holm over the contrasts):

| contrast | success difference [95% CI] | p (Holm) | unsafe difference [95% CI] | p (Holm) |
|---|---|---|---|---|
| rescue_wrist_12 - rescue_wrist_12_samephase | 0.150 [0.067, 0.233] | 0.00781 | -0.067 [-0.133, -0.017] | 0.375 |
| rescue_wrist_12_samephase - rescue_wrist_12_otherphase | 0.383 [0.250, 0.517] | 0.000584 | -0.283 [-0.400, -0.167] | 0.00244 |
| rescue_wrist_12 - rescue_wrist_12_random | 0.233 [0.133, 0.333] | 0.00394 | -0.067 [-0.150, 0.000] | 0.438 |
| rescue_wrist_12 - rescue_wrist_12_otherphase | 0.533 [0.433, 0.633] | 3.26e-09 | -0.350 [-0.450, -0.250] | 6.68e-06 |
| rescue_wrist_12 - rescue_wrist_12_noise | 0.533 [0.433, 0.633] | 3.26e-09 | -0.300 [-0.383, -0.217] | 4.58e-05 |
| rescue_wrist_12_random - rescue_wrist_12_noise | 0.300 [0.183, 0.417] | 0.00213 | -0.233 [-0.333, -0.133] | 0.00525 |
| rescue_readout_20 - rescue_readout_20_random | 0.167 [0.033, 0.300] | 0.0309 | -0.083 [-0.183, 0.017] | 0.438 |

No-op check: putting the stale states back reproduces the unpatched delayed run in 60/60 episodes (same outcome and same number of steps).

