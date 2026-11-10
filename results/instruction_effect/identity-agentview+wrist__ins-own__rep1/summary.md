# Instruction effect under corruption

300 clean episodes, every captured state; IER = IE(corrupted) / IE(clean) on the same state. Intervals: stratified cluster bootstrap (tasks fixed); p: episode-level sign-flip, Holm across cells.

| condition | IER (geometric mean) [95% CI] | tasks resampled | VE [CI] | share difference [CI] | p_holm | verdict |
|---|---|---|---|---|---|---|
| bg_patches-0.8 | 1.17 [1.16, 1.18] | [1.07, 1.30] | 0.63 [0.62, 0.64] | +0.041 [+0.039, +0.042] | 0.00065 | reallocation |
| blur-1 | 1.04 [1.04, 1.04] | [1.03, 1.05] | 0.17 [0.16, 0.17] | +0.010 [+0.009, +0.010] | 0.00065 | reallocation |
| blur-4 | 1.34 [1.32, 1.35] | [1.19, 1.52] | 0.81 [0.80, 0.82] | +0.075 [+0.073, +0.077] | 0.00065 | reallocation |
| blur-8 | 1.39 [1.38, 1.41] | [1.11, 1.84] | 1.43 [1.42, 1.44] | +0.125 [+0.123, +0.128] | 0.00065 | reallocation |
| jpeg-6 | 1.10 [1.09, 1.11] | [1.04, 1.18] | 0.55 [0.54, 0.55] | +0.028 [+0.026, +0.029] | 0.00065 | reallocation |
| jpeg-1 | 1.21 [1.19, 1.22] | [1.04, 1.40] | 1.00 [0.99, 1.01] | +0.068 [+0.065, +0.071] | 0.00065 | reallocation |
| matched_bg_box | 1.01 [1.01, 1.01] | [1.00, 1.03] | 0.06 [0.05, 0.06] | +0.001 [+0.001, +0.002] | 0.00065 | reallocation |
| noise-64 | 1.31 [1.29, 1.32] | [1.17, 1.48] | 0.74 [0.73, 0.75] | +0.062 [+0.059, +0.065] | 0.00065 | reallocation |
| noise-128 | 1.41 [1.39, 1.43] | [1.12, 1.80] | 1.38 [1.37, 1.39] | +0.117 [+0.113, +0.121] | 0.00065 | reallocation |
| photometric-3.5 | 1.06 [1.05, 1.07] | [0.99, 1.15] | 0.56 [0.55, 0.57] | +0.023 [+0.022, +0.025] | 0.00065 | reallocation |
| photometric-4.5 | 1.00 [0.99, 1.02] | [0.87, 1.18] | 0.94 [0.93, 0.95] | +0.033 [+0.030, +0.035] | 0.81 | no detectable change |
| target_occlusion | 1.27 [1.26, 1.29] | [0.97, 1.77] | 1.20 [1.19, 1.21] | +0.063 [+0.060, +0.065] | 0.00065 | reallocation |
| bg_patches-0.8-exactmi | 1.18 [1.17, 1.19] | [1.08, 1.31] | 0.63 [0.62, 0.64] | +0.042 [+0.040, +0.044] | 0.00065 | reallocation |

## Direction of the instruction effect (H1-dir, secondary)

Each instruction contrast split along the clean contrast (IER_par, 1 on clean) and across it (IER_perp, 0 on clean); cos_align = cosine to the clean contrast. Linear scale, arithmetic means; p: sign-flip on IER_par - 1, Holm across cells.

| condition | IER_par [CI] | tasks resampled | IER_perp [CI] | cos_align [CI] | p_holm | verdict |
|---|---|---|---|---|---|---|
| bg_patches-0.8 | 0.887 [0.877, 0.896] | [0.836, 0.936] | 0.817 [0.789, 0.846] | 0.701 [0.697, 0.705] | 0.00065 | aligned fade |
| blur-1 | 0.977 [0.974, 0.980] | [0.946, 1.003] | 0.277 [0.274, 0.281] | 0.920 [0.918, 0.921] | 0.00065 | aligned fade |
| blur-4 | 0.926 [0.914, 0.939] | [0.802, 1.057] | 1.078 [1.035, 1.124] | 0.643 [0.639, 0.647] | 0.00065 | aligned fade |
| blur-8 | 0.733 [0.717, 0.749] | [0.569, 0.906] | 1.671 [1.639, 1.704] | 0.515 [0.511, 0.519] | 0.00065 | aligned fade |
| jpeg-6 | 0.865 [0.857, 0.872] | [0.781, 0.941] | 0.625 [0.615, 0.636] | 0.736 [0.732, 0.740] | 0.00065 | aligned fade |
| jpeg-1 | 0.812 [0.799, 0.824] | [0.693, 0.943] | 1.014 [0.994, 1.035] | 0.595 [0.591, 0.599] | 0.00065 | aligned fade |
| matched_bg_box | 0.978 [0.975, 0.980] | [0.948, 0.996] | 0.122 [0.117, 0.128] | 0.961 [0.958, 0.964] | 0.00065 | aligned fade |
| noise-64 | 0.963 [0.952, 0.975] | [0.833, 1.119] | 0.953 [0.933, 0.973] | 0.670 [0.665, 0.674] | 0.0098 | aligned fade |
| noise-128 | 0.802 [0.784, 0.820] | [0.625, 1.022] | 1.551 [1.511, 1.590] | 0.529 [0.524, 0.533] | 0.00065 | aligned fade |
| photometric-3.5 | 0.814 [0.807, 0.822] | [0.753, 0.863] | 0.662 [0.643, 0.682] | 0.722 [0.718, 0.726] | 0.00065 | aligned fade |
| photometric-4.5 | 0.654 [0.644, 0.665] | [0.593, 0.725] | 0.877 [0.846, 0.908] | 0.593 [0.589, 0.597] | 0.00065 | aligned fade |
| target_occlusion | 0.816 [0.799, 0.834] | [0.650, 1.056] | 1.452 [1.415, 1.490] | 0.579 [0.574, 0.583] | 0.00065 | aligned fade |
| bg_patches-0.8-exactmi | 0.892 [0.883, 0.902] | [0.840, 0.943] | 0.828 [0.798, 0.860] | 0.703 [0.699, 0.707] | 0.00065 | aligned fade |

## Bowl-placement tasks [1, 4, 8] vs the other tasks (H1-task, secondary)

T = -0.300 [-0.312, -0.287] (difference of task-mean log IER over all corrupted cells; ratio 0.74); exact one-sided p = 0.00833 over 120 task relabellings; verdict: lower.

| condition | IER, group | IER, other tasks |
|---|---|---|
| bg_patches-0.8 | 1.00 | 1.25 |
| blur-1 | 1.03 | 1.04 |
| blur-4 | 1.08 | 1.46 |
| blur-8 | 0.93 | 1.66 |
| jpeg-6 | 1.01 | 1.14 |
| jpeg-1 | 0.90 | 1.37 |
| matched_bg_box | 1.00 | 1.02 |
| noise-64 | 1.09 | 1.41 |
| noise-128 | 0.92 | 1.70 |
| photometric-3.5 | 0.95 | 1.11 |
| photometric-4.5 | 0.78 | 1.12 |
| target_occlusion | 0.82 | 1.53 |
| bg_patches-0.8-exactmi | 1.01 | 1.26 |
