# Instruction effect under corruption (EXPLORATORY)

150 clean episodes, every captured state; IER = IE(corrupted) / IE(clean) on the same state. Intervals: stratified cluster bootstrap (tasks fixed); p: episode-level sign-flip, Holm across cells.

| condition | IER (geometric mean) [95% CI] | tasks resampled | VE [CI] | share difference [CI] | p_holm | verdict |
|---|---|---|---|---|---|---|
| bg_patches-0.8 | 1.16 [1.14, 1.18] | [1.08, 1.25] | 0.64 [0.62, 0.65] | +0.041 [+0.038, +0.043] | 0.00065 | reallocation |
| blur-1 | 1.04 [1.03, 1.04] | [1.03, 1.05] | 0.17 [0.16, 0.17] | +0.009 [+0.008, +0.009] | 0.00065 | reallocation |
| blur-4 | 1.32 [1.30, 1.34] | [1.18, 1.48] | 0.80 [0.79, 0.81] | +0.071 [+0.068, +0.074] | 0.00065 | reallocation |
| blur-8 | 1.38 [1.36, 1.40] | [1.12, 1.79] | 1.43 [1.41, 1.44] | +0.122 [+0.118, +0.125] | 0.00065 | reallocation |
| jpeg-6 | 1.08 [1.07, 1.09] | [1.03, 1.15] | 0.54 [0.53, 0.55] | +0.027 [+0.025, +0.029] | 0.00065 | reallocation |
| jpeg-1 | 1.19 [1.17, 1.21] | [1.04, 1.38] | 1.00 [0.99, 1.02] | +0.067 [+0.064, +0.070] | 0.00065 | reallocation |
| matched_bg_box | 1.01 [1.01, 1.01] | [1.00, 1.02] | 0.06 [0.05, 0.06] | +0.001 [+0.001, +0.002] | 0.0009 | reallocation |
| noise-64 | 1.28 [1.27, 1.30] | [1.17, 1.44] | 0.74 [0.73, 0.76] | +0.061 [+0.058, +0.064] | 0.00065 | reallocation |
| noise-128 | 1.40 [1.38, 1.42] | [1.13, 1.75] | 1.37 [1.35, 1.39] | +0.117 [+0.113, +0.121] | 0.00065 | reallocation |
| photometric-3.5 | 1.04 [1.03, 1.06] | [1.00, 1.10] | 0.56 [0.55, 0.57] | +0.022 [+0.020, +0.024] | 0.00065 | reallocation |
| photometric-4.5 | 0.98 [0.97, 1.00] | [0.88, 1.11] | 0.94 [0.92, 0.95] | +0.032 [+0.029, +0.034] | 0.39 | no detectable change |
| target_occlusion | 1.27 [1.25, 1.29] | [0.98, 1.71] | 1.21 [1.19, 1.23] | +0.063 [+0.060, +0.066] | 0.00065 | reallocation |
| bg_patches-0.8-exactmi | 1.17 [1.15, 1.18] | [1.09, 1.26] | 0.63 [0.62, 0.64] | +0.042 [+0.039, +0.044] | 0.00065 | reallocation |

## Direction of the instruction effect (H1-dir, secondary)

Each instruction contrast split along the clean contrast (IER_par, 1 on clean) and across it (IER_perp, 0 on clean); cos_align = cosine to the clean contrast. Linear scale, arithmetic means; p: sign-flip on IER_par - 1, Holm across cells.

| condition | IER_par [CI] | tasks resampled | IER_perp [CI] | cos_align [CI] | p_holm | verdict |
|---|---|---|---|---|---|---|
| bg_patches-0.8 | 0.881 [0.870, 0.892] | [0.823, 0.938] | 0.765 [0.739, 0.793] | 0.699 [0.694, 0.705] | 0.00065 | aligned fade |
| blur-1 | 0.975 [0.972, 0.979] | [0.946, 1.001] | 0.276 [0.272, 0.280] | 0.920 [0.918, 0.922] | 0.00065 | aligned fade |
| blur-4 | 0.926 [0.910, 0.940] | [0.810, 1.051] | 1.035 [0.985, 1.093] | 0.642 [0.637, 0.648] | 0.00065 | aligned fade |
| blur-8 | 0.729 [0.706, 0.751] | [0.574, 0.892] | 1.645 [1.580, 1.714] | 0.513 [0.508, 0.519] | 0.00065 | aligned fade |
| jpeg-6 | 0.862 [0.852, 0.872] | [0.780, 0.936] | 0.591 [0.579, 0.603] | 0.741 [0.736, 0.746] | 0.00065 | aligned fade |
| jpeg-1 | 0.805 [0.789, 0.821] | [0.698, 0.924] | 0.991 [0.961, 1.022] | 0.596 [0.590, 0.601] | 0.00065 | aligned fade |
| matched_bg_box | 0.978 [0.974, 0.982] | [0.949, 0.996] | 0.121 [0.115, 0.127] | 0.963 [0.959, 0.966] | 0.00065 | aligned fade |
| noise-64 | 0.956 [0.942, 0.970] | [0.829, 1.098] | 0.903 [0.879, 0.929] | 0.671 [0.666, 0.676] | 0.021 | aligned fade |
| noise-128 | 0.793 [0.769, 0.816] | [0.634, 0.999] | 1.492 [1.433, 1.554] | 0.527 [0.522, 0.533] | 0.00065 | aligned fade |
| photometric-3.5 | 0.815 [0.807, 0.824] | [0.752, 0.865] | 0.604 [0.585, 0.625] | 0.726 [0.721, 0.731] | 0.00065 | aligned fade |
| photometric-4.5 | 0.649 [0.639, 0.660] | [0.586, 0.721] | 0.814 [0.778, 0.853] | 0.594 [0.588, 0.599] | 0.00065 | aligned fade |
| target_occlusion | 0.812 [0.790, 0.835] | [0.657, 1.034] | 1.400 [1.347, 1.453] | 0.578 [0.573, 0.584] | 0.00065 | aligned fade |
| bg_patches-0.8-exactmi | 0.889 [0.877, 0.900] | [0.830, 0.948] | 0.770 [0.744, 0.798] | 0.702 [0.696, 0.707] | 0.00065 | aligned fade |

## Bowl-placement tasks [1, 4, 8] vs the other tasks (H1-task, secondary)

T = -0.273 [-0.293, -0.254] (difference of task-mean log IER over all corrupted cells; ratio 0.76); exact one-sided p = 0.00833 over 120 task relabellings; verdict: lower.

| condition | IER, group | IER, other tasks |
|---|---|---|
| bg_patches-0.8 | 1.01 | 1.23 |
| blur-1 | 1.03 | 1.04 |
| blur-4 | 1.09 | 1.43 |
| blur-8 | 0.94 | 1.63 |
| jpeg-6 | 1.02 | 1.11 |
| jpeg-1 | 0.90 | 1.35 |
| matched_bg_box | 1.00 | 1.01 |
| noise-64 | 1.10 | 1.37 |
| noise-128 | 0.93 | 1.67 |
| photometric-3.5 | 0.97 | 1.08 |
| photometric-4.5 | 0.79 | 1.08 |
| target_occlusion | 0.85 | 1.50 |
| bg_patches-0.8-exactmi | 1.02 | 1.24 |
