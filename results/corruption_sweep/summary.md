# Corruption sweep (held-out)

300 (task, init state) pairs per condition, held-out confirmatory split, frozen selection and oracle v2. Intervals: stratified cluster bootstrap (tasks fixed, init states resampled), percentile 95%; 'tasks resampled' = sensitivity interval. p_holm: Holm within the family named in the header.

## Success (family: success)

| condition | success | rate % [95% CI] | difference vs clean, pts [CI] | tasks resampled | McNemar n10/n01 | p_holm |
|---|---|---|---|---|---|---|
| identity | 292/300 | 97.3 [95.3, 99.0] | - | - | - | - |
| bg_patches-0.8 | 254/300 | 84.7 [81.0, 88.3] | -12.7 [-16.7, -8.7] | -12.7 [-24.0, -3.0] | 45/7 | 4.2e-07 |
| blur-1 | 295/300 | 98.3 [96.7, 99.7] | 1.0 [-1.0, 3.0] | 1.0 [-2.0, 4.3] | 4/7 | 1 |
| blur-4 | 240/300 | 80.0 [76.3, 83.7] | -17.3 [-21.3, -13.3] | -17.3 [-32.7, -5.3] | 57/5 | 2.5e-11 |
| blur-8 | 73/300 | 24.3 [20.7, 28.0] | -73.0 [-77.0, -69.0] | -73.0 [-88.7, -53.7] | 220/1 | 1.6e-63 |
| jpeg-6 | 290/300 | 96.7 [94.7, 98.3] | -0.7 [-3.3, 2.0] | -0.7 [-7.3, 4.7] | 10/8 | 1 |
| jpeg-1 | 147/300 | 49.0 [45.0, 53.0] | -48.3 [-52.7, -44.0] | -48.3 [-71.0, -25.7] | 148/3 | 4e-39 |
| matched_bg_box | 290/300 | 96.7 [94.7, 98.7] | -0.7 [-2.7, 1.0] | -0.7 [-3.0, 1.7] | 5/3 | 1 |
| noise-64 | 269/300 | 89.7 [87.3, 92.0] | -7.7 [-10.7, -4.7] | -7.7 [-24.3, 2.7] | 30/7 | 0.00096 |
| noise-128 | 95/300 | 31.7 [28.0, 35.3] | -65.7 [-69.7, -61.7] | -65.7 [-84.7, -43.7] | 197/0 | 1.1e-58 |
| photometric-3.5 | 277/300 | 92.3 [90.0, 94.7] | -5.0 [-7.7, -2.3] | -5.0 [-19.0, 3.3] | 22/7 | 0.033 |
| photometric-4.5 | 177/300 | 59.0 [55.3, 62.7] | -38.3 [-42.3, -34.7] | -38.3 [-62.7, -15.7] | 118/3 | 2e-30 |
| target_occlusion | 42/300 | 14.0 [11.0, 17.0] | -83.3 [-86.7, -79.7] | -83.3 [-95.7, -68.0] | 251/1 | 9.1e-73 |
| bg_patches-0.8-exactmi | 246/300 | 82.0 [78.0, 86.0] | -15.3 [-19.7, -11.0] | -15.3 [-25.7, -6.3] | 52/6 | 2.2e-09 |

## H2: intent loss (oracle v2)

| condition | intent-loss episodes | rate difference vs clean, pts [CI] | p_holm (rate) | share among failures (clean) | p_holm (share, Fisher) | destination-aware difference, pts [CI] (exploratory) |
|---|---|---|---|---|---|---|
| bg_patches-0.8 | 4 | 1.0 [-0.3, 2.3] | 1 | 9% (12%) | 1 | 1.3 [0.0, 3.0] |
| blur-1 | 1 | 0.0 [-1.0, 1.0] | 1 | 20% (12%) | 1 | 0.0 [-1.0, 1.0] |
| blur-4 | 2 | 0.3 [-0.7, 1.7] | 1 | 3% (12%) | 1 | 1.3 [0.0, 3.0] |
| blur-8 | 25 | 8.0 [5.3, 10.7] | 8.9e-06 | 11% (12%) | 1 | 13.3 [10.0, 16.7] |
| jpeg-6 | 1 | 0.0 [-1.0, 1.0] | 1 | 10% (12%) | 1 | 0.0 [-1.0, 1.0] |
| jpeg-1 | 8 | 2.3 [0.7, 4.3] | 0.35 | 5% (12%) | 1 | 2.3 [0.7, 4.3] |
| matched_bg_box | 2 | 0.3 [0.0, 1.0] | 1 | 20% (12%) | 1 | 1.3 [0.3, 2.7] |
| noise-64 | 0 | -0.3 [-1.0, 0.0] | 1 | 0% (12%) | 1 | -0.3 [-1.0, 0.0] |
| noise-128 | 33 | 10.7 [7.7, 13.7] | 4.9e-08 | 16% (12%) | 1 | 16.0 [13.0, 19.0] |
| photometric-3.5 | 0 | -0.3 [-1.0, 0.0] | 1 | 0% (12%) | 1 | 0.0 [-1.0, 1.0] |
| photometric-4.5 | 22 | 7.0 [4.7, 9.3] | 5.7e-05 | 18% (12%) | 1 | 7.3 [5.0, 9.7] |
| target_occlusion | 38 | 12.3 [9.7, 15.0] | 1.9e-09 | 15% (12%) | 1 | 15.3 [12.0, 18.7] |
| bg_patches-0.8-exactmi | 1 | 0.0 [-1.0, 1.0] | 1 | 2% (12%) | 1 | 0.0 [-1.0, 1.0] |

## Severity trends (family: trend)

| family | scale | levels | slope (log-odds per unit) [CI] | p_holm |
|---|---|---|---|---|
| blur | log | blur-1, blur-4, blur-8 | -4.38 [-5.55, -3.20] | 8.5e-13 |
| noise | log | noise-64, noise-128 | -8.89 [-11.23, -6.54] | 5e-13 |
| jpeg | log | jpeg-6, jpeg-1 | -3.30 [-4.19, -2.40] | 9.7e-13 |
| photometric | linear | photometric-3.5, photometric-4.5 | -5.19 [-6.60, -3.78] | 9.7e-13 |

## Controls (family: controls)

| contrast | success difference, pts [CI] | McNemar n10/n01 | p_holm |
|---|---|---|---|
| target_occlusion vs matched_bg_box | -82.7 [-86.3, -78.7] | 251/3 | 3.8e-70 |
| bg_patches-0.8 vs bg_patches-0.8-exactmi | 2.7 [-1.0, 6.3] | 12/20 | 0.22 |

## Hazard proxies (reported; Holm within each proxy)

| condition | collision > clean max | difference, pts [CI] | p_holm | off table | p_holm |
|---|---|---|---|---|---|
| bg_patches-0.8 | 6 | 1.0 [-1.0, 3.0] | 1 | 0 | 1 |
| blur-1 | 5 | 0.7 [-1.0, 2.7] | 1 | 0 | 1 |
| blur-4 | 2 | -0.3 [-2.0, 1.0] | 1 | 0 | 1 |
| blur-8 | 32 | 9.7 [7.7, 11.7] | 1.6e-06 | 0 | 1 |
| jpeg-6 | 4 | 0.3 [-1.3, 2.0] | 1 | 0 | 1 |
| jpeg-1 | 4 | 0.3 [-1.3, 2.0] | 1 | 0 | 1 |
| matched_bg_box | 6 | 1.0 [-0.7, 2.7] | 1 | 0 | 1 |
| noise-64 | 8 | 1.7 [-0.3, 3.7] | 1 | 0 | 1 |
| noise-128 | 7 | 1.3 [-0.3, 3.0] | 1 | 0 | 1 |
| photometric-3.5 | 1 | -0.7 [-1.7, 0.0] | 1 | 0 | 1 |
| photometric-4.5 | 10 | 2.3 [0.3, 4.7] | 0.72 | 0 | 1 |
| target_occlusion | 72 | 23.0 [19.3, 26.7] | 4.8e-17 | 0 | 1 |
| bg_patches-0.8-exactmi | 10 | 2.3 [0.0, 4.7] | 0.92 | 0 | 1 |

Clean: 3 collisions above the clean-success maximum (the proxy's false-alarm floor), off table 0.
