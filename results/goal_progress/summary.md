# Physical check on the bowl tasks (post hoc, not pre-registered; held-out, open loop)

90 episodes, 613 states with the bowl held. Margin: commanded 8-step progress toward the instructed destination minus progress toward the other two, horizontal plane, cm per chunk. Picked: the instructed destination gets the most progress.

| condition | margin (cm) | margin vs clean | picked | translation along clean (bowl contrast) |
|---|---|---|---|---|
| identity | 7.23 [6.97, 7.48] | 1 | 0.53 [0.52, 0.54] | 1.00 [1.00, 1.00] |
| matched_bg_box | 7.21 [6.95, 7.46] | 1.00 [0.99, 1.00] | 0.53 [0.51, 0.54] | 0.99 [0.98, 1.00] |
| blur-1 | 7.29 [7.02, 7.54] | 1.01 [1.00, 1.01] | 0.54 [0.52, 0.55] | 0.99 [0.98, 1.00] |
| blur-4 | 7.05 [6.79, 7.29] | 0.97 [0.95, 1.00] | 0.57 [0.56, 0.59] | 0.97 [0.95, 1.01] |
| blur-8 | 6.19 [5.94, 6.41] | 0.86 [0.83, 0.88] | 0.57 [0.56, 0.58] | 0.77 [0.75, 0.80] |
| noise-64 | 7.71 [7.42, 7.98] | 1.07 [1.05, 1.08] | 0.58 [0.56, 0.59] | 1.00 [0.98, 1.03] |
| noise-128 | 5.73 [5.47, 5.95] | 0.79 [0.77, 0.82] | 0.57 [0.55, 0.58] | 0.69 [0.66, 0.71] |
| jpeg-6 | 7.48 [7.20, 7.74] | 1.03 [1.02, 1.05] | 0.55 [0.53, 0.56] | 0.91 [0.90, 0.93] |
| jpeg-1 | 6.55 [6.25, 6.82] | 0.91 [0.88, 0.93] | 0.57 [0.55, 0.58] | 0.76 [0.73, 0.79] |
| photometric-3.5 | 6.92 [6.66, 7.17] | 0.96 [0.94, 0.97] | 0.54 [0.52, 0.55] | 0.83 [0.81, 0.85] |
| photometric-4.5 | 5.72 [5.48, 5.94] | 0.79 [0.77, 0.81] | 0.52 [0.51, 0.53] | 0.60 [0.58, 0.62] |
| bg_patches-0.8 | 7.87 [7.57, 8.14] | 1.09 [1.07, 1.11] | 0.57 [0.56, 0.59] | 1.03 [1.00, 1.06] |
| bg_patches-0.8-exactmi | 7.85 [7.55, 8.12] | 1.09 [1.06, 1.11] | 0.58 [0.56, 0.59] | 1.05 [1.02, 1.08] |
| target_occlusion | 5.83 [5.54, 6.10] | 0.81 [0.77, 0.84] | 0.55 [0.54, 0.56] | 0.72 [0.68, 0.75] |

Spearman across corrupted conditions, margin ratio against translation along clean: 0.95.
