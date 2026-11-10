# Guidance follow-up (exploratory, 150 paired episodes per cell, outside the arm selection)

| arm | condition | plain | arm | change [CI] | discordant (lost/gained) | p | collisions plain / arm |
|---|---|---|---|---|---|---|---|
| cag:1.5 | identity | 1.000 | 0.847 | -0.153 [-0.200, -0.113] | 23/0 | 2.38e-07 | 0.000 / 0.060 |
| cag:1.5 | blur-8 | 0.220 | 0.193 | -0.027 [-0.080, +0.027] | 11/7 | 0.961 (Holm) | 0.120 / 0.173 |
| cag:1.5 | noise-128 | 0.327 | 0.273 | -0.053 [-0.113, +0.007] | 18/10 | 0.555 (Holm) | 0.040 / 0.107 |
| cag:1.5 | target_occlusion | 0.127 | 0.140 | +0.013 [-0.033, +0.060] | 7/9 | 0.961 (Holm) | 0.227 / 0.213 |
