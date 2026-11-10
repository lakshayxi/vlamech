# Mitigation held-out test (300 paired episodes per cell, against the plain sweep episodes)

| arm | condition | plain | arm | change [CI] | discordant (lost/gained) | p_holm | collisions plain / arm |
|---|---|---|---|---|---|---|---|
| cag:3 | identity | 0.973 | 0.390 | -0.583 [-0.620, -0.543] | 179/4 | 7.54e-48 (clean) | 0.010 / 0.260 |
| cag:3 | blur-8 | 0.243 | 0.153 | -0.090 [-0.130, -0.050] | 36/9 | 6.57e-05 | 0.107 / 0.263 |
| cag:3 | noise-128 | 0.317 | 0.133 | -0.183 [-0.227, -0.140] | 71/16 | 5.87e-09 | 0.023 / 0.323 |
| cag:3 | target_occlusion | 0.140 | 0.020 | -0.120 [-0.153, -0.087] | 39/3 | 1.13e-08 | 0.240 / 0.363 |
