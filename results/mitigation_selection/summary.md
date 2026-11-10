# Mitigation arm selection (exploratory, 150 episodes per cell)

Selected: **cag:1**. Held-out arms: cag:3.

| arm | clean success | change vs plain (p) | blur-8 | noise-128 | target_occlusion | mean corrupted |
|---|---|---|---|---|---|---|
| cag:1 | 1.000 | +0.000 (1) | 0.220 | 0.327 | 0.127 | 0.224 |
| cag:3 | 0.380 | -0.620 (2e-28) harm | 0.153 | 0.147 | 0.027 | 0.109 |
| horizon:4 | 0.940 | -0.060 (0.0039) harm | 0.180 | 0.267 | 0.120 | 0.189 |
| horizon:2 | 0.887 | -0.113 (1.5e-05) harm | 0.127 | 0.233 | 0.140 | 0.167 |
| horizon:2:ensemble | 0.940 | -0.060 (0.0039) harm | 0.267 | 0.227 | 0.073 | 0.189 |
