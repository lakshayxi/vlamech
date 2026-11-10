# Mechanism at states reached by the corrupted rollouts

States: offsets 0 and +1 from the first chunk the rollout's own gripper closes (held-out sweep rollouts, 3 bowl tasks). Failing rollouts are tested; failing against succeeding is descriptive.

| condition | states (failing/succeeding episodes) | corrupted IER_par, failing | rho instruction | rho other episode | instr - other | injected: corrupted | injected: other | corrupted - other | verdict |
|---|---|---|---|---|---|---|---|---|---|
| blur-8 | 178 (60/29) | +0.48 [+0.44, +0.52] | +0.52 [+0.46, +0.58] | +0.40 [+0.31, +0.50] | +0.06 [+0.01, +0.11] | +0.47 [+0.45, +0.50] | +0.90 [+0.84, +0.96] | -0.43 [-0.49, -0.37] | instruction route |
| noise-128 | 180 (35/55) | +0.41 [+0.36, +0.45] | +0.52 [+0.48, +0.56] | +0.37 [+0.29, +0.45] | +0.09 [+0.05, +0.13] | +0.47 [+0.44, +0.51] | +0.85 [+0.81, +0.90] | -0.38 [-0.44, -0.33] | instruction route |
| target_occlusion | 180 (56/34) | +0.38 [+0.35, +0.41] | +0.55 [+0.50, +0.60] | +0.39 [+0.33, +0.47] | +0.09 [+0.05, +0.13] | +0.49 [+0.46, +0.52] | +0.87 [+0.82, +0.92] | -0.38 [-0.43, -0.33] | instruction route |

## Failing against succeeding rollouts (descriptive)

| condition | corrupted IER_par, failing | succeeding | failing - succeeding | steering restored, failing - succeeding |
|---|---|---|---|---|
| blur-8 | +0.48 [+0.44, +0.52] | +0.47 [+0.45, +0.50] | +0.00 [-0.05, +0.07] | -0.03 [-0.07, +0.01] |
| noise-128 | +0.41 [+0.36, +0.45] | +0.41 [+0.38, +0.44] | -0.01 [-0.06, +0.05] | -0.00 [-0.04, +0.03] |
| target_occlusion | +0.38 [+0.35, +0.41] | +0.23 [+0.21, +0.25] | +0.15 [+0.10, +0.20] | -0.04 [-0.08, -0.01] |
