# Activation patching (exploratory)

## Patching (bowl states)

rho = fraction of the corrupted run's loss of bowl steering (IER_par) restored by keeping the instruction positions clean up to layer 31. Injected = the clean run's IER_par with instruction states clamped in up to that layer: the corrupted run's, or the clean states of another episode (control).

| condition | corrupted IER_par | rho instruction | rho random donor | instr - donor | injected: corrupted | injected: other episode | corrupted - other | verdict |
|---|---|---|---|---|---|---|---|---|
| blur-8 | 0.44 [0.41, 0.47] | 0.50 [0.46, 0.54] | 0.21 [0.14, 0.28] | 0.29 [0.23, 0.35] | 0.51 [0.48, 0.54] | 0.81 [0.78, 0.84] | -0.30 [-0.33, -0.26] | instruction route |
| noise-128 | 0.35 [0.33, 0.37] | 0.50 [0.47, 0.53] | 0.27 [0.22, 0.32] | 0.23 [0.18, 0.28] | 0.45 [0.42, 0.47] | 0.81 [0.78, 0.84] | -0.36 [-0.40, -0.33] | instruction route |
| target_occlusion | 0.30 [0.28, 0.32] | 0.52 [0.49, 0.56] | 0.28 [0.23, 0.33] | 0.24 [0.19, 0.30] | 0.38 [0.35, 0.41] | 0.81 [0.78, 0.84] | -0.43 [-0.47, -0.38] | instruction route |

### Direction of the steering in the three runs

Along the clean contrast (IER_par, 1 on clean), across it (IER_perp, 0 on clean), cosine to it.

| condition | run | IER_par | IER_perp | cos_align |
|---|---|---|---|---|
| blur-8 | corrupted | 0.44 [0.41, 0.47] | 0.42 [0.40, 0.45] | 0.66 [0.63, 0.69] |
| blur-8 | instruction positions kept clean | 0.72 [0.69, 0.75] | 0.50 [0.47, 0.52] | 0.79 [0.77, 0.81] |
| blur-8 | corrupted instruction states in a clean run | 0.51 [0.48, 0.54] | 0.22 [0.21, 0.24] | 0.89 [0.88, 0.90] |
| noise-128 | corrupted | 0.35 [0.33, 0.37] | 0.34 [0.32, 0.36] | 0.64 [0.61, 0.67] |
| noise-128 | instruction positions kept clean | 0.67 [0.65, 0.70] | 0.43 [0.41, 0.46] | 0.80 [0.78, 0.82] |
| noise-128 | corrupted instruction states in a clean run | 0.45 [0.42, 0.47] | 0.21 [0.20, 0.22] | 0.87 [0.85, 0.88] |
| target_occlusion | corrupted | 0.30 [0.28, 0.32] | 0.31 [0.28, 0.34] | 0.64 [0.61, 0.66] |
| target_occlusion | instruction positions kept clean | 0.67 [0.65, 0.70] | 0.44 [0.41, 0.46] | 0.80 [0.78, 0.82] |
| target_occlusion | corrupted instruction states in a clean run | 0.38 [0.35, 0.41] | 0.20 [0.19, 0.22] | 0.83 [0.81, 0.85] |

## Single camera and single entity (bowl states)

fraction = (1 - IER_par variant) / (1 - IER_par full condition), bowl IER_par; episode means with CIs.

| condition | full IER_par | wrist only | agentview only | bowl only | destination only |
|---|---|---|---|---|---|
| blur-8 | 0.44 [0.41, 0.47] | 0.54 [0.48, 0.60] | 0.54 [0.49, 0.59] | - | - |
| noise-128 | 0.35 [0.33, 0.37] | 0.45 [0.40, 0.51] | 0.66 [0.61, 0.71] | - | - |
| target_occlusion | 0.30 [0.28, 0.32] | 0.66 [0.62, 0.69] | 0.39 [0.36, 0.42] | 0.61 [0.57, 0.64] | 0.22 [0.20, 0.25] |
| matched_bg_box | 1.00 [0.99, 1.00] | no loss to share out | | | |
