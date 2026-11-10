# Activation patching (confirmatory)

## Patching (bowl states)

rho = fraction of the corrupted run's loss of bowl steering (IER_par) restored by keeping the instruction positions clean up to layer 31. Injected = the clean run's IER_par with instruction states clamped in up to that layer: the corrupted run's, or the clean states of another episode (control).

| condition | corrupted IER_par | rho instruction | rho random donor | instr - donor | injected: corrupted | injected: other episode | corrupted - other | verdict |
|---|---|---|---|---|---|---|---|---|
| blur-8 | 0.43 [0.41, 0.45] | 0.50 [0.46, 0.54] | 0.26 [0.20, 0.31] | 0.24 [0.19, 0.30] | 0.49 [0.47, 0.51] | 0.83 [0.81, 0.86] | -0.35 [-0.37, -0.32] | instruction route |
| noise-128 | 0.35 [0.33, 0.36] | 0.51 [0.49, 0.54] | 0.32 [0.29, 0.36] | 0.19 [0.15, 0.22] | 0.43 [0.41, 0.44] | 0.83 [0.81, 0.86] | -0.41 [-0.44, -0.38] | instruction route |
| target_occlusion | 0.30 [0.29, 0.32] | 0.53 [0.50, 0.55] | 0.34 [0.30, 0.38] | 0.19 [0.16, 0.22] | 0.37 [0.35, 0.39] | 0.83 [0.81, 0.86] | -0.46 [-0.49, -0.43] | instruction route |

### Direction of the steering in the three runs

Along the clean contrast (IER_par, 1 on clean), across it (IER_perp, 0 on clean), cosine to it.

| condition | run | IER_par | IER_perp | cos_align |
|---|---|---|---|---|
| blur-8 | corrupted | 0.43 [0.41, 0.45] | 0.43 [0.41, 0.46] | 0.64 [0.62, 0.66] |
| blur-8 | instruction positions kept clean | 0.71 [0.69, 0.73] | 0.50 [0.48, 0.52] | 0.78 [0.77, 0.80] |
| blur-8 | corrupted instruction states in a clean run | 0.49 [0.47, 0.51] | 0.22 [0.21, 0.23] | 0.88 [0.87, 0.89] |
| noise-128 | corrupted | 0.35 [0.33, 0.36] | 0.34 [0.32, 0.36] | 0.65 [0.63, 0.67] |
| noise-128 | instruction positions kept clean | 0.68 [0.66, 0.70] | 0.45 [0.43, 0.47] | 0.80 [0.78, 0.81] |
| noise-128 | corrupted instruction states in a clean run | 0.43 [0.41, 0.44] | 0.21 [0.20, 0.22] | 0.87 [0.85, 0.88] |
| target_occlusion | corrupted | 0.30 [0.29, 0.32] | 0.30 [0.28, 0.31] | 0.64 [0.62, 0.66] |
| target_occlusion | instruction positions kept clean | 0.67 [0.65, 0.69] | 0.43 [0.41, 0.45] | 0.81 [0.80, 0.83] |
| target_occlusion | corrupted instruction states in a clean run | 0.37 [0.35, 0.39] | 0.19 [0.18, 0.20] | 0.83 [0.81, 0.84] |

## Single camera and single entity (bowl states)

fraction = (1 - IER_par variant) / (1 - IER_par full condition), bowl IER_par; episode means with CIs.

| condition | full IER_par | wrist only | agentview only | bowl only | destination only |
|---|---|---|---|---|---|
| blur-8 | 0.43 [0.41, 0.45] | 0.56 [0.52, 0.60] | 0.53 [0.49, 0.56] | - | - |
| noise-128 | 0.35 [0.33, 0.36] | 0.49 [0.46, 0.53] | 0.63 [0.61, 0.66] | - | - |
| target_occlusion | 0.30 [0.29, 0.32] | 0.67 [0.65, 0.69] | 0.37 [0.34, 0.39] | 0.63 [0.60, 0.66] | 0.23 [0.21, 0.24] |
| matched_bg_box | 0.99 [0.98, 1.00] | no loss to share out | | | |

Mean box area hidden, pixels (whole target / bowl only / destination only): main camera 9175 / 626 / 4887; wrist 24064 / 17733 / 4695.
