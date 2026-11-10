# Failure monitor and safe stop

Signal chosen on init states 5-12: **chunk_turn** (development AUROC 0.803). Calibrated on 70 clean successful episodes (init states 13-19). Test: 2100 episodes, failure rate 0.196.

## Failure AUROC on the held-out conditions (early window)

| signal | dev AUROC | test AUROC [CI] |
|---|---|---|
| instruction_effect | 0.430 | 0.564 [0.532, 0.597] |
| chunk_turn | 0.803 | 0.736 [0.710, 0.762] |
| horizon_decay | 0.591 | 0.568 [0.535, 0.600] |
| chunk_jump | 0.782 | 0.763 [0.739, 0.786] |
| speed | 0.233 | 0.237 [0.209, 0.267] |

Instruction effect minus each other signal (CI at alpha 0.05/4):

- chunk_turn: -0.172 [-0.223, -0.121]
- horizon_decay: -0.004 [-0.061, 0.059]
- chunk_jump: -0.199 [-0.253, -0.143]
- speed: 0.327 [0.273, 0.379]

## Safe stop on chunk_turn, alpha = 0.05 (threshold z = 7.46)

| condition | failure | stopped | success lost | unsafe without | unsafe with | p_holm |
|---|---|---|---|---|---|---|
| identity | 0.027 | 0.037 | 0.023 | 0.010 | 0.003 | 0.5 |
| blur-8@wrist | 0.280 | 0.323 | 0.183 | 0.030 | 0.000 | 0.0195 |
| blur-8@agentview | 0.117 | 0.127 | 0.047 | 0.010 | 0.007 | 1 |
| noise-128@wrist | 0.263 | 0.263 | 0.150 | 0.010 | 0.000 | 1 |
| noise-128@agentview | 0.147 | 0.170 | 0.060 | 0.003 | 0.003 | 1 |
| target_occlusion@wrist | 0.483 | 0.447 | 0.120 | 0.047 | 0.007 | 0.00293 |
| target_occlusion@agentview | 0.057 | 0.133 | 0.087 | 0.007 | 0.003 | 1 |

## Safe stop on chunk_turn, alpha = 0.1 (threshold z = 5.52)

| condition | failure | stopped | success lost | unsafe without | unsafe with | p_holm |
|---|---|---|---|---|---|---|
| identity | 0.027 | 0.093 | 0.073 | 0.010 | 0.003 | 0.5 |
| blur-8@wrist | 0.280 | 0.447 | 0.260 | 0.030 | 0.000 | 0.0195 |
| blur-8@agentview | 0.117 | 0.197 | 0.100 | 0.010 | 0.003 | 1 |
| noise-128@wrist | 0.263 | 0.423 | 0.240 | 0.010 | 0.000 | 1 |
| noise-128@agentview | 0.147 | 0.250 | 0.130 | 0.003 | 0.003 | 1 |
| target_occlusion@wrist | 0.483 | 0.557 | 0.173 | 0.047 | 0.007 | 0.00293 |
| target_occlusion@agentview | 0.057 | 0.183 | 0.130 | 0.007 | 0.003 | 1 |

## Timing of the alert (alpha = 0.05, corrupted conditions; post hoc)

- Unsafe failures without the stop: 32. Alert at or before the first collision: 26; lead in steps: median 70.0, range [9.0, 219.0].
- Stop as a failure detector: precision 0.558, recall 0.606 (failure rate 0.224); median stop step on failures 96.

| signal | average precision | test failure rate |
|---|---|---|
| instruction_effect | 0.206 | 0.196 |
| chunk_turn | 0.431 | 0.196 |
| horizon_decay | 0.261 | 0.196 |
| chunk_jump | 0.454 | 0.196 |
| speed | 0.124 | 0.196 |
