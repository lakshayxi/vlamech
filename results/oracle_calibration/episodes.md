# Oracle calibration

Source: `results/severity_calibration/identity-agentview+wrist__ins-own__rep0` (manifest 8f462f16aeea), classifier v2, trace schema 1

| Task | clean baseline successes | disp threshold (µm) | max object force (N) | max env force (N) | env force, 5 calib eps |
|---|---|---|---|---|---|
| 0 | 98 | 0.2786 | 0.0 | 609.6 | 235.4 |
| 1 | 96 | 2.893e+04 | 15.4 | 64.5 | 23.0 |
| 2 | 100 | 4136 | 42.5 | 279.6 | 237.9 |
| 3 | 94 | 3.338e+04 | 24.7 | 840.9 | 404.4 |
| 4 | 100 | 6119 | 18.1 | 150.7 | 37.3 |
| 5 | 100 | 8.907e+04 | 35.8 | 220.6 | 220.6 |
| 6 | 100 | 0.2629 | 21.3 | 194.3 | 187.1 |
| 7 | 100 | 0.2786 | 0.0 | 385.9 | 252.9 |
| 8 | 98 | 3.069e+04 | 9.8 | 0.0 | 0.0 |
| 9 | 98 | 0.2786 | 22.8 | 195.6 | 0.0 |

| Joint | threshold |
|---|---|
| flat_stove_1_button | 0 |
| wooden_cabinet_1_bottom_level | 0 |
| wooden_cabinet_1_middle_level | 0.00166 |
| wooden_cabinet_1_top_level | 0.00173 |

Floors: passive drift of each episode's own task and init state (passive_drift.json).

Calibration episodes by outcome: {'success': 50}

| Episode | steps | outcome | subtype | events |
|---|---|---|---|---|
