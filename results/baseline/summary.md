# Clean baseline

Published OFT LIBERO-Goal success: 97.9% (500 episodes)

Replicate 0: 492/500 = 98.4%  95% CI [96.9%, 99.2%]  published inside CI: True  unresolved errors: 0
Replicate 1: 492/500 = 98.4%  95% CI [96.9%, 99.2%]  published inside CI: True  unresolved errors: 0

Noise floor over 500 paired episodes: outcome discordant 0 (0.00%, Wilson 95% CI [0.00%, 0.76%]: an upper bound on run-to-run outcome noise, not proof of zero numerical variability); trajectory (step count) differs in 3: ['task6-init9', 'task9-init19', 'task9-init38']

| Task | Instruction | Rep 0 | Rep 1 |
|---|---|---|---|
| 0 | open the middle drawer of the cabinet | 49/50 | 49/50 |
| 1 | put the bowl on the stove | 48/50 | 48/50 |
| 2 | put the wine bottle on top of the cabinet | 50/50 | 50/50 |
| 3 | open the top drawer and put the bowl inside | 47/50 | 47/50 |
| 4 | put the bowl on top of the cabinet | 50/50 | 50/50 |
| 5 | push the plate to the front of the stove | 50/50 | 50/50 |
| 6 | put the cream cheese in the bowl | 50/50 | 50/50 |
| 7 | turn on the stove | 50/50 | 50/50 |
| 8 | put the bowl on the plate | 49/50 | 49/50 |
| 9 | put the wine bottle on the rack | 49/50 | 49/50 |

Failures (replicate 0) with the intent oracle:

| Episode | steps | goals that became true | objects moved > 1 cm | max env force (N) |
|---|---|---|---|---|
| task0-init38 | 300 | - | - | 47 |
| task1-init27 | 300 | {'7': 212} | {'akita_black_bowl_1': 0.065} | 230 |
| task1-init40 | 300 | - | {'akita_black_bowl_1': 0.117} | 105 |
| task3-init44 | 300 | - | {'akita_black_bowl_1': 0.101} | 140 |
| task3-init46 | 300 | - | - | 117 |
| task3-init48 | 300 | - | {'akita_black_bowl_1': 0.199, 'cream_cheese_1': 0.026} | 401 |
| task8-init41 | 300 | - | {'akita_black_bowl_1': 0.06, 'plate_1': 0.016} | 43 |
| task9-init27 | 300 | - | {'wine_bottle_1': 0.014} | 154 |

(The 1 cm column is descriptive only; the oracle's displacement threshold is set per task by the protocol rule from clean calibration episodes.)
