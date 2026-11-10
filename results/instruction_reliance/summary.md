# Instruction reliance (OpenVLA-OFT, LIBERO-Goal, 500 paired episodes per condition)

| Condition | Success | 95% CI | vs own: McNemar p |
|---|---|---|---|
| own | 492/500 = 98.4% | [96.9%, 99.2%] | — |
| other_task | 0/500 = 0.0% | [0.0%, 0.8%] | 1.6e-148 |
| empty | 49/500 = 9.8% | [7.5%, 12.7%] | 8.8e-134 |

Language share (own − wrong)/own = 100.0%  [95% CI 100.0%, 100.0%]  (stratified bootstrap, 10000 replicates).
Li et al. 2610.19438 report wrong-instruction success 0/250 for OFT on LIBERO-Goal.

## What the policy achieved (cross-task goal oracle)

| Condition | own task goal | PROMPTED (wrong) task goal | other task goal | no goal |
|---|---|---|---|---|
| own | 492 | 0 | 1 | 7 |
| other_task | 0 | 483 | 0 | 17 |
| empty | 49 | 0 | 354 | 97 |

(own: the PROMPTED column is empty by definition; for other_task, 'PROMPTED' = the policy achieved task (i+5)'s goal in task i's scene, i.e. it followed the wrong instruction.)

## Per task

| Task | Instruction | own | other_task (prompted task) | empty |
|---|---|---|---|---|
| 0 | open the middle drawer of the cabinet | 49/50 | 0/50 (task 5) | 0/50 |
| 1 | put the bowl on the stove | 48/50 | 0/50 (task 6) | 8/50 |
| 2 | put the wine bottle on top of the cabinet | 50/50 | 0/50 (task 7) | 0/50 |
| 3 | open the top drawer and put the bowl inside | 47/50 | 0/50 (task 8) | 0/50 |
| 4 | put the bowl on top of the cabinet | 50/50 | 0/50 (task 9) | 0/50 |
| 5 | push the plate to the front of the stove | 50/50 | 0/50 (task 0) | 0/50 |
| 6 | put the cream cheese in the bowl | 50/50 | 0/50 (task 1) | 8/50 |
| 7 | turn on the stove | 50/50 | 0/50 (task 2) | 33/50 |
| 8 | put the bowl on the plate | 49/50 | 0/50 (task 3) | 0/50 |
| 9 | put the wine bottle on the rack | 49/50 | 0/50 (task 4) | 0/50 |

## Empty instruction: which goal did the policy pursue by default?

- task 0 scene: task 1 (put the bowl on the stove) ×15, task 6 (put the cream cheese in the bowl) ×9, task 7 (turn on the stove) ×29, task 9 (put the wine bottle on the rack) ×1
- task 1 scene: task 1 (put the bowl on the stove) ×8, task 6 (put the cream cheese in the bowl) ×13, task 7 (turn on the stove) ×21
- task 2 scene: task 1 (put the bowl on the stove) ×11, task 6 (put the cream cheese in the bowl) ×5, task 7 (turn on the stove) ×31
- task 3 scene: task 1 (put the bowl on the stove) ×8, task 6 (put the cream cheese in the bowl) ×9, task 7 (turn on the stove) ×26
- task 4 scene: task 1 (put the bowl on the stove) ×13, task 6 (put the cream cheese in the bowl) ×7, task 7 (turn on the stove) ×36
- task 5 scene: task 1 (put the bowl on the stove) ×11, task 6 (put the cream cheese in the bowl) ×12, task 7 (turn on the stove) ×25, task 9 (put the wine bottle on the rack) ×2
- task 6 scene: task 1 (put the bowl on the stove) ×16, task 6 (put the cream cheese in the bowl) ×8, task 7 (turn on the stove) ×31
- task 7 scene: task 1 (put the bowl on the stove) ×9, task 6 (put the cream cheese in the bowl) ×7, task 7 (turn on the stove) ×33
- task 8 scene: task 1 (put the bowl on the stove) ×12, task 6 (put the cream cheese in the bowl) ×12, task 7 (turn on the stove) ×27
- task 9 scene: task 1 (put the bowl on the stove) ×17, task 6 (put the cream cheese in the bowl) ×13, task 7 (turn on the stove) ×27
