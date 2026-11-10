# Study protocol

The hypotheses, tests and decision rules, in the form they were run. Parameters are in `configs/protocol.yaml`
(rendered in `docs/hyperparameters.md`). Dated entries mark rules that were added or changed during the study;
each was fixed before the held-out data it affects existed.

## Setting

- **Policy.** OpenVLA-OFT, LIBERO-Goal checkpoint (`moojink/openvla-7b-oft-finetuned-libero-goal`), bf16, run
  exactly as in the upstream LIBERO evaluation: 256 px renders, 10 settle steps, 300 control steps, 8-step chunks
  executed open loop. Comparison policy: LeRobot's pi0.5 (`lerobot/pi05_libero_finetuned_v044`).
- **Tasks.** The 10 LIBERO-Goal tasks share one scene, so only the instruction says which task to do.
- **Splits of the 50 initial states per task.** Calibration 0-4 (severity levels, oracle thresholds), exploratory
  5-19 (method development), held-out 20-49 (pre-registered tests only, 300 paired episodes per condition).
- **Pairing.** Every condition runs on the same (task, init state) pairs. LIBERO re-samples fixture poses at each
  reset; we replay upstream's sequence once and restore it per init state (`assets/fixture_layouts_libero_goal.npz`),
  so an episode does not depend on run order.

## Freezes

| Date | What | File |
|---|---|---|
| 2026-11-06 17:25 | corruption levels (severity rule on calibration data) and oracle thresholds | `configs/frozen_levels.yaml` |
| 2026-11-09 15:13 | mitigation arm (selection rule on exploratory data) | `configs/frozen_mitigation.yaml` |

Held-out runs refuse any condition or arm that is not frozen (`vlamech.runs.experiments.preflight`).

## Hypotheses and decision rules

Episodes are the unit. Intervals are stratified cluster bootstraps (tasks fixed as strata, init states resampled
with everything observed on them; 10,000 replicates, percentile 95%), with a tasks-resampled interval reported
alongside. Paired success differences use exact McNemar tests; multiplicity is Holm within each family of tests.

**H1, instruction effect (open loop).** For each captured clean state, every frozen condition is re-created offline
and the policy is queried under its own instruction, the 9 others and an empty one. IE = mean distance between the
own-instruction chunk and the others (arm dimensions); IER = IE under corruption / IE clean. H1a (fade): IER < 1;
H1b (reallocation): IER > 1. Per cell: episode-level sign-flip test on log IER, Holm across the 13 cells.
- *H1-dir, added 2026-11-07.* IE is unsigned, so IER > 1 cannot separate more steering from change in other
  directions. Each contrast is split along the clean contrast at the same state (IER_par, 1 on clean) and across it
  (IER_perp, 0 on clean). Test IER_par = 1 per cell; *aligned fade* if the CI is below 1 and significant.
- *H1-task, added 2026-11-07.* The bowl tasks (1, 4, 8: "put the bowl on <destination>") have lower IER than the
  others. Exact one-sided task-relabelling test over all 120 choices of 3 tasks, with a bootstrap CI.

**H2, failure type (closed loop).** Among corruption-induced failures, the share of intent loss (another task's goal
becomes true, or a non-target object moves first) exceeds its share among clean failures. Paired McNemar on the
intent-loss rate and Fisher's exact test on the share, Holm across cells.

**H3, mechanism (activation patching, bowl tasks at the grasp).** States: the first chunk whose clean gripper
command closes, and the next. Metric: steering among the three bowl instructions (IER_par relative to the clean
run). Verdict per condition (blur 8, noise 128, target occlusion), fixed 2026-11-08: "instruction route" if all
three hold, Holm over conditions x tests:
1. keeping the instruction positions clean up to the last layer restores more than the same clamp with another
   episode's clean states (random donor);
2. injecting corrupted instruction states into the clean run lowers steering (CI below 1);
3. it lowers it more than injecting another episode's clean instruction states does.

A vision-position clamp is not a test: corrupted vision is the only difference between the runs, so it reproduces
the clean run by construction. Single-camera and single-entity shares are descriptive.
- *Added 2026-11-09:* the same tests at states the failing held-out rollouts actually reached (test 1 on absolute
  IER_par); failing against succeeding rollouts is descriptive.

**H4, failure monitor.** Signals from commanded actions only (chunk turn, chunk jump, horizon decay, speed) and the
instruction effect. Development on exploratory states 5-12 (normalization, choice of signal) and 13-19
(split-conformal stop threshold); test on held-out states under single-camera conditions no earlier analysis had
seen. H4 holds if the instruction effect has higher failure AUROC than the other signals.

**H5, mitigation (fixed 2026-11-08).** Arms: plain policy, classifier-free-style instruction guidance (CAG, omega 3),
receding horizon (re-plan every 4 or 2 steps) and temporal ensembling. Selection on exploratory states: the highest
mean success under the corrupted conditions among arms whose clean success is not significantly below the plain
policy's. The held-out test runs the selected arm and CAG at omega 3 against the plain sweep episodes.
- *H5b:* a split-conformal safe stop on the chosen monitor signal lowers unsafe failures (failures with a collision
  above the task's clean maximum), evaluated offline on the held-out runs.
- *Added 2026-11-09, outside the selection:* CAG at omega 1.5 on exploratory states, descriptive.

**Cross-model check (added 2026-11-09).** pi0.5 on the held-out states, clean, blur 8 and noise 128: success only,
since the instruction metric is not defined for sampled actions. The drop is compared with OFT's by a bootstrap
interval; this compares two released checkpoints, not two architectures.

## Analyses added afterwards (not pre-registered)

- Exploratory figures of the instruction effect and of closed-loop behaviour (`scripts/figures_*.py`).
- Physical goal check: commanded progress toward the instructed destination on the bowl tasks
  (`scripts/analysis/goal_progress.py`).
- Steering loss against episode outcome (`scripts/analysis/steering_and_failure.py`).
- Timing of the safe stop relative to the first collision (section of `scripts/analysis/failure_monitor.py`).
- The wrong-destination label (`oracle.destination_label`) and the oracle review sheets (`scripts/figures/oracle_review.py`).

## Release note

`results/oracle_calibration/thresholds.json` was regenerated by the released code. Every numeric threshold equals the
one frozen on 2026-11-06; only descriptive text in the file changed, so its hash in `configs/frozen_levels.yaml` was
updated accordingly.
