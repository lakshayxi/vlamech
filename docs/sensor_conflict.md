# When sensors disagree

A follow-up to the corruption study. There we degraded what the cameras showed. Here every input is clean, but the
inputs describe two slightly different worlds: the wrist camera sees the bowl where the third-person camera does
not, the joint encoders report the hand 3 cm from where the cameras show it, or one camera lags the other by 0.4 s.
A robot that acts confidently on contradictory evidence is a safety problem that image-corruption benchmarks do not
measure.

All parameters are in `configs/protocol.yaml` under `sensor_conflict`. They were fixed on 2026-11-17, before any
held-out forward pass; the code was developed on the exploratory captures and on calibration rollouts.

## Questions

1. **Arbitration.** When two inputs disagree, which one decides the action, and does that change between reaching
   for an object and carrying it to its destination?
2. **Relevance.** Does the policy respond to what a change means for the task, or only to how much of the image
   changed?
3. **Mechanism.** At which depth does the conflicting evidence leave its own token positions and reach the positions
   the action head reads?
4. **Detection.** Can the policy's own internal state tell that its inputs disagree before it acts?
5. **Consequences.** What happens in closed loop when a sensor is miscalibrated, stuck or late, and does the same
   0.8 s freeze matter more at some phases than others?

## Design

**Counterfactual worlds** (`src/vlamech/perturbations/worlds.py`). From a captured clean rollout, the simulator
state at a query is restored and rendered twice: as it is, and with one physical fact changed. These are the edits:

| edit | what changes | sources that differ |
|---|---|---|
| object | the object to grasp (before the grasp) or the destination (after it) moves 5 cm | third-person, wrist |
| distractor | an object outside the task moves 5 cm | third-person, wrist |
| arm | the hand moves 3 cm (inverse kinematics; a held object moves with it) | third-person, wrist, proprio |
| gripper | an open gripper is shown closed, a closed one open | third-person, wrist, proprio |
| stale | the observation of the previous query (0.4 s earlier) | third-person, wrist, proprio |

Edits that would push objects into each other are rejected, and the next seeded direction is tried. Each world is
consistent with itself. A conflict is created by mixing them: for example, the third-person frame from the edited
world and the wrist frame and proprio from the real one.

**Shares** (`src/vlamech/analysis/arbitration.py`). The policy is queried on every mix: 4 inputs for two sources, 8
for three. The response to the fully edited world, Δ, is split among the sources by their Shapley values, so the
shares sum to one. A share of 1 for the wrist camera means the action follows what the wrist camera shows and
ignores the rest. Shares are aggregated over episodes, weighted by |Δ|², so states where the policy barely responds
do not dominate.

**Phases.** These come from the policy's own gripper commands in the clean rollout: approach (two queries before
the closing chunk), pre-grasp (one query before), transport (halfway to the release) and place (the query before
the release).

**Mechanism.** At pre-grasp and transport, single-source conflicts are patched one layer at a time. The real run's
states are put back at the conflicting source's positions, at the instruction positions, or at the 56 readout
positions. The curves show how much of the conflict's effect each patch removes at each depth.

**Detection.** A logistic probe on the readout positions' mean state at layers 8, 16 and 24 separates mixed inputs
from consistent ones. It is trained on exploratory states and tested on held-out ones.

**Closed loop** (`src/vlamech/interventions/sensor_faults.py`, experiment `sensor_faults`). Each fault runs on 300
held-out episodes and is paired with a plain run:
- `proprio_offset`: the reported hand position is off by 3 cm.
- `gripper_stuck`: the reported finger positions stay at their initial values.
- `lag_*`: one source, or all of them, is fed from the previous query.
- `freeze_*`: the whole observation is held for two queries, triggered from the simulator at approach, at the grasp
  command, during transport or near the goal. These run on the pick-and-place tasks.

## Pre-specified tests (held-out)

Each test uses a stratified episode bootstrap, with Holm correction within each family:

- **(A) Object edit:** wrist share minus third-person share, at each phase.
- **(B) Arm edit:** proprio share, at each phase.
- **(C) Gripper edit:** proprio share, at each phase.
- **(D) Relevance:** mean log(|Δ object| / |Δ distractor|) > 0, at each phase.
- **(E) Arm edit:** proprio share under blur-8 and under noise-128, against clean images.
- **(F) Probe:** AUROC > 0.5 at layer 16.
- **Closed loop:** success and unsafe outcomes against the paired plain run, using exact McNemar tests.

These are descriptive: the stale-edit shares, the off-axis movement, gripper decision flips and the patching curves.

## Running

```bash
python scripts/run/sensor_conflict.py --split confirmatory --shard 0/2   # and 1/2
python scripts/run/closed_loop.py --experiment sensor_faults --states confirmatory --sensor-fault lag_wrist
python scripts/analysis/sensor_conflict.py
```
