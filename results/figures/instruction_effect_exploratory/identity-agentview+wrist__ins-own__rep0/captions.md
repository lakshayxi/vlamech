# Captions: instruction effect, `identity-agentview+wrist__ins-own__rep0` (EXPLORATORY)

**ier_hero.png.** One state, the instruction effect seen directly.
- *State.* Chosen by rule, not by eye: task 1 ("put the bowl on the stove"), the first closing chunk of each of the
  15 exploratory episodes, the one with the median IER_par under blur-8. This is
  task1-init12, step 40; its IER_par is 0.48.
- *Condition.* blur-8, the condition with the largest mean IER_perp.
- (A) The camera view (true orientation) clean and corrupted.
- (B) The hand path the policy commands over the next 8 steps (normalized x-y translation, summed) when told to put
  the bowl on the stove, on the cabinet or on the plate. Solid: clean. Dashed: corrupted. In 3-D (x, y and height)
  the three path ends are on average 1.43 apart on clean images and 0.74
  apart under blur-8 (normalized units).
- (C) Every corrupted condition on the direction plane: IER_par, the instruction effect along the clean task
  contrast (clean = 1), against IER_perp, the effect across it (clean = 0). Exploratory episode means; intervals are
  in ier_direction.png.

**ier_dose.png.** IER per condition, grouped by family and ordered by severity. Points are geometric means over
episodes. Thick bars: 95% stratified bootstrap CI (tasks fixed, init states resampled, as pre-registered). Thin bars:
tasks also resampled.

**ier_direction.png.** Left: the size of the instruction effect (IER, log scale). Right: its direction, as the
fraction of the clean effect along the clean task contrast (IER_par) and across it (IER_perp), linear scale, 95% CIs
(tasks fixed). Even the matched-box control sits slightly below 1 (any change to the image costs ~2% alignment),
so effects are read against that floor.

**ier_tasks.png.** IER per task and condition (geometric mean over the task's 15 episodes); blue = larger than
clean, orange = smaller.

**ier_grasp.png.** IER_par aligned to the grasp.
- Columns: chunks of 8 steps relative to the first chunk in which the clean policy closes the gripper (raw gripper
  output < 0.5); offsets with at least 10 episodes in each group: -5, -4, -3, -2, -1, +0, +1, +2, +3, +4, +5, +6.
- Left: the bowl-placement tasks (1, 4, 8), where the instruction only chooses the destination.
- Right: the other object carries (2, 3, 6, 9). The stove knob (task 7) is left out: its gripper closes but carries
  nothing.
- Mean over episodes of the per-episode mean at each offset.

**ier_chunk.png.** Inside one 8-step action chunk (step 0 is executed first).
- Left: IER_par computed on each step's 6 arm dimensions.
- Right: the commanded translation speed under the own instruction at each step, divided by the chunk's mean speed;
  the clean row is on top.
- Episode means.
