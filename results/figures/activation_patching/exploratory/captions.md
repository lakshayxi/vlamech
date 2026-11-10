# Captions: activation patching, exploratory

**direction.png.** The three-way contrast between bowl instructions, split along the clean contrast and across it (95% CIs). Clean sits at (1, 0). Moving left is muting; moving up is rotation.

**patch_heatmap.png.** Single-layer denoising on bowl states: the clean run's state of one token group at one depth replaces the corrupted run's; colour = rho, the fraction of the lost bowl steering (IER_par under the three bowl instructions) restored, episode means. Groups are positions, not modalities: after layer 0 each position mixes in all others.

**clamps.png.** Top: prefix clamps (the group kept clean at every depth up to the point) for the instruction positions, with the random-donor (same task, other episode, same phase) and norm-matched-noise controls. Bottom: noising, the clean run with the corrupted run's instruction states clamped in, and the same with another episode's clean states (circles); 1 = steering unaffected.

**inputs.png.** On bowl states, the share of each condition's loss of bowl steering that is reproduced when only one camera is corrupted, or (target occlusion) when only the bowl or only the destination is hidden; 1 = all of it. The bowl/destination split partitions the captured target mask by the segmentation of the restored simulator state; each part then hides its own bounding box, so the two parts neither cover nor equal the box of the whole target.

