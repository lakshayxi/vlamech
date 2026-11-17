# Hyperparameters and design parameters

*Generated from `configs/protocol.yaml` by `scripts/setup/render_hyperparameters.py`. Do not edit by hand.*

Every number used by the experiments is listed here with its category and justification.

| Category | Meaning |
|---|---|
| `upstream` | fixed by a cited prior protocol (OpenVLA-OFT / LIBERO evaluation, or a replicated paper); changing it breaks comparability |
| `definitional` | follows from the definition of the measured quantity; no free choice |
| `rule` | the *value* is the output of a pre-registered selection rule applied to calibration/exploratory data only; the rule (or the ladder it searches) is what is fixed here; selected values are frozen in `configs/frozen_levels.yaml` |
| `convention` | standard statistical or methodological convention (cited) |

## `policy`

| Parameter | Value | Category | Justification |
|---|---|---|---|
| `policy.checkpoint` | moojink/openvla-7b-oft-finetuned-libero-goal | definitional | per-suite OFT checkpoint for LIBERO-Goal, the suite whose task is decided by the instruction alone |
| `policy.revision` | None | definitional | the commit resolved at download is pinned in configs/models.lock.yaml |
| `policy.dtype` | bfloat16 | upstream | OFT LIBERO evaluation uses bf16 |
| `policy.center_crop` | True | upstream | trained with random-crop augmentation; evaluation uses the 0.9-area center crop |
| `policy.num_images` | 2 | upstream | third-person and wrist camera, as trained |
| `policy.attn_implementation` | sdpa | upstream | the bidirectional mask exists only in the OFT fork's SDPA path |
| `policy.global_seed` | 7 | upstream | run_libero_eval seed; the L1 head samples nothing, so it only fixes library RNG state |

## `policies`

| Parameter | Value | Category | Justification |
|---|---|---|---|
| `policies.pi05.family` | pi05 | definitional | LeRobot pi0.5: flow-matching action expert, sampled 50-action chunks |
| `policies.pi05.checkpoint` | lerobot/pi05_libero_finetuned_v044 | definitional | pi0.5 (PaliGemma 2B + 300M flow-matching action expert) fine-tuned by LeRobot on all four LIBERO suites |
| `policies.pi05.lerobot_version` | 0.4.4 | upstream | the release the checkpoint was published for; pinned in env/setup_env_lerobot.sh |
| `policies.pi05.dtype` | bfloat16 | upstream | checkpoint config |
| `policies.pi05.chunk_size` | 50 | upstream | checkpoint config: actions predicted per query |
| `policies.pi05.n_action_steps` | 10 | upstream | actions executed per query in LeRobot's published LIBERO evaluation of this checkpoint (98.0% on LIBERO-Goal) |
| `policies.pi05.num_inference_steps` | 10 | upstream | checkpoint config: Euler steps of the flow-matching sampler |
| `policies.pi05.resolution` | 256 | rule | the training frames' resolution and the frames OFT gets, so a corruption is pixel-identical across policies (LeRobot's evaluation renders 360 px) |
| `policies.pi05.compile_model` | False | definitional | torch.compile autotuning makes numerics run-dependent; eager inference is reproducible |
| `policies.pi05.noise_seed` | derive_seed('pi05_noise', 'rep<replicate>', task, init state, query step) | definitional | every query's sampling noise is seeded, so every visual condition gets the same noise at the same (task, init, step) |

## `env`

| Parameter | Value | Category | Justification |
|---|---|---|---|
| `env.suite` | libero_goal | definitional | see policy.checkpoint |
| `env.resolution` | 256 | upstream | run_libero_eval env_img_res |
| `env.env_seed` | 0 | upstream | get_libero_env calls env.seed(0) |
| `env.settle_steps` | 10 | upstream | num_steps_wait |
| `env.max_steps` | 300 | upstream | TASK_MAX_STEPS['libero_goal'] |
| `env.open_loop_steps` | 8 | upstream | OFT executes the whole 8-step chunk before querying again |

## `splits`

| Parameter | Value | Category | Justification |
|---|---|---|---|
| `splits.n_init_states` | 50 | upstream | benchmark initial states per task |
| `splits.calibration` | [0, 4] | rule | init states 0-4 x 10 tasks = 50 episodes per level, for severity and threshold calibration |
| `splits.exploratory` | [5, 19] | rule | method development: open-loop metrics, patching, monitor signals, mitigation selection |
| `splits.confirmatory` | [20, 49] | rule | held out: 30 x 10 = 300 paired episodes per condition (scripts/calibration/power_analysis.py) |

## `instructions`

| Parameter | Value | Category | Justification |
|---|---|---|---|
| `instructions.wrong_instruction_rule` | (task_id + 5) mod 10 | convention | the fixed wrong-instruction pairing of Li et al. (2610.19438); avoids choosing pairs |

## `corruption`

| Parameter | Value | Category | Justification |
|---|---|---|---|
| `corruption.ladders.photometric_gain_stops` | [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0] | rule | exposure reduction in stops (gain 2^-stops): 0.5 stop is a mild lighting change, 4 stops leaves the scene nearly black |
| `corruption.ladders.noise_sigma` | [2, 4, 8, 16, 32, 64, 128] | rule | Gaussian pixel noise std on the 0-255 scale: 2 is about sensor read noise, 128 swamps the signal |
| `corruption.ladders.blur_sigma_px` | [0.5, 1, 2, 4, 8, 16] | rule | Gaussian defocus at 256 px: 0.5 px is sub-pixel, 16 px erases every object boundary |
| `corruption.ladders.jpeg_quality` | [96, 48, 24, 12, 6, 3, 1] | rule | halving ladder of JPEG quality; 1 is the encoder minimum |
| `corruption.ladders.bg_patch_coverage` | [0.05, 0.1, 0.2, 0.4, 0.8] | rule | fraction of background pixels covered; 0.8 leaves almost no original background |
| `corruption.selection_rule` | For each family: run every ladder level on the calibration split (50 episodes per level, both cameras corrupted). Fit an isotonic (non-increasing) regression of success on ladder index. For each target relative success r, select the level whose fitted success / clean success is closest to r (ties -> milder level). If the top of the ladder stays above the lowest target, extend the ladder once by one step of the same spacing, where the parameter's physical bound allows; if still above, the family is benign at all tested severities and kept only at its top level. | rule | makes families comparable by behavioural harm and removes free choice of levels |
| `corruption.ladder_extension` | photometric: spacing: linear; bound: None; noise: spacing: geometric; bound: None; blur: spacing: geometric; bound: None; jpeg: spacing: geometric; bound: 1; bg_patches: spacing: geometric; bound: 1.0 | definitional | each ladder's own spacing and the parameter's physical bound (JPEG quality >= 1, coverage <= 1), for the one-step extension |
| `corruption.targets` | [0.9, 0.6, 0.3] | rule | mild / moderate / severe harm, keeping enough successes and failures per condition for paired analysis |
| `corruption.cameras_primary` | [agentview, wrist] | definitional | camera-system faults affect every stream; single-camera conditions are a separate ablation |
| `corruption.protected_mask_dilation_px` | 1 | definitional | robosuite renders segmentation with anti-aliasing, so object boundaries decode to wrong instance ids; a 1-px band around protected pixels keeps 'background' to pixels whose label is reliable |
| `corruption.occluder_gray` | 128 | convention | neutral mid-gray occluder, as in occlusion-sensitivity analysis (Zeiler & Fergus 2014) |
| `corruption.target_occlusion_margin` | 0.0 | definitional | tight bounding box of the target mask |
| `corruption.bg_patch_size` | target_object_scale | rule | patch sides drawn from the bounding-box sides of the movable objects in clean calibration frames (object-sized distractors) |
| `corruption.seed_namespace` | vlamech-v1 | definitional | prefix of every stimulus seed; changing it changes every draw |

## `oracle`

| Parameter | Value | Category | Justification |
|---|---|---|---|
| `oracle.displacement_threshold_rule` | per task: maximum displacement (from the post-settle frame) of any non-target movable object over that task's clean successful baseline episodes (both replicates, all init states) | rule | an object counts as manipulated only if it moved more than any incidental disturbance in successful runs of that task |
| `oracle.joint_threshold_rule` | per articulated joint (drawer slides, stove knob): maximum \|q - q_post-settle\| over the clean successful calibration episodes of every task for which it is not a goal joint | rule | a drawer or knob counts as moved only beyond its incidental motion in successful runs; pooled over tasks because a goal joint has no incidental level in its own task |
| `oracle.target_definition` | from the BDDL goal predicates: manipulated objects = movable first arguments of binary predicates; touch entities = manipulated objects + unary-predicate arguments; goal joints = articulated joints of goal arguments; non-target = movable objects outside the task's entities and joints that are not goal joints | definitional | no hand-written target lists; checked against the 10 BDDL files in tests/test_libero_env.py |
| `oracle.outcome_classifier` | vlamech.metrics.outcome: events are first steps after the post-settle frame; precedence success > intent_loss (another goal becomes true, or off-target activity strictly before target progress) > execution_loss (target displaced or joint moved beyond threshold, or touched) > no_progress; ties are not 'before' | definitional | thresholds are applied after collection, from per-step traces, so no outcome depends on thresholds chosen while data were collected |
| `oracle.destination_label` | exploratory: for tasks whose manipulated object is shared with other tasks (bowl: 1, 3, 4, 8; wine bottle: 2, 9), destination mu_k = mean final object position over clean sweep successes of task k. Approach to k = 1 - (closest distance of the object to mu_k) / (its distance at the post-settle frame). Another task k is the heading only if its approach exceeds the largest approach to k over clean successes of the instructed task. A failed episode heading for another task's destination is 'wrong_destination' | definitional | the oracle counts intent loss only when another goal becomes true or another object moves first, so carrying the right object to another task's destination is labelled execution loss; this label is reported alongside, marked exploratory |
| `oracle.force_threshold_rule` | per task, separately for robot-object and robot-environment contacts: maximum contact normal force over that task's clean successful baseline episodes (both replicates, all init states) | rule | safety proxy: forces above anything seen in successful operation (LIBERO has no force specification). Per task because intended fixture contact (drawer handle, stove knob) is far stronger than incidental contact elsewhere |
| `oracle.off_table_rule` | object origin below the table-top plane and outside the table footprint | definitional | the only way an object gets there is by falling off; needs no margin |
| `oracle.task_relevant_entities` | union over all suite tasks of the BDDL objects of interest, regions mapped to their target entity, support surface excluded | definitional | background perturbations may not hide a candidate target of any task, or task substitution would be confounded with which objects stayed visible |

## `metrics`

| Parameter | Value | Category | Justification |
|---|---|---|---|
| `metrics.action_space` | normalized | definitional | the q01/q99 normalization to [-1, 1] makes the 7 dimensions commensurable |
| `metrics.arm_dims` | [0, 1, 2, 3, 4, 5] | definitional | translation and rotation; the gripper (dim 6) is binarized after post-processing and analysed separately |
| `metrics.chunk_steps` | 8 | definitional | the whole chunk is executed open loop, so it is the unit of behaviour |
| `metrics.alternative_instructions` | all_other_tasks | definitional | all 9 other LIBERO-Goal instructions; no subset selection |
| `metrics.ier_aggregation` | per captured clean state s: IER = IE(s, c) / IE(s, clean), both on the arm dims of the normalized 8-step chunk (own instruction against each of the 9 others, mean distance); episode value = mean of log IER over the episode's states; cell estimate = mean of episode values (a geometric-mean ratio) | definitional | a log ratio is scale-free and symmetric; averaging within the episode first makes the episode the independent unit |
| `metrics.share_reference` | VE_state(s_k, c) = distance between the own-instruction chunks at consecutive captured states k and k+1 under condition c; share = IE / (IE + VE_state), compared corrupted against clean per state like IER; the last state of an episode has no successor and is not used | definitional | an action-scale reference under the same condition: if corruption shrinks every action, IE and VE_state shrink together and the share does not move |
| `metrics.ier_decomposition` | per state s, condition c and alternative l': contrast D(s,c,l') = a(s,c,own) - a(s,c,l') on the arm dims; d_par = <D(s,c,l'), D(s,clean,l') / \|D(s,clean,l')\|>, d_perp = sqrt(max(0, \|D\|^2 - d_par^2)), cos = d_par / \|D\|; IER_par = mean_l' d_par / IE(s,clean), IER_perp = mean_l' d_perp / IE(s,clean), cos_align = mean_l' cos; episode value = mean over states; test H0 IER_par = 1 by the episode-level sign-flip test, Holm across cells; IER_perp and cos_align descriptive. Added 2026-11-07, before any held-out output was opened | definitional | IE is an unsigned norm: it grows when the instruction moves actions in any direction, so IER > 1 cannot separate more steering along the clean contrast from change in other directions. Linear scale because d_par can be negative and d_perp is 0 on clean |

## `statistics`

| Parameter | Value | Category | Justification |
|---|---|---|---|
| `statistics.alpha` | 0.05 | convention | two-sided |
| `statistics.multiplicity` | holm | convention | Holm (1979) step-down within each family of tests |
| `statistics.n_boot` | 10000 | convention | Monte-Carlo error of a 95% percentile bound about +/-0.5 percentile points |
| `statistics.bootstrap_strata` | [task] | definitional | the 10 LIBERO tasks are a fixed set, so tasks are strata, not resampled |
| `statistics.bootstrap_levels` | [init_state] | definitional | the init state is the independent unit; both conditions of a pair share it and are resampled together |
| `statistics.bootstrap_sensitivity_levels` | [task, init_state] | definitional | sensitivity analysis treating tasks as a sample of tasks (reported alongside) |
| `statistics.n_perm` | 20000 | convention | permutation p-value resolution 5e-5 |
| `statistics.bowl_tasks` | [1, 4, 8] | rule | the tasks 'put the bowl on <destination>' (stove, top of the cabinet, plate). Added 2026-11-07 from the exploratory per-task table, so tested on held-out states only; task 3 also moves the bowl but opens a drawer first |

## `patching`

| Parameter | Value | Category | Justification |
|---|---|---|---|
| `patching.conditions` | [blur-8, noise-128, target_occlusion, matched_bg_box] | rule | the strongest frozen level of the two families whose IER_par falls with severity (blur, noise), the task-relevant occlusion and the null control. Fixed 2026-11-07, before any forward pass |
| `patching.patch_conditions` | [blur-8, noise-128, target_occlusion] | rule | conditions without the null control (2026-11-08, before any held-out output): the matched box loses no steering, so restoration is undefined there |
| `patching.bowl_states` | tasks in statistics.bowl_tasks; captured queries at offsets 0 and +1 from the first chunk whose clean own-instruction gripper output has chunk mean < 0.5 (raw [0, 1] dimension, binarized upstream at 0.5); every episode of the split | rule | exploratory analysis located the loss of steering in the bowl tasks right after the grasp; offsets 0 and +1 are where it is largest and present in every episode |
| `patching.instructions_bowl` | the instructed bowl task first, then the two other bowl tasks in task order | definitional | at bowl states the question is which destination the instruction selects; the other bowl tasks are the alternatives that share the held object |
| `patching.layer_grid` | [embed, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 31] | rule | the input and every second decoder layer plus the last, to halve the forwards; prefix clamps always cover every layer up to the point |
| `patching.groups` | [vision, agentview, wrist, instruction, readout, control] | definitional | token positions: vision = agentview + wrist patches; readout = the 56 positions the action head reads; control = template positions the head does not read |
| `patching.prefix_groups` | [instruction] | definitional | corrupted vision is the only difference between the runs, so a vision prefix clamp reproduces the clean run by construction (2026-11-08, before any held-out output) |
| `patching.control_points` | [embed, 6, 14, 22, 30, 31] | rule | every fourth grid point plus the last layer, for the random-donor, norm-matched-noise and noising controls |
| `patching.random_donor` | same task, same chunk offset, the next episode of the split in init-state order (cyclic) that has that offset; its clean run under the same instruction | rule | same prompt and task phase, different scene: restores generic task-and-phase information but not this state's |
| `patching.noising_control` | the random donor's clean instruction states injected into the clean run, prefix clamps at patching.control_points, under the 3 bowl instructions | rule | separates damage specific to corruption from the damage any mismatched instruction state does (2026-11-08, before any held-out output) |
| `patching.metric` | bowl IER_par = vlamech.metrics.open_loop.ier_par_against(patched actions under the 3 bowl instructions, clean actions under them); restoration rho = (IER_par_patched - IER_par_corrupted) / (1 - IER_par_corrupted) on states with IER_par_corrupted < 1; episode value = mean over its states | definitional | a contrast between instructions cancels the task-independent visual shift, which would otherwise favour vision patches |
| `patching.verdicts` | per condition of patching.patch_conditions, three paired tests on bowl states (prefix clamps to layer 31; episode sign-flip test and bootstrap CI, Holm over conditions x tests): (a) the instruction clamp's rho exceeds its random-donor control; (b) corrupted instruction states injected into the clean run give IER_par below 1; (c) they lower it more than the donor's clean instruction states do. 'Instruction route' if all three hold, otherwise 'not supported'. Single-camera and single-entity shares are descriptive | rule | every verdict compares estimates with CIs on the same IER_par scale, with no free threshold. Fixed 2026-11-08, before any held-out output (it replaced a rule comparing against a vision clamp, which reproduces the clean run by construction) |
| `patching.rollout_states` | the held-out corrupted sweep rollouts of patching.patch_conditions, bowl tasks; offsets 0 and +1 from the first chunk whose commanded gripper output has chunk mean < 0.5, on the rollout itself; clean = the render of the same simulator state saved at capture, corrupted = the frame fed to the policy | rule | added 2026-11-09 before any such forward: tests the mechanism at states the failing rollouts actually reached |
| `patching.rollout_tests` | on failing rollouts, per condition (episode means; sign-flip test and bootstrap CI, Holm over conditions x tests): (a) IER_par with clean instruction states clamped in exceeds IER_par with another episode's clean states clamped in; (b) corrupted instruction states injected into the clean run give IER_par below 1; (c) lower than another episode's clean states. 'Instruction route' if all three hold. Failing against succeeding rollouts is descriptive | rule | the patching verdicts, with test (a) on absolute IER_par: rollout states can lose little steering, and a per-state ratio over a small loss is unstable |

## `mitigation`

| Parameter | Value | Category | Justification |
|---|---|---|---|
| `mitigation.arms` | [cag:1, cag:3, horizon:4, horizon:2, horizon:2:ensemble] | rule | fixed 2026-11-08 before any run. cag:1 is the plain policy with the instruction effect logged; cag:3 is the guidance strength Fang et al. (2602.17659) use for OFT; k = 4 and 2 halve OFT's 8-step chunk; ensembling follows ACT (Zhao et al. 2023). Listed by forward cost, so ties go to the cheaper arm |
| `mitigation.followup_arms` | [cag:1.5] | rule | added 2026-11-09 after the selection, outside it: whether a milder guidance strength also fails. Descriptive, paired against the plain arm, exact McNemar with Holm over the corrupted conditions |
| `mitigation.selection_conditions` | [identity, blur-8, noise-128, target_occlusion] | rule | clean plus the patching conditions, so the mitigation is tested where the mechanism was studied |
| `mitigation.ensemble_weight_m` | 0.01 | upstream | temporal-ensembling weight exp(-m i), oldest prediction first (ACT reference implementation) |
| `mitigation.cag_variant` | training-free: a = a(o, empty) + omega * (a(o, l) - a(o, empty)) on the normalized arm dims; the gripper comes from the own-instruction branch; the unconditional branch is the empty instruction inside the unchanged prompt template | upstream | Fang et al. (2602.17659) Eq. 4 and App. B. The gripper is a raw [0, 1] output binarized at 0.5; extrapolating it opened commanded-closed grasps, so it is excluded (2026-11-08, before any run) |
| `mitigation.selection_rule` | on the exploratory split, the arm with the highest mean success over the corrupted selection_conditions among arms whose clean success is not significantly below the plain arm's (paired exact McNemar, alpha 0.05); ties go to the arm listed first. The held-out test runs the selected arm and cag:3 | rule | selection on exploratory data only, by a fixed rule |
| `mitigation.safe_stop_rule` | stop at the first query whose per-task normalized score of the monitor's chosen signal exceeds the split-conformal threshold: the ceil((N+1)(1-alpha))-th smallest episode maximum over the N clean successful episodes of init states 13-19, alpha in safe_stop_alphas (first is primary). Evaluated offline on the held-out runs: a stop only ends the episode, so the executed prefix is what would have happened | rule | episode maxima give the guarantee on the share of clean episodes stopped; a per-query quantile would not |
| `mitigation.safe_stop_alphas` | [0.05, 0.1] | convention | share of clean episodes stopped; 1% would need at least 99 calibration episodes |
| `mitigation.monitor_signals` | vlamech.metrics.monitor.SIGNALS: instruction_effect, chunk_turn, horizon_decay, chunk_jump (action-chunk inconsistency), speed; all from commanded actions available before the chunk is executed | rule | signals that cost no extra forward, except the instruction effect (one) |
| `mitigation.monitor_split` | [[5, 12], [13, 19]] | rule | exploratory init states: normalization and choice of signal on the first range, conformal threshold on the second |
| `mitigation.monitor_fit_conditions` | [identity, blur-8, noise-128, target_occlusion, blur-4, noise-64, jpeg-1, photometric-4.5, bg_patches-0.8] | rule | the selection conditions plus frozen levels of the other families and milder blur and noise, so development sees every frozen family and a mix of successes and failures |
| `mitigation.monitor_extra_fit_conditions` | [blur-4, noise-64, jpeg-1, photometric-4.5, bg_patches-0.8] | definitional | monitor_fit_conditions not already run by the arm selection |
| `mitigation.monitor_test_conditions` | [identity, blur-8@wrist, blur-8@agentview, noise-128@wrist, noise-128@agentview, target_occlusion@wrist, target_occlusion@agentview] | rule | held-out init states under single-camera versions of the patching conditions, which no earlier analysis had seen |
| `mitigation.outcome_with_stop` | success / safe stop / unsafe failure (failure with a hazard event) / other failure | definitional | a stop is never a success; a stop before any hazard is the fail-safe outcome |

## `cross_model`

| Parameter | Value | Category | Justification |
|---|---|---|---|
| `cross_model.conditions` | [identity, blur-8, noise-128] | rule | added 2026-11-09 before any run: clean plus the two global patching corruptions, on the held-out init states |
| `cross_model.policy` | pi05 | definitional | a comparison model with a different action head (policies.pi05); success only, because the instruction metric is not defined for sampled actions |
| `cross_model.statistics` | pi0.5 success per condition with Wilson intervals; exact McNemar of each corruption against pi0.5's own clean run, Holm over corruptions; difference of success drops, pi0.5 minus OFT (sweep episodes), stratified cluster bootstrap, descriptive | rule | whether the loss of success is specific to OFT; a comparison of two checkpoints, not of architectures |

## `sensor_conflict`

| Parameter | Value | Category | Justification |
|---|---|---|---|
| `sensor_conflict.tasks` | [1, 2, 4, 6, 8, 9] | definitional | the single pick-and-place tasks: each has an object to grasp and a destination, so the four phases exist. Task 3 opens a drawer before its pick, tasks 0, 5 and 7 have no pick |
| `sensor_conflict.sources` | [agentview, wrist, proprio] | definitional | the three observation inputs of OFT besides the instruction |
| `sensor_conflict.captures` | exploratory: results/clean_capture/identity-agentview+wrist__ins-own__rep0; confirmatory: results/corruption_sweep/identity-agentview+wrist__ins-own__rep1 | definitional | the clean captured rollouts of each split (the held-out ones are the clean repeat, as in the patching study); successful episodes only, so all four phases are reached |
| `sensor_conflict.phases` | from the executed clean chunks: g = first chunk whose gripper output has chunk mean < 0.5, r = first later chunk with mean >= 0.5 (or the last query if the gripper never reopens); approach = g - 2, pregrasp = g - 1, transport = g + floor((r - g) / 2), place = r - 1; an episode is used if g >= 2 and r - g >= 3 | rule | the policy's own grasp and release commands mark the phases; no hand labels. Pregrasp is the last query before the closing chunk: at g itself the fingers already surround the object in most episodes, so moving the object or closing the gripper interpenetrates (exploratory check 2026-11-17: valid in 70% and 10% of states at g, 100% and 87% at g - 1) |
| `sensor_conflict.edits` | object: the manipulated object (approach, pregrasp) or the destination (transport, place) moved 5 cm horizontally; distractor: a movable object outside the task moved 5 cm horizontally (relevance control, same distance); arm: the end effector moved 3 cm horizontally at fixed orientation; at transport and place the held object moves with it; gripper: an open gripper rendered closed, a closed one rendered open; stale: the captured observation of the previous query (8 control steps, 0.4 s earlier) | rule | one physical fact changes per world; 5 cm is about one object radius, 3 cm is the proprio offset of the study plan; the stale world needs no rendering (both are captured observations) |
| `sensor_conflict.directions` | 8 horizontal directions at 45 degrees, tried in the order of a permutation seeded by derive_seed('sensor_conflict', edit, episode key, phase); the first that adds no contact deeper than 3 mm is used; for the distractor the object is the first in a seeded permutation of the non-task movable objects with a valid direction | rule | random but reproducible directions, and no worlds with objects inside each other |
| `sensor_conflict.max_penetration_m` | 0.003 | rule | contacts deeper than this that were not in the base state invalidate an edit |
| `sensor_conflict.vision_conditions` | [identity, blur-8, noise-128] | rule | the arm world is also queried with both cameras corrupted by the two global patching corruptions (same draws in both worlds), to ask whether degraded vision shifts trust towards proprio |
| `sensor_conflict.share` | sources S of an edit; for each subset T of S, a_T = the normalized chunk with the sources in T taken from the edited world and the others from the real one (arm dims); Delta = a_S - a_empty; v(T) = <a_T - a_empty, Delta> / \|Delta\|^2; share phi_m = Shapley value of m in v (sums to 1). Aggregate per edit and phase: sum_e phi_m(e) \|Delta_e\|^2 / sum_e \|Delta_e\|^2 over episodes e | definitional | phi_m is the part of the policy's response to the changed world that follows source m; weighting by \|Delta\|^2 keeps states where the policy barely responds from dominating a ratio |
| `sensor_conflict.off_axis` | for a single-source conflict T = {m}: r = (a_T - a_empty) - v(T) Delta; off-axis = sqrt(sum_e \|r_e\|^2 / sum_e \|Delta_e\|^2) | definitional | how far a conflicting input pushes the action in directions that neither the real nor the edited world calls for |
| `sensor_conflict.probe` | readout positions' mean residual state at layers 8, 16 and 24 for every input of the object, arm and gripper edits; label 1 = inputs mixing two worlds, 0 = the real or the edited world; standardized logistic regression (L2, C = 1) trained on exploratory states, AUROC on held-out states; layer 16 is the primary | rule | whether the policy's own state reveals that its inputs disagree, before it acts |
| `sensor_conflict.patching` | phases pregrasp and transport; conflicts {agentview only, wrist only} of the object edit and {proprio only} of the arm edit; single-point patches of the conflicting source's positions, the instruction positions and the readout positions with the real run's states, at the points of sensor_conflict.patch_grid; removed share = 1 - <a_patched - a_empty, s> / \|s\|^2 with s = a_T - a_empty | rule | single-point patches (not prefix clamps, which reproduce the real run by construction) show at which depth the conflicting evidence leaves its own positions and reaches the positions the action head reads |
| `sensor_conflict.patch_grid` | [embed, 4, 8, 12, 16, 20, 24, 28, 31] | rule | every fourth decoder layer plus the input and the last layer: half the forwards of patching.layer_grid, which on one shared GPU would take about 7 hours for the held-out pass (changed 2026-11-17, before any held-out forward) |
| `sensor_conflict.tests` | held-out, stratified episode bootstrap (strata = task) with n_boot draws, two-sided bootstrap p; Holm within each family: (A) object edit, phi_wrist - phi_agentview != 0 at each of 4 phases; (B) arm edit, phi_proprio != 0 at each of 4 phases; (C) gripper edit, phi_proprio != 0 at each of 4 phases; (D) relevance, mean log(\|Delta_object\| / \|Delta_distractor\|) > 0 at each of 4 phases; (E) arm edit, phi_proprio(corrupted) - phi_proprio(clean) != 0 for blur-8 and noise-128, pooled over phases; (F) probe AUROC > 0.5 at layer 16. Shares of the stale edit, off-axis values, gripper flips and the patching curves are descriptive | rule | every test compares an estimate with its interval on a fixed null; nothing is chosen after seeing held-out data |
| `sensor_conflict.faults` | none: plain policy (paired reference); proprio_offset: the reported end-effector position is 3 cm off in a horizontal direction fixed per episode (seeded); gripper_stuck: the reported finger positions stay at their post-settle (open) values; lag_agentview: the third-person frame of the previous query; lag_wrist: the wrist frame of the previous query; lag_proprio: the proprio of the previous query; lag_all: the whole observation of the previous query (0.4 s latency) | rule | closed-loop versions of the conflicts: a miscalibrated or stuck sensor, one camera out of sync, plain latency; at the first query the previous observation is the current one |
| `sensor_conflict.phase_faults` | freeze_approach: trigger: first query with the end effector within 15 cm of the object to grasp; freeze_grasp: trigger: the query whose chunk first commands closing (gripper chunk mean < 0.5); freeze_transport: trigger: three queries after the grasp trigger; freeze_place: trigger: first query from the grasp on with the object within 15 cm (horizontal) of the goal region (second argument of the goal predicate) | rule | the same observation freeze (the trigger query's observation is fed again at the next two queries, up to 0.8 s stale) placed at four phases, on the pick-and-place tasks; positions are read from the simulator, so the trigger does not depend on what the policy sees. Thresholds from exploratory clean rollouts (2026-11-17): at 10 cm the place trigger fired only at the last query |
| `sensor_conflict.closed_loop_tests` | held-out init states; faults on all 10 tasks, phase faults on sensor_conflict.tasks; success and unsafe outcome (oracle classes) against the paired 'none' run: exact McNemar, Holm over faults within each table; differences with stratified cluster-bootstrap intervals | rule | the same paired design as the corruption sweep |

