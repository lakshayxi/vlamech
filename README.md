# Does the robot stop listening?

Visual corruption and instruction steering in a vision-language-action policy. A course project for
Responsible AI: human control through the instruction, safety under degraded sensing, and transparency about where
inside the model control is lost.

When the camera images of a vision-language-action (VLA) policy are corrupted in ways that have nothing to do with
the task (blur, sensor noise, compression, a dark exposure, background clutter), does the instruction still decide
what the robot does? We study OpenVLA-OFT on LIBERO-Goal, where the scene alone never says which task to do, and
ask what happens to success, to how the instruction steers the actions, where inside the network the steering is
lost, whether failure can be seen coming, and whether training-free fixes help. Every test, threshold and
corruption level was fixed before the held-out data it applies to existed (`docs/protocol.md`).

## Main results

All held-out: 300 paired episodes per condition (10 tasks x init states 20-49), 95% cluster-bootstrap intervals.

| Question | Finding |
|---|---|
| Does corruption hurt? | Success falls from 97.3% (clean) to 24.3% (blur 8 px), 31.7% (noise 128), 14.0% (target hidden). A box of the same size on background costs nothing (96.7%); background patches cost 13-15 points even when every task-relevant pixel the network sees is bit-identical to clean. |
| Does the instruction fade? | No. Its effect on the action grows in 12 of 13 conditions (blur 8: x1.39), but the part along the clean task direction falls in all 13 (blur 8: 0.73 of clean). |
| Where is it lost? | Corrupted images change what the instruction positions carry inside the network: injecting corrupted instruction states into a clean run cuts steering to 0.37-0.49 (another episode's clean states: 0.83). Holds at states failing rollouts reached, in failing and succeeding episodes alike. |
| Can it be detected? | Signals from the commanded actions predict failure (chunk turn AUROC 0.74, chunk jump 0.76); the instruction effect does not beat them (0.56). A conformal safe stop lowers unsafe failures in 2 of 6 conditions, at a cost in successes. |
| Can it be fixed? | Not with the training-free methods tried. Instruction guidance at the published strength lowers success everywhere (clean 97.3% to 39.0%). |
| Is it specific to OFT? | No. pi0.5 drops further: 0.0% under blur 8 and 1.3% under noise 128 (clean 98.7%). |

Full tables with intervals are in `results/<experiment>/summary.md`, figures in `results/figures/<experiment>/`.

## Repository

```
configs/      protocol.yaml (every parameter, with its source and justification), frozen_levels.yaml and
              frozen_mitigation.yaml (selections frozen before the held-out runs), models.lock.yaml (pinned weights)
src/vlamech/  envs/ (LIBERO wrapper, episode loop, oracle traces, capture and offline replay)
              models/ (OFT and pi0.5 wrappers), perturbations/ (visual conditions)
              interventions/ (hooks, activation patching, mitigations), metrics/ (outcomes, open-loop metrics,
              monitor signals, statistics), runs/ (experiment rules, manifests, policy server), analysis/
scripts/      run_*.py / run_*.sh (GPU), analyze_*.py (tables), figures_*.py (exploratory figures)
tests/        correctness tests: parity with upstream, hooks, replay, perturbations, statistics, run integrity
results/      summaries and figures produced by the analysis scripts
docs/         protocol.md, perturbations.md, records.md, hyperparameters.md (generated)
```

## Setup

Linux with a CUDA 12 GPU (about 20 GB per OFT process), conda and git.

```bash
bash env/setup_env.sh                 # upstream code at pinned commits into external/, Python env in .envs/vlamech
source env/env.sh                     # caches, EGL rendering, PYTHONPATH; set VLA_CACHE / HF_HOME to relocate
python scripts/download_models.py     # OFT and pi0.5 weights at the pinned commits, verified by SHA-256
bash env/setup_env_lerobot.sh         # optional: separate environment for pi0.5 (.envs/lerobot)
pytest -m "not gpu and not sim"       # CPU tests; drop the filter to run the simulator and GPU tests too
```

pi0.5 needs the tokenizer of the gated `google/paligemma-3b-pt-224`: accept its license on the Hugging Face Hub and
log in before `download_models.py`. Runs are offline (`HF_HUB_OFFLINE=1`).

## Reproducing the results

Run from a clean checkout: every run other than `debug` refuses uncommitted code. Runs resume where they stopped.
`scripts/serve_policy.py` with `scripts/run_served.sh` lets several simulator processes share one model on the GPU.

| Experiment | Run | Analyse |
|---|---|---|
| Clean baseline (2 x 500 episodes) | `run_closed_loop.py --experiment baseline --states all --replicate {0,1}` | `analyze_baseline.py` |
| Instruction reliance (wrong / empty instruction) | `run_closed_loop.py --experiment instruction_reliance --states all --instruction {other_task,empty}` | `analyze_instruction_reliance.py` |
| Severity calibration | `run_severity_calibration.sh` | `analyze_severity_calibration.py`, `measure_passive_drift.py`, `calibrate_oracle.py`, then `freeze_levels.py` |
| Corruption sweep (held-out) | `run_corruption_sweep.sh`, plus a clean repeat with `--replicate 1` | `analyze_corruption_sweep.py`, `figures_trajectories.py` |
| Instruction effect (open loop) | `run_closed_loop.py --experiment clean_capture --states exploratory`, then `run_instruction_effect.py --source <clean run>` | `analyze_instruction_effect.py`, `figures_instruction_effect.py` |
| Activation patching | `run_activation_patching.py --split {exploratory,confirmatory}`, `run_patching_rollouts.py` | `analyze_activation_patching.py`, `analyze_patching_rollouts.py` |
| Mitigation and failure monitor | `run_monitor_and_mitigation.sh` | `analyze_mitigation.py`, `analyze_failure_monitor.py` |
| Cross-model (pi0.5, LeRobot env) | `run_closed_loop.py --experiment cross_model --policy pi05 --states confirmatory --family {identity,blur,noise} ...` | `analyze_cross_model.py` (OFT env) |
| Post-hoc checks | | `analyze_goal_progress.py`, `analyze_steering_and_failure.py` |

Most of the compute is simulation in the corruption sweep and the mitigation runs. Raw records (traces, captures,
videos; about 50 GB) are not in the repository; `docs/records.md` describes them.

## Notes

- The policy sees a left-right mirror of the true camera view (upstream flips the rendered frame); figures show the
  network input unless stated otherwise.
- LIBERO's segmentation mislabels the robot in tabletop scenes and robosuite's anti-aliased segmentation mislabels
  object boundaries; both are handled in `src/vlamech/envs/libero_env.py`.
- The safety measures are simulator proxies (contact force above anything seen in successful clean runs, objects
  leaving the table), not a physical safety model.

## License

MIT (see `LICENSE`). OpenVLA-OFT, LIBERO and LeRobot are used unmodified at pinned commits under their own licenses.
