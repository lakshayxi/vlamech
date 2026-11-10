# Run records

Every closed-loop run writes to `results/<experiment>/<run>/`. Raw records are large and are not in the repository;
the summaries and figures in `results/` are produced from them by the analysis scripts.

| File | Content |
|---|---|
| `manifest.json` | written once: the request, resolved settings, frozen selection, protocol and asset hashes, verified code, upstream and model identity, and its digest |
| `sessions.jsonl` | one record per process that wrote to the run (time, host, shard, code identity) |
| `episodes*.jsonl` | one record per episode, stamped with the manifest digest; outcome, goals reached, displacements, contact forces |
| `traces/<key>.npz` | the per-step oracle trace (goals, object and joint positions, contacts); its SHA-256 is in the record |
| `actions/<key>.npz` | every commanded action chunk (normalized) and the step it was queried |
| `captures/<key>.npz` | every query's exact inputs: raw and fed frames, segmentation masks, proprio, simulator state |
| `videos/<key>.mp4` | the scene and the exact network inputs, with the goals that hold at each step |

What each experiment records is fixed in `vlamech.runs.experiments.EXPERIMENTS` (a forgotten flag cannot lose data).
Outcome labels are computed after collection from the traces with the frozen thresholds, so no outcome depends on
thresholds chosen while data were being collected.

## Guarantees and the tests behind them

| Guarantee | Test |
|---|---|
| episodes reproduce upstream's evaluation bit-exactly, in any order | `tests/test_libero_env.py` |
| our OFT wrapper gives the upstream action on the same input | `tests/test_oft_parity.py` (GPU) |
| actions from the policy server equal in-process actions | `tests/test_policy_server.py` |
| an offline forward on a capture equals the recorded action | `tests/test_capture_parity.py` (GPU) |
| a corruption re-created offline equals what the closed loop fed | `tests/test_instruction_effect.py`, `tests/test_trace.py` |
| hooks without interventions leave actions unchanged; a full clean clamp gives the clean action | `tests/test_hooks.py`, `tests/test_patching.py` (GPU) |
| background-only conditions never change a protected camera pixel | `tests/test_perturbations.py` |
| a run resumes only with an identical manifest; one process owns each shard | `tests/test_runs.py` |
| analyses refuse incomplete runs, foreign records or uncommitted code | `vlamech.analysis.ingest.require_complete` |

Tests marked `gpu` or `sim` need a GPU or the simulator; `pytest -m "not gpu and not sim"` runs the rest on CPU.
The renderer (EGL) is not bit-deterministic: on rare frames one wrist pixel differs by one intensity level, which
can change a bf16 action. Repeated clean runs of the 300 held-out pairs gave the same outcome in 300 of 300.
