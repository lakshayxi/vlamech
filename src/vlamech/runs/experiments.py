"""Which runs each experiment may make, and the resolved runtime settings, checked before anything is loaded.

resolve_settings()  every runtime parameter, read once from configs/protocol.yaml. With policy="pi05" the policy
                    settings come from protocol policies.pi05. Parameters that fix comparability with the published
                    OFT evaluation must equal the upstream values in every experiment except "debug".
EXPERIMENTS         per experiment: the init-state split, allowed corruption families and values, instruction modes,
                    whether the frozen selection is required, the mitigation mode and the records every run must save.
preflight()         validates a requested run against its experiment; returns the frozen selection it uses.
"debug" is the only experiment that may run uncommitted code or arbitrary settings; it writes to results/debug/.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import yaml

from vlamech.config import FROZEN_MITIGATION, frozen, protocol, split_states
from vlamech.interventions.mitigation import label as mitigation_label

# Fixed by the upstream OFT LIBERO evaluation (run_libero_eval.py / get_libero_env); changing them breaks
# comparability with the published numbers.
UPSTREAM = {
    "env.resolution": 256, "env.env_seed": 0, "env.settle_steps": 10, "env.max_steps": 300, "env.open_loop_steps": 8,
    "policy.center_crop": True, "policy.global_seed": 7,
}
# What the OFT wrapper supports at all (bf16, SDPA, two images).
IMPLEMENTATION = {"policy.dtype": "bfloat16", "policy.attn_implementation": "sdpa", "policy.num_images": 2}
POLICIES = ("oft", "pi05")
PI05_KEYS = ("checkpoint", "lerobot_version", "dtype", "chunk_size", "n_action_steps", "num_inference_steps",
             "compile_model", "noise_seed")


def resolve_settings(experiment: str, policy: str = "oft") -> dict:
    """Every runtime parameter of a run. For pi0.5 the policy keys are replaced by protocol policies.pi05 and the
    environment keys stay (same simulator protocol)."""
    s = {k: protocol(k) for k in (*UPSTREAM, *IMPLEMENTATION, "policy.checkpoint", "env.suite")}
    for k, v in IMPLEMENTATION.items():
        if s[k] != v:
            raise ValueError(f"protocol {k}={s[k]!r} is not supported by the implementation (only {v!r})")
    if experiment != "debug":
        off = {k: (s[k], v) for k, v in UPSTREAM.items() if s[k] != v}
        if off:
            raise ValueError(f"protocol differs from the upstream values required for comparability: {off}")
    if policy == "oft":
        return s
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; known: {POLICIES}")
    s = {k: v for k, v in s.items() if not k.startswith("policy.") or k == "policy.global_seed"}
    s |= {f"policy.{k}": protocol(f"policies.{policy}.{k}") for k in PI05_KEYS}
    s["env.open_loop_steps"] = int(s["policy.n_action_steps"])
    if int(protocol(f"policies.{policy}.resolution")) != int(s["env.resolution"]):
        raise ValueError("pi0.5 must see the same frames as OFT (policies.pi05.resolution == env.resolution)")
    s["policy.family"], s["policy.name"] = protocol(f"policies.{policy}.family"), policy
    return s


@dataclass(frozen=True)
class Experiment:
    split: Optional[str]  # init-state split that must be used exactly; None = any
    families: Optional[frozenset]  # allowed corruption families; None = any
    value_source: str  # "none" | "ladder" | "frozen" | "any"
    instructions: frozenset
    requires_freeze: bool = False
    allow_dirty: bool = False
    # "none" | "grid" (an arm of protocol mitigation.arms) | "frozen" | "monitor" (plain policy, instruction effect logged)
    mitigation: str = "none"
    # evidence every run must save: "video" (scene + exact network input), "actions" (every chunk), "captures"
    # (every query's exact inputs, for offline replay)
    records: frozenset = frozenset()
    policies: frozenset = frozenset({"oft"})


_ALL = frozenset({"video", "actions", "captures"})
EXPERIMENTS = {
    "baseline": Experiment("all", frozenset({"identity"}), "none", frozenset({"own"})),
    "instruction_reliance": Experiment("all", frozenset({"identity"}), "none", frozenset({"other_task", "empty"})),
    "severity_calibration": Experiment("calibration", None, "ladder", frozenset({"own"}), records=frozenset({"video"})),
    # clean exploratory rollouts with every query captured; corruptions are applied offline
    "clean_capture": Experiment("exploratory", frozenset({"identity"}), "none", frozenset({"own"}), records=_ALL),
    "corruption_sweep": Experiment("confirmatory", None, "frozen", frozenset({"own"}), requires_freeze=True, records=_ALL),
    "mitigation_selection": Experiment("exploratory", None, "frozen", frozenset({"own"}), requires_freeze=True,
                                       mitigation="grid", records=frozenset({"video", "actions"})),
    "mitigation_test": Experiment("confirmatory", None, "frozen", frozenset({"own"}), requires_freeze=True,
                                  mitigation="frozen", records=_ALL),
    # plain policy with the instruction effect logged, on single-camera conditions no earlier analysis has seen
    "failure_monitor": Experiment("confirmatory", None, "frozen", frozenset({"own"}), requires_freeze=True,
                                  mitigation="monitor", records=frozenset({"actions"})),
    "cross_model": Experiment("confirmatory", None, "frozen", frozenset({"own"}), requires_freeze=True,
                              records=frozenset({"actions"}), policies=frozenset({"pi05"})),
    # clean check of a comparison model against its published success, before any corrupted run
    "model_check": Experiment("calibration", frozenset({"identity"}), "none", frozenset({"own"}),
                              records=frozenset({"video", "actions"}), policies=frozenset({"pi05"})),
    "debug": Experiment(None, None, "any", frozenset({"own", "other_task", "empty"}), allow_dirty=True, mitigation="any",
                        policies=frozenset(POLICIES)),
}


def experiment_states(split: str, n_states: int) -> list:
    return list(range(n_states)) if split == "all" else split_states(split)


def preflight(experiment: str, states: list, spec, instruction: str, n_states: int,
              mitigation: Optional[dict] = None, policy: str = "oft") -> dict:
    """Validate a requested run; returns {"frozen": <selection used> or None}. Raises on any violation."""
    if experiment not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {experiment!r}; known: {sorted(EXPERIMENTS)}")
    ex = EXPERIMENTS[experiment]
    if policy not in ex.policies:
        raise ValueError(f"{experiment} does not run policy {policy!r}")
    if ex.mitigation == "none" and mitigation:
        raise ValueError(f"{experiment} runs no mitigation")
    if ex.mitigation in ("grid", "frozen", "monitor") and not mitigation:
        raise ValueError(f"{experiment} needs --mitigation")
    if ex.mitigation == "grid":
        if mitigation_label(mitigation) not in [*protocol("mitigation.arms"), *protocol("mitigation.followup_arms")]:
            raise ValueError(f"{mitigation_label(mitigation)} is not on the protocol grid of arms")
        if mitigation.get("safe_stop_below") is not None:
            raise ValueError("the safe-stop threshold is derived after the arm selection, not during it")
    if ex.mitigation == "monitor" and mitigation_label(mitigation) != "cag:1":
        raise ValueError(f"{experiment} runs the plain policy with the instruction effect logged (cag:1)")
    if ex.split is not None:
        expected = experiment_states(ex.split, n_states)
        if sorted(states) != expected:
            raise ValueError(f"{experiment} must use exactly the {ex.split!r} split {expected[0]}-{expected[-1]}")
    if ex.families is not None and spec.family not in ex.families:
        raise ValueError(f"{experiment} does not run family {spec.family!r}")
    if instruction not in ex.instructions:
        raise ValueError(f"{experiment} does not run instruction mode {instruction!r}")
    if ex.value_source == "ladder" and spec.value is not None:
        from vlamech.metrics.selection import next_level
        from vlamech.perturbations.image import LADDER_KEY, ladder

        if spec.family in LADDER_KEY:
            lad = ladder(spec.family)
            ext = protocol("corruption.ladder_extension")[spec.family]
            allowed = lad + [v for v in [next_level(lad, ext["spacing"], ext["bound"])] if v is not None]
            if spec.value not in allowed:  # the ladder plus the selection rule's one-step extension
                raise ValueError(f"{spec.family} value {spec.value} is not on its protocol ladder or its extension {allowed}")
    if ex.value_source == "none" and spec.value is not None:
        raise ValueError(f"{experiment} runs no corruption value")
    if not ex.requires_freeze:
        return {"frozen": None}
    sel = frozen()
    fam = sel.get("corruption", {}).get(spec.family)
    if fam is None:
        raise ValueError(f"family {spec.family!r} was not frozen")
    if spec.family != "identity" and spec.value not in fam.get("levels", []):
        raise ValueError(f"{spec.family} value {spec.value} is not a frozen level {fam.get('levels')}")
    if "oracle" not in sel:
        raise ValueError("the frozen selection has no oracle thresholds")
    if getattr(spec, "exact_model_input", False) and spec.family not in sel.get("variants", {}).get("exact_model_input", []):
        raise ValueError(f"the exact-model-input variant of {spec.family!r} was not frozen")
    out = {"corruption": {spec.family: fam}, "oracle": sel["oracle"]}
    if ex.mitigation == "frozen":
        if not FROZEN_MITIGATION.exists():
            raise ValueError(f"{FROZEN_MITIGATION} missing: the arm selection has not been frozen")
        chosen = yaml.safe_load(FROZEN_MITIGATION.read_text())
        if mitigation_label(mitigation) not in chosen["arms"]:
            raise ValueError(f"mitigation {mitigation_label(mitigation)} is not a frozen arm {chosen['arms']}")
        out["mitigation"] = chosen
    return {"frozen": out}
