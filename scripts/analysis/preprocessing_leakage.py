"""Preprocessing leakage of background-only corruptions into protected model-input pixels.

Usage:  python scripts/analysis/preprocessing_leakage.py  ->  results/preprocessing_leakage/{summary.json, summary.md}

Frames: the post-settle frame of every calibration init state x 10 tasks (benchmark scene), both cameras.
Conditions: every bg_patches ladder level and matched_bg_box (the background-only families), exactly as the
runner builds them (EpisodeCorruptor, masks from LiberoTaskEnv.masks, the cached patch sizes), each as the raw-frame
composite (the default) and as the exact-model-input variant ("-exactmi", composited after preprocessing).
Measured with vlamech.perturbations.leakage: protected raw pixels changed (must be 0), and model-input changes of
protected pixels binned by depth into the protected region. Descriptive; no threshold is applied. Needs no GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vlamech.config import protocol, split_states  # noqa: E402
from vlamech.utils.io import atomic_write_bytes, atomic_write_json  # noqa: E402

OUT = ROOT / "results" / "preprocessing_leakage"


def main():
    from vlamech.envs.libero_env import LiberoTaskEnv
    from vlamech.perturbations.assets import bg_side_lengths
    from vlamech.perturbations.image import CorruptionSpec, EpisodeCorruptor, ladder
    from vlamech.perturbations.leakage import exact_model_input_views, leakage

    suite = protocol("env.suite")
    envs = {}

    def make_env(t):
        if t not in envs:
            envs[t] = LiberoTaskEnv(suite, t, resolution=int(protocol("env.resolution")), segmentation=True,
                                    settle_steps=int(protocol("env.settle_steps")), env_seed=int(protocol("env.env_seed")))
        return envs[t]

    sides = bg_side_lengths(make_env)
    specs = [CorruptionSpec("bg_patches", v, exact_model_input=x) for x in (False, True) for v in ladder("bg_patches")]
    specs += [CorruptionSpec("matched_bg_box", exact_model_input=x) for x in (False, True)]
    center_crop = bool(protocol("policy.center_crop"))
    agg = {}
    for t in range(10):
        env = make_env(t)
        for s in split_states("calibration"):
            frame = env.reset(s)
            masks = env.masks(frame)
            for spec in specs:
                corr = EpisodeCorruptor(spec, f"task{t}-init{s}", sides)
                for cam in ("agentview", "wrist"):
                    raw = getattr(frame, cam)
                    m = masks[cam]
                    img = corr(cam, raw, 0, background_mask=m["background"], target_mask=m["target"])
                    view = (exact_model_input_views(raw, img, ~m["background"], center_crop)
                            if spec.exact_model_input else None)
                    r = leakage(raw, img, ~m["background"], center_crop, corrupted_view=view)
                    a = agg.setdefault((spec.tag, cam), {"frames": 0, "raw_protected_changed": 0, "max_changed_depth": 0,
                                                         "bins": {}, "protected": None, "background": None})
                    a["frames"] += 1
                    a["raw_protected_changed"] += r["raw_protected_changed"]
                    a["max_changed_depth"] = max(a["max_changed_depth"], r["max_changed_depth"] or 0)
                    for name, st in [("protected", r["model"]["protected"]), ("background", r["model"]["background"]),
                                     *[(f"d{k}", v) for k, v in r["model"]["by_depth"].items()]]:
                        tgt = a["bins"].setdefault(name, {"n": 0, "changed": 0, "sum_abs": 0.0, "max_abs": 0})
                        tgt["n"] += st["n"]
                        tgt["changed"] += st["changed"]
                        tgt["sum_abs"] += st["mean_abs"] * st["n"]
                        tgt["max_abs"] = max(tgt["max_abs"], st["max_abs"])
    for e in envs.values():
        e.close()

    out = {"frames_per_condition_camera": len(split_states("calibration")) * 10, "center_crop": center_crop,
           "conditions": {}}
    lines = ["# Background-only corruptions after OFT preprocessing", "",
             "Post-settle frame of every calibration init state x 10 tasks. A model-input pixel is protected if the raw "
             "pixel its centre samples is protected; depth = Chebyshev distance (raw px) to the nearest background "
             "pixel. Changed = any channel differs; |d| in uint8 levels (max over channels).", ""]
    for (tag, cam), a in sorted(agg.items()):
        b = a["bins"]
        row = {k: {"n": v["n"], "changed_frac": v["changed"] / v["n"] if v["n"] else None,
                   "mean_abs": v["sum_abs"] / v["n"] if v["n"] else None, "max_abs": v["max_abs"]} for k, v in b.items()}
        out["conditions"][f"{tag}|{cam}"] = {"frames": a["frames"], "raw_protected_changed": a["raw_protected_changed"],
                                             "max_changed_depth": a["max_changed_depth"], "pixels": row}
        p, bg = row["protected"], row["background"]
        lines.append(f"- **{tag}, {cam}**: raw protected pixels changed {a['raw_protected_changed']}; model input: "
                     f"protected changed {p['changed_frac']:.2%} (mean |d| {p['mean_abs']:.2f}, max {p['max_abs']}), "
                     f"background changed {bg['changed_frac']:.2%} (mean |d| {bg['mean_abs']:.1f}); deepest change "
                     f"{a['max_changed_depth']} px; by depth: " +
                     ", ".join(f"{k[1:]}: {v['changed_frac']:.1%}/{v['max_abs']}" for k, v in row.items()
                               if k.startswith("d") and v["n"]))
    lines += ["", "by depth: `<depth range>: <fraction changed>/<max |d|>`"]
    atomic_write_json(OUT / "summary.json", out)
    atomic_write_bytes(OUT / "summary.md", ("\n".join(lines) + "\n").encode())
    print("\n".join(lines))


if __name__ == "__main__":
    main()
