"""What the closed loop fed the policy at a captured query, re-created offline (open-loop and patching passes).

recreate() applies a condition to a captured clean query exactly as the closed loop would have: same corruptor key
and step, the captured masks, and for the exact-model-input variant the composite after preprocessing.
"""
from __future__ import annotations

from types import SimpleNamespace

from vlamech.config import protocol
from vlamech.envs.episode import exact_views
from vlamech.perturbations.image import NEEDS_BACKGROUND, NEEDS_TARGET, EpisodeCorruptor


def instructions_for(task: int, suite) -> list:
    """[own, the 9 others in task order, ""] (protocol metrics.alternative_instructions = all_other_tasks)."""
    if protocol("metrics.alternative_instructions") != "all_other_tasks":
        raise RuntimeError("metrics.alternative_instructions is not all_other_tasks")
    own = suite.get_task(task).language
    others = [suite.get_task(t).language for t in range(suite.n_tasks) if t != task]
    return [own, *others, ""]


def recreate(corr: EpisodeCorruptor, cap: dict, k: int, center_crop: bool):
    """(agentview, wrist, images_224 or None) the closed loop would have fed under corr.spec at captured query k.

    `corr` must be one corruptor per (episode, condition), called for k = 0, 1, ... in order: episode-level draws
    (patch layout at the first query, matched-box placement relative to the previous query) are state, exactly as in
    the closed loop (tests/test_instruction_effect.py)."""
    spec = corr.spec
    step = int(cap["step"][k])
    masks = {}
    if spec.family in NEEDS_BACKGROUND | NEEDS_TARGET:
        masks = {c: {"background": cap[f"mask_{c}_background"][k], "target": cap[f"mask_{c}_target"][k]}
                 for c in ("agentview", "wrist")}
    imgs = {}
    for cam in ("agentview", "wrist"):
        m = masks.get(cam, {})
        imgs[cam] = corr(cam, cap[cam][k], step, background_mask=m.get("background"), target_mask=m.get("target"))
    if spec.exact_model_input:
        from PIL import Image

        frame = SimpleNamespace(agentview=cap["agentview"][k], wrist=cap["wrist"][k])
        v = exact_views(frame, imgs, masks, spec.cameras, center_crop)
        return imgs["agentview"], imgs["wrist"], tuple(Image.fromarray(v[c]) for c in ("agentview", "wrist"))
    return imgs["agentview"], imgs["wrist"], None
