"""Rollout videos: what happened in the scene next to exactly what the network received.

Used by: scripts/run/closed_loop.py (--record-video).

One video frame per control step (20 Hz, played in real time), three panels:
  scene         the third-person camera as a person would see it: the policy-orientation frame flipped
                left-right (the policy input is a left-right mirror of the true view)
  robot input   policy_view() of the (possibly corrupted) images fed at the most recent query: the exact 224x224
                network input, third-person and wrist. The policy re-queries every 8 steps (OFT; pi0.5: 10), so these panels only
                change at query steps (marked "new observation"): between queries the robot is acting on a held
                image, as in the real loop
A text band shows the instruction, the condition, the step and the goal predicates that currently hold.
H.264 / yuv420p so the files play in browsers and presentation software.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

PANEL = 256
GAP = 8
HEAD, FOOT = 56, 44
WIDTH = 3 * PANEL + 4 * GAP  # 800, a multiple of 16 (H.264 macroblocks)
HEIGHT = HEAD + PANEL + FOOT  # 356 -> padded to 368
HEIGHT_PAD = -(-HEIGHT // 16) * 16
FPS = 20  # LIBERO control rate
HOLD_FINAL = 2 * FPS  # presentation only: the final frame is shown for 2 s


def _font(size: int, bold: bool = False):
    import matplotlib
    from PIL import ImageFont

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name), size)


class RolloutVideo:
    def __init__(self, path: Path, instruction: str, condition: str, task_names: dict, max_steps: int,
                 center_crop: bool = True, inputs: tuple = ("agentview", "wrist"), view_fn=None):
        import imageio.v2 as imageio

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = self.path.with_name(f".{self.path.stem}.tmp.mp4")
        self.w = imageio.get_writer(self.tmp, fps=FPS, codec="libx264", pixelformat="yuv420p", quality=8,
                                    macro_block_size=16, ffmpeg_log_level="error")
        self.instruction, self.condition, self.names, self.max_steps = instruction, condition, task_names, max_steps
        self.center_crop, self.inputs = center_crop, tuple(inputs)  # cameras the policy actually takes
        self.view_fn = view_fn  # frame -> exact network image; None = OFT's policy_view (other policies pass theirs)
        self._seen: Optional[dict] = None
        self._f = {"h": _font(15, True), "s": _font(12), "b": _font(12, True)}

    def add(self, step: int, frame, fed: Optional[dict], fresh: bool, goals: dict, final: Optional[str] = None):
        from PIL import Image, ImageDraw

        if fresh or self._seen is None:
            view = self.view_fn
            if view is None:
                from vlamech.models.oft import policy_view

                view = lambda img: np.asarray(policy_view(img, self.center_crop))  # noqa: E731
            # fed images are raw frames, or already the 224x224 network input (exact-model-input variant)
            self._seen = {c: (None if not fed else fed[c] if fed[c].shape[:2] == (224, 224)
                              else np.asarray(view(fed[c]))) for c in ("agentview", "wrist")}
        canvas = Image.new("RGB", (WIDTH, HEIGHT_PAD), (250, 250, 248))
        d = ImageDraw.Draw(canvas)
        panels = [(np.ascontiguousarray(frame.agentview[:, ::-1]), "scene (true camera view)"),
                  (self._seen["agentview"], "robot input: third-person"),
                  (self._seen["wrist"], "robot input: wrist" if "wrist" in self.inputs else "wrist (NOT a model input)")]
        for k, (img, label) in enumerate(panels):
            x = GAP + k * (PANEL + GAP)
            if img is not None:
                canvas.paste(Image.fromarray(img).resize((PANEL, PANEL), Image.NEAREST), (x, HEAD))
            d.text((x, HEAD - 17), label, fill=(60, 60, 60), font=self._f["s"])
        d.text((GAP, 6), f'"{self.instruction}"' if self.instruction else "(empty instruction)", fill=(20, 20, 20),
               font=self._f["h"])
        d.text((GAP, 24), f"condition: {self.condition}", fill=(90, 90, 90), font=self._f["s"])
        if fresh:
            x = GAP + PANEL + GAP
            d.rectangle([x, HEAD, x + 2 * PANEL + GAP - 1, HEAD + PANEL - 1], outline=(230, 120, 0), width=3)
            d.text((WIDTH - 150, 24), "new observation", fill=(230, 120, 0), font=self._f["b"])
        y = HEAD + PANEL + 6
        d.text((GAP, y), f"step {step}/{self.max_steps}  ({step / FPS:.1f} s)", fill=(20, 20, 20), font=self._f["b"])
        held = [self.names[t] for t, ok in sorted(goals.items()) if ok]
        d.text((GAP + 170, y), "goals holding: " + ("; ".join(held) if held else "none"), fill=(40, 40, 40),
               font=self._f["s"])
        if final:
            d.text((GAP, y + 18), final, fill=(0, 120, 60) if final.startswith("SUCCESS") else (180, 30, 30),
                   font=self._f["b"])
        for _ in range(HOLD_FINAL if final else 1):  # hold the last frame so the result can be read
            self.w.append_data(np.asarray(canvas))

    def close(self, keep: bool = True):
        self.w.close()
        if keep:
            self.tmp.replace(self.path)
        else:
            self.tmp.unlink(missing_ok=True)
