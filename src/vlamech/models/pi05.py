"""pi0.5 (LeRobot, flow-matching action expert) policy wrapper with the interface the closed-loop runner uses.

Used by: scripts/run/closed_loop.py (--policy pi05), tests/test_pi05.py. Runs only in the LeRobot environment
(env/setup_env_lerobot.sh); the OFT environment cannot import lerobot (different transformers build).

Checkpoint: lerobot/pi05_libero_finetuned_v044 (trained on HuggingFaceVLA/libero = all four LIBERO suites, 256x256
frames). Tokenizer: google/paligemma-3b-pt-224. Upstream code: lerobot @ 8fff0fde (tag v0.4.4), pinned in
env/setup_env_lerobot.sh. Every input/output convention is LeRobot's own LIBERO evaluation
(src/lerobot/scripts/lerobot_eval.py `rollout`), reproduced by CALLING LeRobot's code wherever possible:

  observation    lerobot/envs/libero.py LiberoEnv._format_raw_obs: raw robosuite images "agentview_image" -> image,
                 "robot0_eye_in_hand_image" -> image2 (not rotated there); robot_state = eef pos, eef quat (xyzw),
                 gripper qpos. We hand LeRobot the raw (un-rotated) frames: our Frame images are rotated 180 deg
                 (vlamech.envs.libero_env), so they are rotated back ([::-1, ::-1], exact) before LeRobot's steps.
  -> tensors     lerobot/envs/utils.py preprocess_observation: uint8 HWC -> float32 CHW in [0, 1]
  -> env step    lerobot/processor/env_processor.py LiberoProcessorStep: flips H and W (180 deg, the dataset
                 convention), state = [eef_pos(3), axis-angle(eef_quat)(3) computed in float32, gripper_qpos(2)]
  -> task        lerobot/envs/utils.py add_envs_task: observation["task"] = [instruction]
  -> policy pre  checkpoint policy_preprocessor.json (loaded with make_pre_post_processors): MEAN_STD state
                 normalization, state discretized into 256 bins inside the prompt
                 "Task: <instruction>, State: <bins>;\\nAction: " (policies/pi05/processor_pi05.py), PaliGemma
                 tokenizer, max 200 tokens, right padding
  -> model       policies/pi05/modeling_pi05.py: images resize_with_pad (bilinear) to 224x224, scaled to [-1, 1];
                 the third configured camera (empty_camera_0) is a masked -1 image; flow matching from Gaussian
                 noise (B, chunk_size=50, max_action_dim=32) with num_inference_steps=10 Euler steps; actions
                 truncated to 7 dims
  -> policy post checkpoint policy_postprocessor.json: MEAN_STD unnormalization (env-space actions; the gripper is
                 continuous in robosuite's [-1, 1] convention, not OFT's [0, 1] -> binarize -> invert; the runner
                 therefore steps the env with raw actions, LiberoTaskEnv.step(raw=True))
  execution      select_action's queue: the first n_action_steps actions of each 50-action chunk are executed open
                 loop before the next query (modeling_pi05.py PI05Policy.select_action). n_action_steps = 10, the
                 value of LeRobot's published LIBERO evaluation of this checkpoint (docs/source/libero.mdx:
                 lerobot-eval ... --policy.n_action_steps=10, 98.0% on LIBERO-Goal); the checkpoint config says 50
  episode        10 no-op settle steps (LiberoEnv num_steps_wait=10), max 300 steps for libero_goal
                 (TASK_SUITE_MAX_STEPS) - identical to the OFT protocol values our env already uses

Loading note: LeRobot prints "Warning: Could not remap state dict keys ... Missing key(s):
...language_model.embed_tokens.weight". It is cosmetic: Gemma ties the token embedding to lm_head (one Parameter
object), the checkpoint stores lm_head, and PyTorch's strict load copies every present tensor before raising on the
alias. Verified: embed_tokens IS lm_head and equals the checkpoint tensor; all 812 checkpoint tensors equal the
loaded ones (tests/test_pi05.py::test_weights_fully_loaded).

Deliberate differences from lerobot_eval (each fixed here, not configurable by accident):
  * simulator: OUR LiberoTaskEnv (pinned external/LIBERO, fixture-layout replay, seed 0), the same simulator every
    other policy in this study acts in; LeRobot's eval builds its own env (hf-libero).
  * camera resolution 256x256 (the training frames' resolution, and the frames OFT receives, so a corruption is
    pixel-identical across policies); lerobot_eval's LiberoEnv config default renders 360x360.
  * torch.compile off (the checkpoint config sets compile_model=True, mode max-autotune: autotuned kernels make
    numerics depend on the tuning run; eager is reproducible).
  * flow-matching noise is drawn from an explicit CPU torch.Generator seeded with
    derive_seed("pi05_noise", namespace, task, init state, query step) instead of the global CUDA RNG: same
    distribution N(0, 1), but every query is reproducible, and every visual condition of the same (task, init,
    step) gets the same noise (common random numbers, like the corruptions). sample(k) draws k futures for one
    input; future 0 is exactly the closed-loop action: the generator fills the noise tensor sequentially, and each
    future is computed as its own batch-of-1 forward pass, because bf16 GPU kernels give batch-size-dependent
    results (measured: up to 2e-3 per action entry for k = 2-8, as large as the spread between futures near the
    start state). sample(k, batched=True) is faster but not bit-identical to the closed loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from vlamech.utils.seeding import derive_seed

REPO_ID = "lerobot/pi05_libero_finetuned_v044"
TOKENIZER_REPO = "google/paligemma-3b-pt-224"
DOWNLOAD_ALLOW = ["*.json", "*.safetensors"]
NOISE_NAMESPACE = "pi05_noise"
ACTION_DIM = 7


@dataclass
class Pi05Input:
    """One preprocessed query (batch of 1, on the policy device). Created by Pi05Policy.prepare()."""

    instruction: str
    batch: dict
    state: np.ndarray  # (8,) float32 state as LeRobot computed it (before normalization)
    noise_seed: Optional[int]


def raw_observation(agentview: np.ndarray, wrist: np.ndarray, eef_pos: np.ndarray, eef_quat: np.ndarray,
                    gripper_qpos: np.ndarray) -> dict:
    """The observation dict LeRobot's LiberoEnv returns (batched, n_envs = 1), from policy-orientation frames.

    Our frames are rotated 180 deg; LiberoEnv returns the raw robosuite frames, which LiberoProcessorStep rotates.
    Only the robot_state fields LiberoProcessorStep reads are filled (it discards the rest)."""
    for name, img in (("agentview", agentview), ("wrist", wrist)):
        if not (img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3):
            raise TypeError(f"{name}: camera frames must be uint8 HxWx3")
    raw = lambda a: np.ascontiguousarray(a[::-1, ::-1])[None]  # noqa: E731  undo the 180 deg rotation
    f64 = lambda a, n: np.asarray(a, np.float64).reshape(1, n)  # noqa: E731  robosuite observables are float64
    return {"pixels": {"image": raw(agentview), "image2": raw(wrist)},
            "robot_state": {"eef": {"pos": f64(eef_pos, 3), "quat": f64(eef_quat, 4)},
                            "gripper": {"qpos": f64(gripper_qpos, 2)}}}


def noise_seed(namespace: str, task_id: int, init_state: int, step: int) -> int:
    return derive_seed(NOISE_NAMESPACE, namespace, int(task_id), int(init_state), int(step))


def noise(seed: int, k: int, chunk_size: int, max_action_dim: int) -> torch.Tensor:
    """(k, chunk_size, max_action_dim) float32 N(0, 1), CPU generator: future j is the j-th block of the stream."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn((k, chunk_size, max_action_dim), generator=g, dtype=torch.float32)


class Pi05Policy:
    family = "pi05"
    env_space_actions = True  # actions go to robosuite as they are (no OFT gripper processing)
    center_crop = None  # an OFT preprocessing option; meaningless here

    def __init__(self, repo_id: str = REPO_ID, revision: Optional[str] = None, device: str = "cuda:0",
                 seed_namespace: str = "rep0", n_action_steps: int = 10):
        if repo_id != REPO_ID:
            raise ValueError(f"unsupported checkpoint {repo_id}")
        if revision is None:
            raise ValueError("pass a pinned revision; never run on an unpinned 'main'")
        from huggingface_hub import snapshot_download
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05 import PI05Policy
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.env_processor import LiberoProcessorStep

        self.repo_id, self.device = repo_id, torch.device(device)
        self.path = Path(snapshot_download(repo_id, revision=revision, allow_patterns=DOWNLOAD_ALLOW))
        self.revision = self.path.name
        if not self.revision.startswith(revision):
            raise RuntimeError(f"snapshot {self.revision} does not match pinned revision {revision}")
        cfg = PreTrainedConfig.from_pretrained(str(self.path))
        cfg.compile_model = False  # see module docstring
        cfg.n_action_steps = int(n_action_steps)  # as lerobot-eval --policy.n_action_steps (module docstring)
        cfg.device = str(self.device)
        self.config = cfg
        self.model = PI05Policy.from_pretrained(str(self.path), config=cfg).to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(
            cfg, str(self.path), preprocessor_overrides={"device_processor": {"device": str(self.device)}})
        self.env_pre = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])  # make_env_pre_post_processors, LIBERO
        self.actions_per_query = int(cfg.n_action_steps)  # actions returned (and executed) per query
        self.chunk_size = int(cfg.chunk_size)
        self.max_action_dim = int(cfg.max_action_dim)
        self.num_inference_steps = int(cfg.num_inference_steps)
        self.seed_namespace = seed_namespace
        self._context: Optional[tuple] = None

    def check_settings(self, settings: dict) -> None:
        """Fail fast if the loaded checkpoint config disagrees with the protocol values the run records."""
        want = {"n_action_steps": int(settings["policy.n_action_steps"]),
                "num_inference_steps": int(settings["policy.num_inference_steps"]),
                "dtype": settings["policy.dtype"], "compile_model": bool(settings["policy.compile_model"]),
                "chunk_size": int(settings["policy.chunk_size"])}
        have = {k: getattr(self.config, k) for k in want}
        if have != want:
            raise RuntimeError(f"pi0.5 checkpoint config {have} != protocol {want}")

    # ------------------------------------------------------------------ query context (noise seed)
    def set_query_context(self, task_id: int, init_state: int, step: int) -> None:
        """Called by the episode loop before each query; fixes the noise of that query."""
        self._context = (int(task_id), int(init_state), int(step))

    def _seed(self) -> int:
        if self._context is None:
            raise RuntimeError("no query context: call set_query_context(task, init, step) first")
        return noise_seed(self.seed_namespace, *self._context)

    # ------------------------------------------------------------------ inputs
    def prepare(self, agentview: np.ndarray, wrist: np.ndarray, proprio: np.ndarray, instruction: str,
                eef_quat: Optional[np.ndarray] = None, images_224=None) -> Pi05Input:
        """LeRobot's eval preprocessing of one frame. eef_quat (robosuite xyzw) is needed for LeRobot's exact state:
        it computes the axis-angle in float32 from the quaternion. Without it the state is taken from our proprio
        (axis-angle from robosuite's float64 quat2axisangle), which can differ in the last float32 bits."""
        if images_224 is not None:
            raise NotImplementedError("the exact-model-input variant is defined for OFT's preprocessing only")
        from lerobot.envs.utils import preprocess_observation
        from lerobot.utils.constants import OBS_STATE

        proprio = np.asarray(proprio, np.float64)
        if eef_quat is not None:
            obs = preprocess_observation(raw_observation(agentview, wrist, proprio[:3], eef_quat, proprio[6:8]))
        else:
            obs = preprocess_observation({"pixels": raw_observation(agentview, wrist, proprio[:3], [0, 0, 0, 1],
                                                                    proprio[6:8])["pixels"]})
        obs["task"] = [instruction]
        obs = self.env_pre(obs)
        if eef_quat is None:
            obs[OBS_STATE] = torch.from_numpy(proprio[None]).float()
        state = obs[OBS_STATE][0].cpu().numpy().astype(np.float32)
        batch = self.pre(obs)
        return Pi05Input(instruction, batch, state, self._seed() if self._context is not None else None)

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def sample(self, inp: Pi05Input, k: int = 1, seed: Optional[int] = None, batched: bool = False) -> dict:
        """k futures for one input, whole chunks: {"actions": (k, 50, 7) env space, "actions_norm", "seed"}.
        Default: one batch-of-1 pass per future (future 0 == forward() bit for bit); batched=True: one pass of
        batch k (faster, numerically different, see module docstring)."""
        seed = inp.noise_seed if seed is None else seed
        if seed is None:
            raise RuntimeError("no noise seed: set the query context or pass seed=")
        z = noise(seed, k, self.chunk_size, self.max_action_dim).to(self.device)
        if batched:
            batch = {key: (v.repeat(k, *([1] * (v.dim() - 1))) if isinstance(v, torch.Tensor) and v.shape[:1] == (1,)
                           else v * k if isinstance(v, list) and len(v) == 1 else v) for key, v in inp.batch.items()}
            norm = self.model.predict_action_chunk(batch, noise=z)  # (k, 50, 7) normalized
        else:
            norm = torch.cat([self.model.predict_action_chunk(inp.batch, noise=z[j:j + 1]) for j in range(k)])
        acts = self.post(norm)
        return {"actions": acts.float().cpu().numpy().astype(np.float64), "actions_norm": norm.float().cpu().numpy(),
                "seed": int(seed)}

    def forward(self, inp: Pi05Input) -> dict:
        """The closed-loop query: future 0 of sample(); the first n_action_steps actions are executed."""
        out = self.sample(inp, 1)
        n = self.actions_per_query
        return {"actions": out["actions"][0, :n], "actions_norm": out["actions_norm"][0, :n], "seed": out["seed"]}

    # ------------------------------------------------------------------ display
    @staticmethod
    def view(img: np.ndarray, size: int = 224) -> np.ndarray:
        """The image the network receives for one camera frame (uint8 for display): the policy-orientation frame
        (LiberoProcessorStep's flip cancels our rotation) resized with padding exactly as modeling_pi05 does."""
        from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch

        t = torch.from_numpy(np.ascontiguousarray(img)).float().div(255.0)[None]  # (1, H, W, 3), [0, 1]
        if t.shape[1:3] != (size, size):
            t = resize_with_pad_torch(t, size, size)
        return (t[0].clamp(0, 1) * 255.0).round().to(torch.uint8).numpy()
