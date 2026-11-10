"""OpenVLA-OFT policy wrapper for interpretability experiments.

Used by: every experiment that queries the policy (closed loop, open loop, patching, monitor).

What it does
  * Loads the checkpoint with the *reviewed local* classes from external/openvla-oft (no
    trust_remote_code, no rewriting of checkpoint files), plus the L1 action head and proprio projector.
  * Reproduces upstream `predict_action` (L1-regression path) step by step, but returns the
    *normalized* actions, the token layout, and (optionally) cached vision features, so that
      - many instructions can be evaluated on one observation without recomputing the vision tower,
      - interventions can address token groups by name (see TokenLayout).
  * Never mutates its inputs (upstream get_vla_action overwrites obs["state"] in place).
  * Fails fast on silent-mismatch hazards: wrong platform constants, non-SDPA attention (the
    bidirectional mask exists only in the fork's SDPA path), unexpected sequence length.

Parity with upstream `OpenVLAForActionPrediction.predict_action` is tested in tests/test_oft_parity.py.

What you can change: `device`, dtype (bf16 is what OFT was evaluated with; fp32 changes numerics),
`center_crop` (OFT LIBERO checkpoints were trained with random-crop augmentation and evaluated with
center_crop=True; keep it).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

# Upstream's constants module picks the robot platform by searching sys.argv for "aloha"/"bridge"/"libero"
# and defaults to LIBERO. We do not touch sys.argv; we verify the result and refuse to run otherwise
# (e.g. a script path containing "bridge" would silently switch the action chunk size).
from prismatic.vla import constants as _C  # noqa: E402

if (_C.NUM_ACTIONS_CHUNK, _C.ACTION_DIM, _C.PROPRIO_DIM) != (8, 7, 8) or \
        _C.ACTION_PROPRIO_NORMALIZATION_TYPE != _C.NormalizationType.BOUNDS_Q99:
    raise RuntimeError(f"prismatic constants resolved to platform {_C.ROBOT_PLATFORM}, not LIBERO; "
                       "check sys.argv for 'aloha'/'bridge'")

NUM_ACTIONS_CHUNK, ACTION_DIM, PROPRIO_DIM = 8, 7, 8
N_ACTION_SLOTS = NUM_ACTIONS_CHUNK * ACTION_DIM  # 56
EMPTY_TOKEN_ID = 29871  # the "" token OFT appends after "Out:"
PROMPT_TEMPLATE = "In: What action should the robot take to {task}?\nOut:"
# No *.py: we never execute code shipped with a checkpoint. No lora_adapter/: the weights are already merged.
DOWNLOAD_ALLOW = ["*.json", "*.safetensors", "*.model", "*.pt"]
DOWNLOAD_IGNORE = ["lora_adapter/*"]

CHECKPOINTS = {
    # repo id -> (action head file, proprio projector file, unnorm key)
    "moojink/openvla-7b-oft-finetuned-libero-goal": ("action_head--50000_checkpoint.pt", "proprio_projector--50000_checkpoint.pt", "libero_goal_no_noops"),
    "moojink/openvla-7b-oft-finetuned-libero-spatial": ("action_head--150000_checkpoint.pt", "proprio_projector--150000_checkpoint.pt", "libero_spatial_no_noops"),
    "moojink/openvla-7b-oft-finetuned-libero-object": ("action_head--150000_checkpoint.pt", "proprio_projector--150000_checkpoint.pt", "libero_object_no_noops"),
    "moojink/openvla-7b-oft-finetuned-libero-10": ("action_head--150000_checkpoint.pt", "proprio_projector--150000_checkpoint.pt", "libero_10_no_noops"),
    "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10": ("action_head--300000_checkpoint.pt", "proprio_projector--300000_checkpoint.pt", None),
}


@dataclass(frozen=True)
class TokenLayout:
    """Positions of each token group in the LLM input sequence for one prompt.

    Sequence = [BOS] [agentview patches] [wrist patches] [proprio] [prompt tokens ...] [56 action slots] [stop]
    `instruction` = the tokens of the task string inside the prompt; `template` = the other prompt tokens.
    `action_slots` = the 56 zero-embedding placeholder positions.
    `readout` = the 56 positions whose last-layer states the action head reads. OFT keeps the causal-LM
        label shift (finetune.py: hidden[:, num_patches:-1] with shifted label masks; predict_action:
        slice from NUM_PATCHES + NUM_PROMPT_TOKENS), so readout = [last prompt token, slots 0..54]:
        one position BEFORE each slot. Slot 55 and the stop token are never read.
    """

    seq_len: int
    bos: tuple
    agentview: tuple
    wrist: tuple
    proprio: tuple
    template: tuple
    instruction: tuple
    action_slots: tuple
    readout: tuple
    stop: tuple

    def group(self, name: str) -> np.ndarray:
        if name == "vision":
            return np.array(self.agentview + self.wrist)
        if name == "prompt":
            return np.array(self.template + self.instruction)
        if name == "all":
            return np.arange(self.seq_len)
        return np.array(getattr(self, name))

    GROUPS = ("bos", "agentview", "wrist", "proprio", "template", "instruction", "action_slots", "stop")
    # `readout` overlaps template (last prompt token) and action_slots, so it is not part of the partition.


@dataclass
class PolicyInput:
    """Everything needed for one forward pass. Created by OFTPolicy.prepare(); treat as immutable."""

    instruction: str
    input_ids: torch.Tensor  # (1, L) incl. BOS and the trailing empty token
    pixel_values: torch.Tensor  # (1, 12, 224, 224) bf16: [agent(6ch: dino+siglip), wrist(6ch)]
    proprio_norm: np.ndarray  # (8,) normalized
    layout: TokenLayout
    images_224: tuple = field(default_factory=tuple)  # the exact preprocessed PIL images (for inspection)


class OFTPolicy:
    def __init__(self, repo_id: str, revision: Optional[str] = None, device: str = "cuda:0",
                 unnorm_key: Optional[str] = None, center_crop: bool = True):
        if repo_id not in CHECKPOINTS:
            raise ValueError(f"unsupported checkpoint {repo_id}")
        from huggingface_hub import snapshot_download

        self.repo_id, self.device, self.center_crop = repo_id, torch.device(device), center_crop
        if revision is None:
            raise ValueError("pass a pinned revision (vlamech.config.model_revision); never run on an unpinned 'main'")
        self.path = Path(snapshot_download(repo_id, revision=revision, allow_patterns=DOWNLOAD_ALLOW,
                                           ignore_patterns=DOWNLOAD_IGNORE))
        self.revision = _resolved_revision(self.path)
        if not self.revision.startswith(revision):
            raise RuntimeError(f"snapshot {self.revision} does not match pinned revision {revision}")
        head_file, proprio_file, default_key = CHECKPOINTS[repo_id]
        self._load_backbone()
        self._load_components(head_file, proprio_file)
        self.unnorm_key = unnorm_key or default_key
        if self.unnorm_key is None:
            raise ValueError(f"{repo_id} is multi-suite: pass unnorm_key explicitly (one of {list(self.vla.norm_stats)})")
        if self.unnorm_key not in self.vla.norm_stats:
            raise KeyError(f"unnorm key {self.unnorm_key!r} not in {list(self.vla.norm_stats)}")
        self._check_attention_impl()

    # ------------------------------------------------------------------ loading
    def _load_backbone(self):
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
        from transformers import AutoTokenizer

        cfg = OpenVLAConfig.from_pretrained(self.path)
        self.vla = OpenVLAForActionPrediction.from_pretrained(
            self.path, config=cfg, torch_dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True
        ).to(self.device).eval()
        self.vla.vision_backbone.set_num_images_in_input(2)
        stats_path = self.path / "dataset_statistics.json"
        self.vla.norm_stats = json.JSONDecoder().decode(stats_path.read_text())  # immune to json_numpy.patch()
        image_processor = PrismaticImageProcessor.from_pretrained(self.path)
        tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.processor = PrismaticProcessor(image_processor=image_processor, tokenizer=tokenizer)
        self.tokenizer = tokenizer
        for p in self.vla.parameters():
            p.requires_grad_(False)

    def _load_components(self, head_file, proprio_file):
        from prismatic.models.action_heads import L1RegressionActionHead
        from prismatic.models.projectors import ProprioProjector

        llm_dim = self.vla.llm_dim
        self.action_head = L1RegressionActionHead(input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM)
        self.action_head.load_state_dict(_load_component(self.path / head_file))
        self.action_head = self.action_head.to(torch.bfloat16).to(self.device).eval()
        self.proprio_projector = ProprioProjector(llm_dim=llm_dim, proprio_dim=PROPRIO_DIM)
        self.proprio_projector.load_state_dict(_load_component(self.path / proprio_file))
        self.proprio_projector = self.proprio_projector.to(torch.bfloat16).to(self.device).eval()

    def _check_attention_impl(self):
        from transformers.models.llama.modeling_llama import LlamaSdpaAttention

        layers = self.decoder_layers
        bad = [i for i, l in enumerate(layers) if type(l.self_attn) is not LlamaSdpaAttention]
        if bad:
            raise RuntimeError(f"layers {bad} are not SDPA attention: OFT's bidirectional mask would be lost")
        import transformers

        src = Path(transformers.__file__).parent / "models/llama/modeling_llama.py"
        if "is_causal=False" not in src.read_text():
            raise RuntimeError("installed transformers is not the OFT fork (bidirectional attention missing)")

    @property
    def decoder_layers(self):
        return self.vla.language_model.model.layers

    @property
    def n_layers(self) -> int:
        return len(self.decoder_layers)

    # ------------------------------------------------------------------ preprocessing
    def preprocess_images(self, agentview: np.ndarray, wrist: np.ndarray):
        """Exact upstream pipeline: TF JPEG round-trip + lanczos3 resize to 224, then 0.9-area center crop."""
        return tuple(policy_view(img, self.center_crop) for img in (agentview, wrist))

    def normalize_proprio(self, proprio: np.ndarray) -> np.ndarray:
        stats = self.vla.norm_stats[self.unnorm_key]["proprio"]
        lo, hi = np.array(stats["q01"]), np.array(stats["q99"])
        mask = np.array(stats.get("mask", np.ones_like(lo, dtype=bool)))
        p = np.asarray(proprio, dtype=np.float64)
        return np.clip(np.where(mask, 2 * (p - lo) / (hi - lo + 1e-8) - 1, p), -1.0, 1.0)

    def prepare(self, agentview: np.ndarray, wrist: np.ndarray, proprio: np.ndarray, instruction: str,
                images_224: Optional[tuple] = None) -> PolicyInput:
        """Build a PolicyInput. Pass `images_224` to reuse preprocessed images across instructions."""
        imgs = images_224 if images_224 is not None else self.preprocess_images(agentview, wrist)
        prompt = PROMPT_TEMPLATE.format(task=instruction.lower())
        enc_a = self.processor(prompt, imgs[0])
        enc_w = self.processor(prompt, imgs[1])
        pixel_values = torch.cat([enc_a["pixel_values"], enc_w["pixel_values"]], dim=1).to(self.device, torch.bfloat16)
        input_ids = enc_a["input_ids"]
        if not torch.all(input_ids[:, -1] == EMPTY_TOKEN_ID):
            input_ids = torch.cat([input_ids, torch.tensor([[EMPTY_TOKEN_ID]])], dim=1)
        input_ids = input_ids.to(self.device)
        layout = self._layout(input_ids, prompt, instruction.lower())
        return PolicyInput(instruction, input_ids, pixel_values, self.normalize_proprio(proprio), layout, imgs)

    def _layout(self, input_ids: torch.Tensor, prompt: str, task: str) -> TokenLayout:
        n_patch = self.vla.vision_backbone.get_num_patches()  # 256 per image
        n_prompt = input_ids.shape[1] - 1  # excludes BOS
        p0 = 1 + 2 * n_patch + 1  # first prompt position
        # locate task tokens via offsets of the prompt tokenization
        enc = self.tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        if ids != input_ids[0, : len(ids)].tolist():
            raise RuntimeError("tokenizer mismatch between processor and offset tokenization")
        start = prompt.index(task)
        end = start + len(task)
        instr = []
        for j, (a, b) in enumerate(offs):
            if j == 0:  # BOS
                continue
            if b > start and a < end:
                instr.append(p0 + j - 1)
        prompt_pos = list(range(p0, p0 + n_prompt))
        template = [p for p in prompt_pos if p not in set(instr)]
        a0 = p0 + n_prompt
        seq_len = a0 + N_ACTION_SLOTS + 1
        return TokenLayout(
            seq_len=seq_len,
            bos=(0,),
            agentview=tuple(range(1, 1 + n_patch)),
            wrist=tuple(range(1 + n_patch, 1 + 2 * n_patch)),
            proprio=(1 + 2 * n_patch,),
            template=tuple(template),
            instruction=tuple(instr),
            action_slots=tuple(range(a0, a0 + N_ACTION_SLOTS)),
            readout=tuple(range(a0 - 1, a0 - 1 + N_ACTION_SLOTS)),
            stop=(seq_len - 1,),
        )

    # ------------------------------------------------------------------ forward
    @torch.inference_mode()
    def vision_features(self, inp: PolicyInput) -> torch.Tensor:
        """Projected patch embeddings (+ proprio token): (1, 513, 4096). Independent of the instruction (no FiLM)."""
        vla = self.vla
        patches = vla.vision_backbone(inp.pixel_values)
        projected = vla.projector(patches)
        proprio = torch.as_tensor(inp.proprio_norm, dtype=torch.float32).to(self.device, torch.bfloat16).reshape(1, -1)
        proprio_tok = self.proprio_projector(proprio).unsqueeze(1)
        return torch.cat([projected, proprio_tok], dim=1)

    @torch.inference_mode()
    def forward(self, inp: PolicyInput, vision_feats: Optional[torch.Tensor] = None, output_hidden_states: bool = False) -> dict:
        """One OFT forward pass. Returns normalized + unnormalized actions (8x7) and optionally all hidden states.

        Mirrors OpenVLAForActionPrediction.predict_action -> _regression_or_discrete_prediction (L1 head).
        """
        vla = self.vla
        input_ids = inp.input_ids
        # append 56 placeholder action tokens + stop token (ids are irrelevant: their embeddings are zeroed)
        placeholder = torch.ones((1, N_ACTION_SLOTS), dtype=input_ids.dtype, device=input_ids.device)
        stop = torch.full((1, 1), _C.STOP_INDEX, dtype=input_ids.dtype, device=input_ids.device)
        ids = torch.cat([input_ids, placeholder, stop], dim=1)
        emb = vla.get_input_embeddings()(ids)
        n_text = input_ids.shape[1]
        emb[:, n_text : n_text + N_ACTION_SLOTS] = 0  # upstream: input_embeddings * ~all_actions_mask
        vf = vision_feats if vision_feats is not None else self.vision_features(inp)
        seq = torch.cat([emb[:, :1], vf, emb[:, 1:]], dim=1)
        if seq.shape[1] != inp.layout.seq_len:
            raise RuntimeError(f"sequence length {seq.shape[1]} != layout {inp.layout.seq_len}")
        attn = torch.ones(seq.shape[:2], dtype=torch.long, device=seq.device)
        out = vla.language_model(
            input_ids=None, attention_mask=attn, position_ids=None, past_key_values=None, inputs_embeds=seq,
            labels=None, use_cache=False, output_attentions=False, output_hidden_states=True, return_dict=True,
        )
        last = out.hidden_states[-1]
        r0 = inp.layout.readout[0]  # = NUM_PATCHES + NUM_PROMPT_TOKENS upstream (one before the first slot)
        act_h = last[:, r0 : r0 + N_ACTION_SLOTS]
        norm = self.action_head.predict_action(act_h).reshape(NUM_ACTIONS_CHUNK, ACTION_DIM).float().cpu().numpy()
        res = {"actions_norm": norm, "actions": self.unnormalize(norm)}
        if output_hidden_states:
            res["hidden_states"] = out.hidden_states  # tuple of (n_layers+1) x (1, L, 4096); [0] = embeddings
        return res

    def unnormalize(self, norm: np.ndarray) -> np.ndarray:
        """Upstream _unnormalize_actions for BOUNDS_Q99."""
        stats = self.vla.norm_stats[self.unnorm_key]["action"]
        lo, hi = np.array(stats["q01"]), np.array(stats["q99"])
        mask = np.array(stats.get("mask", np.ones_like(lo, dtype=bool)))
        return np.where(mask, 0.5 * (norm + 1) * (hi - lo + 1e-8) + lo, norm)

    @torch.inference_mode()
    def act_many(self, agentview, wrist, proprio, instructions: Sequence[str]) -> dict:
        """Actions for several instructions on the same observation, sharing the vision computation."""
        first = self.prepare(agentview, wrist, proprio, instructions[0])
        vf = self.vision_features(first)
        out = {}
        for ins in instructions:
            inp = first if ins == instructions[0] else self.prepare(agentview, wrist, proprio, ins, images_224=first.images_224)
            out[ins] = self.forward(inp, vision_feats=vf)
        return out


def policy_view(img: np.ndarray, center_crop: bool = True):
    """The exact image the network receives for one camera frame (PIL, 224x224): upstream
    resize_image_for_policy (TF JPEG round-trip + Lanczos-3 to 224) then center_crop_image (0.9 area, resized back).
    Module-level so figures show precisely what the policy sees (same code path as inference)."""
    from PIL import Image

    from experiments.robot.openvla_utils import center_crop_image, resize_image_for_policy

    if not (img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3):
        raise TypeError("camera frames must be uint8 HxWx3")
    pil = Image.fromarray(resize_image_for_policy(np.array(img, copy=True), 224)).convert("RGB")
    return center_crop_image(pil) if center_crop else pil


def _load_component(path: Path) -> dict:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    # checkpoints saved from DDP carry a "module." prefix
    return {k.removeprefix("module."): v for k, v in sd.items()}


def _resolved_revision(path: Path) -> str:
    # snapshot dirs are .../snapshots/<commit-hash>
    return path.name
