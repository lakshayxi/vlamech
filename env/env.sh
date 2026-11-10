# Source this file before running anything in the project:  source env/env.sh
# Every cache, temp dir and model download goes under $VLA_CACHE (default: <repo>/.cache); the Python environment
# is $VLA_ENV_PREFIX (default: <repo>/.envs/vlamech; the pi0.5 environment is <repo>/.envs/lerobot).
# Set HF_HOME to reuse an existing Hugging Face cache.

export VLA_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VLA_CACHE="${VLA_CACHE:-$VLA_PROJECT_ROOT/.cache}"
export VLA_ENV_PREFIX="${VLA_ENV_PREFIX:-$VLA_PROJECT_ROOT/.envs/vlamech}"

# Model and data caches. Runs never download: models are fetched once by scripts/setup/download_models.py.
export HF_HOME="${HF_HOME:-$VLA_CACHE/huggingface}"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PIP_CACHE_DIR="$VLA_CACHE/pip"
export XDG_CACHE_HOME="$VLA_CACHE"
export TORCH_HOME="$VLA_CACHE/torch"
export TRITON_CACHE_DIR="$VLA_CACHE/triton"
export MPLCONFIGDIR="$VLA_CACHE/matplotlib"
export TMPDIR="$VLA_CACHE/tmp"
mkdir -p "$TMPDIR"

# LIBERO writes its config to ~/.libero by default; keep it inside the project instead.
export LIBERO_CONFIG_PATH="$VLA_PROJECT_ROOT/external/libero_config"

# Headless MuJoCo rendering
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# Determinism for cuBLAS (required by torch.use_deterministic_algorithms)
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

export PATH="$VLA_ENV_PREFIX/bin:$PATH"
export PYTHONPATH="$VLA_PROJECT_ROOT/src:$VLA_PROJECT_ROOT/external/openvla-oft:$VLA_PROJECT_ROOT/external/LIBERO${PYTHONPATH:+:$PYTHONPATH}"

# wandb is imported by upstream code; keep it off unless a run opts in explicitly
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_ERROR_REPORTING=false
export WANDB_DIR="$VLA_CACHE/wandb"
