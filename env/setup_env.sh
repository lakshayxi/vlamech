#!/usr/bin/env bash
# OFT environment: upstream checkouts at pinned commits, Python environment at $VLA_ENV_PREFIX (env/env.sh).
# Usage:  bash env/setup_env.sh   (safe to re-run; needs conda, git and a CUDA 12 driver)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

# Never let two installs touch the same env at once.
exec 9>"$VLA_CACHE/vlamech-setup.lock"
flock -n 9 || { echo "another setup_env.sh is running; aborting" >&2; exit 1; }

CONDA="${VLA_CONDA:-conda}"
EXT="$VLA_PROJECT_ROOT/external"
mkdir -p "$EXT"

# ---- pinned upstream sources (read by vlamech.runs.identity) ----
OPENVLA_OFT_COMMIT=e4287e94541f459edc4feabc4e181f537cd569a8
LIBERO_COMMIT=8f1084e3132a39270c3a13ebe37270a43ece2a01
DLIMP_COMMIT=040105d256bd28866cc6620621a3d5f7b6b91b46
TRANSFORMERS_FORK_COMMIT=bc339d9ad707454c0c115970db43c260067c61ab

clone_at () {  # url dir commit
  if [ ! -d "$2/.git" ]; then git clone -q "$1" "$2"; fi
  git -C "$2" fetch -q --all
  git -C "$2" checkout -q "$3"
  echo "[setup] $2 @ $(git -C "$2" rev-parse HEAD)"
}
clone_at https://github.com/moojink/openvla-oft.git "$EXT/openvla-oft" "$OPENVLA_OFT_COMMIT"
clone_at https://github.com/Lifelong-Robot-Learning/LIBERO.git "$EXT/LIBERO" "$LIBERO_COMMIT"

if [ ! -x "$VLA_ENV_PREFIX/bin/python" ]; then
  echo "[setup] creating conda env at $VLA_ENV_PREFIX"
  "$CONDA" create -y -q --prefix "$VLA_ENV_PREFIX" python=3.10
fi
PY="$VLA_ENV_PREFIX/bin/python"
PIP=("$PY" -m pip install --progress-bar off --constraint "$VLA_PROJECT_ROOT/env/constraints.txt")

echo "[setup] pip tooling"
"$PY" -m pip install --progress-bar off --upgrade "pip<25" "setuptools<70" wheel

echo "[setup] torch (cu121 wheels)"
"${PIP[@]}" torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# One resolver call for requirements + the transformers fork, so packages that depend on transformers
# (peft, accelerate) are resolved against the fork instead of pulling a PyPI release.
echo "[setup] requirements + transformers fork"
"${PIP[@]}" -r "$VLA_PROJECT_ROOT/env/requirements.txt" \
  "transformers @ git+https://github.com/moojink/transformers-openvla-oft.git@$TRANSFORMERS_FORK_COMMIT"

# OFT's `prismatic` package imports its RLDS data stack at import time (dlimp, tensorflow_datasets), even for
# inference. Installed without dependency resolution because dlimp declares `tensorflow` (GPU build, ~500 MB)
# while we use `tensorflow-cpu`, which provides the same module; tfds's own deps are listed in requirements.txt.
echo "[setup] OFT data stack (no deps)"
"$PY" -m pip install --progress-bar off --no-deps "tensorflow_datasets==4.9.3" "tensorflow_graphics==2021.12.3" \
  "dlimp @ git+https://github.com/moojink/dlimp_openvla@$DLIMP_COMMIT"

echo "[setup] LIBERO (no deps: its requirements pin torch 1.11 etc.)"
"$PY" -m pip install --progress-bar off --no-deps -e "$EXT/LIBERO"

# LIBERO config lives inside the project, not in $HOME
mkdir -p "$LIBERO_CONFIG_PATH"
cat > "$LIBERO_CONFIG_PATH/config.yaml" <<EOF
benchmark_root: $EXT/LIBERO/libero/libero
bddl_files: $EXT/LIBERO/libero/libero/bddl_files
init_states: $EXT/LIBERO/libero/libero/init_files
datasets: $EXT/LIBERO/libero/datasets
assets: $EXT/LIBERO/libero/libero/assets
EOF

echo "[setup] verifying"
# Accepted pip-check complaints (and only these): `tensorflow` is provided by tensorflow-cpu; tensorflow_graphics'
# extras (openexr, tensorflow-addons, trimesh) are not needed by the only submodule OFT imports
# (tensorflow_graphics.geometry.transformation, checked below).
CHECK=$("$PY" -m pip check 2>&1 | grep -vE "^(dlimp|tensorflow-graphics) [^ ]+ requires (tensorflow|openexr|tensorflow-addons|trimesh), which is not installed.$" || true)
if [ -n "$CHECK" ] && [ "$CHECK" != "No broken requirements found." ]; then echo "$CHECK"; exit 1; fi
"$PY" - <<'EOF'
import torch, transformers, numpy, mujoco, robosuite, tensorflow as tf
import tensorflow_graphics.geometry.transformation, dlimp, tensorflow_datasets  # OFT's import-time data stack
import experiments.robot.openvla_utils  # upstream inference utilities import cleanly
assert torch.__version__.startswith("2.2.0"), torch.__version__
assert transformers.__version__ == "4.40.1", transformers.__version__
assert numpy.__version__ == "1.26.4", numpy.__version__
import pathlib; src = pathlib.Path(transformers.__file__).parent / "models/llama/modeling_llama.py"
assert "is_causal=False" in src.read_text(), "transformers is not the OFT fork"
assert torch.cuda.is_available(), "CUDA not available"
print("[setup] torch", torch.__version__, "| transformers", transformers.__version__, "(OFT fork)",
      "| mujoco", mujoco.__version__, "| robosuite", robosuite.__version__, "| tf", tf.__version__)
EOF
"$PY" -m pip freeze > "$VLA_PROJECT_ROOT/env/requirements.lock.txt"
echo "[setup] Environment ready at $VLA_ENV_PREFIX"
