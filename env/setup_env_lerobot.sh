#!/usr/bin/env bash
# Second environment for the pi0.5 policy (LeRobot). The OFT environment (env/setup_env.sh) is untouched: OFT needs
# its transformers fork, pi0.5 needs LeRobot's patched transformers, and the two cannot share one install.
# Usage:  bash env/setup_env_lerobot.sh   then   VLA_ENV_PREFIX=<repo>/.envs/lerobot source env/env.sh
# Same simulator versions as the OFT env (env/requirements_lerobot.txt), so both policies act in the same LIBERO.
# LIBERO itself comes from external/LIBERO on PYTHONPATH (env/env.sh), at the pin in env/setup_env.sh; LeRobot's
# own LIBERO package (hf-libero, extra "libero") is NOT installed, so `import libero` resolves to the pinned checkout.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VLA_ENV_PREFIX="${VLA_LEROBOT_PREFIX:-$ROOT/.envs/lerobot}"
source "$ROOT/env/env.sh"

exec 9>"$VLA_CACHE/lerobot-setup.lock"
flock -n 9 || { echo "another setup_env_lerobot.sh is running; aborting" >&2; exit 1; }

CONDA="${VLA_CONDA:-conda}"

# ---- pins (read by vlamech.runs.identity for pi0.5 runs) ----
LEROBOT_COMMIT=8fff0fde7c79f23a93d845d1a50e985de01f8b8a           # tag v0.4.4 (checkpoint lerobot/pi05_libero_finetuned_v044)
PI05_TRANSFORMERS_COMMIT=dcddb970176382c0fcf4521b0c0e6fc15894dfe0  # huggingface/transformers fix/lerobot_openpi (lerobot[pi] extra at v0.4.4)

if [ ! -x "$VLA_ENV_PREFIX/bin/python" ]; then
  echo "[setup-lerobot] creating conda env at $VLA_ENV_PREFIX"
  "$CONDA" create -y -q --prefix "$VLA_ENV_PREFIX" python=3.10
fi
PY="$VLA_ENV_PREFIX/bin/python"
# uv's resolver: pip's backtracking did not finish on this set (it walked old contourpy / numba sdists for 30+ min)
"$PY" -m pip install --progress-bar off --upgrade "pip<25" "setuptools<70" wheel "uv==0.9.5"
export UV_CACHE_DIR="$VLA_CACHE/uv"

# One resolver call: lerobot at the pinned commit (its [pi] extra names the transformers BRANCH; the commit pin
# below replaces it), the pinned transformers, and the simulator / analysis stack.
echo "[setup-lerobot] lerobot + patched transformers + simulator"
"$VLA_ENV_PREFIX/bin/uv" pip install --python "$PY" \
  "lerobot @ git+https://github.com/huggingface/lerobot.git@$LEROBOT_COMMIT" \
  "transformers @ git+https://github.com/huggingface/transformers.git@$PI05_TRANSFORMERS_COMMIT" \
  -r "$VLA_PROJECT_ROOT/env/requirements_lerobot.txt" --override "$VLA_PROJECT_ROOT/env/overrides_lerobot.txt"

echo "[setup-lerobot] verifying"
"$PY" - <<'EOF'
import torch, transformers, numpy, mujoco, robosuite, lerobot
from transformers.models.siglip import check  # exists only in the patched transformers
assert check.check_whether_transformers_replace_is_installed_correctly(), "transformers is not LeRobot's patched build"
from lerobot.policies.pi05 import PI05Policy  # noqa: F401
from libero.libero import benchmark  # the pinned external/LIBERO checkout
import libero, pathlib
loc = [str(pathlib.Path(p).resolve()) for p in libero.__path__]  # namespace package: no __file__
assert any("external/LIBERO" in p for p in loc) and not any("site-packages" in p for p in loc), loc
assert numpy.__version__ == "1.26.4" and mujoco.__version__ == "2.3.7" and robosuite.__version__ == "1.4.1"
assert torch.cuda.is_available(), "CUDA not available"
print("[setup-lerobot] torch", torch.__version__, "| transformers", transformers.__version__, "(lerobot_openpi)",
      "| lerobot", lerobot.__version__, "| mujoco", mujoco.__version__, "| robosuite", robosuite.__version__)
EOF
"$PY" -m pip freeze > "$VLA_PROJECT_ROOT/env/requirements_lerobot.lock.txt"
echo "[setup-lerobot] Environment ready at $VLA_ENV_PREFIX"
