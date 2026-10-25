#!/usr/bin/env bash
# Check the existing conda env (default: matching-cd) on the 2x GH200 server.  Read-only unless --install
# (which pip-installs missing numba / scipy / pandas / matplotlib INTO that env only).
#
#   bash check_env.sh            |   bash check_env.sh --install   |   ENV=other bash check_env.sh
set -eo pipefail
cd "$(dirname "$0")"
ENV_NAME=${ENV:-matching-cd}
if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]]; then
    CONDA_BASE=$(conda info --base 2>/dev/null) || { echo "conda not found in PATH"; exit 1; }
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
fi
echo "env: $CONDA_DEFAULT_ENV  python: $(which python)  arch: $(uname -m)"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv || true

python - <<'EOF'
import os, importlib, shutil, subprocess
try:
    import torch
except ImportError:
    raise SystemExit("torch missing in this env")
print("torch", torch.__version__, "| cuda build", torch.version.cuda, "| cuda available", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i} {p.name} {p.total_memory / 2**30:.0f} GiB sm_{p.major}{p.minor}")
for m in ("numpy", "scipy", "numba", "pandas", "matplotlib", "ninja"):
    try:
        print(f"  {m:10s}", importlib.import_module(m).__version__)
    except Exception:
        print(f"  {m:10s} MISSING")
from torch.utils.cpp_extension import CUDA_HOME
nvcc = os.path.join(CUDA_HOME, "bin", "nvcc") if CUDA_HOME else shutil.which("nvcc")
print("CUDA_HOME", CUDA_HOME, "| nvcc", nvcc)
if nvcc and os.path.exists(nvcc):
    print("  ", subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1])
else:
    print("  nvcc NOT FOUND: the fused kernel (pdsb_fused) cannot be built; set CUDA_HOME to a CUDA toolkit "
          "whose major version matches torch.version.cuda")
print("cpus", os.cpu_count())
EOF

missing=$(python - <<'EOF'
import importlib.util
print(" ".join(m for m in ("numpy", "scipy", "numba", "pandas", "matplotlib", "ninja") if importlib.util.find_spec(m) is None))
EOF
)
if [[ -n "$missing" ]]; then
    if [[ "${1:-}" == "--install" ]]; then
        python -m pip install $missing
    else
        echo "missing in $CONDA_DEFAULT_ENV: $missing   (numba: SA baseline; pandas/matplotlib: analysis; ninja: fused-kernel build)"
        echo "install into this env only:  bash check_env.sh --install"
    fi
else
    echo "all packages present."
fi
echo "next:  python selftest.py   (first call builds the fused kernel, ~1 min)"
