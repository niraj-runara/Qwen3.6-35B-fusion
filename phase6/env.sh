#!/usr/bin/env bash
# Phase 6 — dev environment for local SGLang clone (no pip install -e / Rust).
#
# Keeps PyPI sglang + sglang-kernel in phase1 venv; overrides Python sources from
# the sibling clone via PYTHONPATH (see phase6/README.md).
#
# Usage (from repo root):
#   source phase6/env.sh
#
# Optional overrides before sourcing:
#   SGLANG_ROOT=/path/to/sglang  source phase6/env.sh

_PHASE6_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_REPO_ROOT="$(cd "${_PHASE6_DIR}/.." && pwd)"
_VENV="${_REPO_ROOT}/phase1/.venv"
_SGLANG_ROOT="${SGLANG_ROOT:-/sglang}"

if [[ ! -d "${_VENV}" ]]; then
  echo "phase6/env.sh: missing venv at ${_VENV}" >&2
  echo "  Run: source phase1/.venv/bin/activate && bash phase3/setup_sglang.sh" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ ! -d "${_SGLANG_ROOT}/python/sglang" ]]; then
  echo "phase6/env.sh: SGLang clone not found at ${_SGLANG_ROOT}/python" >&2
  echo "  Set SGLANG_ROOT to your local clone (expected @ v0.5.13)." >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck source=/dev/null
source "${_VENV}/bin/activate"

export PYTHONPATH="${_SGLANG_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
export SGLANG_QWEN_FUSION="${SGLANG_QWEN_FUSION:-0}"

_VENV_SITE="$(python -c "import site; print(site.getsitepackages()[0])")"
export LD_LIBRARY_PATH="${_VENV_SITE}/nvidia/cuda_nvrtc/lib:${_VENV_SITE}/nvidia/cuda_runtime/lib:${_VENV_SITE}/nvidia/cu13/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "Phase 6 dev env"
echo "  venv          : ${_VENV}"
echo "  SGLANG_ROOT   : ${_SGLANG_ROOT}"
echo "  PYTHONPATH    : ${_SGLANG_ROOT}/python (first)"
echo "  SGLANG_QWEN_FUSION : ${SGLANG_QWEN_FUSION}"

python -c "import sglang; print('  sglang import  :', sglang.__version__, '←', sglang.__file__)" 2>/dev/null \
  || echo "  sglang import  : FAILED (is phase3/setup_sglang.sh done?)" >&2
