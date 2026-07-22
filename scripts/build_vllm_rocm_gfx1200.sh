#!/bin/bash
# Build vLLM for ROCm gfx1200 (RX 9060 XT) with the HCP patch set.
# Prereqs: conda env with torch 2.13+rocm7.13 and a HIP toolchain.
set -euo pipefail

VLLM_COMMIT="${VLLM_COMMIT:-3f99883d9}"   # v0.23.1rc1.dev905
WORK_DIR="${WORK_DIR:-$HOME}"
PATCH_DIR="$(cd "$(dirname "$0")/.." && pwd)/patches"

cd "${WORK_DIR}"
if [ ! -d vllm/.git ]; then
  git clone https://github.com/vllm-project/vllm.git vllm
fi
cd vllm
git fetch --tags -q
git checkout "${VLLM_COMMIT}"

for p in "${PATCH_DIR}"/*.patch; do
  echo "applying $(basename "$p")"
  git apply "$p"
done

echo "building (pip install -e, no build isolation) ..."
pip install -e . --no-build-isolation
echo "done. Run via scripts/run_vllm_rocm.sh"
