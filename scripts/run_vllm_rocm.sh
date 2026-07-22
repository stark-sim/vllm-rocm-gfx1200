#!/bin/bash
# LD_LIBRARY_PATH wrapper for running vLLM in the vllm-rocm env on gfx1200.
# Usage (after `conda activate vllm-rocm`, and cd somewhere WITHOUT a vllm/ dir):
#   ./run_vllm_rocm.sh python your_script.py
set -euo pipefail
SP="${SP:-/home/stark/miniconda3/envs/vllm-rocm/lib/python3.11/site-packages}"
export LD_LIBRARY_PATH="${SP}/torch/lib:${SP}/_rocm_sdk_core/lib:${SP}/_rocm_sdk_core/lib/host-math/lib:${SP}/_rocm_sdk_core/lib/rocm_sysdeps/lib:${SP}/_rocm_sdk_devel/lib:${SP}/_rocm_sdk_devel/lib/host-math/lib:${SP}/_rocm_sdk_devel/lib/rocm_sysdeps/lib:${LD_LIBRARY_PATH:-}"
exec "$@"
