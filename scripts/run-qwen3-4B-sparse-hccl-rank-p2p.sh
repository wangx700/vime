#!/bin/bash
set -euo pipefail

: "${ASCEND_RT_VISIBLE_DEVICES:?Set ASCEND_RT_VISIBLE_DEVICES explicitly to avoid using occupied NPUs}"
VIME_RUN_ROOT="${VIME_RUN_ROOT:-/home/vllm/c00944022/0623/vime-sparse-hccl-rank-p2p}"
export VLLM_VERSION="${VLLM_VERSION:-0.26.0}"
export UPDATE_WEIGHT_MODE=sparse
export UPDATE_WEIGHT_TRANSPORT=nccl
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.5}"
export RAY_GCS_PORT="${RAY_GCS_PORT:-6401}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8269}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray-vime-sparse-hccl}"

test "$(git -C "${VIME_RUN_ROOT}" rev-parse HEAD)" = 7fce886dccd00686f1802251a9ffb09a5ef5ade9
cd "${VIME_RUN_ROOT}"
exec bash scripts/run-qwen3-4B-npu.sh "$@"
