#!/bin/bash
set -euo pipefail

: "${ASCEND_RT_VISIBLE_DEVICES:?Set ASCEND_RT_VISIBLE_DEVICES explicitly to avoid using occupied NPUs}"
VIME_RUN_ROOT="${VIME_RUN_ROOT:-/home/vllm/c00944022/0623/vime}"
export VLLM_VERSION="${VLLM_VERSION:-0.26.0}"
export UPDATE_WEIGHT_MODE=delta
export UPDATE_WEIGHT_TRANSPORT=sparse_hccl
export UPDATE_WEIGHT_DISK_DIR="${UPDATE_WEIGHT_DISK_DIR:-/home/vllm/c00944022/0623/vime-delta-weights}"
export UPDATE_WEIGHT_LOCAL_CHECKPOINT_DIR="${UPDATE_WEIGHT_LOCAL_CHECKPOINT_DIR:-/tmp/vime-rollout-checkpoint}"
export RAY_GCS_PORT="${RAY_GCS_PORT:-6399}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8267}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray-vime-delta-disk}"

# test "$(git -C "${VIME_RUN_ROOT}" branch --show-current)" = feature/disk-full-weight-sync
cd "${VIME_RUN_ROOT}"
exec bash scripts/run-qwen3-4B-npu.sh "$@"
