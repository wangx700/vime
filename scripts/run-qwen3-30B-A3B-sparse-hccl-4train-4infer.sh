#!/bin/bash

# Isolated validation variant: it does not stop processes outside this run.
set -euo pipefail
set -x

export PYTHONUNBUFFERED=1
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HYDRA_FULL_ERROR=1
export DISABLE_L2_CACHE=1
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_USE_AOT_COMPILE=0
# TP4 Qwen3-30B-A3B has insufficient NPU headroom for a persistent seed copy.
# The baseline stays on host memory and is moved per parameter for the NPU diff.
export VIME_SPARSE_HCCL_SNAPSHOT_DEVICE="${VIME_SPARSE_HCCL_SNAPSHOT_DEVICE:-cpu}"
VIME_WORKSPACE_ROOT="${VIME_WORKSPACE_ROOT:-/home/vllm/c00944022/0623}"
export PYTHONPATH="/home/vllm/c00944022/vime-proj/MegatronAdaptor:/home/vllm/c00944022/vime-proj/TransformerEngineNPU:${VIME_WORKSPACE_ROOT}/Megatron-Bridge/src:${VIME_WORKSPACE_ROOT}/Megatron-LM:${PYTHONPATH:-}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

source "${VIME_WORKSPACE_ROOT}/vime/scripts/models/qwen3-30B-A3B.sh"

MODEL_PATH="${MODEL_PATH:-/home/vllm/weights/Qwen3-30B-A3B}"
PROMPT_DATA_PATH="${PROMPT_DATA_PATH:-/home/vllm/c00944022/datasets/dapo-math-17k/dapo-math-17k.jsonl}"
NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-512}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_PATH}"
   --load "${MODEL_PATH}"
   --ref-load "${MODEL_PATH}"
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA_PATH}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-temperature 1
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --balance-data
)

# The short weight-sync validation is below the original eval interval (20),
# so no evaluation is scheduled and no external DATA_ROOT is required.
EVAL_ARGS=()

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 4
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef "${ENTROPY_COEF:-0.001}"
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

VLLM_ARGS=(
   --rollout-num-gpus-per-engine 4
   --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.6}"
   --vllm-enforce-eager
   --vllm-additional-config '{"weight_nz_mode":0}'
)

UPDATE_WEIGHT_ARGS=(
   --update-weight-delta-batch-diff "${UPDATE_WEIGHT_DELTA_BATCH_DIFF:-32}"
   --update-weight-delta-batch-gather "${UPDATE_WEIGHT_DELTA_BATCH_GATHER:-32}"
   --update-weight-mode "${UPDATE_WEIGHT_MODE:-delta}"
   --update-weight-transport "${UPDATE_WEIGHT_TRANSPORT:-sparse_hccl}"
   --update-weight-delta-verify-every "${UPDATE_WEIGHT_DELTA_VERIFY_EVERY:-0}"
)
if [[ "${UPDATE_WEIGHT_TRANSPORT:-sparse_hccl}" == "disk" ]]; then
   UPDATE_WEIGHT_ARGS+=(--update-weight-disk-dir "${UPDATE_WEIGHT_DISK_DIR:?set UPDATE_WEIGHT_DISK_DIR}")
fi
if [[ "${UPDATE_WEIGHT_MODE:-delta}" == "delta" && "${UPDATE_WEIGHT_TRANSPORT:-sparse_hccl}" == "disk" ]]; then
   UPDATE_WEIGHT_ARGS+=(
      --update-weight-local-checkpoint-dir "${UPDATE_WEIGHT_LOCAL_CHECKPOINT_DIR:?set UPDATE_WEIGHT_LOCAL_CHECKPOINT_DIR}"
      --update-weight-delta-encoding xor
      --update-weight-delta-checksum xxh3-128
   )
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --micro-batch-size 1

   --use-flash-attn
   --no-gradient-accumulation-fusion
)

ray start --head --port="${RAY_GCS_PORT:-6503}" --temp-dir="${RAY_TEMP_DIR:-/tmp/r30b}" \
   --node-ip-address 127.0.0.1 --disable-usage-stats --dashboard-host=0.0.0.0 \
   --dashboard-port="${RAY_DASHBOARD_PORT:-8373}" \
   --dashboard-agent-listen-port="${RAY_DASHBOARD_AGENT_PORT:-52366}"

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT:-8373}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
${VLLM_ARGS[@]} \
${UPDATE_WEIGHT_ARGS[@]} \
${MISC_ARGS[@]}
