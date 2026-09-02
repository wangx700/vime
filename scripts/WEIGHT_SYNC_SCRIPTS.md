# Qwen3-4B weight-sync launchers

All launchers require `ASCEND_RT_VISIBLE_DEVICES` to be set explicitly. They do
not choose NPU cards automatically. `VLLM_VERSION` defaults to `0.26.0`.

## Entry points

| Mode | Transport | Script | VIME checkout |
| --- | --- | --- | --- |
| delta | disk | `run-qwen3-4B-delta-disk.sh` | `/home/vllm/c00944022/0623/vime` |
| full | disk | `run-qwen3-4B-full-disk.sh` | `/home/vllm/c00944022/0623/vime` |
| sparse | HCCL (`nccl` CLI value) | `run-qwen3-4B-sparse-hccl-rank-p2p.sh` | `/home/vllm/c00944022/0623/vime-sparse-hccl-rank-p2p` |
| full | HCCL (`nccl` CLI value) | `run-qwen3-4B-full-hccl-rank-p2p.sh` | `/home/vllm/c00944022/0623/vime-sparse-hccl-rank-p2p` |

The sparse/HCCL and full/HCCL launchers use the same VIME commit so their
performance results are directly comparable.

## Required checkout versions

- VIME sparse/HCCL checkout: branch `feature/sparse-hccl-rank-p2p`, commit
  `7fce886dccd00686f1802251a9ffb09a5ef5ade9`, remote `wangx700/vime`.
- vLLM-Ascend: branch `feature/sparse-hccl-rank-p2p`, GitHub commit
  `c34dc70bcd8b8892d57489da965684b7323fbce8`, remote
  `wangx700/vllm-ascend`.
- vLLM source: `/home/w00899129/vllm-code/vllm_vime/vllm`.
- vLLM-Ascend source: `/home/w00899129/vllm-code/vllm_vime/vllm-ascend`.

## Usage

Choose only currently free cards, then run one entry point from the VIME root:

```bash
export ASCEND_RT_VISIBLE_DEVICES=<explicit-free-card-list>
export VLLM_VERSION=0.26.0
bash scripts/run-qwen3-4B-sparse-hccl-rank-p2p.sh
```

Optional paths can be overridden with `MODEL_PATH`, `PROMPT_DATA_PATH`,
`UPDATE_WEIGHT_DISK_DIR`, and `UPDATE_WEIGHT_LOCAL_CHECKPOINT_DIR`.
Ray ports and state directories can be isolated with `RAY_GCS_PORT`,
`RAY_DASHBOARD_PORT`, and `RAY_TEMP_DIR`.

Warning: the shared base launcher still contains process cleanup commands. Do
not run it in a container that has another active Ray/Python/VLLM workload.
