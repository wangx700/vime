# Delta Weight Sync

Delta weight sync keeps non-colocated rollout engines up to date by shipping only the bytes
that changed between two syncs, instead of a full checkpoint each time. It targets large-model
training/inference disaggregation across clusters or datacenters, where writing the whole actor
every sync is the dominant cost.

Two transports are available: `disk` for shared-storage and cross-cluster deployments, and
`sparse_hccl` for online transfer between Ascend trainers and vLLM-Ascend rollout workers.
The first sync
sends every checkpoint coordinate through the public sparse protocol; later syncs
send only changed coordinates. Both seed and deltas are exported through Megatron-Bridge.

## Configuration

Sparse HCCL:

```bash
--update-weight-mode delta
--update-weight-transport sparse_hccl
--update-weight-delta-batch-gather 32
```

This path requires non-colocated, unquantized ND rollout weights and inference
PP=1, with EPLB disabled. Configure `--vllm-pipeline-parallel-size 1`.
The receiver handles TP/DP/EP loading; inference PP>1 is rejected before
communication initialization. Training-side PP/TP/EP export remains supported.

Sparse export requires Megatron-Bridge in the training environment.

The script `scripts/run-qwen3-30B-A3B-sparse-hccl.sh` uses training
TP2/PP2/EP2 and rollout TP2/PP1/EP2, on four training plus two rollout NPUs.
Set `DATA_ROOT` to the directory containing `weights/` and `datasets/`, and load
the NPU environment before running the script.
VIME normalizes contiguous payloads onto the communicator device and validates
them before invoking the trainer engine.

Disk transport configuration:

```bash
--update-weight-mode delta
--update-weight-transport disk
--update-weight-disk-dir /shared/fs/delta-updates
--update-weight-local-checkpoint-dir /local/nvme/rollout-ckpt
--update-weight-delta-encoding xor          # or: overwrite
--update-weight-delta-checksum xxh3-128     # or: blake3, adler32
```

Table 1: Disk delta transport flags and their roles.

| Flag | Role |
|---|---|
| `--update-weight-disk-dir` | Shared filesystem directory the trainer publishes deltas to and the rollout hosts read from. |
| `--update-weight-local-checkpoint-dir` | Host-local (e.g. NVMe) full HF checkpoint that `/pull_weights` keeps in sync — deltas are applied into it in place; a published full checkpoint replaces it. Each host seeds it from the engine's model path on the first `/pull_weights`. |
| `--update-weight-delta-encoding` | On-disk delta encoding: `xor` (default) or `overwrite`. |
| `--update-weight-delta-checksum` | Per-tensor integrity checksum: `xxh3-128` (default), `blake3`, or `adler32`. |

Deltas are always zstd-compressed (level 1); profiling showed it dominates lz4 / gzip / snappy / brotli on both wire size and decompress speed for this data, so it is not a knob.

## How it works

### Sparse HCCL

Initial and reconnected updates use Seed; subsequent updates use Diff.
Both paths share Prepare, Publish, Apply and Finish.

1. **Prepare.** Pause rollout generation, flush caches and start a weight update
   session. The trainer uses Megatron-Bridge to prepare an export index mapping
   training shards to HF checkpoint weights.
2. **Seed.** On the first update or after reconnection, export all current training
   weights through Bridge and produce complete checkpoint coordinates and target
   values for Publish/Apply. After successful seeding, capture local shard snapshots
   as the baseline for later diffs. Reconnection fully overwrites worker weights.
3. **Diff.** On later updates, compare current training shards with their snapshots
   bitwise. Bridge mappings and NaN probes convert changed positions and values
   into HF checkpoint coordinates.
4. **Publish.** For Diff entries, first gather variable-length payloads to rank zero
   in a consistent directory order. For either Seed or Diff, rank zero validates
   bounds, duplicate indices and NaNs, normalizes device and
   contiguity, and splits payloads by the byte budget. The sender publishes metadata
   through `send_weight_chunk` and runs receive RPCs and HCCL broadcasts concurrently.
5. **Apply.** Rollout workers receive coordinates and values, then use checkpoint-compatible
   MoE views and the model's native loader for TP/EP placement. Values overwrite
   the corresponding weights in place, including zeros; they are target values,
   not additive differences.
6. **Finish.** After all weights are applied successfully, finish the update session,
   record the new weight version and resume rollout generation.

Wire indices are int32; each checkpoint tensor must contain fewer than `2**31`
elements. The receiver requires full-shape NaN staging. Seed sends both indices
and values, so it uses more wire bytes than a dense values-only transfer.

Both dependency patches must be installed together. Sparse transport is unpacked.

### End-to-end checkpoint comparison

`tests/test_qwen3_30B_A3B_sparse_hccl_npu.py` trains on four NPUs, performs a
complete seed and nonempty sparse updates into a two-NPU rollout engine, and
exports the final training weights as a full HF checkpoint through Bridge.
It captures greedy inference, stops the rollout engine, and launches a fresh
engine on the same two devices to load that checkpoint directly from disk.
Both engines receive the same prompt token IDs and sampling parameters.
Generated token IDs and every logprob must match exactly; a missing logprob,
non-finite value or any mismatch fails the run. Repeat inference checks
whether the updated engine is itself exactly repeatable first.

```bash
MODEL_PATH=/mnt/weight/Qwen3-30B-A3B \
PROMPT_DATA_PATH=/path/to/dapo-math-17k.jsonl \
SPARSE_HCCL_RESULT_DIR=/local/nvme/sparse-check-001 \
SPARSE_HCCL_BASELINE_PROMPT="What is 17 + 25? Give the answer in one sentence." \
python tests/test_qwen3_30B_A3B_sparse_hccl_npu.py
```

### Disk transport

1. **Seed.** On the first sync the trainer captures a CPU snapshot of every parameter — seeded
   from `--hf-checkpoint`, which is exactly what each rollout host materializes its local
   checkpoint from. Nothing is published; this snapshot is the base the next sync diffs against.
   The trainer also issues `/pull_weights` with `target_version=0` so every host materializes
   its local base now, overlapped with the snapshot capture.
2. **Publish.** On every later sync the trainer diffs each gathered HF tensor against the
   snapshot, encodes and compresses the change, and writes a new version directory
   `weight_v{N:06d}/` under `--update-weight-disk-dir`. The directory is a canonical HF
   checkpoint — `model-NNNNN.safetensors` files holding the compressed diff tensors plus a
   `model.safetensors.index.json` (tensor name → file) carrying the apply metadata — so the
   artifact is portable, not tied to the trainer's parallelism layout. The snapshot is then
   advanced to the new values for the next diff.
3. **Pull.** The trainer calls `/pull_weights` on each engine. Inside the engine the request is
   broadcast to every rank on every node; each host applies the new version's delta into its
   local checkpoint in place (a per-host file lock collapses co-located ranks to one apply).
   The apply is parallelized across tensors and verified per-tensor (see Integrity); the call
   only reports success once **every host** holds a checksum-verified checkpoint.

   `/pull_weights` is not delta-specific: each published version is self-describing, and a
   version that is an ordinary full HF checkpoint (no delta metadata in its index) is pulled by
   copying it as-is — resetting the chain, so a fresh host joining late seeds from the newest
   full version instead of replaying every delta, and older deltas can be pruned. vime's
   full-mode disk sync uses exactly this when `--update-weight-local-checkpoint-dir` is set.
4. **Reload.** The engines reload the patched local checkpoint through the vanilla
   `update_weights_from_disk` path — the weight-loading code never sees the delta format.

Because the snapshot is seeded from `--hf-checkpoint` (the engine's actual base) rather than
from the current GPU weights, the scheme is correct for any model even where the Megatron→HF
round-trip is not byte-exact (e.g. trimmed vocab-padding rows in the embedding / LM head).

## Encodings

Both encodings are byte-level and dtype-blind, so the same path works for quantized checkpoints.
The engine reads the choice from each version's index metadata.

- **`xor`** (default): writes `new ^ old`. Smallest wire and fastest to apply (sequential,
  cache-friendly; the unchanged bytes are zeros the compressor crushes). It is an involution,
  so it must be applied **exactly once** against the correct base — applying it twice reverts.
- **`overwrite`**: writes the changed positions and their new absolute values. Larger on the
  wire and a less cache-friendly scattered apply, but **idempotent**: re-applying it (or
  finishing a partially-applied delta) converges to the same state regardless of how many times
  it runs. Use it when re-applicability matters more than wire size.

## Integrity

The trainer stores a per-tensor checksum of each tensor's new state in the version. After
applying, every host recomputes the checksum and **raises on any mismatch** — the failure
propagates through the `/pull_weights` response, so a corrupt delta or a wrong base fails loud
instead of serving bad weights. The apply also refuses to run out of order: a version only
applies on top of its declared base version.

`--update-weight-delta-checksum` selects the algorithm. The checksum is not the apply bottleneck
(the apply is decompress + XOR bound), so this is a digest-property choice, not a speed one:
`xxh3-128` (default) is the widest fast non-cryptographic digest; `blake3` is cryptographic, for
untrusted storage; `adler32` is for interop with systems that expect it.

## Shared-filesystem visibility hooks

On a POSIX shared filesystem (NFS, Lustre, …) no extra step is needed. Object-store-backed
mounts that need an explicit publish/refresh to make writes visible across hosts can supply two
optional hooks, loaded by import path — no vendor-specific code lives in vime or vllm:

- `--custom-update-weight-post-write-path` (vime, trainer side): called after a version's files are
  written, before the engines are told to read it (e.g. upload pending writes to the backing object store).
  Signature: `hook(args, version_dir, rollout_engines)`.
- `--custom-update-weight-pre-read-path` (vime, engine side): called on each host
  inside the engine before `/pull_weights` reads the delta directory (e.g. refresh the mount's view).
  Signature: `hook(delta_dir, target_version)`.
