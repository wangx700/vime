# Delta 权重同步

Delta 权重同步只发送两次同步之间发生变化的字节，而不是每次都写一份完整 checkpoint，以此让非 colocate 的 rollout engine 保持最新。它面向大模型、跨集群或跨数据中心的训推解耦场景——这种场景下每次都写整份 actor 权重是主要开销。

它支持两条传输路径：`disk` 面向跨集群共享存储，`sparse_hccl` 面向 Ascend
训练和 vLLM-Ascend rollout 之间的在线传输。首次通过标准 sparse 协议发送全部 checkpoint 坐标，后续只发送变化坐标。
首次全量导出和后续增量索引都使用 Megatron-Bridge。

## 配置

Sparse HCCL：

```bash
--update-weight-mode delta
--update-weight-transport sparse_hccl
--update-weight-delta-batch-gather 32
```

该路径要求非 colocate、未量化 ND rollout 权重、关闭 EPLB，以及推理 PP=1；
设置 `--vllm-pipeline-parallel-size 1`。接收端支持 TP/DP/EP，推理 PP>1
在建组前明确拒绝。训练侧 PP/TP/EP 导出与 gather 保留。

脚本 `scripts/run-qwen3-30B-A3B-sparse-hccl.sh` 使用训练 TP2/PP2/EP2、
推理 TP2/PP1/EP2，四张训练卡加两张推理卡。设置 `DATA_ROOT` 指向包含
`weights/` 和 `datasets/` 的目录，运行前加载 NPU 环境。VIME 在调用训练引擎前完成输入校验、
设备转换及连续化，不改写引擎的发送或接收实现。

磁盘传输配置：

```bash
--update-weight-mode delta
--update-weight-transport disk
--update-weight-disk-dir /shared/fs/delta-updates
--update-weight-local-checkpoint-dir /local/nvme/rollout-ckpt
--update-weight-delta-encoding xor          # 或: overwrite
--update-weight-delta-checksum xxh3-128     # 或: blake3, adler32
```

表 1：磁盘 delta 传输参数及其作用。

| 参数 | 作用 |
|---|---|
| `--update-weight-disk-dir` | 训练端发布 delta、rollout host 读取 delta 的共享文件系统目录。 |
| `--update-weight-local-checkpoint-dir` | host 本地（如 NVMe）的完整 HF checkpoint，由 `/pull_weights` 保持同步——delta 原地 apply，发布的完整 checkpoint 则整份替换。每个 host 在第一次 `/pull_weights` 时由 engine 的 model path seed。 |
| `--update-weight-delta-encoding` | 磁盘上的 delta 编码：`xor`（默认）或 `overwrite`。 |
| `--update-weight-delta-checksum` | 逐 tensor 完整性 checksum：`xxh3-128`（默认）、`blake3` 或 `adler32`。 |

delta 始终用 zstd（level 1）压缩；profiling 显示对这类数据它在 wire 大小和解压速度上都优于 lz4 / gzip / snappy / brotli，所以不做成可配置项。

## 工作原理

### Sparse HCCL

1. 通过 Megatron-Bridge 建立导出索引。首轮全量也使用
   `HfWeightIteratorBridge`，sparse 路径不调用直接映射导出器。
2. 为首轮实际训练权重生成完整 int32 checkpoint 坐标与目标值，按字节预算分片，
   使用`send_weight_chunk` 发送。重连后也能覆盖旧 worker 状态，不假定其
   仍等于初始 checkpoint。接收端仍需完整形状 NaN staging，首轮 wire 大于 dense 值传输。
3. 首轮成功后才捕获本地 shard snapshot。后续 bitwise diff 通过真实 Bridge 映射及
   NaN probe 生成最终 checkpoint 坐标，再按一致目录执行变长 gather 到 rank0。
4. VIME 在通信前检查边界、重复索引及 NaN，并将 tensor 转到 communicator device
   且连续；sender 负责 patch 结构检查、metadata 和并发接收 RPC/broadcast；
   VIME 管理 pause/start/finish/resume。
5. 接收端通过临时 checkpoint-compatible MoE 视图调用模型 loader，处理 TP/EP
   放置；不启用 Ascend MoE 直写。values 是目标值覆盖，零也必须写，不是加法 delta。
6. 显式限制单个 checkpoint tensor 元素数小于 `2**31`，规避当前 checkpoint loader 的
   int32 大张量边界限制。wire 仍为 int32；输入校验和独立正确性 oracle 保留。

两份依赖 patch 必须配套安装。Sparse HCCL 采用 unpacked 传输。
训练环境需安装 Megatron-Bridge。

### 端到端权重与推理对照

`tests/test_qwen3_30B_A3B_sparse_hccl_npu.py` 在四张卡上训练，向两张推理卡
执行完整 seed 和非空 sparse delta；随后从最终训练模型经 Bridge 完整导出 HF
checkpoint。它记录 sparse 更新后的 greedy 推理结果，关闭该推理进程，再在相同两张卡
启动全新推理进程，直接加载导出的本地 checkpoint。两路使用完全相同的 prompt token IDs
和采样参数，要求生成 token IDs 和每个 token 的 logprob 严格相等；缺失、非有限值或
任意差异均失败。对照前也会重复推理一次，检查更新后的模型自身是否可精确复现。

```bash
MODEL_PATH=/mnt/weight/Qwen3-30B-A3B \
PROMPT_DATA_PATH=/path/to/dapo-math-17k.jsonl \
SPARSE_HCCL_RESULT_DIR=/local/nvme/sparse-check-001 \
SPARSE_HCCL_BASELINE_PROMPT="What is 17 + 25? Give the answer in one sentence." \
python tests/test_qwen3_30B_A3B_sparse_hccl_npu.py
```

### 磁盘传输

1. **Seed。** 第一次同步时，训练端为每个参数捕获一份 CPU snapshot——从 `--hf-checkpoint` seed，而这正是每个 rollout host 物化本地 checkpoint 的来源。此次不发布任何东西；这份 snapshot 就是下一次同步 diff 的基准。训练端同时发出 `target_version=0` 的 `/pull_weights`，让每个 host 现在就物化本地 base，与 snapshot 捕获重叠进行。
2. **Publish。** 之后每次同步，训练端把每个 gather 出的 HF tensor 与 snapshot 做 diff，编码、压缩，写到 `--update-weight-disk-dir` 下的新版本目录 `weight_v{N:06d}/`。该目录是一份 canonical HF checkpoint——`model-NNNNN.safetensors` 文件装着压缩后的 diff tensor，外加 `model.safetensors.index.json`（tensor 名 → 文件）承载 apply 元数据——所以这个产物是可移植的，不绑定训练端的并行 layout。随后 snapshot 推进到新值，供下次 diff。
3. **Pull。** 训练端对每个 engine 调用 `/pull_weights`。engine 内部把请求广播到每个节点的每个 rank；每个 host 把新版本的 delta 原地 apply 进它的本地 checkpoint（host 级文件锁把同 host 的多个 rank 合并成一次 apply）。apply 在 tensor 之间并行，并逐 tensor 校验（见"完整性"）；只有**每一个 host** 都持有校验通过的 checkpoint，该调用才报告成功。

   `/pull_weights` 并不绑定 delta：每个发布的版本是自描述的。若某个版本是一份普通的完整 HF
   checkpoint（index 中没有 delta 元数据），pull 就直接整份拷贝——同时重置链条，因此晚加入的
   新 host 从最近的完整版本 seed，而不必回放全部 delta，旧的 delta 也可以被清理。vime 的
   full 模式 disk 同步在设置了 `--update-weight-local-checkpoint-dir` 时正是走这条路径。
4. **Reload。** engine 通过原生 `update_weights_from_disk` 路径 reload 打过补丁的本地 checkpoint——权重加载代码从不接触 delta 格式。

由于 snapshot 是从 `--hf-checkpoint`（engine 真正的 base）seed，而不是从当前 GPU 权重 seed，即使 Megatron→HF 往返不是逐字节相等（例如 embedding / LM head 中被裁掉的 vocab padding 行），该方案对任意模型也都正确。

## 编码

两种编码都是字节级、与 dtype 无关的，所以量化 checkpoint 也走同一条路径。engine 从每个版本的 index 元数据读取所用编码。

- **`xor`**（默认）：写 `new ^ old`。wire 最小、apply 最快（顺序访问、对 cache 友好；未变化的字节是 0，被压缩器压到极小）。它是一个对合（involution），所以必须**恰好对正确的 base apply 一次**——apply 两次会还原。
- **`overwrite`**：写变化的位置及其新的绝对值。wire 更大、apply 是对 cache 不友好的分散写，但**幂等**：重复 apply（或把部分 apply 的 delta 补完）无论执行多少次都收敛到同一状态。当“可重复 apply”比 wire 大小更重要时用它。

## 完整性

训练端把每个 tensor 新状态的逐 tensor checksum 存进版本里。apply 之后每个 host 重新计算 checksum，**任何不匹配都会 raise**——失败会通过 `/pull_weights` 的响应传回，所以损坏的 delta 或错误的 base 会直接报错失败，而不会把坏权重提供出去。apply 还拒绝乱序执行：一个版本只会在它声明的 base 版本之上 apply。

`--update-weight-delta-checksum` 选择算法。checksum 不是 apply 的瓶颈（apply 受解压 + XOR 限制），所以这是一个 digest 属性的选择，而非速度选择：`xxh3-128`（默认）是最宽的快速非加密 digest；`blake3` 是加密 digest，用于不可信存储；`adler32` 用于与期望它的系统互操作。

## 共享文件系统可见性 hook

在 POSIX 共享文件系统（NFS、Lustre……）上不需要额外步骤。对于需要显式 commit/refresh 才能让写入跨 host 可见的对象存储挂载，可以提供两个可选 hook（通过 import 路径加载——vime 和 vllm 里都不存在任何厂商特定代码）：

- `--custom-update-weight-post-write-path`（vime，训练端）：在一个版本的文件写完之后、通知 engine 读取之前调用（例如把待写入数据上传到底层对象存储）。签名：`hook(args, version_dir, rollout_engines)`。
- `--custom-update-weight-pre-read-path`（vime，engine 端）：在每个 host 上、`/pull_weights` 读取 delta 目录之前于 engine 内部调用（例如刷新挂载视图）。签名：`hook(delta_dir, target_version)`。
