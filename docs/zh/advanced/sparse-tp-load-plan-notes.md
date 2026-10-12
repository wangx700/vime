# 接收端 TP load-plan 优化保留说明

本分支 `preserve/sparse-tp-load-plan` 从 PR447 本地代码版本
`9c591df6af370b6c9f547b67648f92dcd6dec9e0` 创建。
保留 `docker/npu_patch/vllm-ascend.patch` 中的备用实现，不改变实际权重更新链路。

## 目的

通用 sparse loader 将变化位置恢复成完整 HF shape 的 NaN 临时张量，
再通过原生 `model.load_weights` 处理 TP 切片和融合参数。
load-plan 方案尝试缓存 HF 参数到运行时参数的映射，直接计算本地索引，
并将 Q/K/V 或 gate/up 等指向同一融合参数的更新合并为 `index_copy_`。
目标是减少完整临时张量、重复映射及 loader 调用；收益需单独测量。

该方案是接收端优化，与发送端 `rank_patches` 定向 P2P 不同；
保持广播传输也可以接入 load-plan。

## 核心接口

```text
apply_sparse_hf_patch：单 patch 包装入口
  → apply_sparse_hf_patches：批量校验、映射、合并和写入
      → _validate_sparse_hf_patch：HF 索引验证
      → _build_sparse_load_plan：解析目标参数与布局
          → _map_qwen3_stacked_name：HF 名称到融合参数名
          → _get_sparse_load_plan：读取实际 TP 范围和融合段偏移
      → SparseLoadPlan.map_indices：HF 全局索引到本地索引
      → index_copy_：覆盖绝对新值
      → _apply_sparse_hf_patch_legacy：不支持布局时原生 loader 回退
```

`SparseLoadPlanCache` 按 HF 名称及完整 shape 缓存计划。
计划包含目标参数、分片维度、HF 起点、分片大小和融合段偏移。
QKV 还利用 KV head 副本信息，不能简单按 TP rank 均分 K/V。

实现位于 patch 新增的
`vllm_ascend/distributed/weight_transfer/sparse_weight_patch.py`。
相关单测位于 patch 新增的
`tests/ut/distributed/weight_transfer/test_sparse_weight_patch.py`。

## 为什么当前未使用

当前 `SparseHCCLWeightTransferEngine.receive_weights` 调用的是
`apply_sparse_hf_patches_with_loader(..., direct_moe=True)`，
不是 `apply_sparse_hf_patch/apply_sparse_hf_patches`。
普通 QKV/线性参数使用原生 loader，支持的 Qwen3-MoE 专家使用专项直写。
本分支保留的 load-plan 接口当前只有单测调用。

可确定的是当前入口没有接入这套实现；没有足够历史证据将原因归结为
某个具体故障或认定该实现错误。
它绕过原生 loader，依赖模块的 TP/融合元数据和可证明的存储布局，
因此不能仅凭单测通过直接替换 NPU 主链路。

## 与 TP2/PP2/EP2 及评测的关系

当前主链路仍处理 TP：原生 loader 完成普通参数 TP 切片和融合；
专家专项路径读取实际 MoE TP 并处理 EP map；PP 在接收入口过滤。
没有使用 load-plan，不意味着没有 TP 支持。

full HCCL 路径不依赖此稀疏 load-plan。
已检查的稀疏优化提交 `c81162b6` 的接收入口使用
`apply_sparse_hf_patches_with_loader`。
不能只凭当前入口或评测数字推断所有历史 benchmark 的实现；
对外引用每组性能数据仍应保留运行代码版本及原始日志。

该优化以已知 Qwen3 未量化 TP 布局为主要目标，不应描述为完整
PP/EP/量化/动态 EPLB 通用方案，也不应描述为只能用于纯 TP。
未来可以与 PP 过滤、EP 专项处理组合，但需独立集成验证。

## 后续接入条件

- 确认真实 Ascend 参数及模块提供所需 TP/融合元数据。
- 保留连续性、ND 格式、packed layout、dtype 等安全检查和回退。
- 对 QKV、KV 复制、行列分片、词表 padding、融合参数进行验证。
- 校验重复/分片 patch 的覆盖顺序以及未变化位置保持不变。
- 实际 NPU 模型上比较直接路径与完整导出或原生 loader 的结果。
- 分开验证纯 TP 和 PP2/TP2/EP2；单测不能替代端到端验证。
- 同条件测量更新耗时与临时内存，不将备用路径归入当前主链路收益。

本次仅创建保留分支及说明，不删除 PR447 中的代码，也不启用新路径。
