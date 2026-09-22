"""Megatron shard-local delta export into final Hugging Face coordinates.

This is the VIME counterpart of verl's ``megatron/delta_export.py``. VIME
does not own a ``MegatronBridge`` conversion-task object, so the same NaN
probe is run through VIME's existing ``convert_to_hf`` mapping. Communication
is synthesized locally: the current TP shard is placed at its real global
location and all peer shards are represented by NaNs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import prod

import torch
import torch.distributed as dist
from megatron.core import mpu

from vime.utils.common import is_npu
from vime.utils.types import ParamInfo

from ..megatron_to_hf import convert_to_hf
from .delta_spec import ShardSpec
from .delta_sync.sparse_gather import shard_delta_indices

logger = logging.getLogger(__name__)


@dataclass
class McoreParamExport:
    """One globally ordered Megatron parameter directory row."""

    info: ParamInfo
    param: torch.Tensor | None
    spec: ShardSpec
    slots: list[tuple[str, tuple[int, ...]]] | None = None

    @property
    def name(self) -> str:
        return self.info.name


def _is_expert(name: str) -> bool:
    return ".experts." in name


def _tp_geometry(info: ParamInfo) -> tuple[int | None, int, int]:
    attrs = info.attrs
    if not attrs.get("tensor_model_parallel", False) or attrs.get("parallel_mode") == "duplicated":
        return None, 1, 0
    expert = _is_expert(info.name)
    size = (
        mpu.get_expert_tensor_parallel_world_size()
        if expert
        else mpu.get_tensor_model_parallel_world_size()
    )
    rank = (
        mpu.get_expert_tensor_parallel_rank()
        if expert
        else mpu.get_tensor_model_parallel_rank()
    )
    if size == 1:
        return None, 1, 0
    dim = int(attrs.get("partition_dim", -1))
    if "linear_fc1.weight" in info.name or "linear_fc1.bias" in info.name:
        if is_npu():
            # NPU Megatron shards linear_fc1 (GLU) along dim 0; mirrors the
            # platform layer's adjust_tp_partition_dim used upstream.
            dim = 0
    if "linear_fc2.weight" in info.name and dim == 0:
        dim = 1
    if dim < 0 or dim >= len(info.shape):
        raise ValueError(f"invalid TP partition dim {dim} for {info.name} shape={tuple(info.shape)}")
    return dim, size, rank


def _full_shape(info: ParamInfo) -> tuple[int, ...]:
    shape = [int(x) for x in info.shape]
    dim, size, _rank = _tp_geometry(info)
    if dim is not None:
        shape[dim] *= size
    return tuple(shape)


def _synthesize_full_probe(info: ParamInfo, local_probe: torch.Tensor) -> torch.Tensor:
    """Place this rank's NaN-masked TP shard into a NaN full tensor.

    The layout exactly mirrors ``all_gather_params_async`` including the GLU
    reordering used by ``linear_fc1``.
    """
    dim, size, rank = _tp_geometry(info)
    if dim is None:
        return local_probe
    full = torch.full(
        _full_shape(info),
        float("nan"),
        dtype=local_probe.dtype,
        device=local_probe.device,
    )
    local_width = int(local_probe.shape[dim])
    if "linear_fc1.weight" in info.name or "linear_fc1.bias" in info.name:
        if local_width % 2:
            raise ValueError(f"GLU TP shard is not even for {info.name}: {local_width}")
        half = local_width // 2
        first, second = local_probe.chunk(2, dim=dim)
        first_slice = [slice(None)] * local_probe.ndim
        first_slice[dim] = slice(rank * half, (rank + 1) * half)
        second_slice = [slice(None)] * local_probe.ndim
        second_start = size * half + rank * half
        second_slice[dim] = slice(second_start, second_start + half)
        full[tuple(first_slice)] = first
        full[tuple(second_slice)] = second
    else:
        target = [slice(None)] * local_probe.ndim
        target[dim] = slice(rank * local_width, (rank + 1) * local_width)
        full[tuple(target)] = local_probe
    return full


def _route_spec(info: ParamInfo, owner: bool) -> ShardSpec:
    """Apply the DP/CP/TP/EP/PP routing rules documented by verl."""
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    tp_dim, tp_size, _ = _tp_geometry(info)
    expert = _is_expert(info.name)
    dp_cp_primary = mpu.get_data_parallel_rank(with_context_parallel=True) == 0

    if pp_size > 1:
        if expert:
            contributes = owner and mpu.get_expert_data_parallel_rank() == 0
        elif tp_dim is not None and tp_size > 1:
            contributes = owner and dp_cp_primary
        else:
            contributes = owner and mpu.get_tensor_model_parallel_rank() == 0 and dp_cp_primary
        return ShardSpec(_full_shape(info), gather_group=dist.group.WORLD, contributes=contributes)

    if expert:
        contributes = owner and mpu.get_expert_data_parallel_rank() == 0
        return ShardSpec(
            _full_shape(info),
            gather_group=mpu.get_expert_tensor_and_model_parallel_group(),
            contributes=contributes,
        )
    if tp_dim is not None and tp_size > 1:
        return ShardSpec(
            _full_shape(info),
            gather_group=mpu.get_tensor_model_parallel_group(),
            contributes=owner and dp_cp_primary,
        )
    return ShardSpec(
        _full_shape(info),
        gather_group=None,
        contributes=owner and dist.get_rank() == 0,
    )


def _probe_outputs(
    args,
    model_name: str,
    quantization_config,
    record: McoreParamExport,
    local_indices: torch.Tensor,
    local_values: torch.Tensor,
) -> list[tuple[str, torch.Tensor]]:
    if record.param is None:
        return []
    if not record.param.is_floating_point():
        raise TypeError(f"delta NaN probe requires floating point weights: {record.name}")
    local_probe = torch.full_like(record.param, float("nan"))
    if local_indices.numel():
        local_probe.view(-1)[local_indices] = local_values
    full_probe = _synthesize_full_probe(record.info, local_probe)
    return convert_to_hf(
        args,
        model_name,
        record.name,
        full_probe,
        quantization_config,
    )


def build_export_index(
    args,
    model_name: str,
    quantization_config,
    param_infos: list[ParamInfo],
    local_weights: dict[str, torch.Tensor],
) -> list[McoreParamExport]:
    """Build the identical global directory and per-row HF slot table."""
    rank = dist.get_rank()
    index: list[McoreParamExport] = []
    local_rows: list[list[tuple[str, tuple[int, ...]]] | None] = []
    for info in param_infos:
        param = local_weights.get(info.name) if info.src_rank == rank else None
        record = McoreParamExport(info, param, _route_spec(info, param is not None))
        index.append(record)
        if param is None:
            local_rows.append(None)
            continue
        empty_indices = torch.empty(0, dtype=torch.int64, device=param.device)
        empty_values = torch.empty(0, dtype=param.dtype, device=param.device)
        outputs = _probe_outputs(
            args,
            model_name,
            quantization_config,
            record,
            empty_indices,
            empty_values,
        )
        local_rows.append([(name, tuple(int(x) for x in tensor.shape)) for name, tensor in outputs])

    gathered: list = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_rows)
    if any(len(rows) != len(index) for rows in gathered):
        raise RuntimeError("Megatron delta directory row counts differ across ranks")
    for row_index, record in enumerate(index):
        union: dict[tuple[str, tuple[int, ...]], None] = {}
        for rows in gathered:
            row = rows[row_index]
            if row is not None:
                for name, shape in row:
                    union[(name, tuple(shape))] = None
        record.slots = list(union)
        if not record.slots:
            logger.warning("delta export drops unowned row %s", record.name)
    return [record for record in index if record.slots]


def prime_delta_snapshots(
    index: list[McoreParamExport],
    snapshots: dict[str, torch.Tensor],
    *,
    pin: bool = False,
) -> None:
    """Snapshot this rank's owned shards after the dense seed completes."""
    for record in index:
        if record.param is None:
            continue
        local = record.param.detach().contiguous().view(-1)
        snapshot = snapshots.get(record.name)
        if snapshot is None or snapshot.numel() != local.numel():
            snapshot = torch.empty_like(local, device="cpu", pin_memory=pin)
            snapshots[record.name] = snapshot
        snapshot.copy_(local, non_blocking=True)


def iter_delta_entries(
    args,
    model_name: str,
    quantization_config,
    index: list[McoreParamExport],
    snapshots: dict[str, torch.Tensor],
):
    """Yield verl-compatible final-HF-coordinate sparse directory entries."""
    for record in index:
        slots = record.slots
        assert slots is not None
        if record.param is None:
            dtype = record.info.dtype
            local_indices = torch.empty(0, dtype=torch.int64, device=torch.accelerator.current_device())
            local_values = torch.empty(0, dtype=dtype, device=local_indices.device)
            outputs: list[tuple[str, torch.Tensor]] = []
        else:
            local = record.param.detach().contiguous().view(-1)
            snapshot = snapshots.get(record.name)
            if snapshot is None or snapshot.numel() != local.numel():
                raise RuntimeError(f"{record.name}: no seed snapshot for this shard")
            if record.spec.contributes:
                base = snapshot.to(local.device, non_blocking=True)
                local_indices, local_values = shard_delta_indices(local, base, 0)
            else:
                local_indices = torch.empty(0, dtype=torch.int64, device=local.device)
                local_values = torch.empty(0, dtype=local.dtype, device=local.device)
            snapshot.copy_(local, non_blocking=True)
            outputs = (
                _probe_outputs(
                    args,
                    model_name,
                    quantization_config,
                    record,
                    local_indices,
                    local_values,
                )
                if local_indices.numel()
                else []
            )

        output_map = dict(outputs)
        unknown = set(output_map) - {name for name, _shape in slots}
        if unknown:
            raise RuntimeError(f"{record.name}: delta probe produced unknown HF slots: {sorted(unknown)}")
        counts = torch.zeros(len(slots), dtype=torch.int64)
        index_parts: list[torch.Tensor] = []
        value_parts: list[torch.Tensor] = []
        for slot_index, (slot_name, slot_shape) in enumerate(slots):
            if prod(slot_shape) >= 2**31:
                raise OverflowError(f"{slot_name}: sparse int32 position limit exceeded")
            output = output_map.get(slot_name)
            if output is None:
                continue
            flat = output.reshape(-1)
            changed = (~torch.isnan(flat)).nonzero(as_tuple=False).view(-1)
            counts[slot_index] = changed.numel()
            if changed.numel():
                index_parts.append(changed.to(torch.int32))
                value_parts.append(flat[changed])
        device = local_values.device
        hf_indices = (
            torch.cat(index_parts)
            if index_parts
            else torch.empty(0, dtype=torch.int32, device=device)
        )
        hf_values = (
            torch.cat(value_parts)
            if value_parts
            else torch.empty(0, dtype=local_values.dtype, device=device)
        )
        yield (
            slots,
            str(local_values.dtype).replace("torch.", ""),
            counts,
            hf_indices,
            hf_values,
            record.spec.gather_group,
        )
