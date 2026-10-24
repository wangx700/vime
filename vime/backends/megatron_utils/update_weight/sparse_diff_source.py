"""Sparse weight sync: diff each HF export and drive the sparse trainer engine.

The sparse backends take per-round ``(indices, values)`` patches instead of a
``WeightSource`` (a stable, full-parameter stream). VIME already exports the full
HF-layout weights every sync, so the only new piece is a diff against the
previous export: keep the previous export on the host and emit the changed
elements as flat-index patches.

The sending engine is the platform's *trainer* engine
(``SparseHCCLTrainerWeightTransferEngine`` on Ascend), which owns the HCCL
rendezvous and the broadcast; the inference side is
``SparseHCCLWeightTransferEngine`` inside each rollout engine and needs no VIME
changes beyond the ``sparse_hccl`` backend selection.
"""

from __future__ import annotations

import socket
import time
from collections.abc import Iterator, Sequence
from typing import Any

import torch
import torch.distributed as dist

from vime.utils.distributed_utils import get_gloo_group


def _sparse_engine_classes() -> tuple[Any, Any]:
    """Return the platform's sparse trainer init-info and engine classes."""
    try:
        from vllm_ascend.distributed.weight_transfer.sparse_hccl_engine import (
            SparseHCCLTrainerInitInfo,
            SparseHCCLTrainerWeightTransferEngine,
        )
    except ImportError:
        from vllm.distributed.weight_transfer.sparse_nccl_engine import (
            SparseNCCLTrainerInitInfo as SparseHCCLTrainerInitInfo,
        )
        from vllm.distributed.weight_transfer.sparse_nccl_engine import (
            SparseNCCLTrainerWeightTransferEngine as SparseHCCLTrainerWeightTransferEngine,
        )
    return SparseHCCLTrainerInitInfo, SparseHCCLTrainerWeightTransferEngine


def _sparse_weight_patch() -> Any:
    from vllm.distributed.weight_transfer.sparse_nccl_engine import SparseWeightPatch

    return SparseWeightPatch


class SparseDiffSource:
    """Turn consecutive full HF exports into one round's sparse patches.

    Only the trainer sender rank diffs; every other rank consumes the export
    without building patches because it never sends.
    """

    def __init__(self, source: Any, *, snapshot_in_host: bool = True) -> None:
        self._source = source
        self._snapshot_in_host = snapshot_in_host
        self._previous: dict[str, torch.Tensor] = {}
        self.stats: dict[str, float] = {}
        self.patches: list[Any] = []

    @property
    def draft(self) -> bool:
        return self._source.draft

    @draft.setter
    def draft(self, value: bool) -> None:
        if value:
            raise RuntimeError("Sparse weight sync has no draft (MTP) path.")
        self._source.draft = False

    def collect(self) -> list[Any]:
        """Iterate one full export and return the changed-element patches."""
        SparseWeightPatch = _sparse_weight_patch()
        patches: list[Any] = []
        changed = 0
        exported = 0
        wire = 0
        diff_seconds = 0.0
        started = time.perf_counter()

        for name, tensor in self._source:
            if not self._snapshot_in_host:
                continue
            step = time.perf_counter()
            current = tensor.detach().to("cpu", copy=True).contiguous()
            exported += current.numel()
            previous = self._previous.get(name)
            if previous is None:
                # First sync has no baseline: the reference server was loaded
                # from the same checkpoint, so nothing has changed yet.
                self._previous[name] = current
                diff_seconds += time.perf_counter() - step
                continue

            flat_current = current.view(-1)
            flat_previous = previous.view(-1)
            indices = torch.nonzero(flat_current != flat_previous, as_tuple=False).flatten()
            if indices.numel():
                values = flat_current[indices].clone()
                patches.append(
                    SparseWeightPatch(
                        name=name,
                        full_shape=tuple(current.shape),
                        indices=indices.int(),
                        values=values,
                    )
                )
                changed += int(indices.numel())
                wire += int(indices.numel()) * (indices.element_size() + values.element_size())
            self._previous[name] = current
            diff_seconds += time.perf_counter() - step

        self.patches = patches
        self.stats = {
            "sparse_exported_elements": exported,
            "sparse_changed_elements": changed,
            "sparse_changed_ratio": (changed / exported) if exported else 0.0,
            "sparse_index_value_bytes": wire,
            "sparse_patch_count": len(patches),
            "sparse_collect_seconds": diff_seconds,
            "sparse_collect_inclusive_seconds": time.perf_counter() - started,
        }
        return patches

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        return iter(self._source)


class SparseTrainerAdapter:
    """Expose the sparse trainer engine through the dense trainer surface.

    ``UpdateWeightFromDistributed`` expects ``.client``, ``send_weights()`` and
    ``shutdown()``; the sparse engine instead takes its payload per call.
    """

    def __init__(self, client: Any, source: SparseDiffSource, engine: Any) -> None:
        self.client = client
        self.source = source
        self.engine = engine

    def send_weights(self) -> None:
        self.engine.send_weights(self.source.collect())

    def pop_metrics(self) -> dict[str, float]:
        metrics, self.source.stats = self.source.stats, {}
        return metrics

    def shutdown(self) -> None:
        self.engine.shutdown()


def create_sparse_trainer(
    client: Any,
    source: Any,
    engine_gpu_counts: Sequence[int],
) -> SparseTrainerAdapter:
    """Rendezvous with the rollout engines and build the sparse trainer engine."""
    import ray

    SparseHCCLTrainerInitInfo, SparseHCCLTrainerWeightTransferEngine = _sparse_engine_classes()
    diff_source = SparseDiffSource(source)
    rendezvous: list[Any] = [None]
    if dist.get_rank() == 0:
        with socket.socket() as sock:
            sock.bind(("", 0))
            rendezvous[0] = (ray._private.services.get_node_ip_address(), sock.getsockname()[1])
    dist.broadcast_object_list(rendezvous, src=0, group=get_gloo_group())
    master_address, master_port = rendezvous[0]
    engine = SparseHCCLTrainerWeightTransferEngine.trainer_init(
        SparseHCCLTrainerInitInfo(
            master_address,
            master_port,
            sum(engine_gpu_counts) + 1,
            rank=dist.get_rank(),
        ),
        client=client,
    )
    return SparseTrainerAdapter(client, diff_source, engine)
