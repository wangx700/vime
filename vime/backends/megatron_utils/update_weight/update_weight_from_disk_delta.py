from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import time
from argparse import Namespace
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import ray
import safetensors.numpy
import torch
import torch.distributed as dist
import zstandard
from ray.actor import ActorHandle

from vime.utils.disk_delta import NUM_WORKERS, checksum, make_tensor_reader, overwrite_encode
from vime.utils.distributed_utils import get_gloo_group
from vime.utils.profile_utils import should_profile_update_weight

from .hf_weight_iterator_base import HfWeightIteratorBase

logger = logging.getLogger(__name__)


class UpdateWeightFromDiskDelta:
    """Publish byte-level HF weight deltas and reload them through local checkpoints.

    This is intentionally independent of ``UpdateWeightFromDistributed``: all
    training ranks still participate in the Ascend PP/TP/EP conversion iterator,
    while only global rank zero publishes the canonical result.  No HCCL group is
    created for this transport.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        self.args = args
        self.weights_getter = weights_getter
        self.weight_version = 0
        self._update_weight_call = 0
        self.update_weight_metrics: dict[str, float] = {}
        self.rollout_engines: list[ActorHandle] = []
        self.delta_dir = args.update_weight_disk_dir
        self.delta_encoding = args.update_weight_delta_encoding
        self.checksum_algorithm = args.update_weight_delta_checksum
        self._iterator = HfWeightIteratorBase.create(
            args=args,
            model=model,
            model_name=model_name,
            quantization_config=quantization_config,
        )
        self._snapshot: dict[str, np.ndarray] = {}
        self._baseline_captured = False
        self._post_write_hook: Callable | None = None
        if args.custom_update_weight_post_write_path:
            from vime.utils.misc import load_function

            self._post_write_hook = load_function(args.custom_update_weight_post_write_path)

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        del rollout_engine_lock, engine_gpu_counts, engine_gpu_offsets
        self.rollout_engines = list(rollout_engines)

    def disconnect_rollout_engines(self) -> None:
        return

    def pop_metrics(self) -> dict[str, float]:
        metrics, self.update_weight_metrics = self.update_weight_metrics, {}
        return metrics

    @torch.no_grad()
    def update_weights(self) -> None:
        self._update_weight_call += 1
        if not self._baseline_captured:
            self._capture_baseline()
            self._baseline_captured = True
            return

        self.weight_version += 1
        self._stage_metrics: dict[str, float] = {}
        self._publish()
        self._reload_engines()
        self._record_metrics()

    def _capture_baseline(self) -> None:
        """Use the serving checkpoint as the byte-exact version-zero baseline."""
        pulls = []
        if dist.get_rank() == 0:
            shutil.rmtree(self.delta_dir, ignore_errors=True)
            os.makedirs(self.delta_dir, exist_ok=True)
            if self._post_write_hook is not None:
                self._post_write_hook(self.args, self.delta_dir, self.rollout_engines)
            pulls = [engine.pull_weights.remote(target_version=0) for engine in self.rollout_engines]
        dist.barrier(group=get_gloo_group())

        read_hf = make_tensor_reader(self.args.hf_checkpoint) # 本地HF权重？
        for name, tensor in self._iter_hf_tensors(progress_desc="Capture disk delta baseline"):
            try:
                self._snapshot[name] = read_hf(name)
            except KeyError:
                self._snapshot[name] = _tensor_bytes(tensor)[0]
                logger.warning("delta baseline: %s absent from hf_checkpoint; using current converted weight", name)

        if dist.get_rank() == 0:
            ray.get(pulls)
            logger.info("[disk delta] captured baseline for %d tensors", len(self._snapshot))

    def _publish(self) -> None:
        started = time.perf_counter()
        self._encode_delta()
        encoded = time.perf_counter()
        dist.barrier(group=get_gloo_group())
        if dist.get_rank() == 0:
            write_started = time.perf_counter()
            self._write_delta_files()
            self._stage_metrics["perf/update_weights_delta_write_time"] = time.perf_counter() - write_started
        dist.barrier(group=get_gloo_group())
        if dist.get_rank() == 0:
            self._stage_metrics["perf/update_weights_delta_encode_time"] = encoded - started
            self._stage_metrics["perf/update_weights_delta_publish_time"] = time.perf_counter() - started

    def _iter_hf_tensors(self, *, progress_desc: str):
        """All ranks execute conversion collectives; only rank zero publishes tensors."""
        for chunk in self._iterator.get_hf_weight_chunks(self.weights_getter(), progress_desc=progress_desc):
            if dist.get_rank() == 0:
                yield from chunk

    def _encode_delta(self) -> None:
        self._version_dir = os.path.join(self.delta_dir, f"weight_v{self.weight_version:06d}")
        self._delta: dict[str, np.ndarray] = {}
        self._checksums: dict[str, str] = {}
        self.changed_bytes = 0
        self.total_bytes = 0
        self.wire_bytes = 0
        self._stage_metrics.update(
            {
                "perf/update_weights_delta_buffer_pool_time": 0.0,
                "perf/update_weights_delta_hf_conversion_time": 0.0,
                "perf/update_weights_delta_tensor_copy_time": 0.0,
                "perf/update_weights_delta_xor_cpu_time": 0.0,
                "perf/update_weights_delta_nonzero_cpu_time": 0.0,
                "perf/update_weights_delta_compress_cpu_time": 0.0,
                "perf/update_weights_delta_checksum_cpu_time": 0.0,
                "perf/update_weights_delta_worker_cpu_time": 0.0,
                "perf/update_weights_delta_worker_wait_time": 0.0,
                "perf/update_weights_delta_serialize_time": 0.0,
                "perf/update_weights_delta_data_write_time": 0.0,
                "perf/update_weights_delta_index_write_time": 0.0,
            }
        )

        if dist.get_rank() != 0:
            # Do not return: every rank must execute the conversion iterator's collectives.
            for _name, _tensor in self._iter_hf_tensors(progress_desc="Encode disk delta"):
                pass
            return

        os.makedirs(self._version_dir, exist_ok=True)
        buffer_pool_started = time.perf_counter()
        max_bytes = max((value.nbytes for value in self._snapshot.values()), default=0)
        free_buffers: queue.Queue[torch.Tensor] = queue.Queue()
        use_pinned = max_bytes > 0
        if use_pinned:
            try:
                pool_size = max(2, min(2 * NUM_WORKERS, (8 << 30) // max(max_bytes, 1)))
                for _ in range(pool_size):
                    free_buffers.put(torch.empty(max_bytes, dtype=torch.uint8, pin_memory=True))
            except RuntimeError as exc:
                logger.warning("Pinned host buffers unavailable (%s); using pageable copies", exc)
                use_pinned = False
        self._stage_metrics["perf/update_weights_delta_buffer_pool_time"] = (
            time.perf_counter() - buffer_pool_started
        )

        def diff_and_compress(
            name: str, new: np.ndarray, leased_buffer: torch.Tensor | None
        ) -> tuple[str, np.ndarray, np.ndarray | None, str | None, int, dict[str, float]]:
            old = self._snapshot[name]
            try:
                if new.nbytes != old.nbytes:
                    raise ValueError(f"Delta tensor size changed for {name}: {old.nbytes} != {new.nbytes}")
                diff_started = time.perf_counter()
                if self.delta_encoding == "xor":
                    diff = new ^ old
                    diff_finished = time.perf_counter()
                    nonzero_started = diff_finished
                    changed = int(np.count_nonzero(diff))
                else:
                    mask = new != old
                    diff_finished = time.perf_counter()
                    nonzero_started = diff_finished
                    changed = int(np.count_nonzero(mask))
                    diff = overwrite_encode(new, mask)
                nonzero_finished = time.perf_counter()
                timings = {
                    "perf/update_weights_delta_xor_cpu_time": diff_finished - diff_started,
                    "perf/update_weights_delta_nonzero_cpu_time": nonzero_finished - nonzero_started,
                    "perf/update_weights_delta_compress_cpu_time": 0.0,
                    "perf/update_weights_delta_checksum_cpu_time": 0.0,
                }
                compressed = None
                digest = None
                if changed:
                    compress_started = time.perf_counter()
                    compressed = np.frombuffer(
                        zstandard.ZstdCompressor(level=1).compress(diff), dtype=np.uint8
                    )
                    timings["perf/update_weights_delta_compress_cpu_time"] = (
                        time.perf_counter() - compress_started
                    )
                    # The version chain already proves which base is being patched.
                    # Hash the wire representation so corruption is still detected
                    # without another full-model pass over the materialized tensor.
                    checksum_started = time.perf_counter()
                    digest = checksum(self.checksum_algorithm, compressed)
                    timings["perf/update_weights_delta_checksum_cpu_time"] = (
                        time.perf_counter() - checksum_started
                    )

                if leased_buffer is not None:
                    # Keep the NPU-to-host copy leased until the worker is done.
                    # Moving the unavoidable snapshot copy here overlaps it with
                    # conversion and compression of neighbouring tensors.
                    if old.flags.writeable:
                        np.copyto(old, new)
                        new = old
                    else:
                        new = new.copy()
                return name, new, compressed, digest, changed, timings
            finally:
                if leased_buffer is not None:
                    free_buffers.put(leased_buffer)

        pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)
        inflight: deque = deque()
        try:
            tensor_iterator = iter(self._iter_hf_tensors(progress_desc="Encode disk delta"))
            while True:
                conversion_started = time.perf_counter()
                try:
                    name, tensor = next(tensor_iterator)
                except StopIteration:
                    self._stage_metrics["perf/update_weights_delta_hf_conversion_time"] += (
                        time.perf_counter() - conversion_started
                    )
                    break
                self._stage_metrics["perf/update_weights_delta_hf_conversion_time"] += (
                    time.perf_counter() - conversion_started
                )
                copy_started = time.perf_counter()
                new, leased_buffer = _tensor_bytes(
                    tensor,
                    free_buffers=free_buffers if use_pinned else None,
                    lease_buffer=use_pinned,
                )
                self._stage_metrics["perf/update_weights_delta_tensor_copy_time"] += (
                    time.perf_counter() - copy_started
                )
                self.total_bytes += new.nbytes
                inflight.append(pool.submit(diff_and_compress, name, new, leased_buffer))
                if len(inflight) >= 2 * NUM_WORKERS:
                    wait_started = time.perf_counter()
                    self._collect_encoded(inflight.popleft())
                    self._stage_metrics["perf/update_weights_delta_worker_wait_time"] += (
                        time.perf_counter() - wait_started
                    )
            while inflight:
                wait_started = time.perf_counter()
                self._collect_encoded(inflight.popleft())
                self._stage_metrics["perf/update_weights_delta_worker_wait_time"] += (
                    time.perf_counter() - wait_started
                )
        finally:
            pool.shutdown()
        self._stage_metrics["perf/update_weights_delta_worker_cpu_time"] = sum(
            self._stage_metrics[name]
            for name in (
                "perf/update_weights_delta_xor_cpu_time",
                "perf/update_weights_delta_nonzero_cpu_time",
                "perf/update_weights_delta_compress_cpu_time",
                "perf/update_weights_delta_checksum_cpu_time",
            )
        )

    def _collect_encoded(self, future) -> None:
        name, new, compressed, digest, changed, timings = future.result()
        for metric, elapsed in timings.items():
            self._stage_metrics[metric] += elapsed
        self._snapshot[name] = new
        if changed:
            self.changed_bytes += changed
            assert compressed is not None and digest is not None
            self._delta[name] = compressed
            self._checksums[name] = digest

    def _write_delta_files(self) -> None:
        if self._delta:
            filename = "model-00000-of-00001.safetensors"
            serialize_started = time.perf_counter()
            blob = safetensors.numpy.save(self._delta, metadata=self._checksums)
            self._stage_metrics["perf/update_weights_delta_serialize_time"] = (
                time.perf_counter() - serialize_started
            )
            self.wire_bytes = len(blob)
            data_write_started = time.perf_counter()
            _atomic_write(os.path.join(self._version_dir, filename), blob)
            self._stage_metrics["perf/update_weights_delta_data_write_time"] = (
                time.perf_counter() - data_write_started
            )
        else:
            filename = None
        index = {
            "metadata": {
                "version": f"{self.weight_version:06d}",
                "base_version": f"{self.weight_version - 1:06d}",
                "delta_encoding": self.delta_encoding,
                "compression_format": "zstd",
                "checksum_format": self.checksum_algorithm,
                "checksum_scope": "compressed_delta",
            },
            "weight_map": {name: filename for name in self._delta},
        }
        index_write_started = time.perf_counter()
        _atomic_write(
            os.path.join(self._version_dir, "model.safetensors.index.json"),
            json.dumps(index).encode(),
        )
        self._stage_metrics["perf/update_weights_delta_index_write_time"] = (
            time.perf_counter() - index_write_started
        )

    def _reload_engines(self) -> None:
        if self._post_write_hook is not None:
            self._post_write_hook(self.args, self._version_dir, self.rollout_engines)
        dist.barrier(group=get_gloo_group())
        if dist.get_rank() == 0:
            profile_inference = should_profile_update_weight(self._update_weight_call)
            inference_profile_started = False
            if profile_inference:
                ray.get([engine.start_profile.remote() for engine in self.rollout_engines])
                inference_profile_started = True
            # Include pull_weights: it materializes the version from the shared
            # disk into the rollout host's local checkpoint before reload.
            started = time.perf_counter()
            try:
                ray.get([engine.pull_weights.remote(self.weight_version) for engine in self.rollout_engines])
                pulled = time.perf_counter()
                ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
                paused = time.perf_counter()
                try:
                    ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
                    flushed = time.perf_counter()
                    ray.get(
                        [
                            engine.update_weights_from_disk.remote(
                                self.args.update_weight_local_checkpoint_dir,
                                weight_version=str(self.weight_version),
                            )
                            for engine in self.rollout_engines
                        ]
                    )
                    reloaded = time.perf_counter()
                finally:
                    ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
                resumed = time.perf_counter()
            finally:
                if inference_profile_started:
                    ray.get([engine.stop_profile.remote() for engine in self.rollout_engines])
            self._stage_metrics.update(
                {
                    "perf/update_weights_delta_materialize_time": pulled - started,
                    "perf/update_weights_delta_pause_time": paused - pulled,
                    "perf/update_weights_delta_flush_time": flushed - paused,
                    "perf/update_weights_delta_reload_time": reloaded - flushed,
                    "perf/update_weights_delta_resume_time": resumed - reloaded,
                    "perf/update_weights_delta_rollout_time": resumed - started,
                }
            )
        dist.barrier(group=get_gloo_group())

    def _record_metrics(self) -> None:
        device = _metric_device()
        counts = torch.tensor(
            [self.changed_bytes, self.total_bytes, self.wire_bytes], dtype=torch.int64, device=device
        )
        dist.all_reduce(counts)
        changed, total, wire = counts.tolist()
        self.update_weight_metrics["perf/update_weights_density"] = changed / max(total, 1)
        self.update_weight_metrics["perf/update_weights_wire_bytes"] = wire
        if dist.get_rank() == 0:
            self.update_weight_metrics.update(self._stage_metrics)
            logger.info("[disk delta timings v=%s] %s", self.weight_version, self._stage_metrics)
            logger.info("[disk delta v=%s] density=%.2f%% wire=%.2f GB", self.weight_version, 100 * changed / max(total, 1), wire / 1e9)


def _tensor_bytes(
    tensor: torch.Tensor,
    *,
    free_buffers: queue.Queue[torch.Tensor] | None = None,
    lease_buffer: bool = False,
) -> tuple[np.ndarray, torch.Tensor | None]:
    flat = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    if flat.device.type == "cpu":
        return flat.numpy().copy(), None
    if free_buffers is None:
        return flat.cpu().numpy().copy(), None

    buffer = free_buffers.get()
    try:
        buffer[: flat.numel()].copy_(flat, non_blocking=True)
        if flat.device.type == "npu":
            torch.npu.current_stream().synchronize()
        elif flat.device.type == "cuda":
            torch.cuda.current_stream().synchronize()
        if lease_buffer:
            return buffer[: flat.numel()].numpy(), buffer
        return buffer[: flat.numel()].numpy().copy(), None
    finally:
        if not lease_buffer:
            free_buffers.put(buffer)


def _metric_device() -> torch.device:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu", torch.npu.current_device())
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _atomic_write(path: str, data: bytes) -> None:
    temporary = path + ".tmp"
    with open(temporary, "wb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
