"""Delta weight synchronization primitives adapted from verl."""

from .sparse_gather import gather_slot_entries_to_rank0, shard_delta_indices

__all__ = [
    "gather_slot_entries_to_rank0",
    "shard_delta_indices",
]
