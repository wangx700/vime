"""Delta weight synchronization primitives adapted from verl."""

from .sparse_gather import gather_slot_entries_to_rank0

__all__ = [
    "gather_slot_entries_to_rank0",
]
