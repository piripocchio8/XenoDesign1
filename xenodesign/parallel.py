"""Round-robin sharding for multi-GPU parallel runs."""
from __future__ import annotations

from typing import Sequence


def shard_round_robin(items: Sequence, n_shards: int) -> list[list]:
    """Split items into n_shards round-robin (shard i gets items[i::n_shards])."""
    return [list(items[i::n_shards]) for i in range(n_shards)]
