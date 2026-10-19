#!/usr/bin/env python3
"""Capability-unique batch sampler for the Delveta BGE-M3 dual-target set.

The dual-target set gives Task A and Task B of one query the SAME
``gold_capability_id``. With FlagEmbedding's default in-batch negatives, any
batch that carries two records of one capability turns one record's positive
into another record's in-batch *negative* (a false negative). This sampler
removes that class of error by guaranteeing, at the DataLoader level, that
every batch carries at most one record per ``gold_capability_id``.

That single invariant subsumes the four batch requirements:

  * one record per capability per batch                (by construction),
  * a query's Task A / Task B never co-batch            (they share a cap),
  * two queries of one capability never co-batch        (they share a cap).

Batches are produced as explicit index *lists* (a ``batch_sampler``), so
``torch.utils.data.DataLoader`` yields them verbatim -- it never re-chunks them
by ``batch_size``. The class subclasses the fork's ``BatchRebalanceSampler`` so
``Trainer._get_dataloader`` routes it through the ``batch_sampler`` branch
(transformers/trainer.py:1008) and neutralises the accelerator shard
(trainer.py:1026), keeping the default in-batch negatives intact.

Read-only w.r.t. the frozen dataset: capabilities are read from the manifest,
never inferred from query text.
"""
from __future__ import annotations

import heapq
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Sequence, Union

from transformers.trainer_pt_utils import BatchRebalanceSampler


def capabilities_from_manifest(manifest_path: Union[str, Path]) -> List[str]:
    """Return the row-aligned ``gold_capability_id`` list for a manifest.

    Row ``i`` of the manifest corresponds to row ``i`` of the paired dataset
    file, so the returned list is the sampler's index -> capability map.
    """
    with open(manifest_path, encoding="utf-8") as fh:
        return [json.loads(line)["gold_capability_id"] for line in fh if line.strip()]


class CapabilityUniqueBatchSampler(BatchRebalanceSampler):
    """Yield index batches whose ``gold_capability_id`` values are unique.

    Contract per epoch:

    * every dataset index is emitted exactly once (index_loss = 0);
    * no index is emitted twice (index_dup = 0);
    * each yielded batch has distinct capabilities;
    * a trailing batch may be smaller than ``batch_size`` -- it is emitted,
      never silently dropped;
    * ``dp_size`` ranks partition the batches by ``index % dp_size``; the union
      over all ranks is the full, once-only index set.

    ``batch_size`` greater than the number of distinct capabilities is rejected
    (a capability-unique batch is then impossible); the sampler never clamps.
    """

    def __init__(
        self,
        capabilities: Sequence[str],
        batch_size: int,
        dp_size: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 42,
    ):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        n_caps = len(set(capabilities))
        if batch_size > n_caps:
            raise ValueError(
                f"batch_size={batch_size} exceeds the number of distinct capabilities "
                f"({n_caps}); a capability-unique batch is impossible. Refusing to clamp."
            )
        if dp_size < 1:
            raise ValueError(f"dp_size must be >= 1, got {dp_size}")
        if not (0 <= rank < dp_size):
            raise ValueError(f"rank must be in [0, {dp_size}), got {rank}")

        self.capabilities = list(capabilities)
        self.batch_size = int(batch_size)
        self.dp_size = int(dp_size)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        # set by the parent in its own __init__; harmless defaults for the paths
        # that only ever run the isinstance() routing / logging.
        self.grad_accum = 1
        self._rows_cache = None
        self._rows_cache_epoch = None

    # -- epoch handling -----------------------------------------------------
    def set_epoch(self, epoch: int) -> None:
        """Set the epoch; the next iteration rebuilds rows under a new seed."""
        self.epoch = int(epoch)
        self._rows_cache = None
        self._rows_cache_epoch = None

    # -- row construction ---------------------------------------------------
    def _all_rows(self) -> List[List[int]]:
        """All capability-unique rows covering every index exactly once.

        Rows are filled by a min-heap "smallest row first" assignment, so every
        row gets one sample per capability at most and row sizes stay balanced.
        The row count is ``max(ceil(N / batch_size), max_capability_count)`` --
        the second term is the feasibility floor: a capability with ``k``
        samples needs at least ``k`` rows (<= 1 per row). Only trailing rows
        can be smaller than ``batch_size``, and only when that floor forces it.
        """
        if self._rows_cache is not None and self._rows_cache_epoch == self.epoch:
            return self._rows_cache

        rng = random.Random(self.seed + (self.epoch if self.shuffle else 0))
        by_cap: dict[str, list[int]] = defaultdict(list)
        for idx, cap in enumerate(self.capabilities):
            by_cap[cap].append(idx)
        n = len(self.capabilities)
        if n == 0:
            self._rows_cache, self._rows_cache_epoch = [], self.epoch
            return []

        for cap in by_cap:
            rng.shuffle(by_cap[cap])
        max_count = max(len(v) for v in by_cap.values())
        num_rows = max(math.ceil(n / self.batch_size), max_count)
        rows: List[List[int]] = [[] for _ in range(num_rows)]

        # Always fill the currently smallest row -> balanced sizes, and each
        # capability is popped from the heap at most once per pass, so no row
        # can receive two samples of the same capability.
        heap = [(0, j) for j in range(num_rows)]
        heapq.heapify(heap)
        cap_list = list(by_cap)
        rng.shuffle(cap_list)
        for cap in cap_list:
            samples = by_cap[cap]
            popped = [heapq.heappop(heap) for _ in range(len(samples))]
            for (size, j), s in zip(popped, samples):
                rows[j].append(s)
                heapq.heappush(heap, (size + 1, j))

        if self.shuffle:
            rng.shuffle(rows)  # rows stay intact; only their order changes
        self._rows_cache = rows
        self._rows_cache_epoch = self.epoch
        return rows

    def _rank_rows(self) -> List[List[int]]:
        return self._all_rows()[self.rank :: self.dp_size]

    # -- sampler protocol ---------------------------------------------------
    def __iter__(self):
        yield from self._rank_rows()

    def __len__(self) -> int:
        return len(self._rank_rows())
