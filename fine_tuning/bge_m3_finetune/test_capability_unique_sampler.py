#!/usr/bin/env python3
"""Tests for CapabilityUniqueBatchSampler.

Run through the REAL torch DataLoader (never just sampler.__iter__) and assert
the batch-level invariant on the ``gold_capability_id`` the model actually sees:

    sampler -> DataLoader -> actual batch -> gold_capability_id values unique

Also covers multi-epoch, distriuted (world_size 1/2) partitioning, the tail
batch, and a read-only smoke test over the frozen train manifest.

Run:  .venv-bge-train/Scripts/python.exe -m unittest test_capability_unique_sampler -v
"""
from __future__ import annotations

import json
import random
import sys
import unittest
from collections import Counter
from pathlib import Path

from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capability_unique_sampler import (  # noqa: E402
    CapabilityUniqueBatchSampler,
    capabilities_from_manifest,
)
from transformers.trainer_pt_utils import BatchRebalanceSampler  # noqa: E402

HERE = Path(__file__).resolve().parent
TRAIN_MANIFEST = HERE / "data" / "bge_m3_train_manifest.jsonl"
CAPS = [f"cap-{i:02d}" for i in range(18)]


class RowDataset(Dataset):
    """Index-aligned dataset exposing idx / cap / qsha, like the training set."""

    def __init__(self, caps, qsha):
        assert len(caps) == len(qsha)
        self.caps = list(caps)
        self.qsha = list(qsha)

    def __len__(self):
        return len(self.caps)

    def __getitem__(self, i):
        return {"idx": i, "cap": self.caps[i], "qsha": self.qsha[i]}


def make_synthetic(seed=0):
    """Unequal per-capability counts; each query -> Task A + Task B sharing a
    capability AND a query_sha256 (the two must never co-batch)."""
    rng = random.Random(seed)
    caps, qsha = [], []
    for cap in CAPS:
        for q in range(rng.randint(20, 60)):
            qs = f"{cap}-q{q:03d}"
            caps.append(cap)
            qsha.append(qs)          # Task A
            caps.append(cap)
            qsha.append(qs)          # Task B (same query -> same cap, same sha)
    order = list(range(len(caps)))
    rng.shuffle(order)
    return [caps[i] for i in order], [qsha[i] for i in order]


def _collect(dataloader):
    """Return actual DataLoader batches as (idx_list, cap_list, qsha_list)."""
    batches = []
    for b in dataloader:
        batches.append((list(b["idx"]), list(b["cap"]), list(b["qsha"])))
    return batches


class TestCapabilityUniqueBatchSampler(unittest.TestCase):
    # ------------------------------------------------------------------ basic
    def test_isinstance_routes_to_batch_sampler(self):
        """The fork routes it as batch_sampler only if isinstance(BatchRebalanceSampler)."""
        s = CapabilityUniqueBatchSampler(CAPS, 8)
        self.assertIsInstance(s, BatchRebalanceSampler)

    def test_batch_size_above_caps_raises(self):
        for bs in (19, 20, 32, 64):
            with self.assertRaises(ValueError):
                CapabilityUniqueBatchSampler(CAPS, bs)
        # exactly at the cap count is allowed
        CapabilityUniqueBatchSampler(CAPS, 18)

    def test_len_matches_yielded_and_tail_not_dropped(self):
        caps, qsha = make_synthetic(1)
        n = len(caps)
        ds = RowDataset(caps, qsha)
        for bs in (8, 16, 18):
            s = CapabilityUniqueBatchSampler(caps, bs)
            dl = DataLoader(ds, batch_sampler=s, num_workers=0)
            got = _collect(dl)
            self.assertEqual(len(s), len(dl), f"bs={bs}: __len__ vs dataloader")
            self.assertEqual(len(s), len(got), f"bs={bs}: __len__ vs yielded")
            idx = [i for _i, _c, _q in got for i in _i]
            self.assertEqual(len(idx), n, f"bs={bs}: index_loss != 0")
            # tail (last batch) may be small but must exist and carry data
            self.assertTrue(got[-1][0], f"bs={bs}: empty tail")

    # --------------------------------------------- real DataLoader, uniqueness
    def test_actual_dataloader_uniqueness(self):
        caps, qsha = make_synthetic(2)
        ds = RowDataset(caps, qsha)
        for bs in (8, 16, 18):
            s = CapabilityUniqueBatchSampler(caps, bs)
            dl = DataLoader(ds, batch_sampler=s, num_workers=0)
            got = _collect(dl)
            same_cap, same_query = 0, 0
            for _idx, c, q in got:
                if len(c) != len(set(c)):
                    same_cap += 1
                if len(q) != len(set(q)):
                    same_query += 1
            idx = [i for _i, _c, _q in got for i in _i]
            self.assertEqual(same_cap, 0, f"bs={bs}: same-capability collision")
            self.assertEqual(same_query, 0, f"bs={bs}: same-query A/B collision")
            self.assertEqual(len(idx) - len(set(idx)), 0, f"bs={bs}: index_dup != 0")
            self.assertEqual(len(set(idx)), len(caps), f"bs={bs}: index_loss != 0")

    # -------------------------------------------------- 100 epochs uniqueness
    def test_100_epochs_uniqueness(self):
        caps, qsha = make_synthetic(3)
        ds = RowDataset(caps, qsha)
        s = CapabilityUniqueBatchSampler(caps, 8, seed=123)
        n = len(caps)
        for epoch in range(100):
            s.set_epoch(epoch)
            dl = DataLoader(ds, batch_sampler=s, num_workers=0)
            got = _collect(dl)
            idx = [i for _i, _c, _q in got for i in _i]
            for _idx, c, q in got:
                self.assertEqual(len(c), len(set(c)), f"epoch {epoch}: same-cap")
                self.assertEqual(len(q), len(set(q)), f"epoch {epoch}: same-query")
            self.assertEqual(len(idx) - len(set(idx)), 0, f"epoch {epoch}: dup")
            self.assertEqual(len(set(idx)), n, f"epoch {epoch}: loss")

    # ------------------------------------------------------ distributed ranks
    def _distributed(self, dp_size, bs=8, seed=7):
        caps, qsha = make_synthetic(4)
        ds = RowDataset(caps, qsha)
        merged, per_rank = [], []
        for r in range(dp_size):
            s = CapabilityUniqueBatchSampler(caps, bs, dp_size=dp_size, rank=r, seed=seed)
            dl = DataLoader(ds, batch_sampler=s, num_workers=0)
            got = _collect(dl)
            per_rank.append(got)
            for _idx, c, q in got:
                self.assertEqual(len(c), len(set(c)), f"rank {r}: same-cap")
                self.assertEqual(len(q), len(set(q)), f"rank {r}: same-query")
            merged += [i for _i, _c, _q in got for i in _i]
        return merged, per_rank, len(caps)

    def test_world_size_1(self):
        merged, _per, n = self._distributed(1)
        self.assertEqual(len(merged) - len(set(merged)), 0, "w=1: dup")
        self.assertEqual(len(set(merged)), n, "w=1: loss")

    def test_world_size_2(self):
        merged, _per, n = self._distributed(2)
        self.assertEqual(len(merged) - len(set(merged)), 0, "w=2: dup")
        self.assertEqual(len(set(merged)), n, "w=2: loss")

    # ------------------------------------------------- frozen real-data smoke
    def test_real_manifest_smoke(self):
        if not TRAIN_MANIFEST.is_file():
            self.skipTest("train manifest not present")
        caps = capabilities_from_manifest(TRAIN_MANIFEST)
        with open(TRAIN_MANIFEST, encoding="utf-8") as fh:
            qsha = [json.loads(l)["query_sha256"] for l in fh if l.strip()]
        ds = RowDataset(caps, qsha)
        s = CapabilityUniqueBatchSampler(caps, 8)
        dl = DataLoader(ds, batch_sampler=s, num_workers=0)
        got = _collect(dl)
        idx = [i for _i, _c, _q in got for i in _i]
        same_cap = sum(1 for _i, c, _q in got if len(c) != len(set(c)))
        same_query = sum(1 for _i, _c, q in got if len(q) != len(set(q)))
        self.assertEqual(same_cap, 0)
        self.assertEqual(same_query, 0)
        self.assertEqual(len(idx) - len(set(idx)), 0)
        self.assertEqual(len(set(idx)), len(caps))
        self.assertEqual(len(s), len(dl))


if __name__ == "__main__":
    unittest.main(verbosity=2)
