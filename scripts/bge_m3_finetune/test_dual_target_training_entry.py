#!/usr/bin/env python3
"""Offline smoke test for the dual-target BGE-M3 training entry-point.

Proves the capability-unique sampler sits on the REAL training path::

    dataset -> manifest alignment -> CapabilityUniqueBatchSampler
            -> Trainer.get_train_dataloader() -> actual batch

Every training-path assertion goes through the vendored ``transformers.Trainer``
and a real ``torch.utils.data.DataLoader``; no test inspects ``sampler.__iter__``
directly. FlagEmbedding is NOT required: the sampler injection is proven with the
plain ``Trainer``, so this file is trivially runnable offline.

``accelerate`` is a runtime requirement of ``transformers.Trainer``. This module
first tries to import it; if absent it looks for a cached copy (uv archive) and
prepends it to ``sys.path``; if it is still unavailable the Trainer-based tests
skip with an explicit message (they never silently pass).

Run:  .venv-laya/Scripts/python.exe -m unittest test_dual_target_training_entry -v
"""
from __future__ import annotations

import glob
import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DATA = HERE / "data"
TRAIN_MANIFEST = DATA / "bge_m3_train_manifest.jsonl"
TRAIN_DATA = DATA / "bge_m3_train.jsonl"

# Local temporary workspace (never the system TEMP).
TMP_BASE = Path(os.environ.get("DELVETA_TMP", "F:/WorkSpace/Delveta/.tmp"))


def _ensure_accelerate() -> bool:
    """Make ``import accelerate`` work, else return False (tests then skip)."""
    try:
        import accelerate  # noqa: F401

        return True
    except ImportError:
        pass

    override = os.environ.get("DELVETA_ACCELERATE_SITE")
    if override and Path(override).is_dir():
        sys.path.insert(0, override)

    roots = [
        os.environ.get("UV_CACHE_DIR"),
        os.path.expanduser("~/.cache/uv"),
        os.path.expanduser("~/AppData/Local/uv/cache"),
    ]
    for root in filter(None, roots):
        for init in glob.glob(os.path.join(root, "archive-v0", "*", "accelerate", "__init__.py")):
            site = str(Path(init).parent.parent)
            sys.path.insert(0, site)
            try:
                import accelerate  # noqa: F401

                return True
            except ImportError:
                sys.path.pop(0)
    return False


HAVE_ACCELERATE = _ensure_accelerate()

import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from capability_unique_sampler import (  # noqa: E402
    CapabilityUniqueBatchSampler,
    capabilities_from_manifest,
)
from capability_unique_trainer import CapabilityUniqueSamplerMixin  # noqa: E402
from dual_target_alignment import AlignmentError, validate  # noqa: E402
import train_bge_m3_dual_target as entry  # noqa: E402

if HAVE_ACCELERATE:
    from transformers import Trainer, TrainingArguments  # noqa: E402

CAPS = [f"cap-{i:02d}" for i in range(18)]


# --------------------------------------------------------------------------- #
# Stub dataset: index-aligned rows carrying idx / cap / query_sha256.
# The collator keeps strings verbatim (the real FlagEmbedding collator is also a
# custom callable that consumes tuple features).
# --------------------------------------------------------------------------- #
class RowDataset(Dataset):
    def __init__(self, caps, qsha):
        assert len(caps) == len(qsha)
        self.caps = list(caps)
        self.qsha = list(qsha)

    def __len__(self):
        return len(self.caps)

    def __getitem__(self, i):
        return {"idx": i, "cap": self.caps[i], "qsha": self.qsha[i]}


def collate(features):
    return {
        "idx": [f["idx"] for f in features],
        "cap": [f["cap"] for f in features],
        "qsha": [f["qsha"] for f in features],
    }


def make_synthetic(seed=0):
    """18 caps, unequal counts; every query -> two records sharing cap AND qsha
    (the two positive sources must never co-batch)."""
    rng = random.Random(seed)
    caps, qsha = [], []
    for cap in CAPS:
        for q in range(rng.randint(20, 60)):
            qs = f"{cap}-q{q:03d}"
            caps.append(cap)
            qsha.append(qs)  # positive source 1
            caps.append(cap)
            qsha.append(qs)  # positive source 2 (same query -> same cap, same sha)
    order = list(range(len(caps)))
    rng.shuffle(order)
    return [caps[i] for i in order], [qsha[i] for i in order]


class StubTrainer(CapabilityUniqueSamplerMixin, Trainer if HAVE_ACCELERATE else object):
    """Real Trainer + the capability-unique sampler mixin (no FlagEmbedding)."""


def build_trainer(caps, qsha, batch_size, tmpdir):
    ds = RowDataset(caps, qsha)
    args = TrainingArguments(
        output_dir=str(tmpdir),
        per_device_train_batch_size=batch_size,
        remove_unused_columns=False,
        use_cpu=True,
        report_to=[],
        dataloader_num_workers=0,
    )
    model = nn.Linear(2, 2)
    return StubTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collate,
        capabilities=caps,
    )


@unittest.skipUnless(HAVE_ACCELERATE, "accelerate unavailable: Trainer-based tests skipped")
class TestTrainingPath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TMP_BASE.mkdir(parents=True, exist_ok=True)
        cls.tmp = Path(tempfile.mkdtemp(dir=str(TMP_BASE), prefix="bge_entry_"))

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls.tmp, ignore_errors=True)

    # ---- 2. sampler injection -------------------------------------------------
    def test_sampler_injected_through_real_trainer(self):
        caps, qsha = make_synthetic(1)
        tr = build_trainer(caps, qsha, 8, self.tmp)
        dl = tr.get_train_dataloader()
        self.assertIsInstance(dl, DataLoader)
        self.assertIs(dl.batch_sampler, tr.train_sampler)      # 2. injection PASS
        self.assertIsInstance(tr.train_sampler, CapabilityUniqueBatchSampler)

    # ---- 3/4/5/6/7/8. real DataLoader batch invariants ------------------------
    def test_batch_invariants_bs_8_16_18(self):
        caps, qsha = make_synthetic(2)
        n = len(caps)
        for bs in (8, 16, 18):
            with self.subTest(batch_size=bs):
                tr = build_trainer(caps, qsha, bs, self.tmp)
                dl = tr.get_train_dataloader()
                batches = list(dl)                              # real DataLoader output
                self.assertEqual(len(dl), len(tr.train_sampler), f"bs={bs}: __len__ vs dl")

                seen_idx, cap_batch, qsha_batch = [], {}, {}
                for bid, b in enumerate(batches):
                    c, q = b["cap"], b["qsha"]
                    # 3. capability unique within the actual batch
                    self.assertEqual(len(c), len(set(c)), f"bs={bs} batch{bid}: same-cap")
                    # 4. a query's two records never co-batch
                    self.assertEqual(len(q), len(set(q)), f"bs={bs} batch{bid}: same-query")
                    for cap in c:
                        cap_batch.setdefault(cap, set()).add(bid)
                    for s in q:
                        qsha_batch.setdefault(s, set()).add(bid)
                    seen_idx += b["idx"]

                # 6/7. every index exactly once
                self.assertEqual(len(seen_idx), n, f"bs={bs}: index_loss")
                self.assertEqual(len(seen_idx), len(set(seen_idx)), f"bs={bs}: index_dup")

                # 5. same capability records land in distinct batches
                per_cap_records = {}
                for i, cap in enumerate(caps):
                    per_cap_records.setdefault(cap, []).append(i)
                for cap, idxs in per_cap_records.items():
                    self.assertEqual(len(cap_batch[cap]), len(idxs), f"bs={bs} cap {cap}: co-batch")
                # 4 (cont.) each query's two records land in distinct batches
                per_q_records = {}
                for i, s in enumerate(qsha):
                    per_q_records.setdefault(s, []).append(i)
                for s, idxs in per_q_records.items():
                    self.assertEqual(len(qsha_batch[s]), len(idxs), f"bs={bs} q {s}: co-batch")

    # ---- 9. batch_size > 18 must FAIL ----------------------------------------
    def test_batch_size_above_capabilities_fails(self):
        caps, qsha = make_synthetic(3)
        for bs in (19, 20, 32, 64):
            with self.subTest(batch_size=bs):
                tr = build_trainer(caps, qsha, bs, self.tmp)
                with self.assertRaises(ValueError):
                    tr.get_train_dataloader()

    # ---- runtime alignment guard ---------------------------------------------
    def test_dataset_manifest_length_mismatch_raises(self):
        caps, qsha = make_synthetic(4)
        tr = build_trainer(caps, qsha, 8, self.tmp)
        tr._capability_ids = caps[:-1]  # simulate a stale/truncated manifest
        with self.assertRaises(ValueError):
            tr.get_train_dataloader()


class TestOfflineComponents(unittest.TestCase):
    # ---- 1. alignment PASS on the frozen data --------------------------------
    def test_alignment_passes_on_frozen_data(self):
        if not (TRAIN_DATA.is_file() and TRAIN_MANIFEST.is_file()):
            self.skipTest("frozen dual-target data not present")
        for data, man in (
            (TRAIN_DATA, TRAIN_MANIFEST),
            (DATA / "bge_m3_test.jsonl", DATA / "bge_m3_test_manifest.jsonl"),
        ):
            rep = validate(data, man)
            self.assertEqual(rep["sha256_mismatches"], 0)
            self.assertEqual(rep["rows"], rep["sha256_checked"])
            self.assertGreater(rep["rows"], 0)
        # capability map row-aligned to the data rows
        caps = capabilities_from_manifest(TRAIN_MANIFEST)
        self.assertEqual(len(caps), validate(TRAIN_DATA, TRAIN_MANIFEST)["rows"])
        self.assertEqual(len(set(caps)), 18)

    def test_alignment_detects_reordering(self):
        if not TRAIN_DATA.is_file():
            self.skipTest("frozen data not present")
        rows = TRAIN_DATA.read_text(encoding="utf-8").splitlines()
        rows[0], rows[1] = rows[1], rows[0]  # swap two rows -> order broken
        with tempfile.TemporaryDirectory(dir=str(TMP_BASE)) as d:
            tampered = Path(d) / "bge_m3_train.jsonl"
            tampered.write_text("\n".join(rows) + "\n", encoding="utf-8")
            with self.assertRaises(AlignmentError):
                validate(tampered, TRAIN_MANIFEST)

    def test_alignment_detects_row_count_mismatch(self):
        if not TRAIN_MANIFEST.is_file():
            self.skipTest("frozen data not present")
        lines = TRAIN_MANIFEST.read_text(encoding="utf-8").splitlines()
        with tempfile.TemporaryDirectory(dir=str(TMP_BASE)) as d:
            trimmed = Path(d) / "bge_m3_train_manifest.jsonl"
            trimmed.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            with self.assertRaises(AlignmentError):
                validate(TRAIN_DATA, trimmed)

    # ---- 10. default in-batch negatives stay ENABLED -------------------------
    def test_in_batch_negatives_enabled(self):
        # The frozen file has no `.no_in_batch_neg` suffix -> FlagEmbedding keeps
        # no_in_batch_neg_flag=False (in-batch negatives ON).
        self.assertTrue(entry.uses_in_batch_negatives(TRAIN_DATA))
        self.assertFalse(entry.uses_in_batch_negatives("something.no_in_batch_neg.jsonl"))
        self.assertTrue(entry.uses_in_batch_negatives("plain.jsonl"))

    # ---- entry remains importable without FlagEmbedding ----------------------
    def test_entry_imports_without_flagembedding(self):
        self.assertTrue(hasattr(entry, "main"))
        self.assertTrue(hasattr(entry, "build_trainer_class"))
        self.assertNotIn("FlagEmbedding", sys.modules)  # never imported at module load


if __name__ == "__main__":
    unittest.main(verbosity=2)
