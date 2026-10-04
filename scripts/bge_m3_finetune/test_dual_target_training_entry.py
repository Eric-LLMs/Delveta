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
import inspect
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

import torch  # noqa: E402
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
        # Proven in a clean interpreter: other tests DO import FlagEmbedding, so an
        # in-process sys.modules check would be order-dependent. This asserts the
        # real property -- the module loads with FlagEmbedding absent.
        code = (
            "import sys;"
            f"sys.path.insert(0, r'{HERE}');"
            "import train_bge_m3_dual_target as e;"
            "print('HAS_MAIN', hasattr(e, 'main'));"
            "print('HAS_BTC', hasattr(e, 'build_trainer_class'));"
            "print('FLAG_IMPORTED', 'FlagEmbedding' in sys.modules)"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(HERE)
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("HAS_MAIN True", proc.stdout)
        self.assertIn("HAS_BTC True", proc.stdout)
        self.assertIn("FLAG_IMPORTED False", proc.stdout)


# --------------------------------------------------------------------------- #
# Tiny in-memory model + stub tokenizer for the FlagEmbedding 1.4.2 contract
# tests. No BGE-M3 weights are loaded and nothing is downloaded.
# --------------------------------------------------------------------------- #
def _fake_batch(bs, seq=5, vocab=64):
    return {
        "input_ids": torch.randint(0, vocab, (bs, seq)),
        "attention_mask": torch.ones(bs, seq, dtype=torch.long),
    }


def _tiny_dense_m3():
    """A dense-only ``EncoderOnlyEmbedderM3Model`` on a tiny random XLM-R encoder."""
    from transformers import XLMRobertaConfig, XLMRobertaModel
    from FlagEmbedding.finetune.embedder.encoder_only.m3.modeling import (
        EncoderOnlyEmbedderM3Model,
    )

    cfg = XLMRobertaConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
    )
    base_model = {"model": XLMRobertaModel(cfg)}
    return EncoderOnlyEmbedderM3Model(
        base_model=base_model,
        tokenizer=None,
        negatives_cross_device=False,
        temperature=1.0,
        sub_batch_size=-1,
        kd_loss_type="kl_div",
        sentence_pooling_method="cls",
        normalize_embeddings=False,
        unified_finetuning=False,
        use_self_distill=False,
        self_distill_start_step=-1,
    )


class _StubTokenizer:
    """Minimal tokenizer surface consumed by ``AbsEmbedderCollator.__call__``."""

    padding_side = "right"

    def __call__(self, texts, truncation=True, max_length=None, return_tensors=None):
        return {
            "input_ids": [[1, 2, 3] for _ in texts],
            "attention_mask": [[1, 1, 1] for _ in texts],
        }

    def pad(self, features, padding=True, max_length=None, pad_to_multiple_of=None, return_tensors="pt"):
        ids, am = features["input_ids"], features["attention_mask"]
        width = max(len(row) for row in ids)
        pad_row = lambda row: row + [0] * (width - len(row))  # noqa: E731
        return {
            "input_ids": torch.tensor([pad_row(r) for r in ids]),
            "attention_mask": torch.tensor([pad_row(r) for r in am]),
        }


class TestM3RuntimeContract(unittest.TestCase):
    """STEP-2B contract: the entry must wire the official FlagEmbedding 1.4.2 M3
    runtime (Runner -> Model -> Trainer -> TrainingArguments), dense-only."""

    @classmethod
    def setUpClass(cls):
        TMP_BASE.mkdir(parents=True, exist_ok=True)
        cls.tmp = Path(tempfile.mkdtemp(dir=str(TMP_BASE), prefix="bge_m3contract_"))

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls.tmp, ignore_errors=True)

    # 1 + 5. trainer inheritance, MRO, sampler-hook resolution
    def test_trainer_class_inheritance_and_mro(self):
        from FlagEmbedding.finetune.embedder.encoder_only.m3.trainer import (
            EncoderOnlyEmbedderM3Trainer,
        )
        from FlagEmbedding.abc.finetune.embedder import AbsEmbedderTrainer

        cls = entry.build_trainer_class()
        self.assertTrue(issubclass(cls, EncoderOnlyEmbedderM3Trainer))
        self.assertTrue(issubclass(cls, AbsEmbedderTrainer))
        names = [c.__name__ for c in cls.__mro__]
        self.assertEqual(names[0], "M3DualTargetTrainer")
        self.assertEqual(names[1], "CapabilityUniqueSamplerMixin")
        self.assertEqual(names[2], "EncoderOnlyEmbedderM3Trainer")
        self.assertIn("AbsEmbedderTrainer", names)
        self.assertIn("Trainer", names)
        # the sampler hook must resolve to the mixin, never to the base Trainer
        self.assertTrue(
            cls._get_train_sampler.__qualname__.startswith("CapabilityUniqueSamplerMixin")
        )

    # 2. model-construction contract (the STEP-2 blocker)
    def test_model_has_no_from_pretrained_and_entry_uses_official_ctor(self):
        from FlagEmbedding.finetune.embedder.encoder_only.m3.modeling import (
            EncoderOnlyEmbedderM3Model,
        )

        # the unavailable API the old entry wrongly assumed
        self.assertFalse(hasattr(EncoderOnlyEmbedderM3Model, "from_pretrained"))
        # the entry wires the official constructor path instead
        self.assertTrue(callable(entry.build_model_and_tokenizer))
        self.assertTrue(callable(entry.build_training_arguments))

    # 4. M3 TrainingArguments, dense-only
    @unittest.skipUnless(HAVE_ACCELERATE, "accelerate unavailable")
    def test_training_arguments_are_m3_and_dense_only(self):
        from FlagEmbedding.finetune.embedder.encoder_only.m3 import (
            EncoderOnlyEmbedderM3TrainingArguments,
        )

        ns = entry.build_arg_parser().parse_args(["--output_dir", str(self.tmp / "ta")])
        ta = entry.build_training_arguments(ns)
        self.assertIsInstance(ta, EncoderOnlyEmbedderM3TrainingArguments)
        self.assertFalse(ta.unified_finetuning)          # model default is True -> pinned False
        self.assertFalse(ta.negatives_cross_device)

    # 3 + 7. dense-only forward + default in-batch negatives path
    def test_dense_only_forward_uses_in_batch_negatives(self):
        model = _tiny_dense_m3()
        self.assertIsNone(model.colbert_linear)  # dense-only: head discarded
        self.assertIsNone(model.sparse_linear)   # dense-only: head discarded
        model.train()

        with (
            patch.object(
                model, "_compute_in_batch_neg_loss", wraps=model._compute_in_batch_neg_loss
            ) as spy_in,
            patch.object(
                model, "_compute_no_in_batch_neg_loss", wraps=model._compute_no_in_batch_neg_loss
            ) as spy_no,
            patch.object(
                model, "_compute_cross_device_neg_loss", wraps=model._compute_cross_device_neg_loss
            ) as spy_x,
            patch.object(model, "_sparse_embedding", wraps=model._sparse_embedding) as spy_s,
            patch.object(model, "_colbert_embedding", wraps=model._colbert_embedding) as spy_c,
        ):
            out = model(queries=_fake_batch(2), passages=_fake_batch(4), no_in_batch_neg_flag=False)

        self.assertEqual(out.loss.ndim, 0)
        self.assertTrue(torch.isfinite(out.loss))
        spy_in.assert_called_once()   # default in-batch negatives path taken
        spy_no.assert_not_called()
        spy_x.assert_not_called()
        spy_s.assert_not_called()     # dense-only: no sparse head used
        spy_c.assert_not_called()     # dense-only: no colbert head used

    # 3 + 7. the loss path takes no task input
    def test_forward_signature_has_no_task(self):
        from FlagEmbedding.finetune.embedder.encoder_only.m3.modeling import (
            EncoderOnlyEmbedderM3Model,
        )

        params = set(inspect.signature(EncoderOnlyEmbedderM3Model.forward).parameters)
        self.assertNotIn("task", params)
        self.assertEqual(
            params,
            {"self", "queries", "passages", "teacher_scores", "no_in_batch_neg_flag"},
        )

    # 6. the training path never consumes a `task` column
    def test_dataset_output_independent_of_task_column(self):
        from FlagEmbedding.abc.finetune.embedder import (
            AbsEmbedderDataArguments,
            AbsEmbedderTrainDataset,
        )

        base = {"query": "q", "pos": ["p"], "neg": ["n1"]}  # single neg -> deterministic
        plain = self.tmp / "no_task.jsonl"
        tagged = self.tmp / "with_task.jsonl"
        plain.write_text(json.dumps(base) + "\n", encoding="utf-8")
        tagged.write_text(json.dumps(dict(base, task="A")) + "\n", encoding="utf-8")

        item_plain = AbsEmbedderTrainDataset(
            args=AbsEmbedderDataArguments(
                train_data=[str(plain)], train_group_size=2, cache_path=str(self.tmp)
            ),
            tokenizer=None,
        )[0]
        item_tagged = AbsEmbedderTrainDataset(
            args=AbsEmbedderDataArguments(
                train_data=[str(tagged)], train_group_size=2, cache_path=str(self.tmp)
            ),
            tokenizer=None,
        )[0]

        self.assertEqual(len(item_plain), 3)          # (query, passages, teacher_scores)
        self.assertEqual(item_plain, item_tagged)     # `task` changes nothing
        self.assertEqual(item_plain[0], "q")

    # 7. in-batch negatives stay ENABLED at the collator boundary
    def test_collator_sets_no_in_batch_neg_flag_false(self):
        from FlagEmbedding.abc.finetune.embedder import AbsEmbedderCollator

        collator = AbsEmbedderCollator(
            tokenizer=_StubTokenizer(), query_max_len=8, passage_max_len=8
        )
        batch = collator([("q", ["p"], None)])
        self.assertIn("no_in_batch_neg_flag", batch)
        self.assertFalse(batch["no_in_batch_neg_flag"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
