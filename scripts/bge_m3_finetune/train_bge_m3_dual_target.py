#!/usr/bin/env python3
"""Minimal BGE-M3 training entry for the Delveta dual-target set.

One unified contrastive objective::

    query -> positive target text -> negative target texts

A positive target is either a capability's canonical description or a
standard/similar corpus query. Both are the same contrastive task for the
embedder: there is no task-specific loss, no capability-classification head, and
no capability label in the loss. The ``task`` field is provenance only and is
never read here.

The single manifest field training consumes is ``gold_capability_id``, used
solely to keep at most one record per capability in each batch via
``CapabilityUniqueBatchSampler`` (mixed in through
``CapabilityUniqueSamplerMixin``). Default in-batch negatives stay ON: the
training file is a plain ``.jsonl`` with no ``.no_in_batch_neg`` suffix, so the
FlagEmbedding collator keeps ``no_in_batch_neg_flag=False``.

Startup order (fail-fast):

  1. hard gate: ``train_data`` must be a single JSONL file (never a directory,
     which would concatenate files and break row alignment);
  2. ``dual_target_alignment.validate`` -- data file <-> manifest strict
     row-alignment (count / row-order / ``query_sha256``);
  3. ``capabilities_from_manifest`` -- the row-aligned capability map;
  4. ``per_device_train_batch_size`` must be <= the number of distinct
     capabilities (the sampler also enforces this and refuses to clamp).

FlagEmbedding is an optional, external runtime and is imported lazily: this
module imports cleanly without it, and only ``main`` needs it installed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from capability_unique_sampler import capabilities_from_manifest
from capability_unique_trainer import CapabilityUniqueSamplerMixin
from dual_target_alignment import validate

# FlagEmbedding module paths for the abc-level building blocks. These follow the
# current FlagEmbedding layout; adjust here if a different version is installed.
_ABC_MODULE = "FlagEmbedding.abc.finetune.embedder"
_M3_MODEL_MODULE = "FlagEmbedding.finetune.embedder.encoder_only.m3.modeling"


def uses_in_batch_negatives(data_path) -> bool:
    """Whether FlagEmbedding keeps default in-batch negatives for ``data_path``.

    FlagEmbedding switches to no-in-batch mode only for a file whose last
    dotted segment before the extension ends with ``no_in_batch_neg``
    (``AbsEmbedderSameDatasetTrainDataset``). This mirrors that rule without
    importing FlagEmbedding. The frozen dual-target file does not carry the
    suffix, so in-batch negatives stay enabled.
    """
    return not str(data_path).split(".")[-2].endswith("no_in_batch_neg")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_data = HERE / "data" / "bge_m3_train.jsonl"
    parser.add_argument("--train_data", default=str(default_data), help="single training JSONL file")
    parser.add_argument(
        "--train_manifest",
        default=str(HERE / "data" / "bge_m3_train_manifest.jsonl"),
        help="row-aligned manifest for --train_data",
    )
    parser.add_argument("--model_name_or_path", default="BAAI/bge-m3")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--train_group_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--query_max_len", type=int, default=512)
    parser.add_argument("--passage_max_len", type=int, default=512)
    return parser


def build_trainer_class():
    """Return ``M3DualTargetTrainer`` = capability-unique mixin + FlagEmbedder trainer."""
    try:
        from FlagEmbedding.abc.finetune.embedder import AbsEmbedderTrainer
    except ImportError as exc:  # pragma: no cover - requires FlagEmbedding
        raise RuntimeError(
            "FlagEmbedding is required to run training. Install it and re-run "
            f"(could not import {_ABC_MODULE}.AbsEmbedderTrainer)."
        ) from exc

    class M3DualTargetTrainer(CapabilityUniqueSamplerMixin, AbsEmbedderTrainer):
        def _save(self, output_dir=None, state_dict=None):
            output_dir = output_dir or self.args.output_dir
            self.save_model(output_dir)
            self._save_processing_class(output_dir)

    return M3DualTargetTrainer


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - requires FlagEmbedding
    args = build_arg_parser().parse_args(argv)

    data_path = Path(args.train_data).resolve()
    manifest_path = Path(args.train_manifest).resolve()

    # 1. hard gate: single file, not a directory.
    if data_path.is_dir():
        raise SystemExit(f"--train_data must be a single JSONL file, got directory {data_path}")

    # 2. strict row alignment (fail-fast on count / order / query_sha256).
    report = validate(data_path, manifest_path)

    # 3. row-aligned capability map (the only manifest field training consumes).
    capabilities = capabilities_from_manifest(manifest_path)
    n_caps = len(set(capabilities))

    # 4. batch size cannot exceed the distinct-capability count.
    if args.per_device_train_batch_size > n_caps:
        raise SystemExit(
            f"--per_device_train_batch_size={args.per_device_train_batch_size} exceeds the "
            f"{n_caps} distinct capabilities; a capability-unique batch is impossible."
        )

    if not uses_in_batch_negatives(data_path):
        raise SystemExit(
            f"{data_path.name} carries the .no_in_batch_neg suffix; the dual-target recipe "
            "requires default in-batch negatives. Rename the file or drop the suffix."
        )

    # --- FlagEmbedding runtime (lazy) ---------------------------------------
    from FlagEmbedding.abc.finetune.embedder import (
        AbsEmbedderCollator,
        AbsEmbedderDataArguments,
        AbsEmbedderTrainDataset,
        AbsEmbedderTrainingArguments,
    )
    from transformers import AutoTokenizer

    m3_model_module = __import__(_M3_MODEL_MODULE, fromlist=["EncoderOnlyEmbedderM3Model"])
    EncoderOnlyEmbedderM3Model = m3_model_module.EncoderOnlyEmbedderM3Model

    data_args = AbsEmbedderDataArguments(
        train_data=[str(data_path)],
        train_group_size=args.train_group_size,
        query_max_len=args.query_max_len,
        passage_max_len=args.passage_max_len,
    )
    train_args = AbsEmbedderTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        # The dataset yields tuples (query, passages, teacher_scores); column
        # removal is a no-op for tuples, kept off for clarity.
        remove_unused_columns=False,
        # Default training sampler is irrelevant -- the mixin replaces it.
        train_sampling_strategy="random",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = EncoderOnlyEmbedderM3Model.from_pretrained(args.model_name_or_path)
    train_dataset = AbsEmbedderTrainDataset(args=data_args, tokenizer=tokenizer)
    data_collator = AbsEmbedderCollator(
        tokenizer=tokenizer,
        query_max_len=args.query_max_len,
        passage_max_len=args.passage_max_len,
    )

    TrainerClass = build_trainer_class()
    trainer = TrainerClass(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        capabilities=capabilities,
    )

    print(
        f"[dual-target] aligned rows={report['rows']} capabilities={n_caps} "
        f"in_batch_negatives={uses_in_batch_negatives(data_path)} "
        f"per_device_batch={args.per_device_train_batch_size}",
        flush=True,
    )
    trainer.train()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
