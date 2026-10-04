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

Training is dense-only (``unified_finetuning=False``): the model produces and
trains only the dense embedding. The M3 sparse / ColBERT heads and their losses
are not built and not used.

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

# FlagEmbedding 1.4.2 module paths. The M3 fine-tuning runtime is imported
# lazily; the model class has no ``from_pretrained`` (see
# ``build_model_and_tokenizer``), so the entry assembles it from an ``AutoModel``
# plus the M3 heads via the official runner helper.
_M3_PACKAGE = "FlagEmbedding.finetune.embedder.encoder_only.m3"
_M3_TRAINER_MODULE = f"{_M3_PACKAGE}.trainer"


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
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="load the tokenizer/model from the local HF cache only (no network)",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--train_group_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--query_max_len", type=int, default=512)
    parser.add_argument("--passage_max_len", type=int, default=512)
    return parser


def build_trainer_class():
    """Return ``M3DualTargetTrainer`` = capability-unique mixin + official M3 trainer.

    The mixin stays left-most so its ``_get_train_sampler`` override wins and the
    vendored ``Trainer._get_dataloader`` routes the sampler through
    ``batch_sampler=``. The loss and the checkpoint layout come from the official
    ``EncoderOnlyEmbedderM3Trainer`` and are never re-implemented here.
    """
    try:
        from FlagEmbedding.finetune.embedder.encoder_only.m3.trainer import (
            EncoderOnlyEmbedderM3Trainer,
        )
    except ImportError as exc:  # pragma: no cover - requires FlagEmbedding
        raise RuntimeError(
            "FlagEmbedding is required to run training. Install it and re-run "
            f"(could not import {_M3_TRAINER_MODULE}.EncoderOnlyEmbedderM3Trainer)."
        ) from exc

    class M3DualTargetTrainer(CapabilityUniqueSamplerMixin, EncoderOnlyEmbedderM3Trainer):
        """Capability-unique batched training on the official M3 loss/save path."""

    return M3DualTargetTrainer


def build_training_arguments(ns) -> "EncoderOnlyEmbedderM3TrainingArguments":
    """Build the M3 training arguments, explicitly dense-only.

    ``unified_finetuning`` is set explicitly: the M3 model constructor defaults it
    to ``True`` while the official M3 ``TrainingArguments`` default is ``False``.
    The dense-only recipe is pinned here instead of relying on either default.
    """
    from FlagEmbedding.finetune.embedder.encoder_only.m3 import (
        EncoderOnlyEmbedderM3TrainingArguments,
    )

    return EncoderOnlyEmbedderM3TrainingArguments(
        output_dir=ns.output_dir,
        per_device_train_batch_size=ns.per_device_train_batch_size,
        learning_rate=ns.learning_rate,
        num_train_epochs=ns.num_train_epochs,
        # The dataset yields tuples (query, passages, teacher_scores); column
        # removal is a no-op for tuples, kept off for clarity.
        remove_unused_columns=False,
        # Default training sampler is irrelevant -- the mixin replaces it.
        train_sampling_strategy="random",
        # Dense-only contrastive training: no sparse / ColBERT heads or losses.
        unified_finetuning=False,
    )


def build_model_and_tokenizer(model_name_or_path, train_args, local_files_only=False):
    """Assemble the BGE-M3 embedder through the official FlagEmbedding 1.4.2 runtime.

    FlagEmbedding 1.4.2 exposes no ``EncoderOnlyEmbedderM3Model.from_pretrained``:
    the model is built from a ``base_model`` dict produced by the official
    ``EncoderOnlyEmbedderM3Runner.get_model`` (an ``AutoModel`` plus the M3 heads).
    With ``unified_finetuning=False`` the constructor discards the two heads, so
    only the dense encoder is trained.
    """
    from transformers import AutoTokenizer
    from FlagEmbedding.finetune.embedder.encoder_only.m3.runner import (
        EncoderOnlyEmbedderM3Runner,
    )
    from FlagEmbedding.finetune.embedder.encoder_only.m3.modeling import (
        EncoderOnlyEmbedderM3Model,
    )

    source = str(model_name_or_path)
    if local_files_only and not Path(source).exists():
        # Resolve a hub id to its cached snapshot so nothing touches the network.
        from huggingface_hub import snapshot_download

        source = snapshot_download(repo_id=source, local_files_only=True)

    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=local_files_only)
    base_model = EncoderOnlyEmbedderM3Runner.get_model(source)
    model = EncoderOnlyEmbedderM3Model(
        base_model=base_model,
        tokenizer=tokenizer,
        negatives_cross_device=train_args.negatives_cross_device,
        temperature=train_args.temperature,
        sub_batch_size=train_args.sub_batch_size,
        kd_loss_type=train_args.kd_loss_type,
        sentence_pooling_method=train_args.sentence_pooling_method,
        normalize_embeddings=train_args.normalize_embeddings,
        unified_finetuning=train_args.unified_finetuning,
        use_self_distill=train_args.use_self_distill,
        self_distill_start_step=train_args.self_distill_start_step,
    )
    return tokenizer, model


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
    )

    train_args = build_training_arguments(args)
    tokenizer, model = build_model_and_tokenizer(
        args.model_name_or_path, train_args, local_files_only=args.local_files_only
    )

    data_args = AbsEmbedderDataArguments(
        train_data=[str(data_path)],
        train_group_size=args.train_group_size,
        query_max_len=args.query_max_len,
        passage_max_len=args.passage_max_len,
    )
    train_dataset = AbsEmbedderTrainDataset(args=data_args, tokenizer=tokenizer)
    data_collator = AbsEmbedderCollator(
        tokenizer=tokenizer,
        query_max_len=args.query_max_len,
        passage_max_len=args.passage_max_len,
        sub_batch_size=train_args.sub_batch_size,
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
        f"in_batch_negatives={uses_in_batch_negatives(data_path)} dense_only=True "
        f"per_device_batch={args.per_device_train_batch_size}",
        flush=True,
    )
    trainer.train()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
