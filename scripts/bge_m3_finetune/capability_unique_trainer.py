#!/usr/bin/env python3
"""Trainer mixin that injects ``CapabilityUniqueBatchSampler`` into training.

A BGE-M3 / FlagEmbedding trainer is given the capability-unique sampler by
mixing this in *before* its base trainer::

    class M3DualTargetTrainer(CapabilityUniqueSamplerMixin, AbsEmbedderTrainer):
        ...

The mixin's whole job is to override ``_get_train_sampler`` -- the single hook
that HuggingFace ``Trainer.get_train_dataloader`` calls. Returning a
``CapabilityUniqueBatchSampler`` (a ``BatchRebalanceSampler`` subclass) makes the
vendored Transformers ``Trainer._get_dataloader`` route the dataloader through
``batch_sampler=`` and neutralise the accelerator's batch shard, so the sampler's
index batches reach the model verbatim (see transformers/trainer.py:1008).

Responsibilities are kept separate from the sampler *algorithm* (which lives in
``capability_unique_sampler.py``): this module only wires that algorithm into the
Trainer. It imports ``transformers`` + the sampler, never FlagEmbedding, so the
injection can be proven with the plain ``Trainer`` -- no external runtime needed.

The sampled ``batch_size`` is ``per_device_train_batch_size`` (the per-process
micro-batch), not ``train_batch_size`` (= per-device x world size). Under data
parallelism each rank takes ``rows[rank::dp_size]``, so one rank's batch is one
sampler row.
"""
from __future__ import annotations

from typing import List, Sequence

from capability_unique_sampler import CapabilityUniqueBatchSampler


class CapabilityUniqueSamplerMixin:
    """Mixin that returns a :class:`CapabilityUniqueBatchSampler` for training.

    ``capabilities`` is the row-aligned ``gold_capability_id`` list (row ``i`` of
    the training dataset -> ``capabilities[i]``). It is passed as a keyword to the
    trainer constructor and consumed only here -- never fed to the model.
    """

    def __init__(
        self,
        *args,
        capabilities: Sequence[str],
        shuffle: bool = True,
        seed: int = 42,
        **kwargs,
    ):
        self._capability_ids: List[str] = list(capabilities)
        self._sampler_shuffle = bool(shuffle)
        self._sampler_seed = int(seed)
        self._train_sampler: CapabilityUniqueBatchSampler | None = None
        super().__init__(*args, **kwargs)

    @property
    def capability_ids(self) -> List[str]:
        """Row-aligned ``gold_capability_id`` list the sampler was built from."""
        return list(self._capability_ids)

    @property
    def train_sampler(self) -> CapabilityUniqueBatchSampler | None:
        """The sampler produced by the most recent ``_get_train_sampler`` call."""
        return self._train_sampler

    def _get_train_sampler(self, train_dataset=None):
        train_dataset = train_dataset if train_dataset is not None else self.train_dataset
        if train_dataset is None:
            raise ValueError("CapabilityUniqueSamplerMixin: train_dataset is None")

        # Runtime alignment guard: the sampler index -> capability map is only
        # valid when the dataset and the manifest have the same length.
        if len(train_dataset) != len(self._capability_ids):
            raise ValueError(
                f"dataset/manifest length mismatch: len(dataset)={len(train_dataset)} "
                f"!= len(capabilities)={len(self._capability_ids)}; the sampler's "
                f"index -> capability map would be wrong"
            )

        world_size = max(1, self.args.world_size)
        rank = self.args.process_index if self.args.world_size > 1 else 0
        self._train_sampler = CapabilityUniqueBatchSampler(
            capabilities=self._capability_ids,
            batch_size=self.args.per_device_train_batch_size,
            dp_size=world_size,
            rank=rank,
            shuffle=self._sampler_shuffle,
            seed=self._sampler_seed,
        )
        return self._train_sampler
