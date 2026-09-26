# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.
import abc

from typing import List, Tuple

import torch

class NegativesSampler(torch.nn.Module):
    def __init__(self, l2_norm: bool, l2_norm_eps: float) -> None:
        super().__init__()

        self._l2_norm: bool = l2_norm
        self._l2_norm_eps: float = l2_norm_eps

    def normalize_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self._maybe_l2_norm(x)

    def _maybe_l2_norm(self, x: torch.Tensor) -> torch.Tensor:
        if self._l2_norm:
            x = x / torch.clamp(
                torch.linalg.norm(x, ord=2, dim=-1, keepdim=True),
                min=self._l2_norm_eps,
            )
        return x

    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def process_batch(
        self,
        ids: torch.Tensor,
        presences: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> None:
        pass

    @abc.abstractmethod
    def forward(
        self,
        positive_ids: torch.Tensor,
        num_to_sample: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            A tuple of (sampled_ids, sampled_negative_embeddings).
        """
        pass

class LocalNegativesSampler(NegativesSampler):
    def __init__(
        self,
        num_items: int,
        item_emb: torch.nn.Embedding,
        all_item_ids: List[int],
        l2_norm: bool,
        l2_norm_eps: float,
    ) -> None:
        super().__init__(l2_norm=l2_norm, l2_norm_eps=l2_norm_eps)

        self._num_items: int = len(all_item_ids)
        self._item_emb: torch.nn.Embedding = item_emb
        self.register_buffer("_all_item_ids", torch.tensor(all_item_ids))

    def debug_str(self) -> str:
        sampling_debug_str = (
            f"local{f'-l2-eps{self._l2_norm_eps}' if self._l2_norm else ''}"
        )
        return sampling_debug_str

    def process_batch(
        self,
        ids: torch.Tensor,
        presences: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> None:
        pass

    def forward(
        self,
        positive_ids: torch.Tensor,
        num_to_sample: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            A tuple of (sampled_ids, sampled_negative_embeddings).
        """
        # assert torch.max(torch.abs(self._item_emb(positive_ids) - positive_embeddings)) < 1e-4
        output_shape = positive_ids.size() + (num_to_sample,)
        sampled_offsets = torch.randint(
            low=0,
            high=self._num_items,
            size=output_shape,
            dtype=positive_ids.dtype,
            device=positive_ids.device,
        )
        sampled_ids = self._all_item_ids[sampled_offsets.view(-1)].reshape(output_shape)
        return sampled_ids, self.normalize_embeddings(self._item_emb(sampled_ids))

class AutoregressiveLoss(torch.nn.Module):
    @abc.abstractmethod
    def jagged_forward(
        self,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        negatives_sampler: NegativesSampler,
    ) -> torch.Tensor:
        """
        Variant of forward() when the tensors are already in jagged format.

        Args:
            output_embeddings: [N', D] x float, embeddings for the current
                input sequence.
            supervision_ids: [N'] x int64, (positive) supervision ids.
            supervision_embeddings: [N', D] x float.
            supervision_weights: Optional [N'] x float. Optional weights for
                masking out invalid positions, or reweighting supervision labels.
            negatives_sampler: sampler used to obtain negative examples paired with
                positives.

        Returns:
            (1), loss for the current engaged sequence.
        """
        pass

    @abc.abstractmethod
    def forward(
        self,
        lengths: torch.Tensor,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        negatives_sampler: NegativesSampler,
    ) -> torch.Tensor:
        """
        Args:
            lengths: [B] x int32 representing number of non-zero elements per row.
            output_embeddings: [B, N, D] x float, embeddings for the current
                input sequence.
            supervision_ids: [B, N] x int64, (positive) supervision ids.
            supervision_embeddings: [B, N, D] x float.
            supervision_weights: Optional [B, N] x float. Optional weights for
                masking out invalid positions, or reweighting supervision labels.
            negatives_sampler: sampler used to obtain negative examples paired with
                positives.

        Returns:
            (1), loss for the current engaged sequence.
        """
        pass
