# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""Frozen candidate-aware MLP scoring for candidate-symmetric DeltaRec."""

from __future__ import annotations

from numbers import Integral
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class CandidateMLPSelector(nn.Module):
    """A frozen adapter for the registered ``SmallCandidateMLP``.

    The source MLP consumes ``[history, candidate, history * candidate]``.
    This adapter copies that module's weights into immutable buffers and splits
    its first layer into the exact contiguous ``W_h``, ``W_c``, and ``W_i``
    blocks.  :meth:`materialized_scores` is the transparent padded dense oracle;
    :meth:`factorized_scores` computes the same algebra without constructing a
    ``[B, K, L, 3 * D]`` feature tensor.  :meth:`exact_packed_scores` preserves
    the frozen selector's valid-event packing and GEMM shape exactly; it is the
    strict optimized-executor path because algebraic factorization can perturb
    boundary near-ties by a few FP32 ulps.

    Both raw embedding inputs are divided by ``embedding_rms`` before scoring,
    matching the frozen training and reference-inference path.  Parameters are
    intentionally absent from this module: gradients flow to its embedding
    inputs, while every copied selector weight remains frozen.
    """

    def __init__(
        self,
        source_mlp: nn.Module,
        *,
        embedding_rms: float | torch.Tensor,
        utility_scale: Optional[float | torch.Tensor] = None,
        validate_runtime: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(validate_runtime, bool):
            raise TypeError("validate_runtime must be boolean")
        input_layer = getattr(source_mlp, "input", None)
        output_layer = getattr(source_mlp, "output", None)
        if not isinstance(input_layer, nn.Linear) or not isinstance(
            output_layer, nn.Linear
        ):
            raise TypeError(
                "source_mlp must expose SmallCandidateMLP-equivalent "
                "Linear input and output layers"
            )
        if input_layer.bias is None or output_layer.bias is None:
            raise ValueError("candidate MLP input and output layers require biases")
        if input_layer.in_features % 3:
            raise ValueError("candidate MLP input width must equal 3 * embedding_dim")
        embedding_dim = input_layer.in_features // 3
        hidden_dim = input_layer.out_features
        if embedding_dim < 1 or hidden_dim < 1:
            raise ValueError("candidate MLP dimensions must be positive")
        if output_layer.in_features != hidden_dim or output_layer.out_features != 1:
            raise ValueError("candidate MLP output must map hidden_dim to one score")
        if input_layer.weight.device != output_layer.weight.device:
            raise ValueError("candidate MLP layers must share a device")
        if input_layer.weight.dtype != output_layer.weight.dtype:
            raise ValueError("candidate MLP layers must share a dtype")

        weight = input_layer.weight.detach()
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.validate_runtime = validate_runtime
        self.register_buffer("W_h", weight[:, :embedding_dim].clone())
        self.register_buffer(
            "W_c", weight[:, embedding_dim : 2 * embedding_dim].clone()
        )
        self.register_buffer("W_i", weight[:, 2 * embedding_dim :].clone())
        self.register_buffer("b_1", input_layer.bias.detach().clone())
        self.register_buffer("W_2", output_layer.weight.detach().clone())
        self.register_buffer("b_2", output_layer.bias.detach().clone())

        rms = self._scalar_like(embedding_rms, weight, "embedding_rms")
        if utility_scale is None:
            if not hasattr(source_mlp, "utility_scale"):
                raise ValueError(
                    "utility_scale must be provided when source_mlp has no "
                    "utility_scale buffer"
                )
            utility_scale = getattr(source_mlp, "utility_scale")
        scale = self._scalar_like(utility_scale, weight, "utility_scale")
        if not bool(torch.isfinite(rms)) or float(rms) <= 0.0:
            raise ValueError("embedding_rms must be finite and positive")
        if not bool(torch.isfinite(scale)) or float(scale) <= 0.0:
            raise ValueError("utility_scale must be finite and positive")
        self.register_buffer("embedding_rms", rms)
        self.register_buffer("utility_scale", scale)
        # Retain the registered scalar as a Python value so the exact packed
        # scoring path does not synchronize CUDA merely to recover a divisor
        # that never changes after construction.
        self._embedding_rms_float = float(rms.detach().cpu())

    @staticmethod
    def _scalar_like(
        value: float | torch.Tensor,
        reference: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        scalar = torch.as_tensor(
            value,
            dtype=reference.dtype,
            device=reference.device,
        )
        if scalar.numel() != 1:
            raise ValueError(f"{name} must be scalar")
        return scalar.detach().clone().reshape(())

    @classmethod
    def from_small_candidate_mlp(
        cls,
        source_mlp: nn.Module,
        *,
        embedding_rms: float | torch.Tensor,
        utility_scale: Optional[float | torch.Tensor] = None,
        validate_runtime: bool = True,
    ) -> "CandidateMLPSelector":
        """Copy and freeze a ``SmallCandidateMLP``-equivalent module."""

        return cls(
            source_mlp,
            embedding_rms=embedding_rms,
            utility_scale=utility_scale,
            validate_runtime=validate_runtime,
        )

    @classmethod
    def from_module(
        cls,
        source_mlp: nn.Module,
        *,
        embedding_rms: float | torch.Tensor,
        utility_scale: Optional[float | torch.Tensor] = None,
        validate_runtime: bool = True,
    ) -> "CandidateMLPSelector":
        """Alias for callers that avoid depending on the historical class name."""

        return cls.from_small_candidate_mlp(
            source_mlp,
            embedding_rms=embedding_rms,
            utility_scale=utility_scale,
            validate_runtime=validate_runtime,
        )

    @property
    def history_weight(self) -> torch.Tensor:
        return self.W_h

    @property
    def candidate_weight(self) -> torch.Tensor:
        return self.W_c

    @property
    def interaction_weight(self) -> torch.Tensor:
        return self.W_i

    @property
    def input_bias(self) -> torch.Tensor:
        return self.b_1

    @property
    def output_weight(self) -> torch.Tensor:
        return self.W_2

    @property
    def output_bias(self) -> torch.Tensor:
        return self.b_2

    def _validate_inputs(
        self,
        history_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        history_lengths: torch.Tensor,
    ) -> tuple[int, int, int]:
        if history_embeddings.ndim != 3:
            raise ValueError("history_embeddings must have shape [B,L,D]")
        batch, width, dimension = history_embeddings.shape
        if dimension != self.embedding_dim:
            raise ValueError("history embedding dimension does not match selector")
        if candidate_embeddings.ndim != 3 or candidate_embeddings.shape[0] != batch:
            raise ValueError("candidate_embeddings must have shape [B,K,D]")
        if candidate_embeddings.shape[2] != self.embedding_dim:
            raise ValueError("candidate embedding dimension does not match selector")
        candidates = candidate_embeddings.shape[1]
        if candidates < 1:
            raise ValueError("at least one candidate is required")
        if history_lengths.shape != (batch,) or history_lengths.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("history_lengths must be an integer tensor with shape [B]")
        if (
            history_embeddings.device != candidate_embeddings.device
            or history_embeddings.device != history_lengths.device
        ):
            raise ValueError("selector inputs must share a device")
        if history_embeddings.device != self.W_h.device:
            raise ValueError("selector buffers and inputs must share a device")
        # Value checks synchronize CUDA.  The high-level executor validates a
        # request once and may construct this adapter with validation disabled
        # to keep repeated candidate-chunk scoring fully asynchronous.
        if self.validate_runtime and (
            bool((history_lengths < 1).any())
            or bool((history_lengths > width).any())
        ):
            raise ValueError("history_lengths must address nonempty valid prefixes")
        return batch, candidates, width

    @staticmethod
    def _chunk_size(candidate_chunk_size: Optional[int], candidates: int) -> int:
        if candidate_chunk_size is None:
            return candidates
        if isinstance(candidate_chunk_size, bool) or not isinstance(
            candidate_chunk_size, Integral
        ):
            raise TypeError("candidate_chunk_size must be an integer or None")
        if candidate_chunk_size < 1:
            raise ValueError("candidate_chunk_size must be positive")
        return min(int(candidate_chunk_size), candidates)

    def _normalized_inputs(
        self,
        history_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        history_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The registered weights are FP32.  Converting to their dtype also
        # reproduces the historical ``raw_embedding.float() / rms`` boundary
        # for BF16 model embeddings while preserving autograd to both inputs.
        dtype = self.W_h.dtype
        rms = self.embedding_rms.to(dtype=dtype)
        history = history_embeddings.to(dtype=dtype) / rms
        # Do not merely overwrite final padding scores: sanitizing the raw
        # suffix also prevents NaN/Inf poison from entering otherwise-zero
        # backward paths through SiLU and the interaction contraction.
        positions = torch.arange(history.shape[1], device=history.device)
        valid = positions[None, :] < history_lengths[:, None]
        history = history.masked_fill(~valid[..., None], 0.0)
        return history, candidate_embeddings.to(dtype=dtype) / rms

    def materialized_scores(
        self,
        history_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        history_lengths: torch.Tensor,
        *,
        candidate_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Return the dense feature-concatenation reference scores ``[B,K,L]``."""

        batch, candidates, width = self._validate_inputs(
            history_embeddings, candidate_embeddings, history_lengths
        )
        chunk_size = self._chunk_size(candidate_chunk_size, candidates)
        history, candidate = self._normalized_inputs(
            history_embeddings, candidate_embeddings, history_lengths
        )
        input_weight = torch.cat((self.W_h, self.W_c, self.W_i), dim=1)
        chunks: list[torch.Tensor] = []
        for start in range(0, candidates, chunk_size):
            current = candidate[:, start : start + chunk_size]
            count = current.shape[1]
            event = history[:, None, :, :].expand(
                batch, count, width, self.embedding_dim
            )
            conditioned = current[:, :, None, :].expand(
                batch, count, width, self.embedding_dim
            )
            features = torch.cat((event, conditioned, event * conditioned), dim=-1)
            hidden = F.silu(F.linear(features, input_weight, self.b_1))
            chunks.append(F.linear(hidden, self.W_2, self.b_2).squeeze(-1))
        scores = torch.cat(chunks, dim=1)
        scores = scores * self.utility_scale.to(scores.dtype)
        positions = torch.arange(width, device=scores.device)
        invalid = positions[None, None, :] >= history_lengths[:, None, None]
        return scores.masked_fill(invalid, float("-inf"))

    # A deliberately explicit alias makes call sites self-document which path
    # is the allocation-heavy correctness oracle.
    score_materialized_reference = materialized_scores

    def exact_packed_scores(
        self,
        history_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        history_lengths: torch.Tensor,
        *,
        candidate_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Return scores with the frozen selector's exact packed GEMM shape.

        The registered PC-MLP reference repeats each history for a candidate
        chunk, removes padded events, constructs the concatenated
        ``[event, candidate, event*candidate]`` features for only those valid
        events, and invokes the two frozen linear layers on that packed matrix.
        Reproducing that ordering and matrix shape is necessary for bitwise
        selector scores: splitting the first linear layer into algebraically
        equivalent terms changes FP32 reduction order and can flip a boundary
        near-tie.  Candidate chunking bounds the packed feature allocation while
        invalid suffix values never enter the computation.
        """

        batch, candidates, width = self._validate_inputs(
            history_embeddings, candidate_embeddings, history_lengths
        )
        chunk_size = self._chunk_size(candidate_chunk_size, candidates)
        dtype = self.W_h.dtype
        history = history_embeddings.to(dtype=dtype)
        candidate = candidate_embeddings.to(dtype=dtype)
        rms = self._embedding_rms_float
        input_weight = torch.cat((self.W_h, self.W_c, self.W_i), dim=1)
        positions = torch.arange(width, device=history.device)
        chunks: list[torch.Tensor] = []
        for start in range(0, candidates, chunk_size):
            current = candidate[:, start : start + chunk_size]
            count = current.shape[1]
            repeated_history = history[:, None, :, :].expand(
                batch, count, width, self.embedding_dim
            ).reshape(batch * count, width, self.embedding_dim)
            repeated_lengths = history_lengths.repeat_interleave(count)
            valid = positions[None, :] < repeated_lengths[:, None]
            packed_history = repeated_history[valid] / rms
            normalized_candidate = current.reshape(
                batch * count, self.embedding_dim
            ) / rms
            sequence_ids = torch.repeat_interleave(
                torch.arange(
                    len(repeated_lengths),
                    device=history.device,
                ),
                repeated_lengths,
            )
            conditioned = normalized_candidate.index_select(0, sequence_ids)
            features = torch.cat(
                (
                    packed_history,
                    conditioned,
                    packed_history * conditioned,
                ),
                dim=-1,
            )
            hidden = F.silu(F.linear(features, input_weight, self.b_1))
            packed_scores = F.linear(hidden, self.W_2, self.b_2).squeeze(-1)
            packed_scores = packed_scores * self.utility_scale.to(
                packed_scores.dtype
            )
            dense = history.new_full(
                (batch * count, width),
                float("-inf"),
                dtype=torch.float32,
            )
            dense[valid] = packed_scores.float()
            chunks.append(dense.reshape(batch, count, width))
        return torch.cat(chunks, dim=1)

    score_exact_packed = exact_packed_scores

    def factorized_scores(
        self,
        history_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        history_lengths: torch.Tensor,
        *,
        candidate_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Return factorized candidate scores without a ``3 * D`` pair tensor.

        Candidate chunking bounds the only pairwise activation at
        ``[B, chunk, L, hidden_dim]``.  The diagonal bilinear interaction uses
        candidate-weighted hidden rows and batched matrix multiplication, so
        no ``[B, chunk, L, D]`` intermediate is requested.
        """

        _, candidates, width = self._validate_inputs(
            history_embeddings, candidate_embeddings, history_lengths
        )
        chunk_size = self._chunk_size(candidate_chunk_size, candidates)
        history, candidate = self._normalized_inputs(
            history_embeddings, candidate_embeddings, history_lengths
        )
        history_term = F.linear(history, self.W_h, self.b_1)
        chunks: list[torch.Tensor] = []
        for start in range(0, candidates, chunk_size):
            current = candidate[:, start : start + chunk_size]
            batch = current.shape[0]
            count = current.shape[1]
            candidate_term = F.linear(current, self.W_c, None)
            # Weight each candidate once per hidden unit, then use one batched
            # matrix multiplication.  The largest pairwise result is
            # [B, chunk, L, H]; no [B, chunk, L, D] product is materialized.
            weighted_candidate = (
                current[:, :, None, :] * self.W_i[None, None, :, :]
            ).reshape(batch, count * self.hidden_dim, self.embedding_dim)
            interaction = torch.bmm(
                history,
                weighted_candidate.transpose(1, 2),
            ).reshape(
                batch,
                width,
                count,
                self.hidden_dim,
            ).permute(
                0,
                2,
                1,
                3,
            )
            hidden = F.silu(
                history_term[:, None, :, :]
                + candidate_term[:, :, None, :]
                + interaction
            )
            chunks.append(F.linear(hidden, self.W_2, self.b_2).squeeze(-1))
        scores = torch.cat(chunks, dim=1)
        scores = scores * self.utility_scale.to(scores.dtype)
        positions = torch.arange(width, device=scores.device)
        invalid = positions[None, None, :] >= history_lengths[:, None, None]
        return scores.masked_fill(invalid, float("-inf"))

    score_factorized = factorized_scores

    def forward(
        self,
        history_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        history_lengths: torch.Tensor,
        *,
        candidate_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        return self.factorized_scores(
            history_embeddings,
            candidate_embeddings,
            history_lengths,
            candidate_chunk_size=candidate_chunk_size,
        )


__all__ = ["CandidateMLPSelector"]

