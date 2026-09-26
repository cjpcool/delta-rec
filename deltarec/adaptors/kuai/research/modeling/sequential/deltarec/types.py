# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""Public tensor contracts for candidate-symmetric DeltaRec execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import torch


@dataclass(frozen=True)
class DeltaRecRequest:
    """Frozen candidate-symmetric request.

    Histories are valid prefixes of ``[B, L]``.  Raw history and candidate
    embeddings are inputs to the frozen selector, while ranking candidate
    embeddings are consumed only by the matched scorer.
    """

    history_ids: torch.Tensor
    history_lengths: torch.Tensor
    history_embeddings: torch.Tensor
    candidate_ids: torch.Tensor
    candidate_embeddings: torch.Tensor
    ranking_candidate_embeddings: torch.Tensor
    history_payloads: Mapping[str, torch.Tensor] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return int(self.history_ids.shape[0])

    @property
    def history_width(self) -> int:
        return int(self.history_ids.shape[1])

    @property
    def candidate_count(self) -> int:
        return int(self.candidate_ids.shape[1])

    def validate(self) -> None:
        if self.history_ids.ndim != 2 or self.history_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("history_ids must be an integer tensor with shape [B,L]")
        batch, width = self.history_ids.shape
        if self.history_lengths.shape != (batch,) or self.history_lengths.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("history_lengths must be an integer tensor with shape [B]")
        if bool((self.history_lengths < 1).any()) or bool(
            (self.history_lengths > width).any()
        ):
            raise ValueError("history_lengths must address nonempty valid prefixes")
        if self.history_embeddings.ndim != 3 or self.history_embeddings.shape[:2] != (
            batch,
            width,
        ):
            raise ValueError("history_embeddings must have shape [B,L,D]")
        if self.candidate_ids.ndim != 2 or self.candidate_ids.shape[0] != batch:
            raise ValueError("candidate_ids must have shape [B,K]")
        if self.candidate_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("candidate_ids must be integer")
        candidates = self.candidate_ids.shape[1]
        if candidates < 1:
            raise ValueError("candidate_ids must include at least one candidate")
        if (
            self.candidate_embeddings.ndim != 3
            or self.candidate_embeddings.shape[:2] != (batch, candidates)
        ):
            raise ValueError("candidate_embeddings must have shape [B,K,D]")
        if self.candidate_embeddings.shape[-1] != self.history_embeddings.shape[-1]:
            raise ValueError("selector history and candidate embedding dimensions differ")
        if (
            self.ranking_candidate_embeddings.ndim != 3
            or self.ranking_candidate_embeddings.shape[:2] != (batch, candidates)
        ):
            raise ValueError("ranking_candidate_embeddings must have shape [B,K,Dq]")
        device = self.history_ids.device
        tensors = (
            self.history_lengths,
            self.history_embeddings,
            self.candidate_ids,
            self.candidate_embeddings,
            self.ranking_candidate_embeddings,
        )
        if any(tensor.device != device for tensor in tensors):
            raise ValueError("all DeltaRec request tensors must share a device")
        for name, payload in self.history_payloads.items():
            if payload.ndim < 2 or payload.shape[:2] != (batch, width):
                raise ValueError(f"history payload {name!r} must start with [B,L]")
            if payload.device != device:
                raise ValueError(f"history payload {name!r} is on a different device")

    def candidate_slice(self, start: int, end: int) -> "DeltaRecRequest":
        if not 0 <= start < end <= self.candidate_count:
            raise ValueError("candidate slice is outside [0,K]")
        return DeltaRecRequest(
            history_ids=self.history_ids,
            history_lengths=self.history_lengths,
            history_embeddings=self.history_embeddings,
            history_payloads=self.history_payloads,
            candidate_ids=self.candidate_ids[:, start:end],
            candidate_embeddings=self.candidate_embeddings[:, start:end],
            ranking_candidate_embeddings=self.ranking_candidate_embeddings[:, start:end],
        )


@dataclass(frozen=True)
class SelectionOutput:
    """Exact candidate-specific membership and chronological packed layout."""

    indices: torch.Tensor
    counts: torch.Tensor
    budgets: torch.Tensor
    packed_source_indices: torch.Tensor
    source_positions: torch.Tensor
    offsets: torch.Tensor
    dense_mask: Optional[torch.Tensor] = None

    @property
    def batch_size(self) -> int:
        return int(self.counts.shape[0])

    @property
    def candidate_count(self) -> int:
        return int(self.counts.shape[1])

    @property
    def selected_tokens(self) -> int:
        return int(self.offsets[-1])

    def validate(self, history_width: int) -> None:
        if isinstance(history_width, bool) or not isinstance(history_width, int):
            raise TypeError("history_width must be an integer")
        if history_width < 1:
            raise ValueError("history_width must be positive")
        if self.counts.ndim != 2 or self.counts.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("counts must be an integer tensor with shape [B,K]")
        batch, candidates = self.counts.shape
        if self.budgets.shape != (batch,) or self.budgets.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("budgets must be an integer tensor with shape [B]")
        if not torch.equal(
            self.counts.to(self.budgets),
            self.budgets[:, None].expand(batch, candidates),
        ):
            raise ValueError("every candidate row must use its user's exact budget")
        if self.indices.ndim != 3 or self.indices.shape[:2] != (batch, candidates):
            raise ValueError("indices must have shape [B,K,Smax]")
        if self.indices.dtype != torch.int64:
            raise ValueError("indices must be int64")
        device = self.counts.device
        if any(
            tensor.device != device
            for tensor in (
                self.budgets,
                self.indices,
                self.packed_source_indices,
                self.source_positions,
                self.offsets,
            )
        ):
            raise ValueError("all selection tensors must share a device")
        if self.offsets.shape != (batch * candidates + 1,) or self.offsets.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("offsets must have shape [B*K+1]")
        if int(self.offsets[0]) != 0 or int(self.offsets[-1]) != len(
            self.packed_source_indices
        ):
            raise ValueError("offsets must span every packed index")
        if bool((self.offsets[1:] < self.offsets[:-1]).any()):
            raise ValueError("offsets must be nondecreasing")
        if not torch.equal(
            (self.offsets[1:] - self.offsets[:-1]).reshape(batch, candidates).to(
                self.counts
            ),
            self.counts,
        ):
            raise ValueError("offset lengths and counts disagree")
        tokens = len(self.packed_source_indices)
        if self.packed_source_indices.shape != (tokens,) or self.source_positions.shape != (
            tokens,
        ):
            raise ValueError("packed indices and source positions must be vectors")
        if self.packed_source_indices.dtype != torch.int64 or self.source_positions.dtype != torch.int64:
            raise ValueError("packed indices and source positions must be int64")
        if tokens and (
            int(self.packed_source_indices.min()) < 0
            or int(self.packed_source_indices.max()) >= batch * history_width
            or int(self.source_positions.min()) < 0
            or int(self.source_positions.max()) >= history_width
        ):
            raise ValueError("packed selection addresses an invalid history position")
        if self.dense_mask is not None:
            if self.dense_mask.shape != (batch, candidates, history_width):
                raise ValueError("dense_mask must have shape [B,K,L]")
            if self.dense_mask.dtype != torch.bool:
                raise ValueError("dense_mask must be boolean")
            if self.dense_mask.device != device:
                raise ValueError("dense_mask and compact selection must share a device")
        flat_counts = self.counts.reshape(-1).to(torch.int64)
        packed_rows = torch.repeat_interleave(
            torch.arange(batch * candidates, device=device, dtype=torch.int64),
            flat_counts,
        )
        if len(packed_rows) != tokens:
            raise ValueError("packed row count disagrees with selected tokens")
        if tokens > 1:
            same_row = packed_rows[1:] == packed_rows[:-1]
            if bool(
                (
                    same_row
                    & (self.source_positions[1:] <= self.source_positions[:-1])
                ).any()
            ):
                raise ValueError("packed positions must be unique and chronological")
        packed_users = torch.div(packed_rows, candidates, rounding_mode="floor")
        if not torch.equal(
            self.packed_source_indices,
            self.source_positions + packed_users * history_width,
        ):
            raise ValueError("packed source indices cross a user boundary")
        if self.dense_mask is not None:
            if not torch.equal(
                self.dense_mask.sum(dim=-1).to(self.counts),
                self.counts,
            ):
                raise ValueError("dense_mask counts disagree with compact selection")
            dense_coordinates = torch.nonzero(self.dense_mask, as_tuple=False)
            dense_rows = (
                dense_coordinates[:, 0].to(torch.int64) * candidates
                + dense_coordinates[:, 1].to(torch.int64)
            )
            if not torch.equal(dense_rows, packed_rows) or not torch.equal(
                dense_coordinates[:, 2].to(torch.int64),
                self.source_positions,
            ):
                raise ValueError("dense_mask membership disagrees with compact selection")


@dataclass(frozen=True)
class DeltaRecDiagnostics:
    backend: str
    architecture: str
    logical_states: int
    selected_transitions: int
    physical_gdr_calls: Optional[int] = None
    stage_times_ms: Mapping[str, float] = field(default_factory=dict)
    temporary_bytes: Mapping[str, int] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeltaRecOutput:
    scores: torch.Tensor
    queries: Optional[torch.Tensor] = None
    selection: Optional[SelectionOutput] = None
    final_states: Optional[torch.Tensor] = None
    diagnostics: Optional[DeltaRecDiagnostics] = None

    def validate(self, batch_size: int, candidate_count: int) -> None:
        if self.scores.shape != (batch_size, candidate_count):
            raise ValueError("DeltaRec scores must have shape [B,K]")
        if self.queries is not None and self.queries.shape[:2] != (
            batch_size,
            candidate_count,
        ):
            raise ValueError("DeltaRec queries must start with shape [B,K]")
        if not bool(torch.isfinite(self.scores).all()):
            raise ValueError("DeltaRec scores must be finite")

