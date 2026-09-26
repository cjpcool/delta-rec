# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""Exact, stable, candidate-specific event selection for DeltaRec."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .types import SelectionOutput


SUPPORTED_RETENTION_RATIOS = (0.25, 0.50)
HEADLINE_WRITE_RATIO = 0.50
HEADLINE_RECENT_FLOOR = 32


def validate_retention_ratio(retention_ratio: float) -> float:
    """Return one of the two preregistered ratios or fail closed."""

    if isinstance(retention_ratio, bool) or not isinstance(
        retention_ratio, (int, float)
    ):
        raise TypeError("retention_ratio must be numeric")
    ratio = float(retention_ratio)
    if ratio not in SUPPORTED_RETENTION_RATIOS:
        raise ValueError("retention_ratio must be one of {0.25, 0.50}")
    return ratio


def exact_budget(
    history_lengths: torch.Tensor,
    *,
    retention_ratio: float,
    recent_floor: int = HEADLINE_RECENT_FLOOR,
    validate_runtime: bool = True,
) -> torch.Tensor:
    """Return ``max(ceil(rho*n), min(recent_floor,n))`` for each history.

    The ratio multiplication and ceiling are explicitly FP32, matching the
    preregistered controller.  Only 25% and 50% are admitted.  The default
    recent floor is the headline value 32; callers using a diagnostic floor
    must pass it explicitly so cache versions can bind that concrete value.
    """

    if history_lengths.ndim != 1 or history_lengths.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("history_lengths must be a one-dimensional integer tensor")
    if not isinstance(validate_runtime, bool):
        raise TypeError("validate_runtime must be boolean")
    ratio = validate_retention_ratio(retention_ratio)
    if isinstance(recent_floor, bool) or not isinstance(recent_floor, int):
        raise TypeError("recent_floor must be an integer")
    if recent_floor < 0:
        raise ValueError("recent_floor must be nonnegative")
    if validate_runtime and bool((history_lengths < 0).any()):
        raise ValueError("history_lengths must be nonnegative")
    lengths_i64 = history_lengths.to(torch.int64)
    ratio_budget = torch.ceil(
        history_lengths.to(torch.float32) * ratio
    ).to(torch.int64)
    floor_budget = torch.minimum(
        lengths_i64,
        lengths_i64.new_full(lengths_i64.shape, recent_floor),
    )
    return torch.minimum(
        lengths_i64,
        torch.maximum(ratio_budget, floor_budget),
    )


def exact_quarter_budget(
    history_lengths: torch.Tensor,
    *,
    recent_floor: int = HEADLINE_RECENT_FLOOR,
    validate_runtime: bool = True,
) -> torch.Tensor:
    """Return the preregistered 25% budget including the recent floor."""

    return exact_budget(
        history_lengths,
        retention_ratio=0.25,
        recent_floor=recent_floor,
        validate_runtime=validate_runtime,
    )


# Plural form reads naturally at call sites and remains a true alias, so there
# is only one budget implementation to audit.
exact_quarter_budgets = exact_quarter_budget


def exact_half_budget(
    history_lengths: torch.Tensor,
    *,
    recent_floor: int = HEADLINE_RECENT_FLOOR,
    validate_runtime: bool = True,
) -> torch.Tensor:
    """Return the preregistered 50% budget including the recent floor."""

    return exact_budget(
        history_lengths,
        retention_ratio=HEADLINE_WRITE_RATIO,
        recent_floor=recent_floor,
        validate_runtime=validate_runtime,
    )


exact_half_budgets = exact_half_budget


def _validate_selection_inputs(
    scores: torch.Tensor,
    history_lengths: torch.Tensor,
    recent_floor: int,
    validate_runtime: bool,
) -> tuple[int, int, int, torch.Tensor]:
    if scores.ndim != 3 or not torch.is_floating_point(scores):
        raise ValueError("scores must be floating point with shape [B,K,L]")
    batch, candidates, width = scores.shape
    if batch < 1 or candidates < 1 or width < 1:
        raise ValueError("scores must contain at least one user, candidate, and event")
    if history_lengths.shape != (batch,) or history_lengths.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("history_lengths must be an integer tensor with shape [B]")
    if history_lengths.device != scores.device:
        raise ValueError("scores and history_lengths must share a device")
    if not isinstance(validate_runtime, bool):
        raise TypeError("validate_runtime must be boolean")
    if validate_runtime and (
        bool((history_lengths < 1).any())
        or bool((history_lengths > width).any())
    ):
        raise ValueError("history_lengths must address nonempty valid prefixes")
    if isinstance(recent_floor, bool) or not isinstance(recent_floor, int):
        raise TypeError("recent_floor must be an integer")
    if recent_floor < 0:
        raise ValueError("recent_floor must be nonnegative")
    positions = torch.arange(width, device=scores.device)
    valid = positions[None, :] < history_lengths[:, None]
    if validate_runtime and not bool(
        torch.isfinite(scores.masked_select(valid[:, None, :])).all()
    ):
        raise ValueError("valid selector scores must be finite")
    return batch, candidates, width, positions


def select_candidate_events(
    scores: torch.Tensor,
    history_lengths: torch.Tensor,
    *,
    recent_floor: int = 32,
    retention_ratio: float = HEADLINE_WRITE_RATIO,
    return_dense_mask: bool = False,
    validate_runtime: bool = True,
) -> SelectionOutput:
    """Select exact candidate-specific budgets and pack them chronologically.

    Selection is fully batched over the flattened ``B*K`` logical rows.  The
    mandatory recent suffix counts inside each user's registered budget.  Remaining
    membership is chosen by stable descending score, which makes an exact tie
    prefer the earlier source position.  Padding is overwritten with
    ``-inf`` before sorting and can therefore contain arbitrary poison values.

    ``SelectionOutput.indices`` stores chronological source positions with
    ``-1`` outside each row's count.  ``packed_source_indices`` addresses the
    flattened original history ``[B,L]`` and is laid out in final row-major
    ``(b,k)`` order.
    """

    batch, candidates, width, positions = _validate_selection_inputs(
        scores, history_lengths, recent_floor, validate_runtime
    )
    if not isinstance(return_dense_mask, bool):
        raise TypeError("return_dense_mask must be boolean")

    budgets = exact_budget(
        history_lengths,
        retention_ratio=retention_ratio,
        recent_floor=recent_floor,
        validate_runtime=validate_runtime,
    )
    recent_counts = torch.minimum(
        budgets,
        budgets.new_full(budgets.shape, recent_floor),
    )
    recent_starts = history_lengths.to(torch.int64) - recent_counts

    # Every nonmandatory eligible event lies strictly before recent_starts.
    # Invalid suffixes and mandatory events are replaced before argsort, so a
    # NaN/large-value padding poison cannot affect valid membership.
    eligible = positions[None, :] < recent_starts[:, None]
    ranked_scores = scores.masked_fill(~eligible[:, None, :], float("-inf"))
    order = torch.argsort(
        ranked_scores,
        dim=-1,
        descending=True,
        stable=True,
    )
    remaining = budgets - recent_counts
    take_by_rank = (
        positions[None, None, :] < remaining[:, None, None]
    ).expand(batch, candidates, width)
    selected_earlier = torch.zeros_like(scores, dtype=torch.bool).scatter(
        dim=-1,
        index=order,
        src=take_by_rank,
    )
    mandatory_recent = (
        (positions[None, :] >= recent_starts[:, None])
        & (positions[None, :] < history_lengths[:, None])
    )
    dense_mask = selected_earlier | mandatory_recent[:, None, :]

    counts = budgets[:, None].expand(batch, candidates)
    if validate_runtime:
        if not torch.equal(dense_mask.sum(dim=-1).to(counts), counts):
            raise RuntimeError("candidate selection did not satisfy the exact budget")
        if bool((mandatory_recent[:, None, :] & ~dense_mask).any()):
            raise RuntimeError("candidate selection omitted a mandatory recent event")
        valid = positions[None, :] < history_lengths[:, None]
        if bool((dense_mask & ~valid[:, None, :]).any()):
            raise RuntimeError("candidate selection included a padded event")

    # ``nonzero`` is lexicographic in (b, k, t), hence chronological within
    # every logical row.  Repacking these coordinates avoids a second full
    # argsort over B*K*L after score-based membership is already known.
    coordinates = torch.nonzero(dense_mask, as_tuple=False)
    source_positions = coordinates[:, 2].to(torch.int64)
    max_budget = int(budgets.max())
    valid_slots = (
        torch.arange(max_budget, device=scores.device)[None, None, :]
        < counts[..., None]
    )
    indices = torch.full(
        (batch, candidates, max_budget),
        -1,
        dtype=torch.int64,
        device=scores.device,
    )
    indices.masked_scatter_(valid_slots, source_positions)

    flat_counts = counts.reshape(-1).to(torch.int64)
    offsets = torch.cat(
        (flat_counts.new_zeros(1), torch.cumsum(flat_counts, dim=0)),
        dim=0,
    )
    packed_source_indices = coordinates[:, 0].to(torch.int64) * width + source_positions

    return SelectionOutput(
        indices=indices,
        counts=counts,
        budgets=budgets,
        packed_source_indices=packed_source_indices,
        source_positions=source_positions,
        offsets=offsets,
        dense_mask=dense_mask if return_dense_mask else None,
    )


def _padded_packed_field(
    selection: SelectionOutput,
    field: torch.Tensor,
    padded_width: int,
    *,
    fill_value: int,
) -> torch.Tensor:
    """Restore a packed integer field to ``[B,K,S]`` without row loops."""

    batch, candidates = selection.counts.shape
    if field.ndim != 1 or field.dtype != torch.int64:
        raise ValueError("packed selection fields must be int64 vectors")
    slots = (
        torch.arange(padded_width, device=selection.counts.device)[None, None, :]
        < selection.counts[..., None]
    )
    if field.numel() != int(slots.sum()):
        raise ValueError("packed selection field length disagrees with counts")
    padded = torch.full(
        (batch, candidates, padded_width),
        fill_value,
        dtype=torch.int64,
        device=selection.counts.device,
    )
    padded.masked_scatter_(slots, field)
    return padded


def _right_pad_indices(indices: torch.Tensor, width: int) -> torch.Tensor:
    if indices.shape[-1] == width:
        return indices
    output = indices.new_full((*indices.shape[:-1], width), -1)
    output[..., : indices.shape[-1]] = indices
    return output


def merge_candidate_selections(
    selections: Sequence[SelectionOutput],
) -> SelectionOutput:
    """Merge candidate-chunk outputs into final row-major ``B,K`` packing.

    Simply concatenating chunk-packed vectors is incorrect for ``B > 1``: it
    produces chunk-major physical streams.  This routine restores each chunk
    to a padded ``[B,K_chunk,S]`` representation, concatenates candidates, and
    repacks once in final ``b*K+k`` order.
    """

    chunks = tuple(selections)
    if not chunks:
        raise ValueError("at least one candidate selection is required")
    first = chunks[0]
    if first.counts.ndim != 2:
        raise ValueError("selection counts must have shape [B,K]")
    batch = first.batch_size
    device = first.counts.device
    budgets = first.budgets
    max_budget = max(chunk.indices.shape[-1] for chunk in chunks)
    if max_budget < 1:
        raise ValueError("candidate selections must contain nonempty rows")

    indices_chunks: list[torch.Tensor] = []
    counts_chunks: list[torch.Tensor] = []
    packed_source_chunks: list[torch.Tensor] = []
    source_position_chunks: list[torch.Tensor] = []
    dense_chunks: list[torch.Tensor] = []
    all_dense = all(chunk.dense_mask is not None for chunk in chunks)
    dense_width = None

    for chunk in chunks:
        if chunk.counts.ndim != 2 or chunk.batch_size != batch:
            raise ValueError("candidate chunks must share a batch size")
        if chunk.candidate_count < 1:
            raise ValueError("candidate chunks must be nonempty")
        if chunk.counts.device != device or chunk.indices.device != device:
            raise ValueError("candidate chunks must share a device")
        if not torch.equal(chunk.budgets.to(budgets), budgets):
            raise ValueError("candidate chunks must share per-user budgets")
        expected_counts = budgets[:, None].expand_as(chunk.counts)
        if not torch.equal(chunk.counts.to(expected_counts), expected_counts):
            raise ValueError("candidate chunk counts do not match exact budgets")
        if chunk.indices.ndim != 3 or chunk.indices.shape[:2] != chunk.counts.shape:
            raise ValueError("chunk indices must have shape [B,K_chunk,S]")
        if chunk.indices.dtype != torch.int64:
            raise ValueError("chunk indices must be int64")

        indices_chunks.append(_right_pad_indices(chunk.indices, max_budget))
        counts_chunks.append(chunk.counts)
        packed_source_chunks.append(
            _padded_packed_field(
                chunk,
                chunk.packed_source_indices,
                max_budget,
                fill_value=-1,
            )
        )
        source_position_chunks.append(
            _padded_packed_field(
                chunk,
                chunk.source_positions,
                max_budget,
                fill_value=-1,
            )
        )
        if all_dense:
            assert chunk.dense_mask is not None
            if chunk.dense_mask.ndim != 3 or chunk.dense_mask.shape[:2] != chunk.counts.shape:
                raise ValueError("chunk dense masks must have shape [B,K_chunk,L]")
            if chunk.dense_mask.dtype != torch.bool:
                raise ValueError("chunk dense masks must be boolean")
            if dense_width is None:
                dense_width = chunk.dense_mask.shape[-1]
            elif chunk.dense_mask.shape[-1] != dense_width:
                raise ValueError("candidate chunk dense masks must share a history width")
            dense_chunks.append(chunk.dense_mask)

    indices = torch.cat(indices_chunks, dim=1)
    counts = torch.cat(counts_chunks, dim=1)
    padded_source_indices = torch.cat(packed_source_chunks, dim=1)
    padded_source_positions = torch.cat(source_position_chunks, dim=1)
    candidates = counts.shape[1]
    slots = (
        torch.arange(max_budget, device=device)[None, None, :]
        < counts[..., None]
    )
    packed_source_indices = padded_source_indices.masked_select(slots)
    source_positions = padded_source_positions.masked_select(slots)
    if not torch.equal(indices.masked_select(slots), source_positions):
        raise ValueError("chunk indices and packed source positions disagree")
    flat_counts = counts.reshape(-1).to(torch.int64)
    offsets = torch.cat(
        (flat_counts.new_zeros(1), torch.cumsum(flat_counts, dim=0)),
        dim=0,
    )
    dense_mask = torch.cat(dense_chunks, dim=1) if all_dense else None

    return SelectionOutput(
        indices=indices,
        counts=counts,
        budgets=budgets,
        packed_source_indices=packed_source_indices,
        source_positions=source_positions,
        offsets=offsets,
        dense_mask=dense_mask,
    )


__all__ = [
    "HEADLINE_RECENT_FLOOR",
    "HEADLINE_WRITE_RATIO",
    "SUPPORTED_RETENTION_RATIOS",
    "exact_budget",
    "exact_half_budget",
    "exact_half_budgets",
    "exact_quarter_budget",
    "exact_quarter_budgets",
    "merge_candidate_selections",
    "select_candidate_events",
    "validate_retention_ratio",
]

