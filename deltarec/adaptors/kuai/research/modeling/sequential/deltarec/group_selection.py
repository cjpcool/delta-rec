# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""Exact group-conditioned history selection for Subplan 5C."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .selection import select_candidate_events
from .types import SelectionOutput


@dataclass(frozen=True)
class GroupSelectionOutput:
    """Exact chronological selection for all ``B*G`` semantic group rows."""

    indices: torch.Tensor
    counts: torch.Tensor
    budgets: torch.Tensor
    packed_source_indices: torch.Tensor
    source_positions: torch.Tensor
    group_offsets: torch.Tensor
    dense_mask: torch.Tensor | None = None

    @property
    def offsets(self) -> torch.Tensor:
        """Compatibility alias for generic packed-sequence consumers."""

        return self.group_offsets

    @property
    def selected_indices(self) -> torch.Tensor:
        return self.indices

    @property
    def selected_counts(self) -> torch.Tensor:
        return self.counts

    @property
    def batch_size(self) -> int:
        return int(self.counts.shape[0])

    @property
    def group_count(self) -> int:
        return int(self.counts.shape[1])

    @property
    def selected_tokens(self) -> int:
        return int(self.group_offsets[-1])

    def validate(self, history_width: int) -> None:
        """Reuse the audited generic packed-selection validation contract."""

        SelectionOutput(
            indices=self.indices,
            counts=self.counts,
            budgets=self.budgets,
            packed_source_indices=self.packed_source_indices,
            source_positions=self.source_positions,
            offsets=self.group_offsets,
            dense_mask=self.dense_mask,
        ).validate(history_width)


def select_group_events(
    group_scores: torch.Tensor,
    history_lengths: torch.Tensor,
    *,
    recent_floor: int = 32,
    retention_ratio: float = 0.50,
    return_dense_mask: bool = False,
    validate_runtime: bool = True,
) -> GroupSelectionOutput:
    """Select each group's exact registered budget and pack it chronologically.

    Mandatory recent events count inside the budget.  Remaining events use a
    stable descending score order, so exact ties prefer the earlier source
    position.  Padding is excluded before ranking.
    """

    selection = select_candidate_events(
        group_scores,
        history_lengths,
        recent_floor=recent_floor,
        retention_ratio=retention_ratio,
        return_dense_mask=return_dense_mask,
        validate_runtime=validate_runtime,
    )
    output = GroupSelectionOutput(
        indices=selection.indices,
        counts=selection.counts,
        budgets=selection.budgets,
        packed_source_indices=selection.packed_source_indices,
        source_positions=selection.source_positions,
        group_offsets=selection.offsets,
        dense_mask=selection.dense_mask,
    )
    if validate_runtime:
        output.validate(group_scores.shape[-1])
    return output

