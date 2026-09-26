# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""Physical packing for group-shared candidate-conditioned DeltaRec.

Each logical row contains contextual writes, one chronologically selected
group history, and every candidate query assigned to that group.  Candidate
queries are appended as identity/read-only GDR events and their output indices
are scattered back to the original ``[B, K]`` candidate order.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .grouping import GroupingOutput
from .global_group_state_cache import GlobalGroupStateLookup
from .group_selection import GroupSelectionOutput
from .projections import HSTUFixedWidthLayout


@dataclass(frozen=True)
class PackedGroupSequence:
    """One physical buffer containing all ``B * G`` group streams."""

    values: torch.Tensor
    offsets: torch.Tensor
    event_gate: torch.Tensor
    query_indices: torch.Tensor
    source_indices: torch.Tensor
    source_roles: torch.Tensor
    history_lengths: torch.Tensor
    group_sizes: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.query_indices.shape[0])

    @property
    def candidate_count(self) -> int:
        return int(self.query_indices.shape[1])

    @property
    def group_count(self) -> int:
        return int(self.group_sizes.shape[1])

    @property
    def sequence_count(self) -> int:
        return self.batch_size * self.group_count

    @property
    def packed_tokens(self) -> int:
        return len(self.values)

    @property
    def write_tokens(self) -> int:
        # This shape-only count intentionally avoids a CUDA synchronization.
        return self.packed_tokens - self.batch_size * self.candidate_count

    @property
    def read_tokens(self) -> int:
        return self.batch_size * self.candidate_count

    def validate(self) -> None:
        if self.values.ndim < 1:
            raise ValueError("packed group values must have a token dimension")
        tokens = len(self.values)
        if self.offsets.shape != (self.sequence_count + 1,) or self.offsets.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("group offsets must have shape [B*G+1]")
        if int(self.offsets[0]) != 0 or int(self.offsets[-1]) != tokens:
            raise ValueError("group offsets must span every packed token")
        if bool((self.offsets[1:] <= self.offsets[:-1]).any()):
            raise ValueError("every packed group stream must be nonempty")
        if self.event_gate.shape != (tokens,) or not torch.is_floating_point(
            self.event_gate
        ):
            raise ValueError("event_gate must be floating point with shape [tokens]")
        if self.source_indices.shape != (tokens,) or self.source_indices.dtype != torch.int64:
            raise ValueError("source_indices must be an int64 token vector")
        if self.source_roles.shape != (tokens,) or self.source_roles.dtype != torch.int8:
            raise ValueError("source_roles must be an int8 token vector")
        if self.query_indices.dtype != torch.int64:
            raise ValueError("query_indices must be int64")
        if self.history_lengths.shape != (self.batch_size,):
            raise ValueError("history_lengths must have shape [B]")
        device = self.values.device
        fields = (
            self.offsets,
            self.event_gate,
            self.query_indices,
            self.source_indices,
            self.source_roles,
            self.history_lengths,
            self.group_sizes,
        )
        if any(field.device != device for field in fields):
            raise ValueError("all packed group tensors must share a device")
        # The upper bound depends on the original source tensor and is checked
        # by ``pack_group_streams``.  This immutable result deliberately does
        # not retain that usually much larger source allocation.
        if tokens and int(self.source_indices.min()) < 0:
            raise ValueError("packed source indices must be nonnegative")
        if not bool(torch.isfinite(self.event_gate).all()):
            raise ValueError("event gates must be finite")
        if bool((self.event_gate < 0).any()) or bool((self.event_gate > 1).any()):
            raise ValueError("event gates must lie in [0,1]")
        if bool((self.query_indices < 0).any()) or bool(
            (self.query_indices >= tokens).any()
        ):
            raise ValueError("query_indices must address packed candidate tokens")
        if not bool((self.event_gate.index_select(0, self.query_indices.reshape(-1)) == 0).all()):
            raise ValueError("every candidate query must be a read-only event")


@dataclass(frozen=True)
class PackedGlobalGroupQueries:
    """Candidate-only streams for occupied global groups across a ragged batch."""

    values: torch.Tensor
    offsets: torch.Tensor
    event_gate: torch.Tensor
    query_indices: torch.Tensor
    source_indices: torch.Tensor
    candidate_to_state_row: torch.Tensor
    occupied_user_rows: torch.Tensor
    occupied_group_ids: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.query_indices.shape[0])

    @property
    def candidate_count(self) -> int:
        return int(self.query_indices.shape[1])

    @property
    def sequence_count(self) -> int:
        return int(self.offsets.numel() - 1)

    @property
    def packed_tokens(self) -> int:
        return int(self.values.shape[0])

    def validate(self) -> None:
        tokens = self.batch_size * self.candidate_count
        if self.values.ndim != 2 or self.values.shape[0] != tokens:
            raise ValueError("global-group packed values must contain B*K tokens")
        if self.offsets.shape != (self.sequence_count + 1,) or self.offsets.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("global-group offsets must be an integer vector")
        if self.sequence_count < 1:
            raise ValueError("global-group packed queries require an occupied state")
        if int(self.offsets[0]) != 0 or int(self.offsets[-1]) != tokens:
            raise ValueError("global-group offsets must span every candidate")
        if bool((self.offsets[1:] <= self.offsets[:-1]).any()):
            raise ValueError("global-group packed rows must be nonempty")
        if self.event_gate.shape != (tokens,) or not bool((self.event_gate == 0).all()):
            raise ValueError("global-group candidates must be identity/read-only events")
        if self.query_indices.shape != (self.batch_size, self.candidate_count):
            raise ValueError("query_indices must restore [B,K]")
        if self.source_indices.shape != (tokens,) or self.source_indices.dtype != torch.int64:
            raise ValueError("source_indices must be an int64 permutation")
        if not torch.equal(
            torch.sort(self.source_indices).values,
            torch.arange(tokens, dtype=torch.int64, device=self.values.device),
        ):
            raise ValueError("source_indices must be a permutation of B*K")
        if self.candidate_to_state_row.shape != self.query_indices.shape:
            raise ValueError("candidate_to_state_row must have shape [B,K]")
        if bool((self.candidate_to_state_row < 0).any()) or bool(
            (self.candidate_to_state_row >= self.sequence_count).any()
        ):
            raise ValueError("candidate_to_state_row addresses an invalid state")
        if self.occupied_user_rows.shape != (self.sequence_count,) or (
            self.occupied_group_ids.shape != (self.sequence_count,)
        ):
            raise ValueError("occupied row metadata must have one entry per sequence")
        fields = (
            self.offsets,
            self.event_gate,
            self.query_indices,
            self.source_indices,
            self.candidate_to_state_row,
            self.occupied_user_rows,
            self.occupied_group_ids,
        )
        if any(field.device != self.values.device for field in fields):
            raise ValueError("global-group packed fields must share a device")


def pack_group_streams(
    *,
    x: torch.Tensor,
    x_lengths: torch.Tensor,
    x_offsets: torch.Tensor,
    num_targets: torch.Tensor,
    grouping: GroupingOutput,
    selection: GroupSelectionOutput,
    contextual_seq_len: int = 0,
    validate_runtime: bool = True,
) -> PackedGroupSequence:
    """Pack ``[context, selected history, candidate queries]`` per group.

    The source ``x`` uses the production layout
    ``[context, valid history, candidates]`` for every user.  Candidate rows
    inside a group follow the grouping module's canonical item-ID order, while
    ``query_indices`` restores the caller's original candidate axis.
    """

    if x.ndim != 2:
        raise ValueError("x must have shape [source_tokens,D]")
    if x_lengths.ndim != 1 or x_lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("x_lengths must be an integer vector")
    batch = len(x_lengths)
    if num_targets.shape != (batch,) or num_targets.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("num_targets must be an integer vector with shape [B]")
    if x_offsets.shape != (batch + 1,) or x_offsets.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("x_offsets must have shape [B+1]")
    if isinstance(contextual_seq_len, bool) or not isinstance(contextual_seq_len, int):
        raise TypeError("contextual_seq_len must be an integer")
    if contextual_seq_len < 0:
        raise ValueError("contextual_seq_len must be nonnegative")
    candidate_to_group = grouping.candidate_to_group
    if candidate_to_group.ndim != 2 or candidate_to_group.shape[0] != batch:
        raise ValueError("candidate_to_group must have shape [B,K]")
    candidates = int(candidate_to_group.shape[1])
    groups = int(grouping.group_sizes.shape[1])
    if selection.counts.shape != (batch, groups):
        raise ValueError("group selection must have shape [B,G]")
    if grouping.group_sizes.shape != (batch, groups):
        raise ValueError("group sizes must have shape [B,G]")
    if validate_runtime and bool((num_targets != candidates).any()):
        raise ValueError("packed group execution currently requires uniform K")
    device = x.device
    fields = (
        x_lengths,
        x_offsets,
        num_targets,
        candidate_to_group,
        grouping.group_sizes,
        grouping.packed_candidate_indices,
        grouping.group_offsets,
        selection.counts,
        selection.source_positions,
        selection.group_offsets,
    )
    if any(field.device != device for field in fields):
        raise ValueError("packing inputs must share the source x device")

    history_lengths = (
        x_lengths.to(torch.int64)
        - int(contextual_seq_len)
        - num_targets.to(torch.int64)
    )
    if validate_runtime:
        if int(x_offsets[0]) != 0 or int(x_offsets[-1]) != len(x):
            raise ValueError("x_offsets must span the source stream")
        if bool((x_offsets[1:] - x_offsets[:-1] != x_lengths).any()):
            raise ValueError("x offsets and lengths disagree")
        if bool((history_lengths < 1).any()):
            raise ValueError("group-shared execution requires nonempty histories")
        if not torch.equal(grouping.group_sizes.sum(dim=1), num_targets.to(torch.int64)):
            raise ValueError("group sizes must cover every user candidate")
        if bool((grouping.group_sizes < 0).any()):
            raise ValueError("semantic candidate group sizes must be nonnegative")

    selected_counts = selection.counts.reshape(-1).to(torch.int64)
    candidate_counts = grouping.group_sizes.reshape(-1).to(torch.int64)
    context_counts = selected_counts.new_full(
        selected_counts.shape, int(contextual_seq_len)
    )
    row_lengths = context_counts + selected_counts + candidate_counts
    packed_offsets = torch.cat(
        (row_lengths.new_zeros(1), row_lengths.cumsum(dim=0)), dim=0
    )
    packed_token_count = (
        batch * groups * int(contextual_seq_len)
        + selection.source_positions.numel()
        + grouping.packed_candidate_indices.numel()
    )
    row_count = batch * groups
    row_axis = torch.arange(row_count, device=device, dtype=torch.int64)
    source_offsets = x_offsets[:-1].to(torch.int64)
    source_indices = torch.empty(
        packed_token_count, device=device, dtype=torch.int64
    )
    source_roles = torch.empty(
        packed_token_count, device=device, dtype=torch.int8
    )

    # Context, selected history, and candidates occupy contiguous segments in
    # every logical row.  Their output sizes are known from immutable shapes,
    # so construct physical indices analytically rather than using CUDA
    # boolean indexing/nonzero, both of which require dynamic-size allocation.
    context_tokens = row_count * int(contextual_seq_len)
    if context_tokens:
        context_rows = torch.repeat_interleave(
            row_axis,
            int(contextual_seq_len),
            output_size=context_tokens,
        )
        context_within = torch.arange(
            context_tokens, device=device, dtype=torch.int64
        ) - context_rows * int(contextual_seq_len)
        context_physical = packed_offsets[:-1].index_select(0, context_rows) + (
            context_within
        )
        context_users = torch.div(context_rows, groups, rounding_mode="floor")
        context_source = source_offsets.index_select(0, context_users) + (
            context_within
        )
        source_indices.index_copy_(0, context_physical, context_source)
        source_roles.index_fill_(0, context_physical, 0)

    selected_tokens = selection.source_positions.numel()
    history_rows = torch.repeat_interleave(
        row_axis,
        selected_counts,
        output_size=selected_tokens,
    )
    history_within = torch.arange(
        selected_tokens, device=device, dtype=torch.int64
    ) - selection.group_offsets[:-1].to(torch.int64).index_select(
        0, history_rows
    )
    history_physical = (
        packed_offsets[:-1].index_select(0, history_rows)
        + int(contextual_seq_len)
        + history_within
    )
    history_users = torch.div(history_rows, groups, rounding_mode="floor")
    history_source = (
        source_offsets.index_select(0, history_users)
        + int(contextual_seq_len)
        + selection.source_positions
    )
    source_indices.index_copy_(0, history_physical, history_source)
    source_roles.index_fill_(0, history_physical, 1)

    candidate_tokens = grouping.packed_candidate_indices.numel()
    candidate_rows = torch.repeat_interleave(
        row_axis,
        candidate_counts,
        output_size=candidate_tokens,
    )
    candidate_within = torch.arange(
        candidate_tokens, device=device, dtype=torch.int64
    ) - grouping.group_offsets[:-1].to(torch.int64).index_select(
        0, candidate_rows
    )
    candidate_physical_indices = (
        packed_offsets[:-1].index_select(0, candidate_rows)
        + int(contextual_seq_len)
        + selected_counts.index_select(0, candidate_rows)
        + candidate_within
    )
    ordered_flat_candidates = grouping.packed_candidate_indices.to(torch.int64)
    candidate_users = torch.div(
        ordered_flat_candidates, candidates, rounding_mode="floor"
    )
    candidate_slots = ordered_flat_candidates.remainder(candidates)
    if validate_runtime and not torch.equal(
        candidate_users,
        torch.div(candidate_rows, groups, rounding_mode="floor"),
    ):
        raise ValueError("candidate packing crossed a user boundary")
    candidate_source = (
        source_offsets.index_select(0, candidate_users)
        + int(contextual_seq_len)
        + history_lengths.index_select(0, candidate_users)
        + candidate_slots
    )
    source_indices.index_copy_(
        0, candidate_physical_indices, candidate_source
    )
    source_roles.index_fill_(0, candidate_physical_indices, 2)
    query_indices_flat = torch.empty(
        batch * candidates, device=device, dtype=torch.int64
    )
    query_indices_flat.scatter_(
        0, ordered_flat_candidates, candidate_physical_indices
    )
    query_indices = query_indices_flat.reshape(batch, candidates)
    event_gate = torch.ones(packed_token_count, device=device, dtype=x.dtype)
    event_gate.index_fill_(0, candidate_physical_indices, 0)

    if validate_runtime and (
        bool((source_indices < 0).any()) or bool((source_indices >= len(x)).any())
    ):
        raise ValueError("packed group stream addresses an invalid source token")
    packed = PackedGroupSequence(
        values=x.index_select(0, source_indices),
        offsets=packed_offsets,
        event_gate=event_gate,
        query_indices=query_indices,
        source_indices=source_indices,
        source_roles=source_roles,
        history_lengths=history_lengths,
        group_sizes=grouping.group_sizes,
    )
    if validate_runtime:
        packed.validate()
    return packed


def pack_group_queries(
    candidate_x: torch.Tensor,
    grouping: GroupingOutput,
    *,
    validate_runtime: bool = True,
) -> PackedGroupSequence:
    """Pack candidate-only group rows for execution from cached GDR states.

    ``candidate_x`` retains the caller's ``[B,K]`` candidate order.  The
    physical rows follow ``grouping.packed_candidate_indices`` while
    ``query_indices`` restores that original order after execution.  Every
    token is a read-only GDR event because the corresponding group history has
    already been summarized into an initial recurrent state.
    """

    if candidate_x.ndim != 3:
        raise ValueError("candidate_x must have shape [B,K,D]")
    if not torch.is_floating_point(candidate_x):
        raise ValueError("candidate_x must be floating point")
    batch, candidates, embedding_dim = candidate_x.shape
    if batch < 1 or candidates < 1 or embedding_dim < 1:
        raise ValueError("candidate_x dimensions must be positive")
    if grouping.candidate_to_group.shape != (batch, candidates):
        raise ValueError("grouping and candidate_x must agree on B and K")
    groups = grouping.group_count
    device = candidate_x.device
    grouping_fields = (
        grouping.candidate_to_group,
        grouping.group_sizes,
        grouping.packed_candidate_indices,
        grouping.group_offsets,
    )
    if any(field.device != device for field in grouping_fields):
        raise ValueError("grouping and candidate_x must share a device")
    if validate_runtime:
        grouping.validate()

    ordered_flat_candidates = grouping.packed_candidate_indices.to(torch.int64)
    token_count = batch * candidates
    if ordered_flat_candidates.shape != (token_count,):
        raise ValueError("grouping must pack exactly B*K candidates")
    physical_indices = torch.arange(
        token_count,
        device=device,
        dtype=torch.int64,
    )
    query_indices_flat = torch.empty_like(physical_indices)
    query_indices_flat.scatter_(
        0,
        ordered_flat_candidates,
        physical_indices,
    )
    packed = PackedGroupSequence(
        values=candidate_x.reshape(token_count, embedding_dim).index_select(
            0,
            ordered_flat_candidates,
        ),
        offsets=grouping.group_offsets,
        event_gate=candidate_x.new_zeros(token_count),
        query_indices=query_indices_flat.reshape(batch, candidates),
        source_indices=ordered_flat_candidates,
        source_roles=torch.full(
            (token_count,),
            2,
            device=device,
            dtype=torch.int8,
        ),
        history_lengths=torch.zeros(batch, device=device, dtype=torch.int64),
        group_sizes=grouping.group_sizes,
    )
    if validate_runtime:
        packed.validate()
    return packed


def pack_global_group_queries(
    candidate_x: torch.Tensor,
    lookup: GlobalGroupStateLookup,
    *,
    validate_runtime: bool = True,
) -> PackedGlobalGroupQueries:
    """Pack one strictly nonempty row per occupied ``(user, global_group)``."""

    if candidate_x.ndim != 3 or not torch.is_floating_point(candidate_x):
        raise ValueError("candidate_x must be floating point [B,K,D]")
    batch, candidates, dimension = candidate_x.shape
    if lookup.candidate_to_state_row.shape != (batch, candidates):
        raise ValueError("global-group lookup and candidate_x disagree on B or K")
    if lookup.packed_candidate_indices.shape != (batch * candidates,):
        raise ValueError("global-group lookup must pack every candidate")
    if lookup.states.device != candidate_x.device:
        raise ValueError("global-group lookup states and candidates must share a device")
    source_indices = lookup.packed_candidate_indices.to(torch.int64)
    physical = torch.arange(
        batch * candidates, dtype=torch.int64, device=candidate_x.device
    )
    query_indices_flat = torch.empty_like(physical)
    query_indices_flat.scatter_(0, source_indices, physical)
    packed = PackedGlobalGroupQueries(
        values=candidate_x.reshape(batch * candidates, dimension).index_select(
            0, source_indices
        ),
        offsets=lookup.offsets,
        event_gate=candidate_x.new_zeros(batch * candidates),
        query_indices=query_indices_flat.reshape(batch, candidates),
        source_indices=source_indices,
        candidate_to_state_row=lookup.candidate_to_state_row,
        occupied_user_rows=lookup.occupied_user_rows,
        occupied_group_ids=lookup.occupied_group_ids,
    )
    if validate_runtime:
        packed.validate()
        packed_rows = torch.repeat_interleave(
            torch.arange(
                packed.sequence_count,
                dtype=torch.int64,
                device=candidate_x.device,
            ),
            packed.offsets[1:] - packed.offsets[:-1],
        )
        source_rows = torch.div(source_indices, candidates, rounding_mode="floor")
        source_slots = source_indices.remainder(candidates)
        if not torch.equal(
            packed.occupied_user_rows.index_select(0, packed_rows), source_rows
        ):
            raise ValueError("global-group packing crossed a user boundary")
        if not torch.equal(
            lookup.candidate_group_ids[source_rows, source_slots],
            packed.occupied_group_ids.index_select(0, packed_rows),
        ):
            raise ValueError("global-group packing crossed a group boundary")
    return packed


def build_group_fixed_width_layout(
    packed: PackedGroupSequence,
) -> HSTUFixedWidthLayout:
    """Build optional projection-only padding for shape-invariant GEMMs."""

    lengths = packed.offsets[1:] - packed.offsets[:-1]
    if lengths.numel() < 1 or bool((lengths < 1).any()):
        raise ValueError("packed group streams must be nonempty")
    sequence_count = len(lengths)
    token_width = int(lengths.max())
    sequence_ids = torch.repeat_interleave(
        torch.arange(
            sequence_count,
            device=packed.offsets.device,
            dtype=torch.int64,
        ),
        lengths.to(torch.int64),
    )
    positions = (
        torch.arange(
            packed.packed_tokens,
            device=packed.offsets.device,
            dtype=torch.int64,
        )
        - packed.offsets[:-1]
        .to(torch.int64)
        .index_select(0, sequence_ids)
    )
    return HSTUFixedWidthLayout(
        offsets=packed.offsets,
        flat_padded_indices=sequence_ids * token_width + positions,
        sequence_count=sequence_count,
        token_width=token_width,
        packed_tokens=packed.packed_tokens,
    )


__all__ = [
    "PackedGroupSequence",
    "PackedGlobalGroupQueries",
    "build_group_fixed_width_layout",
    "pack_global_group_queries",
    "pack_group_queries",
    "pack_group_streams",
]

