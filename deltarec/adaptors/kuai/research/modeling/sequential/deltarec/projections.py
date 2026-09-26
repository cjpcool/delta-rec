# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""Safe candidate-independent projection reuse for HSTU DeltaRec.

Only input preprocessing and the first HSTU layer's UVQK/gate projection are
candidate independent.  This module materializes those tensors once for each
event in a user's valid history, then gathers them into the row-major
``(user, candidate)`` layout selected by DeltaRec.  Recurrent execution and
all later-layer projections deliberately remain outside this module because
their inputs are candidate specific.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from deltarec.adaptors.kuai.research.modeling.sequential.hstu_delta_rec import (
    HSTUDeltaRecAdapter,
)
from deltarec.adaptors.kuai.research.modeling.sequential.selective_gdr import (
    GDRKernelInput,
    SelectedSequence,
)

from .types import DeltaRecRequest, SelectionOutput


@dataclass(frozen=True)
class HSTULayerZeroProjection:
    """Selected layer-0 tensors after one projection per valid source event.

    ``x`` and ``u`` are token aligned with ``selected``. ``kernel_input`` has
    the same ``[B*K+1]`` offsets and owns no state shared across those rows.
    ``valid_source_tokens`` records the amount of candidate-independent work
    and excludes every padded suffix token.
    """

    selected: SelectedSequence
    x: torch.Tensor
    u: torch.Tensor
    kernel_input: GDRKernelInput
    valid_source_tokens: int


@dataclass(frozen=True)
class HSTULayerZeroSourceProjection:
    """Candidate-independent layer-0 tensors computed once per request.

    The identity fields bind a prepared projection to the adapter and exact
    history tensors that produced it. Candidate slices preserve those tensor
    objects, so request-level reuse is validated without reading a CUDA scalar.
    ``kernel_input`` contains projected source events, but its ``[B+1]``
    offsets must never be used for candidate execution; gathering replaces
    them with the independent ``[B*K_chunk+1]`` selection offsets.
    """

    x: torch.Tensor
    u: torch.Tensor
    kernel_input: GDRKernelInput
    valid_offsets: torch.Tensor
    batch_size: int
    history_width: int
    valid_source_tokens: int
    adapter_identity: int
    history_ids_identity: int
    history_lengths_identity: int
    history_payload_identities: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class HSTUFixedWidthLayout:
    """Map a physically packed stream to fixed-width logical sequences.

    Candidate chunking changes the leading dimension of ordinary packed
    ``torch.mm`` calls.  CUDA GEMM implementations may consequently choose a
    different reduction schedule and perturb later recurrent states.  This
    layout gives every logical ``(user, candidate)`` row the same token width
    ``S`` (the request's maximum exact selection budget), while keeping only
    real events in the GDR input itself.

    ``flat_padded_indices`` maps each real packed token into a conceptual
    ``[B*K, S, ...]`` tensor.  Padding is used only by projection GEMMs and is
    never passed to the recurrence.
    """

    offsets: torch.Tensor
    flat_padded_indices: torch.Tensor
    sequence_count: int
    token_width: int
    packed_tokens: int

    @property
    def padded_tokens(self) -> int:
        return self.sequence_count * self.token_width

    @property
    def padding_tokens(self) -> int:
        return self.padded_tokens - self.packed_tokens


def build_hstu_fixed_width_layout(
    selection: SelectionOutput,
    selected: SelectedSequence,
    *,
    validate_runtime: bool = True,
) -> HSTUFixedWidthLayout:
    """Build a label-blind fixed-width projection layout for one chunk."""

    if selection.indices.ndim != 3:
        raise ValueError("selection indices must have shape [B,K,S]")
    sequence_count = selection.batch_size * selection.candidate_count
    token_width = selection.indices.shape[-1]
    if token_width < 1:
        raise ValueError("fixed-width projection requires a positive token width")
    if selected.offsets.shape != (sequence_count + 1,):
        raise ValueError("selected offsets must describe B*K logical sequences")
    if selected.offsets.device != selection.indices.device:
        raise ValueError("selection and selected offsets must share a device")

    lengths = selected.offsets[1:] - selected.offsets[:-1]
    sequence_ids = torch.repeat_interleave(
        torch.arange(
            sequence_count,
            dtype=torch.int64,
            device=selected.offsets.device,
        ),
        lengths.to(torch.int64),
    )
    packed_tokens = len(selected.values)
    positions = torch.arange(
        packed_tokens,
        dtype=torch.int64,
        device=selected.offsets.device,
    ) - selected.offsets[:-1].to(torch.int64).index_select(0, sequence_ids)
    flat_padded_indices = sequence_ids * token_width + positions

    if validate_runtime:
        if selected.offsets.shape != selection.offsets.shape or not torch.equal(
            selected.offsets, selection.offsets
        ):
            raise ValueError("selected and selection offsets disagree")
        expected_lengths = selection.counts.reshape(-1).to(lengths)
        if not torch.equal(lengths, expected_lengths):
            raise ValueError("selection counts and selected offsets disagree")
        if bool((lengths < 1).any()) or bool((lengths > token_width).any()):
            raise ValueError("selected lengths must lie in [1, S]")
        if len(flat_padded_indices) != packed_tokens:
            raise ValueError("fixed-width indices must align with packed tokens")

    return HSTUFixedWidthLayout(
        offsets=selected.offsets,
        flat_padded_indices=flat_padded_indices,
        sequence_count=sequence_count,
        token_width=token_width,
        packed_tokens=packed_tokens,
    )


def fixed_width_sequence_linear(
    x: torch.Tensor,
    right_weight: torch.Tensor,
    bias: torch.Tensor | None,
    layout: HSTUFixedWidthLayout,
) -> torch.Tensor:
    """Apply ``x @ right_weight + bias`` with one fixed GEMM per sequence.

    The batched matrices all have shape ``[S, Din] @ [Din, Dout]``.  Changing
    candidate chunk width changes only the number of independent matrices,
    never the reduction shape seen by a logical sequence.  Returned values are
    gathered back into the original physically packed chronological order.
    """

    if x.ndim != 2 or len(x) != layout.packed_tokens:
        raise ValueError("fixed-width linear input must align with packed tokens")
    if right_weight.ndim != 2 or right_weight.shape[0] != x.shape[1]:
        raise ValueError("right_weight must have shape [input_dim, output_dim]")
    if right_weight.device != x.device or right_weight.dtype != x.dtype:
        raise ValueError("fixed-width linear tensors must share device and dtype")
    output_dim = right_weight.shape[1]
    if bias is not None:
        if bias.shape != (output_dim,):
            raise ValueError("fixed-width linear bias has the wrong shape")
        if bias.device != x.device or bias.dtype != x.dtype:
            raise ValueError("fixed-width linear bias must match input device and dtype")

    padded = x.new_zeros((layout.padded_tokens, x.shape[1]))
    padded = padded.index_copy(0, layout.flat_padded_indices, x).view(
        layout.sequence_count,
        layout.token_width,
        x.shape[1],
    )
    projected = torch.bmm(
        padded,
        right_weight.unsqueeze(0).expand(layout.sequence_count, -1, -1),
    )
    if bias is not None:
        projected = projected + bias
    return projected.reshape(layout.padded_tokens, output_dim).index_select(
        0,
        layout.flat_padded_indices,
    )


def _valid_history_fields(
    request: DeltaRecRequest,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    torch.Tensor,
]:
    """Pack valid source prefixes once, preserving original source positions."""

    batch, width = request.history_ids.shape
    positions = torch.arange(width, device=request.history_ids.device).expand(
        batch, width
    )
    valid = positions < request.history_lengths[:, None]
    flat_valid_indices = torch.nonzero(
        valid.reshape(-1), as_tuple=False
    ).squeeze(-1)

    item_ids = request.history_ids.reshape(-1).index_select(0, flat_valid_indices)
    source_positions = positions.reshape(-1).index_select(0, flat_valid_indices)
    payloads = {
        name: payload.reshape(batch * width, *payload.shape[2:]).index_select(
            0, flat_valid_indices
        )
        for name, payload in request.history_payloads.items()
    }
    offsets = torch.cat(
        [
            request.history_lengths.new_zeros(1),
            torch.cumsum(request.history_lengths, dim=0),
        ]
    )
    return item_ids, source_positions, payloads, offsets


def _history_source_identity(
    request: DeltaRecRequest,
) -> tuple[int, int, tuple[tuple[str, int], ...]]:
    return (
        id(request.history_ids),
        id(request.history_lengths),
        tuple(
            sorted(
                (name, id(payload))
                for name, payload in request.history_payloads.items()
            )
        ),
    )


def _validate_hstu_layer_zero_source(
    adapter: HSTUDeltaRecAdapter,
    request: DeltaRecRequest,
    source: HSTULayerZeroSourceProjection,
    *,
    validate_values: bool,
) -> None:
    if not isinstance(source, HSTULayerZeroSourceProjection):
        raise TypeError(
            "prepared_layer_zero must be an HSTULayerZeroSourceProjection"
        )
    if source.adapter_identity != id(adapter):
        raise ValueError("prepared layer-0 source belongs to a different adapter")
    history_ids_identity, history_lengths_identity, payload_identities = (
        _history_source_identity(request)
    )
    if (
        source.history_ids_identity != history_ids_identity
        or source.history_lengths_identity != history_lengths_identity
        or source.history_payload_identities != payload_identities
    ):
        raise ValueError("prepared layer-0 source belongs to a different history")
    if source.batch_size != request.batch_size or source.history_width != (
        request.history_width
    ):
        raise ValueError("prepared layer-0 source has incompatible request dimensions")
    if source.valid_source_tokens < 1:
        raise ValueError("prepared layer-0 source must contain valid events")
    if source.valid_offsets.shape != (request.batch_size + 1,):
        raise ValueError("prepared valid offsets must have shape [B+1]")
    if source.x.ndim != 2 or len(source.x) != source.valid_source_tokens:
        raise ValueError("prepared layer-0 inputs do not align with valid events")
    if source.u.ndim != 2 or len(source.u) != source.valid_source_tokens:
        raise ValueError("prepared layer-0 U does not align with valid events")
    projected = source.kernel_input
    for name, value in (
        ("q", projected.q),
        ("k", projected.k),
        ("v", projected.v),
        ("decay_logits", projected.decay_logits),
        ("beta_logits", projected.beta_logits),
    ):
        if len(value) != source.valid_source_tokens:
            raise ValueError(
                f"prepared layer-0 {name} does not align with valid events"
            )
    if projected.event_gate is not None and len(projected.event_gate) != (
        source.valid_source_tokens
    ):
        raise ValueError("prepared layer-0 event gate does not align with valid events")
    if projected.offsets.shape != source.valid_offsets.shape:
        raise ValueError("prepared projection and valid offsets have different shapes")
    if validate_values:
        expected_offsets = torch.cat(
            [
                request.history_lengths.new_zeros(1),
                torch.cumsum(request.history_lengths, dim=0),
            ]
        )
        if not torch.equal(source.valid_offsets, expected_offsets):
            raise ValueError("prepared valid offsets do not match history lengths")
        if not torch.equal(projected.offsets, source.valid_offsets):
            raise ValueError("prepared projection offsets do not match valid offsets")


def project_hstu_layer_zero_source(
    adapter: HSTUDeltaRecAdapter,
    request: DeltaRecRequest,
    *,
    validate_runtime: bool = True,
) -> HSTULayerZeroSourceProjection:
    """Preprocess and layer-0-project every valid history event exactly once."""

    if validate_runtime:
        request.validate()
    item_ids, source_positions, payloads, valid_offsets = _valid_history_fields(
        request
    )
    x_source = adapter._preprocess(item_ids, source_positions, payloads)
    layer = adapter.layers[0]
    source_write_mask = torch.ones(
        len(item_ids), dtype=torch.bool, device=item_ids.device
    )
    u_source, source_projection = adapter._project_mixed(
        layer,
        x_source,
        source_write_mask,
        valid_offsets,
    )
    history_ids_identity, history_lengths_identity, payload_identities = (
        _history_source_identity(request)
    )
    source = HSTULayerZeroSourceProjection(
        x=x_source,
        u=u_source,
        kernel_input=source_projection,
        valid_offsets=valid_offsets,
        batch_size=request.batch_size,
        history_width=request.history_width,
        valid_source_tokens=len(item_ids),
        adapter_identity=id(adapter),
        history_ids_identity=history_ids_identity,
        history_lengths_identity=history_lengths_identity,
        history_payload_identities=payload_identities,
    )
    if validate_runtime:
        _validate_hstu_layer_zero_source(
            adapter,
            request,
            source,
            validate_values=True,
        )
    return source


def gather_hstu_layer_zero_projection(
    adapter: HSTUDeltaRecAdapter,
    request: DeltaRecRequest,
    selection: SelectionOutput,
    selected: SelectedSequence,
    source: HSTULayerZeroSourceProjection,
    *,
    validate_runtime: bool = True,
) -> HSTULayerZeroProjection:
    """Gather one request-level source projection into candidate chunk rows."""

    if validate_runtime:
        request.validate()
        selection.validate(request.history_width)
        selected.validate()
    _validate_hstu_layer_zero_source(
        adapter,
        request,
        source,
        validate_values=validate_runtime,
    )
    if (
        selection.batch_size != request.batch_size
        or selection.candidate_count != request.candidate_count
    ):
        raise ValueError("selection and request batch/candidate shapes disagree")
    if selected.offsets.shape != selection.offsets.shape:
        raise ValueError("selected and selection offsets disagree")
    if len(selected.values) != len(selection.packed_source_indices):
        raise ValueError("selected values and packed source indices disagree")
    if (
        selected.write_mask.shape != (len(selected.values),)
        or selected.write_mask.dtype != torch.bool
    ):
        raise ValueError("selected write_mask must be a token-aligned boolean vector")
    if validate_runtime and not torch.equal(selected.offsets, selection.offsets):
        raise ValueError("selected and selection offsets disagree")
    if validate_runtime and not bool(selected.write_mask.all()):
        raise ValueError("candidate-selected history events must all be full writes")

    selected_source_indices = selection.packed_source_indices
    selected_users = torch.div(
        selected_source_indices,
        source.history_width,
        rounding_mode="floor",
    )
    gather_indices = source.valid_offsets.index_select(
        0, selected_users.long()
    ).long()
    gather_indices = gather_indices + selection.source_positions.long()

    if validate_runtime and gather_indices.numel():
        selected_lengths = request.history_lengths.index_select(
            0, selected_users.long()
        )
        if bool((selection.source_positions >= selected_lengths).any()):
            raise ValueError("selection addresses a padded history suffix")
        if (
            int(gather_indices.min()) < 0
            or int(gather_indices.max()) >= source.valid_source_tokens
        ):
            raise ValueError("selection cannot be mapped into valid-prefix storage")

    source_projection = source.kernel_input
    event_gate = source_projection.event_gate
    gathered_projection = GDRKernelInput(
        q=source_projection.q.index_select(0, gather_indices),
        k=source_projection.k.index_select(0, gather_indices),
        v=source_projection.v.index_select(0, gather_indices),
        decay_logits=source_projection.decay_logits.index_select(0, gather_indices),
        beta_logits=source_projection.beta_logits.index_select(0, gather_indices),
        log_decay_scale=source_projection.log_decay_scale,
        decay_bias=source_projection.decay_bias,
        offsets=selected.offsets,
        event_gate=(
            None if event_gate is None else event_gate.index_select(0, gather_indices)
        ),
        offsets_cpu=selected.offsets_cpu,
    )
    return HSTULayerZeroProjection(
        selected=selected,
        x=source.x.index_select(0, gather_indices),
        u=source.u.index_select(0, gather_indices),
        kernel_input=gathered_projection,
        valid_source_tokens=source.valid_source_tokens,
    )


def project_hstu_layer_zero_once(
    adapter: HSTUDeltaRecAdapter,
    request: DeltaRecRequest,
    selection: SelectionOutput,
    selected: SelectedSequence,
    *,
    validate_runtime: bool = True,
) -> HSTULayerZeroProjection:
    """Compose request-level projection and candidate-chunk gathering.

    The gather index converts a selected source location ``(b, t)`` into the
    valid-prefix layout ``valid_offsets[b] + t``.  Thus padded suffixes are
    neither embedded nor projected, while positional features and payloads
    continue to use their original history positions.
    """

    source = project_hstu_layer_zero_source(
        adapter,
        request,
        validate_runtime=validate_runtime,
    )
    return gather_hstu_layer_zero_projection(
        adapter,
        request,
        selection,
        selected,
        source,
        validate_runtime=validate_runtime,
    )


__all__ = [
    "HSTUFixedWidthLayout",
    "HSTULayerZeroProjection",
    "HSTULayerZeroSourceProjection",
    "build_hstu_fixed_width_layout",
    "fixed_width_sequence_linear",
    "gather_hstu_layer_zero_projection",
    "project_hstu_layer_zero_once",
    "project_hstu_layer_zero_source",
]

