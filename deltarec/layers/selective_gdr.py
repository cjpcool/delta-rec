# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""Interfaces for selecting and packing events before a GDR attention kernel."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Mapping, Optional

import torch


@dataclass(frozen=True)
class PackedSequence:
    """A jagged sequence whose payloads are aligned with ``values``."""

    values: torch.Tensor
    offsets: torch.Tensor
    payloads: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def validate(self) -> None:
        if self.values.ndim < 1:
            raise ValueError("values must have a token dimension")
        if self.offsets.ndim != 1 or self.offsets.numel() < 1:
            raise ValueError("offsets must have shape [batch + 1]")
        if self.offsets.dtype not in (torch.int32, torch.int64):
            raise ValueError("offsets must be an integer tensor")
        if int(self.offsets[0]) != 0 or int(self.offsets[-1]) != len(self.values):
            raise ValueError("offsets must span all values")
        if bool((self.offsets[1:] < self.offsets[:-1]).any()):
            raise ValueError("offsets must be nondecreasing")
        for name, payload in self.payloads.items():
            if payload.ndim < 1 or len(payload) != len(self.values):
                raise ValueError(f"payload {name!r} must be token-aligned")


@dataclass(frozen=True)
class SelectionRequest:
    """Low-dimensional inputs and hard constraints for pre-GDR selection.

    ``budgets`` counts every selected event, including mandatory and read-only
    events. ``read_only_mask`` must be a subset of ``mandatory_mask``.
    """

    token_features: torch.Tensor
    offsets: torch.Tensor
    budgets: torch.Tensor
    mandatory_mask: torch.Tensor
    read_only_mask: torch.Tensor
    chunk_size: int = 64
    max_per_chunk: int = 16

    def validate(self, source: PackedSequence) -> None:
        source.validate()
        tokens = len(source.values)
        batch = len(source.offsets) - 1
        if self.token_features.ndim != 2 or len(self.token_features) != tokens:
            raise ValueError("token_features must have shape [tokens, selector_dim]")
        if not torch.equal(self.offsets, source.offsets):
            raise ValueError("selector and source offsets must match")
        if self.budgets.shape != (batch,) or self.budgets.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("budgets must be an integer tensor with shape [batch]")
        for name, mask in (
            ("mandatory_mask", self.mandatory_mask),
            ("read_only_mask", self.read_only_mask),
        ):
            if mask.shape != (tokens,) or mask.dtype != torch.bool:
                raise ValueError(f"{name} must be boolean with shape [tokens]")
        if bool((self.read_only_mask & ~self.mandatory_mask).any()):
            raise ValueError("read_only_mask must be a subset of mandatory_mask")
        lengths = self.offsets[1:] - self.offsets[:-1]
        if bool((self.budgets < 0).any()) or bool((self.budgets > lengths).any()):
            raise ValueError("budgets must lie between zero and the request length")
        mandatory_counts = torch.stack(
            [
                self.mandatory_mask[int(self.offsets[i]) : int(self.offsets[i + 1])].sum()
                for i in range(batch)
            ]
        )
        if bool((self.budgets < mandatory_counts).any()):
            raise ValueError("budgets must include every mandatory event")
        if self.chunk_size < 1 or self.max_per_chunk < 1:
            raise ValueError("chunk_size and max_per_chunk must be positive")


@dataclass(frozen=True)
class SelectionPlan:
    """Chronological packed layout produced by a selector.

    ``indices`` addresses the flattened input sequence. ``offsets`` describes
    the selected output sequence. ``write_mask`` is false for packed query or
    candidate events that may read state but must not change it.
    """

    indices: torch.Tensor
    offsets: torch.Tensor
    source_positions: torch.Tensor
    write_mask: torch.Tensor
    scores: Optional[torch.Tensor] = None
    offsets_cpu: Optional[torch.Tensor] = None

    def validate(
        self,
        source: PackedSequence,
        request: Optional[SelectionRequest] = None,
    ) -> None:
        source.validate()
        selected = len(self.indices)
        if self.indices.ndim != 1 or self.indices.dtype != torch.int64:
            raise ValueError("indices must be an int64 tensor with shape [selected]")
        if self.offsets.shape != source.offsets.shape:
            raise ValueError("selected offsets must match the source batch shape")
        if self.offsets.dtype not in (torch.int32, torch.int64):
            raise ValueError("selected offsets must be an integer tensor")
        if int(self.offsets[0]) != 0 or int(self.offsets[-1]) != selected:
            raise ValueError("selected offsets must span all selected events")
        if bool((self.offsets[1:] < self.offsets[:-1]).any()):
            raise ValueError("selected offsets must be nondecreasing")
        if self.source_positions.shape != (selected,):
            raise ValueError("source_positions must have shape [selected]")
        if self.write_mask.shape != (selected,) or self.write_mask.dtype != torch.bool:
            raise ValueError("write_mask must be boolean with shape [selected]")
        if self.scores is not None and self.scores.shape != (selected,):
            raise ValueError("scores must have shape [selected]")
        if self.offsets_cpu is not None:
            if (
                self.offsets_cpu.device.type != "cpu"
                or self.offsets_cpu.dtype != torch.int64
                or self.offsets_cpu.shape != self.offsets.shape
            ):
                raise ValueError("offsets_cpu must be an int64 CPU copy of offsets")
            if not torch.equal(self.offsets_cpu, self.offsets.detach().cpu().long()):
                raise ValueError("offsets_cpu does not match offsets")
        if selected and (
            int(self.indices.min()) < 0 or int(self.indices.max()) >= len(source.values)
        ):
            raise ValueError("selected indices are outside the source sequence")

        for batch_index in range(len(source.offsets) - 1):
            source_start = int(source.offsets[batch_index])
            source_end = int(source.offsets[batch_index + 1])
            selected_start = int(self.offsets[batch_index])
            selected_end = int(self.offsets[batch_index + 1])
            indices = self.indices[selected_start:selected_end]
            positions = self.source_positions[selected_start:selected_end]
            if len(indices) and (
                int(indices[0]) < source_start or int(indices[-1]) >= source_end
            ):
                raise ValueError("a selected event belongs to the wrong request")
            if len(indices) > 1 and bool((indices[1:] <= indices[:-1]).any()):
                raise ValueError("selected events must be unique and chronological")
            if not torch.equal(positions, indices - source_start):
                raise ValueError("source_positions do not match selected indices")
        if request is not None:
            request.validate(source)
            selected_lengths = self.offsets[1:] - self.offsets[:-1]
            if not torch.equal(selected_lengths.to(request.budgets), request.budgets):
                raise ValueError("selected lengths must equal the requested budgets")
            selected_indicator = torch.zeros(
                len(source.values), dtype=torch.bool, device=self.indices.device
            )
            selected_indicator[self.indices] = True
            if bool((request.mandatory_mask & ~selected_indicator).any()):
                raise ValueError("the selection plan omitted a mandatory event")
            expected_write_mask = ~request.read_only_mask.index_select(0, self.indices)
            if not torch.equal(self.write_mask, expected_write_mask):
                raise ValueError("write_mask does not match the requested event roles")

    def apply(self, source: PackedSequence) -> SelectedSequence:
        return SelectedSequence(
            values=source.values.index_select(0, self.indices),
            offsets=self.offsets,
            payloads={
                name: value.index_select(0, self.indices)
                for name, value in source.payloads.items()
            },
            source_indices=self.indices,
            source_positions=self.source_positions,
            write_mask=self.write_mask,
            offsets_cpu=self.offsets_cpu,
        )


@dataclass(frozen=True)
class SelectedSequence(PackedSequence):
    """The physical event stream materialized for projection and GDR."""

    source_indices: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    source_positions: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    write_mask: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    offsets_cpu: Optional[torch.Tensor] = None


class PreGDRSelector(torch.nn.Module):
    """Produces a fixed-budget, chronological plan from cheap token features."""

    @abc.abstractmethod
    def forward(self, request: SelectionRequest) -> SelectionPlan:
        pass


@dataclass(frozen=True)
class GDRKernelInput:
    """Projected packed events consumed by a GDR kernel."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    decay_logits: torch.Tensor
    beta_logits: torch.Tensor
    log_decay_scale: torch.Tensor
    decay_bias: torch.Tensor
    offsets: torch.Tensor
    event_gate: Optional[torch.Tensor] = None
    offsets_cpu: Optional[torch.Tensor] = None


def apply_gdr_event_gate(
    decay: torch.Tensor,
    beta: torch.Tensor,
    event_gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the canonical GDR event-gate identity interpolation.

    A no-write event has ``decay=1`` and ``beta=0``; a full-write event keeps
    the original transition. Fractional gates linearly interpolate those two
    gate parameters as ``1 - z * (1 - decay)`` and ``z * beta``. Values are
    not transformed by the event gate.
    """

    decay_gate = event_gate.to(decay.dtype)
    while decay_gate.ndim < decay.ndim:
        decay_gate = decay_gate.unsqueeze(-1)
    beta_gate = event_gate.to(beta.dtype)
    while beta_gate.ndim < beta.ndim:
        beta_gate = beta_gate.unsqueeze(-1)
    effective_decay = (1.0 - decay_gate) + decay_gate * decay
    effective_beta = beta_gate * beta
    return effective_decay, effective_beta


@dataclass(frozen=True)
class GDRProjection:
    """Kernel inputs plus architecture-owned tensors used after attention."""

    kernel_input: GDRKernelInput
    auxiliary: Mapping[str, torch.Tensor] = field(default_factory=dict)


class GDRProjector(torch.nn.Module):
    """Runs the main embedding/projection path only on selected events."""

    @abc.abstractmethod
    def forward(self, selected: SelectedSequence) -> GDRProjection:
        pass


@dataclass(frozen=True)
class GDRKernelOutput:
    context: torch.Tensor
    final_state: Optional[torch.Tensor] = None


class GDRKernel(torch.nn.Module):
    """Selector-agnostic full-write GDR kernel interface."""

    @abc.abstractmethod
    def forward(
        self,
        projected: GDRKernelInput,
        initial_state: Optional[torch.Tensor] = None,
        return_final_state: bool = False,
    ) -> GDRKernelOutput:
        pass


@dataclass(frozen=True)
class SelectiveAttentionOutput:
    kernel_output: GDRKernelOutput
    selected: SelectedSequence
    auxiliary: Mapping[str, torch.Tensor]
    plan: SelectionPlan


class SelectiveGDRAttention(torch.nn.Module):
    """Composes selection, physical packing, projection, and GDR execution."""

    def __init__(
        self,
        selector: PreGDRSelector,
        projector: GDRProjector,
        kernel: GDRKernel,
        validate_plan: bool = False,
    ) -> None:
        super().__init__()
        self.selector = selector
        self.projector = projector
        self.kernel = kernel
        self.validate_plan = validate_plan

    def forward(
        self,
        source: PackedSequence,
        request: SelectionRequest,
        initial_state: Optional[torch.Tensor] = None,
        return_final_state: bool = False,
    ) -> SelectiveAttentionOutput:
        if source.offsets.shape != request.offsets.shape:
            raise ValueError("source and selector offsets must match")
        if self.validate_plan and not torch.equal(source.offsets, request.offsets):
            raise ValueError("source and selector offsets must match")
        plan = self.selector(request)
        if self.validate_plan:
            plan.validate(source, request)
        selected = plan.apply(source)
        projection = self.projector(selected)
        if projection.kernel_input.offsets.shape != selected.offsets.shape:
            raise ValueError("projected and selected offsets must match")
        if self.validate_plan and not torch.equal(
            projection.kernel_input.offsets, selected.offsets
        ):
            raise ValueError("projected and selected offsets must match")
        kernel_output = self.kernel(
            projection.kernel_input,
            initial_state,
            return_final_state=return_final_state,
        )
        return SelectiveAttentionOutput(
            kernel_output=kernel_output,
            selected=selected,
            auxiliary=projection.auxiliary,
            plan=plan,
        )
