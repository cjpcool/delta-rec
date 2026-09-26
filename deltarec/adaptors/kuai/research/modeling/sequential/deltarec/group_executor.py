# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""Packed group-shared GDR execution behind the Subplan 5C contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional

import torch
from torch import nn

from deltarec.adaptors.kuai.research.modeling.sequential.selective_gdr import (
    GDRKernelInput,
)

from .group_packing import (
    PackedGlobalGroupQueries,
    PackedGroupSequence,
    build_group_fixed_width_layout,
)
from .selection import validate_retention_ratio


GroupingPolicy = Literal[
    "frozen_embedding_cluster_v1",
    "canonical_id_balanced",
    "fixed_category_prototype_v1",
]
GroupPool = Literal["logmeanexp", "mean", "max", "mask_vote"]
ProjectionMode = Literal["packed_mm", "fixed_width_batched"]


@dataclass(frozen=True)
class GroupSharedDeltaRecConfig:
    """Frozen semantic knobs plus explicitly separate scheduling knobs."""

    candidate_group_count: int = 4
    grouping_policy: GroupingPolicy = "frozen_embedding_cluster_v1"
    group_pool: GroupPool = "logmeanexp"
    retention_ratio: float = 0.50
    recent_floor: int = 32
    execution_chunk_size: int = 32
    projection_dtype: str = "bfloat16"
    state_dtype: str = "float32"
    projection_mode: ProjectionMode = "packed_mm"
    validate_runtime: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.candidate_group_count, bool)
            or not isinstance(self.candidate_group_count, int)
            or self.candidate_group_count < 1
        ):
            raise ValueError("candidate_group_count must be a positive integer")
        if self.grouping_policy not in (
            "frozen_embedding_cluster_v1",
            "canonical_id_balanced",
            "fixed_category_prototype_v1",
        ):
            raise ValueError(f"unsupported grouping policy {self.grouping_policy!r}")
        if self.group_pool not in ("logmeanexp", "mean", "max", "mask_vote"):
            raise ValueError(f"unsupported group pool {self.group_pool!r}")
        if (
            self.grouping_policy == "fixed_category_prototype_v1"
            and self.group_pool != "logmeanexp"
        ):
            raise ValueError(
                "fixed_category_prototype_v1 uses frozen prototype scores and "
                "requires the canonical logmeanexp configuration label"
            )
        object.__setattr__(
            self,
            "retention_ratio",
            validate_retention_ratio(self.retention_ratio),
        )
        if (
            isinstance(self.recent_floor, bool)
            or not isinstance(self.recent_floor, int)
            or self.recent_floor < 0
        ):
            raise ValueError("recent_floor must be a nonnegative integer")
        if (
            isinstance(self.execution_chunk_size, bool)
            or not isinstance(self.execution_chunk_size, int)
            or self.execution_chunk_size < 1
        ):
            raise ValueError("execution_chunk_size must be a positive integer")
        if self.projection_dtype != "bfloat16":
            raise ValueError("the registered optimized row requires BF16 projections")
        if self.state_dtype != "float32":
            raise ValueError("GDR recurrent state storage must be FP32")
        if self.projection_mode not in ("packed_mm", "fixed_width_batched"):
            raise ValueError(f"unsupported projection mode {self.projection_mode!r}")
        if not isinstance(self.validate_runtime, bool):
            raise TypeError("validate_runtime must be boolean")


@dataclass(frozen=True)
class GroupExecutorOutput:
    """Candidate-aligned reads and optional group-aligned recurrent states."""

    queries: torch.Tensor
    final_states: Optional[torch.Tensor]
    packed_embeddings: torch.Tensor
    physical_gdr_calls: int
    auxiliary: Mapping[str, Any] = field(default_factory=dict)

    def validate(
        self,
        *,
        batch_size: int,
        candidate_count: int,
        group_count: int,
        layer_count: int,
    ) -> None:
        if self.queries.ndim != 3 or self.queries.shape[:2] != (
            batch_size,
            candidate_count,
        ):
            raise ValueError("group executor queries must have shape [B,K,D]")
        if self.packed_embeddings.ndim != 2:
            raise ValueError("packed group embeddings must have shape [tokens,D]")
        if self.final_states is not None:
            if self.final_states.ndim != 6 or self.final_states.shape[:3] != (
                batch_size,
                group_count,
                layer_count,
            ):
                raise ValueError(
                    "group final states must have shape [B,G,layers,H,Dk,Dv]"
                )
            if self.final_states.dtype != torch.float32:
                raise ValueError("group GDR final states must use FP32 storage")
        if self.physical_gdr_calls != layer_count:
            raise ValueError("optimized group execution requires one GDR call per layer")
        if not bool(torch.isfinite(self.queries).all()):
            raise ValueError("group executor queries must be finite")


class GroupSharedGDRExecutor(nn.Module):
    """Run all group streams through one packed GDR call per layer.

    The wrapped layers follow the narrow production GDR protocol exposed by
    ``DLRMv3GDRSTULayer``: ``_project``, ``forward_gdr_projected``,
    ``forward_gdr``, and ``last_final_state``.  Keeping orchestration here
    makes grouping, selection, and packing reusable without importing dataset
    artifacts or experiment code into the core package.
    """

    def __init__(
        self,
        layers: nn.ModuleList | list[nn.Module] | tuple[nn.Module, ...],
        *,
        config: Optional[GroupSharedDeltaRecConfig] = None,
    ) -> None:
        super().__init__()
        if len(layers) < 1:
            raise ValueError("group-shared execution requires at least one layer")
        self.layers = (
            layers if isinstance(layers, nn.ModuleList) else nn.ModuleList(layers)
        )
        for layer in self.layers:
            for method in ("_project", "forward_gdr_projected", "forward_gdr"):
                if not callable(getattr(layer, method, None)):
                    raise TypeError(f"group GDR layer must expose {method}")
        self.config = config or GroupSharedDeltaRecConfig()

    @staticmethod
    def _gather_layer_zero_projection(
        source_projection: GDRKernelInput,
        source_indices: torch.Tensor,
        packed: PackedGroupSequence,
    ) -> GDRKernelInput:
        indices = source_indices.to(torch.int64)
        return GDRKernelInput(
            q=source_projection.q.index_select(0, indices),
            k=source_projection.k.index_select(0, indices),
            v=source_projection.v.index_select(0, indices),
            decay_logits=source_projection.decay_logits.index_select(0, indices),
            beta_logits=source_projection.beta_logits.index_select(0, indices),
            log_decay_scale=source_projection.log_decay_scale,
            decay_bias=source_projection.decay_bias,
            offsets=packed.offsets,
            event_gate=packed.event_gate,
        )

    def forward(
        self,
        *,
        source_x: torch.Tensor,
        packed: PackedGroupSequence,
        return_final_states: bool = True,
    ) -> GroupExecutorOutput:
        """Execute a previously frozen group layout.

        Layer zero is projected once over the original request stream and then
        gathered.  Later layers operate directly on the much smaller packed
        group buffer.  Candidate queries keep ``event_gate=0`` at every layer.
        """

        if source_x.ndim != 2 or packed.values.ndim != 2:
            raise ValueError("source and packed group inputs must be matrices")
        if source_x.shape[1] != packed.values.shape[1]:
            raise ValueError("source and packed group embedding widths differ")
        if source_x.device != packed.values.device:
            raise ValueError("source and packed group tensors must share a device")
        if self.config.projection_dtype == "bfloat16" and source_x.is_cuda:
            if source_x.dtype != torch.bfloat16:
                raise RuntimeError("registered A100 group execution requires BF16 x")
        if self.config.validate_runtime:
            packed.validate()

        fixed_width_layout = (
            build_group_fixed_width_layout(packed)
            if self.config.projection_mode == "fixed_width_batched"
            else None
        )
        layer_zero = self.layers[0]
        source_u, q, k, v, decay_logits, beta_logits = layer_zero._project(source_x)
        source_projected = GDRKernelInput(
            q=q,
            k=k,
            v=v,
            decay_logits=decay_logits,
            beta_logits=beta_logits,
            log_decay_scale=layer_zero.gdr_log_decay_scale,
            decay_bias=layer_zero.gdr_decay_bias,
            # Source offsets are deliberately never consumed by group
            # recurrence; gathering below installs the B*G offsets.
            offsets=packed.offsets.new_tensor([0, len(source_x)]),
            event_gate=None,
        )
        projected = self._gather_layer_zero_projection(
            source_projected,
            packed.source_indices,
            packed,
        )
        packed_x = packed.values
        packed_u = source_u.index_select(0, packed.source_indices.to(torch.int64))
        packed_x = layer_zero.forward_gdr_projected(
            x=packed_x,
            u=packed_u,
            projected=projected,
            return_final_state=return_final_states,
            fixed_width_layout=fixed_width_layout,
        )
        states: list[torch.Tensor] = []
        if return_final_states:
            if layer_zero.last_final_state is None:
                raise RuntimeError("group GDR layer omitted a requested final state")
            states.append(layer_zero.last_final_state.float())

        for layer in self.layers[1:]:
            packed_x = layer.forward_gdr(
                x=packed_x,
                x_offsets=packed.offsets,
                event_gate=packed.event_gate,
                return_final_state=return_final_states,
                fixed_width_layout=fixed_width_layout,
            )
            if return_final_states:
                if layer.last_final_state is None:
                    raise RuntimeError(
                        "group GDR layer omitted a requested final state"
                    )
                states.append(layer.last_final_state.float())

        flat_queries = packed_x.index_select(
            0, packed.query_indices.reshape(-1).to(torch.int64)
        )
        queries = flat_queries.reshape(
            packed.batch_size,
            packed.candidate_count,
            flat_queries.shape[-1],
        )
        final_states = None
        if states:
            final_states = torch.stack(states, dim=1).reshape(
                packed.batch_size,
                packed.group_count,
                len(self.layers),
                *states[0].shape[1:],
            )
        output = GroupExecutorOutput(
            queries=queries,
            final_states=final_states,
            packed_embeddings=packed_x,
            physical_gdr_calls=len(self.layers),
            auxiliary={
                "logical_encodes": packed.sequence_count,
                "logical_states": packed.sequence_count * len(self.layers),
                "packed_tokens_per_layer": packed.packed_tokens,
                "write_tokens_per_layer": packed.write_tokens,
                "read_tokens_per_layer": packed.read_tokens,
                "physical_gdr_calls": len(self.layers),
                "projection_linear_mode": self.config.projection_mode,
                "fixed_width_padded_tokens": (
                    fixed_width_layout.padded_tokens
                    if fixed_width_layout is not None
                    else packed.packed_tokens
                ),
                "fixed_width_padding_tokens": (
                    fixed_width_layout.padding_tokens
                    if fixed_width_layout is not None
                    else 0
                ),
                "actual_layer_backends": [
                    getattr(layer, "kernel_backend", "unknown")
                    for layer in self.layers
                ],
            },
        )
        if self.config.validate_runtime:
            output.validate(
                batch_size=packed.batch_size,
                candidate_count=packed.candidate_count,
                group_count=packed.group_count,
                layer_count=len(self.layers),
            )
        return output

    def forward_from_states(
        self,
        *,
        candidate_x: torch.Tensor,
        packed: PackedGroupSequence,
        states: torch.Tensor,
        return_final_states: bool = True,
    ) -> GroupExecutorOutput:
        """Execute candidate-only reads from cached per-layer group states.

        ``states`` uses the public ``[B,G,layers,H,Dk,Dv]`` FP32 layout.  The
        selected history has already been prefetched through every layer, so
        each online layer runs only the packed candidate queries from the
        corresponding cached state.  Candidate events must be identity writes
        (``event_gate=0``), hence requested final states are exactly the cached
        input tensor and do not require a kernel-side state materialization.
        """

        if candidate_x.ndim != 3:
            raise ValueError("candidate_x must have shape [B,K,D]")
        batch, candidates, embedding_dim = candidate_x.shape
        if packed.values.ndim != 2 or packed.values.shape != (
            batch * candidates,
            embedding_dim,
        ):
            raise ValueError("packed queries must contain exactly B*K embeddings")
        if packed.batch_size != batch or packed.candidate_count != candidates:
            raise ValueError("packed queries and candidate_x disagree on B or K")
        if candidate_x.device != packed.values.device:
            raise ValueError("candidate_x and packed queries must share a device")
        if candidate_x.dtype != packed.values.dtype:
            raise ValueError("candidate_x and packed queries must share a dtype")
        if self.config.projection_dtype == "bfloat16" and candidate_x.is_cuda:
            if candidate_x.dtype != torch.bfloat16:
                raise RuntimeError(
                    "registered A100 state-cache execution requires BF16 candidate_x"
                )
        expected_state_prefix = (
            batch,
            packed.group_count,
            len(self.layers),
        )
        if states.ndim != 6 or states.shape[:3] != expected_state_prefix:
            raise ValueError(
                "states must have shape [B,G,layers,H,Dk,Dv]"
            )
        if states.dtype != torch.float32:
            raise ValueError("cached GDR states must use FP32 storage")
        if states.device != candidate_x.device:
            raise ValueError("cached states and candidate_x must share a device")

        if self.config.validate_runtime:
            packed.validate()
            if packed.write_tokens != 0 or not bool((packed.event_gate == 0).all()):
                raise ValueError("state-cache execution requires read-only candidates")
            expected_values = candidate_x.reshape(
                batch * candidates,
                embedding_dim,
            ).index_select(0, packed.source_indices)
            if not torch.equal(expected_values, packed.values):
                raise ValueError("packed queries do not match candidate_x")
            if not bool(torch.isfinite(states).all()):
                raise ValueError("cached GDR states must be finite")

        fixed_width_layout = (
            build_group_fixed_width_layout(packed)
            if self.config.projection_mode == "fixed_width_batched"
            else None
        )
        # FLA's initialized-state Triton kernel addresses sequence states as a
        # dense ``N * H * Dk * Dv`` buffer and does not consume arbitrary
        # leading strides.  A slice of the public B,G,layers layout retains an
        # inter-layer gap, so materialize one layer-major buffer before the
        # loop rather than issuing one contiguous copy per layer.
        layer_major_states = states.permute(2, 0, 1, 3, 4, 5).contiguous()
        packed_x = packed.values
        for layer_index, layer in enumerate(self.layers):
            layer_initial_state = layer_major_states[layer_index].reshape(
                packed.sequence_count,
                *states.shape[3:],
            )
            packed_x = layer.forward_gdr(
                x=packed_x,
                x_offsets=packed.offsets,
                event_gate=packed.event_gate,
                initial_state=layer_initial_state,
                return_final_state=False,
                fixed_width_layout=fixed_width_layout,
            )

        flat_queries = packed_x.index_select(
            0,
            packed.query_indices.reshape(-1).to(torch.int64),
        )
        queries = flat_queries.reshape(batch, candidates, flat_queries.shape[-1])
        final_states = states if return_final_states else None
        output = GroupExecutorOutput(
            queries=queries,
            final_states=final_states,
            packed_embeddings=packed_x,
            physical_gdr_calls=len(self.layers),
            auxiliary={
                "logical_encodes": packed.sequence_count,
                "logical_states": packed.sequence_count * len(self.layers),
                "packed_tokens_per_layer": packed.packed_tokens,
                "write_tokens_per_layer": 0,
                "read_tokens_per_layer": packed.read_tokens,
                "physical_gdr_calls": len(self.layers),
                "projection_linear_mode": self.config.projection_mode,
                "fixed_width_padded_tokens": (
                    fixed_width_layout.padded_tokens
                    if fixed_width_layout is not None
                    else packed.packed_tokens
                ),
                "fixed_width_padding_tokens": (
                    fixed_width_layout.padding_tokens
                    if fixed_width_layout is not None
                    else 0
                ),
                "initial_state_source": "cached_all_layers",
                "initial_state_layout": "B,G,layers,H,Dk,Dv",
                "kernel_initial_state_layout": "layers,B*G,H,Dk,Dv_contiguous",
                "online_history_tokens": 0,
                "actual_layer_backends": [
                    getattr(layer, "kernel_backend", "unknown")
                    for layer in self.layers
                ],
            },
        )
        if self.config.validate_runtime:
            output.validate(
                batch_size=batch,
                candidate_count=candidates,
                group_count=packed.group_count,
                layer_count=len(self.layers),
            )
        return output

    def forward_from_global_states(
        self,
        *,
        candidate_x: torch.Tensor,
        packed: PackedGlobalGroupQueries,
        states: torch.Tensor,
        return_final_states: bool = True,
    ) -> GroupExecutorOutput:
        """Read a ragged batch of occupied global-group FP32 prefix states.

        ``states`` is ``[N_occupied,layers,H,Dk,Dv]`` and each strictly
        nonempty packed row addresses the state with the same leading index.
        Candidate events remain identity writes, so returned states alias the
        input cache materialization.
        """

        if candidate_x.ndim != 3 or not torch.is_floating_point(candidate_x):
            raise ValueError("candidate_x must be floating point [B,K,D]")
        batch, candidates, dimension = candidate_x.shape
        if packed.values.shape != (batch * candidates, dimension):
            raise ValueError("global-group packed queries must contain B*K rows")
        if packed.query_indices.shape != (batch, candidates):
            raise ValueError("global-group query restoration must have shape [B,K]")
        if states.ndim != 5 or states.shape[:2] != (
            packed.sequence_count,
            len(self.layers),
        ):
            raise ValueError("states must have shape [N_occupied,layers,H,Dk,Dv]")
        if states.dtype != torch.float32:
            raise ValueError("cached global-group states must use FP32")
        if states.device != candidate_x.device or packed.values.device != candidate_x.device:
            raise ValueError("global-group states, packing, and candidates must share a device")
        if self.config.projection_dtype == "bfloat16" and candidate_x.is_cuda:
            if candidate_x.dtype != torch.bfloat16:
                raise RuntimeError(
                    "registered A100 global-group execution requires BF16 candidate_x"
                )
        if self.config.validate_runtime:
            packed.validate()
            expected = candidate_x.reshape(batch * candidates, dimension).index_select(
                0, packed.source_indices
            )
            if not torch.equal(expected, packed.values):
                raise ValueError("global-group packed queries do not match candidate_x")
            if not bool(torch.isfinite(states).all()):
                raise ValueError("cached global-group states must be finite")

        fixed_width_layout = (
            build_group_fixed_width_layout(packed)  # type: ignore[arg-type]
            if self.config.projection_mode == "fixed_width_batched"
            else None
        )
        layer_major_states = states.permute(1, 0, 2, 3, 4).contiguous()
        packed_x = packed.values
        for layer_index, layer in enumerate(self.layers):
            packed_x = layer.forward_gdr(
                x=packed_x,
                x_offsets=packed.offsets,
                event_gate=packed.event_gate,
                initial_state=layer_major_states[layer_index],
                return_final_state=False,
                fixed_width_layout=fixed_width_layout,
            )
        flat_queries = packed_x.index_select(
            0, packed.query_indices.reshape(-1).to(torch.int64)
        )
        queries = flat_queries.reshape(batch, candidates, flat_queries.shape[-1])
        if self.config.validate_runtime and not bool(torch.isfinite(queries).all()):
            raise ValueError("global-group candidate queries must be finite")
        return GroupExecutorOutput(
            queries=queries,
            final_states=states if return_final_states else None,
            packed_embeddings=packed_x,
            physical_gdr_calls=len(self.layers),
            auxiliary={
                "logical_encodes": packed.sequence_count,
                "logical_states": packed.sequence_count * len(self.layers),
                "packed_tokens_per_layer": packed.packed_tokens,
                "write_tokens_per_layer": 0,
                "read_tokens_per_layer": packed.packed_tokens,
                "physical_gdr_calls": len(self.layers),
                "projection_linear_mode": self.config.projection_mode,
                "initial_state_source": "global_group_cache",
                "initial_state_layout": "N_occupied,layers,H,Dk,Dv",
                "kernel_initial_state_layout": "layers,N_occupied,H,Dk,Dv_contiguous",
                "online_history_tokens": 0,
                "actual_layer_backends": [
                    getattr(layer, "kernel_backend", "unknown")
                    for layer in self.layers
                ],
            },
        )


__all__ = [
    "GroupExecutorOutput",
    "GroupPool",
    "GroupSharedDeltaRecConfig",
    "GroupSharedGDRExecutor",
    "GroupingPolicy",
    "ProjectionMode",
]

