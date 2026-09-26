# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import hashlib

import math

from dataclasses import dataclass

from typing import Any, Optional

import torch

import torch.nn.functional as F

from deltarec.adaptors.kuai.common import HammerKernel

from deltarec.adaptors.kuai.modules.stu import STU, STULayer, STUStack

from deltarec.adaptors.kuai.ops.hstu_compute import hstu_compute_output, hstu_compute_uqvk

from deltarec.adaptors.kuai.research.modeling.sequential.delta_rec import DenseEventBatch, SelectionConstraints

from deltarec.adaptors.kuai.research.modeling.sequential.gdr_kernels import FLAGDRKernel, ReferenceGDRKernel

from deltarec.adaptors.kuai.research.modeling.sequential.selective_gdr import GDRKernelInput

from torchrec import KeyedJaggedTensor

PRODUCTION_HASH_SEED = 20260818

def stable_budget_mask(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    *,
    budget: float = 0.25,
    recent_floor: int = 32,
) -> torch.Tensor:
    """Stable score selection with exact per-row budgets and no padding."""

    if scores.ndim != 2 or lengths.shape != (scores.shape[0],):
        raise ValueError("scores and lengths must have shapes [batch,time] and [batch]")
    if not 0 < budget <= 1:
        raise ValueError("budget must lie in (0, 1]")
    if recent_floor < 0:
        raise ValueError("recent_floor must be nonnegative")
    if bool((lengths < 0).any()) or bool((lengths > scores.shape[1]).any()):
        raise ValueError("lengths are outside the score tensor")
    # headline_v1 freezes FP32 multiply followed by ceil and the recent floor.
    budgets = torch.ceil(
        lengths.to(torch.float32)
        * torch.tensor(float(budget), dtype=torch.float32, device=lengths.device)
    ).long()
    floor = torch.minimum(lengths.long(), torch.full_like(lengths.long(), recent_floor))
    budgets = torch.maximum(budgets, floor)
    output = torch.zeros_like(scores, dtype=torch.bool)
    for row, length_tensor in enumerate(lengths):
        length = int(length_tensor)
        count = int(budgets[row])
        if count:
            order = torch.argsort(scores[row, :length], descending=True, stable=True)
            output[row, order[:count]] = True
    return output

def compact_keyed_jagged_history(
    features: KeyedJaggedTensor,
    *,
    reference_key: str,
    keep_mask: torch.Tensor,
) -> tuple[KeyedJaggedTensor, torch.Tensor, torch.Tensor]:
    """Stably compact every history-aligned KJT feature.

    Features whose per-request lengths differ from ``reference_key`` are
    contextual metadata and are preserved verbatim.  The returned source
    positions are the original chronological positions of retained history
    tokens and are used to preserve positional embeddings after compaction.
    """

    reference = features[reference_key]
    reference_lengths = reference.lengths().long()
    if keep_mask.ndim != 2 or keep_mask.dtype != torch.bool:
        raise ValueError("keep_mask must be boolean with shape [batch,time]")
    if keep_mask.shape[0] != len(reference_lengths):
        raise ValueError("keep_mask batch does not match the KJT")
    if keep_mask.shape[1] < int(reference_lengths.max() if len(reference_lengths) else 0):
        raise ValueError("keep_mask is shorter than a history row")
    if features.weights_or_none() is not None:
        raise ValueError("weighted KJT history compaction is not supported")
    positions = torch.arange(keep_mask.shape[1], device=keep_mask.device)[None, :]
    valid = positions < reference_lengths.to(keep_mask.device)[:, None]
    keep = keep_mask & valid
    new_reference_lengths = keep.sum(dim=1).long()
    source_positions = positions.expand_as(keep)[keep]

    all_values: list[torch.Tensor] = []
    all_lengths: list[torch.Tensor] = []
    for key in features.keys():
        jagged = features[key]
        lengths = jagged.lengths().long()
        values = jagged.values()
        if torch.equal(lengths.cpu(), reference_lengths.cpu()):
            offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
            rows = []
            for row in range(len(lengths)):
                start = int(offsets[row])
                end = int(offsets[row + 1])
                rows.append(
                    values[start:end][
                        keep[row, : int(lengths[row])].to(values.device)
                    ]
                )
            compact_values = (
                torch.cat(rows) if rows else values.new_empty((0, *values.shape[1:]))
            )
            all_values.append(compact_values)
            all_lengths.append(new_reference_lengths.to(lengths.device))
        else:
            all_values.append(values)
            all_lengths.append(lengths)
    return (
        KeyedJaggedTensor(
            keys=features.keys(),
            values=torch.cat(all_values),
            lengths=torch.cat(all_lengths),
        ),
        source_positions.long(),
        new_reference_lengths,
    )

@dataclass(frozen=True)
class DLRMv3SelectionBatch:
    """Frozen selector fields for UIH events before contextual/target insertion."""

    history: DenseEventBatch
    constraints: Optional[SelectionConstraints] = None

class DLRMv3GDRSTULayer(STU):
    """GDR aggregation around one stock production :class:`STULayer`."""

    def __init__(
        self,
        base: STULayer,
        *,
        seed: int = PRODUCTION_HASH_SEED,
        kernel_backend: str = "triton",
        assume_binary_event_gate: bool = False,
        state_attention_dim: Optional[int] = None,
        state_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__(is_inference=base.is_inference)
        if kernel_backend not in ("reference", "fla", "triton"):
            raise ValueError(
                "kernel_backend must be 'reference', 'fla', or 'triton'"
            )
        self.base = base
        self.state_attention_dim = int(
            base._attention_dim if state_attention_dim is None else state_attention_dim
        )
        self.state_hidden_dim = int(
            base._hidden_dim if state_hidden_dim is None else state_hidden_dim
        )
        if not 0 < self.state_attention_dim <= base._attention_dim:
            raise ValueError("GDR state key dimension must be in (0, base Dk]")
        if not 0 < self.state_hidden_dim <= base._hidden_dim:
            raise ValueError("GDR state value dimension must be in (0, base Dv]")
        self.kernel_backend = kernel_backend
        self.reference_kernel = ReferenceGDRKernel()
        # Binary-gate specialization is opt-in because this layer also exposes
        # fractional-gate training/teacher entry points.  Production adapters
        # that construct gates from boolean event roles enable it explicitly.
        self.fla_kernel = FLAGDRKernel(
            validate_inputs=False,
            assume_binary_event_gate=assume_binary_event_gate,
        )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        decay_scale = torch.empty(base._num_heads).uniform_(0, 16, generator=generator)
        self.gdr_log_decay_scale = torch.nn.Parameter(
            torch.log(decay_scale.clamp_min(1e-4))
        )
        dt = torch.exp(
            torch.rand(base._num_heads, generator=generator)
            * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp_min(1e-4)
        self.gdr_decay_bias = torch.nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.gdr_gate_weight = torch.nn.Parameter(
            torch.zeros(base._embedding_dim, 2 * base._num_heads)
        )
        self.last_final_state: Optional[torch.Tensor] = None
        self.profile_stages = False
        self._last_stage_events = None

    def _project(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        base = self.base
        kernel = HammerKernel.PYTORCH if x.device.type == "cpu" else base.hammer_kernel()
        u, q, k, v = hstu_compute_uqvk(
            x=x,
            norm_weight=base._input_norm_weight.to(x.dtype),
            norm_bias=base._input_norm_bias.to(x.dtype),
            norm_eps=1e-6,
            num_heads=base._num_heads,
            attn_dim=base._attention_dim,
            hidden_dim=base._hidden_dim,
            uvqk_weight=base._uvqk_weight.to(x.dtype),
            uvqk_bias=base._uvqk_beta.to(x.dtype),
            kernel=kernel,
        )
        normed = F.layer_norm(
            x,
            (base._embedding_dim,),
            base._input_norm_weight.to(x.dtype),
            base._input_norm_bias.to(x.dtype),
            1e-6,
        )
        gates = torch.mm(normed, self.gdr_gate_weight.to(x.dtype))
        decay_logits, beta_logits = gates.chunk(2, dim=-1)
        return (
            u,
            q[..., : self.state_attention_dim],
            k[..., : self.state_attention_dim],
            v[..., : self.state_hidden_dim],
            decay_logits,
            beta_logits,
        )

    def _project_fixed_width(
        self,
        x: torch.Tensor,
        layout: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project each logical candidate stream with a fixed GEMM shape."""

        from deltarec.adaptors.kuai.research.modeling.sequential.deltarec.projections import (
            fixed_width_sequence_linear,
        )

        linear = (
            self._reference_fixed_width_linear
            if self.kernel_backend == "reference"
            else fixed_width_sequence_linear
        )

        base = self.base
        normed = F.layer_norm(
            x,
            (base._embedding_dim,),
            base._input_norm_weight.to(x.dtype),
            base._input_norm_bias.to(x.dtype),
            1e-6,
        )
        uvqk = linear(
            normed,
            base._uvqk_weight.to(x.dtype),
            base._uvqk_beta.to(x.dtype),
            layout,
        )
        u, v, q, k = torch.split(
            uvqk,
            [
                base._hidden_dim * base._num_heads,
                base._hidden_dim * base._num_heads,
                base._attention_dim * base._num_heads,
                base._attention_dim * base._num_heads,
            ],
            dim=1,
        )
        u = F.silu(u)
        q = q.view(-1, base._num_heads, base._attention_dim)
        k = k.view(-1, base._num_heads, base._attention_dim)
        v = v.view(-1, base._num_heads, base._hidden_dim)
        gates = linear(
            normed,
            self.gdr_gate_weight.to(x.dtype),
            None,
            layout,
        )
        decay_logits, beta_logits = gates.chunk(2, dim=-1)
        return (
            u,
            q[..., : self.state_attention_dim],
            k[..., : self.state_attention_dim],
            v[..., : self.state_hidden_dim],
            decay_logits,
            beta_logits,
        )

    def _restore_value_width(self, context: torch.Tensor) -> torch.Tensor:
        """Zero-pad a reduced recurrent value state for the stock output MLP."""

        missing = int(self.base._hidden_dim) - int(context.shape[-1])
        if missing < 0:
            raise ValueError("GDR context exceeds the stock HSTU value width")
        return context if missing == 0 else F.pad(context, (0, missing))

    @staticmethod
    def _reference_fixed_width_linear(
        x: torch.Tensor,
        right_weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        layout: Any,
    ) -> torch.Tensor:
        """Transparent per-sequence GEMM oracle independent of chunk width."""

        padded = x.new_zeros((layout.padded_tokens, x.shape[1]))
        padded = padded.index_copy(
            0,
            layout.flat_padded_indices,
            x,
        ).view(layout.sequence_count, layout.token_width, x.shape[1])
        rows = []
        for sequence in range(layout.sequence_count):
            value = torch.mm(padded[sequence], right_weight)
            if bias is not None:
                value = value + bias
            rows.append(value)
        projected = torch.stack(rows, dim=0)
        return projected.reshape(
            layout.padded_tokens,
            right_weight.shape[1],
        ).index_select(0, layout.flat_padded_indices)

    def _output_fixed_width(
        self,
        *,
        x: torch.Tensor,
        u: torch.Tensor,
        context: torch.Tensor,
        layout: Any,
    ) -> torch.Tensor:
        """Run the stock HSTU output/residual path with fixed-width GEMMs."""

        from deltarec.adaptors.kuai.research.modeling.sequential.deltarec.projections import (
            fixed_width_sequence_linear,
        )

        base = self.base
        attn = context.reshape(-1, base._hidden_dim * base._num_heads)
        attn_fp32 = attn.float()
        u_fp32 = u.float()
        if base._use_group_norm:
            normalized = F.group_norm(
                attn_fp32.view(-1, base._num_heads, base._hidden_dim),
                num_groups=base._num_heads,
                weight=base._output_norm_weight.float(),
                bias=base._output_norm_bias.float(),
                eps=1e-6,
            ).view(-1, base._num_heads * base._hidden_dim)
        else:
            normalized = F.layer_norm(
                attn_fp32,
                (attn_fp32.shape[-1],),
                base._output_norm_weight.float(),
                base._output_norm_bias.float(),
                1e-6,
            )
        output_input = torch.cat(
            (u_fp32, attn_fp32, u_fp32 * normalized),
            dim=-1,
        ).to(x.dtype)
        if base._output_dropout_ratio:
            output_input = F.dropout(
                output_input,
                p=base._output_dropout_ratio,
                training=self.training,
            )
        linear = (
            self._reference_fixed_width_linear
            if self.kernel_backend == "reference"
            else fixed_width_sequence_linear
        )
        return x + linear(
            output_input,
            base._output_weight.to(x.dtype),
            None,
            layout,
        )

    def forward_gdr_projected(
        self,
        *,
        x: torch.Tensor,
        u: torch.Tensor,
        projected: GDRKernelInput,
        initial_state: Optional[torch.Tensor] = None,
        return_final_state: bool = False,
        fixed_width_layout: Any = None,
    ) -> torch.Tensor:
        """Apply GDR and the stock HSTU output path to an explicit projection.

        Candidate-symmetric execution may safely share the candidate-independent
        layer-0 projection and gather it into ``B*K`` packed streams.  Later
        layers still call :meth:`forward_gdr` because their inputs are already
        candidate-specific.  Keeping this boundary explicit prevents a
        metadata-only "project once" path from claiming work it did not do.
        """

        if x.ndim != 2 or u.ndim != 2 or len(x) != len(u):
            raise ValueError("projected GDR x and u must be token-aligned matrices")
        if projected.q.shape[0] != len(x):
            raise ValueError("projected GDR tensors must align with x")
        base = self.base
        stage_events = None
        if self.profile_stages:
            # Projection occurred outside this method.  Record a zero-width
            # projection span so the existing diagnostics schema remains
            # stable while the caller accounts for shared projection time.
            stage_events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            stage_events[0].record()
            stage_events[1].record()
        if self.kernel_backend == "reference":
            kernel_output = self.reference_kernel(
                projected,
                initial_state=initial_state,
                return_final_state=return_final_state,
            )
        elif self.kernel_backend == "fla":
            kernel_output = self.fla_kernel(
                projected,
                initial_state=initial_state,
                return_final_state=return_final_state,
            )
        else:
            if not x.is_cuda:
                raise RuntimeError(
                    "the production Triton GDR backend requires CUDA; use "
                    "kernel_backend='reference' for CPU correctness"
                )
            if self.training and torch.is_grad_enabled() and x.requires_grad:
                raise RuntimeError(
                    "the Triton GDR prefill backend is serving-only; use "
                    "kernel_backend='fla' for differentiable training"
                )
            from deltarec.adaptors.kuai.ops.triton.triton_delta_rec import (
                triton_gdr_packed_prefill,
            )

            kernel_output = triton_gdr_packed_prefill(
                projected,
                initial_state=initial_state,
                return_final_state=return_final_state,
            )
        if stage_events is not None:
            stage_events[2].record()
        self.last_final_state = (
            None
            if kernel_output.final_state is None
            else kernel_output.final_state.float()
        )
        if fixed_width_layout is None:
            restored_context = self._restore_value_width(kernel_output.context)
            output = hstu_compute_output(
                attn=restored_context.reshape(-1, base._hidden_dim * base._num_heads),
                u=u,
                x=x,
                norm_weight=base._output_norm_weight.to(x.dtype),
                norm_bias=base._output_norm_bias.to(x.dtype),
                norm_eps=1e-6,
                dropout_ratio=base._output_dropout_ratio,
                output_weight=base._output_weight.to(x.dtype),
                group_norm=base._use_group_norm,
                num_heads=base._num_heads,
                linear_dim=base._hidden_dim,
                concat_u=True,
                concat_x=True,
                mul_u_activation_type="none",
                training=self.training,
                kernel=HammerKernel.PYTORCH if x.device.type == "cpu" else base.hammer_kernel(),
                recompute_y_in_backward=base._recompute_y,
            )
        else:
            output = self._output_fixed_width(
                x=x,
                u=u,
                context=self._restore_value_width(kernel_output.context),
                layout=fixed_width_layout,
            )
        if stage_events is not None:
            stage_events[3].record()
            self._last_stage_events = stage_events
        else:
            self._last_stage_events = None
        return output

    def forward_gdr(
        self,
        *,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        event_gate: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_final_state: bool = False,
        fixed_width_layout: Any = None,
    ) -> torch.Tensor:
        stage_events = None
        if self.profile_stages:
            stage_events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            stage_events[0].record()
        if fixed_width_layout is None:
            u, q, k, v, decay_logits, beta_logits = self._project(x)
        else:
            u, q, k, v, decay_logits, beta_logits = self._project_fixed_width(
                x,
                fixed_width_layout,
            )
        if stage_events is not None:
            stage_events[1].record()
        projected = GDRKernelInput(
            q=q,
            k=k,
            v=v,
            decay_logits=decay_logits,
            beta_logits=beta_logits,
            log_decay_scale=self.gdr_log_decay_scale,
            decay_bias=self.gdr_decay_bias,
            offsets=x_offsets,
            event_gate=event_gate,
        )
        # The shared helper records its own events, so retain the true
        # projection start/done events from this call and replace only the
        # recurrence/output markers afterward.
        output = self.forward_gdr_projected(
            x=x,
            u=u,
            projected=projected,
            initial_state=initial_state,
            return_final_state=return_final_state,
            fixed_width_layout=fixed_width_layout,
        )
        if stage_events is not None and self._last_stage_events is not None:
            _, _, prefill_done, output_done = self._last_stage_events
            self._last_stage_events = [
                stage_events[0],
                stage_events[1],
                prefill_done,
                output_done,
            ]
        return output

    def last_stage_times_ms(self) -> dict[str, float]:
        if self._last_stage_events is None:
            return {}
        torch.cuda.synchronize()
        for event in self._last_stage_events:
            event.synchronize()
        start, projection_done, prefill_done, output_done = self._last_stage_events
        return {
            "projection_ms": start.elapsed_time(projection_done),
            "prefill_ms": projection_done.elapsed_time(prefill_done),
            "output_ms": prefill_done.elapsed_time(output_done),
        }

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del max_seq_len, max_kv_caching_len, kv_caching_lengths
        positions = torch.arange(len(x), device=x.device) - torch.repeat_interleave(
            x_offsets[:-1], x_lengths
        )
        target_start = torch.repeat_interleave(x_lengths - num_targets, x_lengths)
        event_gate = (positions < target_start).to(x.dtype)
        return self.forward_gdr(x=x, x_offsets=x_offsets, event_gate=event_gate)

    def cached_forward(
        self,
        delta_x: torch.Tensor,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "DLRMv3 GDR cached serving uses the versioned DeltaRec state-cache adapter"
        )

class DLRMv3DeltaRecSTUStack(STU):
    """Shared full-GDR/DeltaRec stack installed at the production STU boundary."""

    def __init__(
        self,
        original: STUStack,
        *,
        mode: str,
        compactor: Optional[torch.nn.Module] = None,
        seed: int = PRODUCTION_HASH_SEED,
        kernel_backend: str = "triton",
        contextual_seq_len: int = 0,
        retention_ratio: float = 1.0,
        recent_floor: int = 32,
        state_attention_dim: Optional[int] = None,
        state_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__(is_inference=original.is_inference)
        if mode not in (
            "full_gdr",
            "dense_gated",
            "target_similarity_25",
            "rank_only_25",
            "recent_only",
            "rank_only",
            "deltarec",
        ):
            raise ValueError(f"unsupported production GDR mode {mode!r}")
        if mode in ("dense_gated", "deltarec") and compactor is None:
            raise ValueError("selected GDR modes require the frozen threshold compactor")
        self.mode = mode
        if mode in ("recent_only", "rank_only") and retention_ratio not in (0.25, 0.50):
            raise ValueError("headline sparse shared GDR requires ratio 0.25 or 0.50")
        if recent_floor != 32:
            raise ValueError("headline sparse shared GDR recent floor must remain 32")
        self.retention_ratio = float(retention_ratio)
        self.recent_floor = int(recent_floor)
        self.compactor = compactor
        self.contextual_seq_len = contextual_seq_len
        self.seed = seed
        self.layers = torch.nn.ModuleList(
            [
                DLRMv3GDRSTULayer(
                    layer,
                    seed=seed + index,
                    kernel_backend=kernel_backend,
                    state_attention_dim=state_attention_dim,
                    state_hidden_dim=state_hidden_dim,
                )
                for index, layer in enumerate(original._stu_layers)
            ]
        )
        base_parameter = next(original.parameters(), None)
        if base_parameter is not None and base_parameter.device.type != "meta":
            # GDR parameters are created independently of the stock STU
            # modules. Keep the wrapper colocated when a boundary adapter is
            # constructed around an already-materialized CUDA checkpoint.
            self.to(base_parameter.device)
        initialization = hashlib.sha256()
        initialization.update(b"dlrmv3-gdr-v1")
        for index, layer in enumerate(self.layers):
            for name in (
                "gdr_log_decay_scale",
                "gdr_decay_bias",
                "gdr_gate_weight",
            ):
                value = getattr(layer, name).detach().cpu().contiguous()
                initialization.update(f"{index}:{name}:{tuple(value.shape)}".encode())
                initialization.update(value.numpy().tobytes())
        self.gdr_initialization_hash = initialization.hexdigest()
        self._pending_selection: Optional[DLRMv3SelectionBatch] = None
        self._history_precompacted = False
        self._precompacted_history_lengths: Optional[torch.Tensor] = None
        self._precompacted_history_write_mask: Optional[torch.Tensor] = None
        self.last_history_source_positions: Optional[torch.Tensor] = None
        self.last_original_history_lengths: Optional[torch.Tensor] = None
        self.last_selected_tokens = 0
        self.last_write_tokens = 0
        self.last_total_tokens = 0
        self.last_sparse_history_tokens = 0
        self.last_sparse_selected_writes = 0
        self.last_sparse_budgets: Optional[torch.Tensor] = None

    def set_stage_profiling(self, enabled: bool) -> None:
        for layer in self.layers:
            layer.profile_stages = enabled
        if self.compactor is not None and hasattr(self.compactor, "profile_stages"):
            self.compactor.profile_stages = enabled

    def last_stage_times_ms(self) -> dict[str, float]:
        totals = {"projection_ms": 0.0, "prefill_ms": 0.0, "output_ms": 0.0}
        for layer in self.layers:
            for name, value in layer.last_stage_times_ms().items():
                totals[name] += value
        if self.compactor is not None and hasattr(self.compactor, "last_stage_times_ms"):
            totals.update(self.compactor.last_stage_times_ms())
        return totals

    def prepare_selection(
        self,
        history: DenseEventBatch,
        constraints: Optional[SelectionConstraints] = None,
    ) -> None:
        """Install one request's pre-bucketed causal fields for the next forward."""

        if self._pending_selection is not None:
            raise RuntimeError("the previous DeltaRec selection batch was not consumed")
        if constraints is not None:
            constraints.validate(history.shape)
        self._pending_selection = DLRMv3SelectionBatch(
            history=history,
            constraints=constraints,
        )

    def precompact_history_features(
        self,
        features: KeyedJaggedTensor,
        *,
        reference_key: str,
    ) -> KeyedJaggedTensor:
        """Run the frozen DeltaRec threshold before embedding/projection."""

        if self.mode != "deltarec":
            return features
        pending = self._pending_selection
        self._pending_selection = None
        if pending is None:
            raise RuntimeError("DeltaRec forward requires prepare_delta_rec_selection()")
        history = pending.history
        constraints = pending.constraints or SelectionConstraints.none(
            history.shape, history.item_ids.device
        )
        assert self.compactor is not None
        selected = self.compactor.compact(
            history,
            constraints=constraints,
        ).selected
        keep = torch.zeros(history.shape, dtype=torch.bool, device=history.item_ids.device)
        lengths = history.lengths.long()
        sequence_ids = torch.repeat_interleave(
            torch.arange(len(lengths), device=lengths.device),
            selected.offsets[1:] - selected.offsets[:-1],
        )
        keep[sequence_ids, selected.source_positions.long()] = True
        compacted, positions, compact_lengths = compact_keyed_jagged_history(
            features,
            reference_key=reference_key,
            keep_mask=keep,
        )
        if not torch.equal(compact_lengths.cpu(), (selected.offsets[1:] - selected.offsets[:-1]).cpu()):
            raise RuntimeError("selector and KJT compaction lengths disagree")
        self._history_precompacted = True
        self._precompacted_history_lengths = compact_lengths.detach().clone()
        self._precompacted_history_write_mask = selected.write_mask.detach().clone()
        self.last_history_source_positions = positions
        self.last_original_history_lengths = history.lengths.detach().clone()
        self.last_total_tokens = int(history.lengths.sum())
        self.last_selected_tokens = len(selected.values)
        self.last_write_tokens = int(selected.write_mask.sum())
        return compacted

    def _precompacted_event_gate(
        self,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        num_targets: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Rebuild context/write, history/policy, and candidate/read roles."""

        history_lengths = self._precompacted_history_lengths
        history_write_mask = self._precompacted_history_write_mask
        if history_lengths is None or history_write_mask is None:
            raise RuntimeError("precompacted DeltaRec role metadata is missing")
        history_lengths = history_lengths.to(x_lengths.device).long()
        history_write_mask = history_write_mask.to(x_lengths.device)
        targets = num_targets.to(x_lengths.device).long()
        contextual = x_lengths.long() - history_lengths - targets
        if bool((contextual < 0).any()):
            raise RuntimeError("precompacted sequence roles exceed the packed lengths")
        if int(history_lengths.sum()) != len(history_write_mask):
            raise RuntimeError("precompacted history role metadata is inconsistent")
        positions = self._packed_positions(x_lengths.long(), x_offsets.long())
        contextual_per_token = torch.repeat_interleave(contextual, x_lengths.long())
        history_per_token = torch.repeat_interleave(history_lengths, x_lengths.long())
        history_tokens = (positions >= contextual_per_token) & (
            positions < contextual_per_token + history_per_token
        )
        gate = (positions < contextual_per_token).to(dtype)
        gate[history_tokens] = history_write_mask.to(dtype)
        return gate

    @staticmethod
    def _packed_positions(
        lengths: torch.Tensor, offsets: torch.Tensor
    ) -> torch.Tensor:
        return torch.arange(int(offsets[-1]), device=lengths.device) - torch.repeat_interleave(
            offsets[:-1], lengths
        )

    def _compact(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
    ):
        pending = self._pending_selection
        self._pending_selection = None
        if pending is None:
            raise RuntimeError("DeltaRec forward requires prepare_selection()")
        history = pending.history
        batch = len(x_lengths)
        if history.shape[0] != batch:
            raise ValueError("selector batch does not match the production batch")
        history_lengths = history.lengths.to(device=x.device, dtype=torch.int64)
        targets = num_targets.to(device=x.device, dtype=torch.int64)
        contextual = x_lengths.to(torch.int64) - history_lengths - targets
        if bool((contextual < 0).any()):
            raise ValueError("production sequence is shorter than history plus targets")

        positions = torch.arange(max_seq_len, device=x.device)[None, :]
        history_positions = positions - contextual[:, None]
        history_mask = (history_positions >= 0) & (
            history_positions < history_lengths[:, None]
        )
        target_mask = (positions >= (contextual + history_lengths)[:, None]) & (
            positions < x_lengths[:, None]
        )
        contextual_mask = positions < contextual[:, None]
        gather_index = history_positions.clamp(0, history.shape[1] - 1).long()

        def align(value: torch.Tensor) -> torch.Tensor:
            value = value.to(x.device)
            gathered = torch.gather(value, 1, gather_index)
            return torch.where(history_mask, gathered, torch.zeros_like(gathered)).contiguous()

        valid = positions < x_lengths[:, None]
        dense_x = x.new_zeros((batch, max_seq_len, x.shape[-1]))
        dense_x[valid] = x
        source = DenseEventBatch(
            item_ids=align(history.item_ids),
            rating_buckets=align(history.rating_buckets),
            time_gap_buckets=align(history.time_gap_buckets),
            position_buckets=align(history.position_buckets),
            popularity_buckets=align(history.popularity_buckets),
            repeat_buckets=align(history.repeat_buckets),
            lengths=x_lengths,
            payloads={"x": dense_x},
        )
        constraints = SelectionConstraints(
            force_write_mask=contextual_mask.contiguous(),
            force_read_mask=target_mask.contiguous(),
            require_output_mask=target_mask.contiguous(),
        )
        if pending.constraints is not None:
            history_constraints = pending.constraints

            def align_mask(value: torch.Tensor) -> torch.Tensor:
                gathered = torch.gather(value.to(x.device), 1, gather_index)
                return torch.where(
                    history_mask,
                    gathered,
                    torch.zeros_like(gathered),
                ).contiguous()

            constraints = SelectionConstraints(
                force_write_mask=(
                    constraints.force_write_mask
                    | align_mask(history_constraints.force_write_mask)
                ),
                force_read_mask=(
                    constraints.force_read_mask
                    | align_mask(history_constraints.force_read_mask)
                ),
                require_output_mask=(
                    constraints.require_output_mask
                    | align_mask(history_constraints.require_output_mask)
                ),
            )
        assert self.compactor is not None
        return self.compactor.compact(source, constraints=constraints)

    def _screening_compact(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        num_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the registered exact-budget screening controls.

        Target similarity scores embedded histories against candidate slot zero.
        Rank-only evaluates online ridge leverage from the first layer's full K
        projection, so its screening cost is intentionally retained.
        """

        batch = len(x_lengths)
        history_lengths = x_lengths.long() - num_targets.long() - self.contextual_seq_len
        if bool((history_lengths < 0).any()):
            raise ValueError("sequence is shorter than context plus candidates")
        max_history = int(history_lengths.max()) if batch else 0
        scores = x.new_full((batch, max_history), float("-inf"), dtype=torch.float32)
        projected_k: Optional[torch.Tensor] = None
        if self.mode in ("rank_only_25", "rank_only"):
            with torch.no_grad():
                projected_k = self.layers[0]._project(x)[2].float()
                projected_k = F.normalize(projected_k, dim=-1, eps=1e-6)
        for row in range(batch):
            start = int(x_offsets[row]) + self.contextual_seq_len
            history = int(history_lengths[row])
            if history == 0:
                continue
            if self.mode == "target_similarity_25":
                candidate = int(x_offsets[row + 1]) - int(num_targets[row])
                if int(num_targets[row]) < 1:
                    raise ValueError("target similarity requires candidate slot zero")
                history_x = F.normalize(x[start : start + history].float(), dim=-1)
                target_x = F.normalize(x[candidate].float(), dim=-1)
                scores[row, :history] = torch.mv(history_x, target_x)
            elif self.mode in ("rank_only_25", "rank_only"):
                assert projected_k is not None
                keys = projected_k[start : start + history]
                heads, dimension = keys.shape[1:]
                inverse = torch.eye(
                    dimension, device=x.device, dtype=torch.float32
                )[None].repeat(heads, 1, 1)
                inverse = inverse / 1e-3
                for position in range(history):
                    key = keys[position]
                    inv_key = torch.bmm(inverse, key.unsqueeze(-1)).squeeze(-1)
                    leverage = (key * inv_key).sum(-1).clamp_min(0)
                    scores[row, position] = leverage.mean()
                    denominator = (1.0 + leverage).clamp_min(1e-8)
                    inverse = inverse - (
                        inv_key[:, :, None]
                        * inv_key[:, None, :]
                        / denominator[:, None, None]
                    )
            else:
                # Higher chronological position wins, giving the exact most
                # recent B history events with stable deterministic ties.
                scores[row, :history] = torch.arange(
                    history, device=x.device, dtype=torch.float32
                )
        ratio = 0.25 if self.mode.endswith("_25") else self.retention_ratio
        floor = 0 if self.mode.endswith("_25") else self.recent_floor
        history_selected = stable_budget_mask(
            scores, history_lengths, budget=ratio, recent_floor=floor
        )
        budgets = history_selected.sum(dim=1).long()
        self.last_sparse_history_tokens = int(history_lengths.sum().item())
        self.last_sparse_selected_writes = int(budgets.sum().item())
        self.last_sparse_budgets = budgets.detach().clone()
        selected_indices: list[torch.Tensor] = []
        selected_lengths: list[int] = []
        write_masks: list[torch.Tensor] = []
        for row in range(batch):
            row_start = int(x_offsets[row])
            context_end = row_start + self.contextual_seq_len
            history = int(history_lengths[row])
            target_start = context_end + history
            row_end = int(x_offsets[row + 1])
            context_indices = torch.arange(row_start, context_end, device=x.device)
            history_indices = (
                torch.nonzero(history_selected[row, :history], as_tuple=False).flatten()
                + context_end
            )
            target_indices = torch.arange(target_start, row_end, device=x.device)
            indices = torch.cat((context_indices, history_indices, target_indices))
            selected_indices.append(indices)
            selected_lengths.append(len(indices))
            write_masks.append(
                torch.cat(
                    (
                        torch.ones(
                            len(context_indices) + len(history_indices),
                            device=x.device,
                            dtype=torch.bool,
                        ),
                        torch.zeros(
                            len(target_indices), device=x.device, dtype=torch.bool
                        ),
                    )
                )
            )
        indices = torch.cat(selected_indices) if selected_indices else x.new_empty(0, dtype=torch.long)
        compact_x = x.index_select(0, indices.long())
        lengths = x_lengths.new_tensor(selected_lengths)
        offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
        gate = torch.cat(write_masks).to(x.dtype) if write_masks else x.new_empty(0)
        return compact_x, offsets, gate, indices.long()

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del max_kv_caching_len, kv_caching_lengths
        if not self._history_precompacted:
            self.last_total_tokens = len(x)
        if self.mode == "full_gdr":
            positions = self._packed_positions(x_lengths, x_offsets)
            target_start = torch.repeat_interleave(x_lengths - num_targets, x_lengths)
            gate = (positions < target_start).to(x.dtype)
            for layer in self.layers:
                x = layer.forward_gdr(x=x, x_offsets=x_offsets, event_gate=gate)
            self.last_selected_tokens = len(x)
            self.last_write_tokens = int(gate.sum())
            return x

        if self.mode in (
            "target_similarity_25",
            "rank_only_25",
            "recent_only",
            "rank_only",
        ):
            compact_x, compact_offsets, gate, indices = self._screening_compact(
                x,
                x_lengths,
                x_offsets,
                num_targets,
            )
            for layer in self.layers:
                compact_x = layer.forward_gdr(
                    x=compact_x,
                    x_offsets=compact_offsets,
                    event_gate=gate,
                )
            restored = torch.zeros_like(x)
            restored.index_copy_(0, indices, compact_x)
            self.last_selected_tokens = len(compact_x)
            self.last_write_tokens = int(gate.sum())
            return restored

        if self.mode == "deltarec" and self._history_precompacted:
            self._history_precompacted = False
            gate = self._precompacted_event_gate(
                x_lengths,
                x_offsets,
                num_targets,
                dtype=x.dtype,
            )
            self._precompacted_history_lengths = None
            self._precompacted_history_write_mask = None
            for layer in self.layers:
                x = layer.forward_gdr(x=x, x_offsets=x_offsets, event_gate=gate)
            self.last_selected_tokens = len(x)
            self.last_write_tokens = int(gate.sum())
            return x

        compacted = self._compact(x, x_lengths, max_seq_len, num_targets)
        selected = compacted.selected
        if self.mode == "dense_gated":
            gate = x.new_zeros(len(x))
            gate.index_copy_(
                0,
                selected.source_indices.long(),
                selected.write_mask.to(x.dtype),
            )
            for layer in self.layers:
                x = layer.forward_gdr(x=x, x_offsets=x_offsets, event_gate=gate)
            self.last_selected_tokens = len(selected.values)
            self.last_write_tokens = int(selected.write_mask.sum())
            return x
        compact_x = selected.payloads["x"]
        gate = selected.write_mask.to(compact_x.dtype)
        for layer in self.layers:
            compact_x = layer.forward_gdr(
                x=compact_x,
                x_offsets=selected.offsets,
                event_gate=gate,
            )
        restored = torch.zeros_like(x)
        restored.index_copy_(0, selected.source_indices.long(), compact_x)
        self.last_selected_tokens = len(compact_x)
        self.last_write_tokens = int(selected.write_mask.sum())
        return restored

    def cached_forward(
        self,
        delta_x: torch.Tensor,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "use the versioned DeltaRec cache write/read interface for cached serving"
        )

