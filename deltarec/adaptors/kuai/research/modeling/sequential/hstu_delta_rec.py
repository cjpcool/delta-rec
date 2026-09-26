# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""HSTU plugin for physically compacted and cached DeltaRec execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Optional

import torch
import torch.nn.functional as F

from deltarec.adaptors.kuai.research.modeling.sequential.delta_rec_cache import (
    DeltaRecBackboneAdapter,
    DeltaRecCacheVersion,
    DeltaRecExecutionOutput,
    GDRStepInput,
    TensorGDRStateCache,
    reference_gdr_read,
    reference_gdr_step,
)
from deltarec.adaptors.kuai.research.modeling.sequential.hstu import (
    HSTU,
    SequentialTransductionUnitJagged,
)
from deltarec.adaptors.kuai.research.modeling.sequential.selective_gdr import (
    GDRKernelInput,
    SelectedSequence,
)

if TYPE_CHECKING:
    from deltarec.adaptors.kuai.research.modeling.sequential.deltarec.projections import (
        HSTUFixedWidthLayout,
    )


class HSTUDeltaRecAdapter(DeltaRecBackboneAdapter):
    """Preserves HSTU projection/output semantics around sparse GDR writes."""

    def __init__(
        self,
        model: HSTU,
        *,
        cache_version: DeltaRecCacheVersion,
        step_backend: str = "reference",
        prefill_backend: str = "model",
        fused_binary_reads: bool = False,
        strict_positions: bool = True,
        validate_runtime: bool = True,
    ) -> None:
        super().__init__()
        if model._attention_backend != "gdr":
            raise ValueError("HSTUDeltaRecAdapter requires an HSTU-GDR model")
        if step_backend not in ("reference", "triton"):
            raise ValueError("step_backend must be 'reference' or 'triton'")
        if prefill_backend not in ("model", "triton"):
            raise ValueError("prefill_backend must be 'model' or 'triton'")
        self.model = model
        self.cache_version = cache_version
        self.step_backend = step_backend
        self.prefill_backend = prefill_backend
        self.fused_binary_reads = fused_binary_reads
        self.strict_positions = strict_positions
        self.validate_runtime = validate_runtime
        layers = list(model._hstu._attention_layers)
        if not layers:
            raise ValueError("HSTU-GDR must contain at least one layer")
        layout = (
            layers[0]._num_heads,
            layers[0]._attention_dim,
            layers[0]._linear_dim,
        )
        if any(
            (layer._num_heads, layer._attention_dim, layer._linear_dim) != layout
            for layer in layers
        ):
            raise ValueError("cached HSTU layers must share one GDR state layout")
        self._read_projection_names: list[str] = []
        for index, layer in enumerate(layers):
            if layer._gdr_kernel is not None and hasattr(
                layer._gdr_kernel, "validate_inputs"
            ):
                layer._gdr_kernel.validate_inputs = validate_runtime
            linear = layer._linear_dim * layer._num_heads
            attention = layer._attention_dim * layer._num_heads
            q_start = 2 * linear
            weight = torch.cat(
                [
                    layer._uvqk[:, :linear],
                    layer._uvqk[:, q_start : q_start + attention],
                ],
                dim=1,
            ).detach()
            name = f"_read_projection_{index}"
            self.register_buffer(name, weight, persistent=False)
            self._read_projection_names.append(name)

    @property
    def layers(self) -> list[SequentialTransductionUnitJagged]:
        return list(self.model._hstu._attention_layers)

    @staticmethod
    def _activate(layer: SequentialTransductionUnitJagged, value: torch.Tensor) -> torch.Tensor:
        if layer._linear_activation == "silu":
            return F.silu(value)
        if layer._linear_activation == "none":
            return value
        raise ValueError(f"unsupported HSTU linear activation {layer._linear_activation}")

    @staticmethod
    def _postprocess(
        layer: SequentialTransductionUnitJagged,
        x: torch.Tensor,
        u: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        attn_output = context.reshape(-1, layer._num_heads * layer._linear_dim)
        if layer._concat_ua:
            a = layer._norm_attn_output(attn_output)
            output_input = torch.cat([u, a, u * a], dim=-1)
        else:
            output_input = u * layer._norm_attn_output(attn_output)
        return layer._o(
            F.dropout(
                output_input,
                p=layer._dropout_ratio,
                training=layer.training,
            )
        ) + x

    @staticmethod
    def _postprocess_fixed_width(
        layer: SequentialTransductionUnitJagged,
        x: torch.Tensor,
        u: torch.Tensor,
        context: torch.Tensor,
        layout: "HSTUFixedWidthLayout",
    ) -> torch.Tensor:
        """Postprocess with a chunk-shape-invariant per-sequence GEMM.

        This is an inference-oriented numerical execution policy used by the
        candidate-symmetric project-once architecture.  It computes the same
        linear map as ``layer._o`` but fixes the GEMM reduction shape for each
        logical sequence, independent of candidate chunk width.
        """

        from deltarec.adaptors.kuai.research.modeling.sequential.deltarec.projections import (
            fixed_width_sequence_linear,
        )

        attn_output = context.reshape(-1, layer._num_heads * layer._linear_dim)
        if layer._concat_ua:
            a = layer._norm_attn_output(attn_output)
            output_input = torch.cat([u, a, u * a], dim=-1)
        else:
            output_input = u * layer._norm_attn_output(attn_output)
        output_input = F.dropout(
            output_input,
            p=layer._dropout_ratio,
            training=layer.training,
        )
        return fixed_width_sequence_linear(
            output_input,
            layer._o.weight.t(),
            layer._o.bias,
            layout,
        ) + x

    def _preprocess(
        self,
        item_ids: torch.Tensor,
        source_positions: torch.Tensor,
        payloads: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if item_ids.ndim != 1 or item_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("packed item_ids must be an integer vector")
        if source_positions.shape != item_ids.shape:
            raise ValueError("source_positions must align with item_ids")
        for name, payload in payloads.items():
            if payload.shape[0] != len(item_ids):
                raise ValueError(f"payload {name!r} must align with item_ids")
        embeddings = self.model.get_item_embeddings(item_ids.long())
        return self.model._input_features_preproc.forward_packed(
            item_ids.long(),
            embeddings,
            dict(payloads),
            source_positions,
            validate=self.validate_runtime,
        )

    def _project_mixed(
        self,
        layer: SequentialTransductionUnitJagged,
        x: torch.Tensor,
        write_mask: torch.Tensor,
        offsets: torch.Tensor,
        offsets_cpu: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, GDRKernelInput]:
        if write_mask.shape != (len(x),) or write_mask.dtype != torch.bool:
            raise ValueError("write_mask must be boolean and token-aligned")
        normed = layer._norm_input(x)
        linear = layer._linear_dim * layer._num_heads
        attention = layer._attention_dim * layer._num_heads
        projected = self._activate(layer, torch.mm(normed, layer._uvqk))
        u, v, q, k = torch.split(
            projected,
            [linear, linear, attention, attention],
            dim=-1,
        )
        assert layer._gdr_gate is not None
        decay_logits, beta_logits = layer._gdr_gate(normed).chunk(2, dim=-1)

        assert layer._gdr_log_decay_scale is not None
        assert layer._gdr_decay_bias is not None
        event_gate: Optional[torch.Tensor]
        if self.prefill_backend == "model" and self.fused_binary_reads:
            identity_logits = torch.full_like(decay_logits, float("-inf"))
            mask = write_mask[:, None]
            decay_logits = torch.where(mask, decay_logits, identity_logits)
            beta_logits = torch.where(mask, beta_logits, identity_logits)
            event_gate = None
        else:
            event_gate = write_mask.to(normed.dtype)
        return u, GDRKernelInput(
            q=q.view(-1, layer._num_heads, layer._attention_dim),
            k=k.view(-1, layer._num_heads, layer._attention_dim),
            v=v.view(-1, layer._num_heads, layer._linear_dim),
            decay_logits=decay_logits,
            beta_logits=beta_logits,
            log_decay_scale=layer._gdr_log_decay_scale,
            decay_bias=layer._gdr_decay_bias,
            offsets=offsets,
            event_gate=event_gate,
            offsets_cpu=offsets_cpu,
        )

    def _project_mixed_fixed_width(
        self,
        layer: SequentialTransductionUnitJagged,
        x: torch.Tensor,
        write_mask: torch.Tensor,
        offsets: torch.Tensor,
        layout: "HSTUFixedWidthLayout",
        offsets_cpu: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, GDRKernelInput]:
        """Project packed events with a fixed ``[S,D]`` GEMM per sequence."""

        from deltarec.adaptors.kuai.research.modeling.sequential.deltarec.projections import (
            fixed_width_sequence_linear,
        )

        if write_mask.shape != (len(x),) or write_mask.dtype != torch.bool:
            raise ValueError("write_mask must be boolean and token-aligned")
        layout_offsets = getattr(layout, "offsets", None)
        if not isinstance(layout_offsets, torch.Tensor) or (
            layout_offsets.shape != offsets.shape
        ):
            raise ValueError("fixed-width layout must align with packed offsets")
        if self.validate_runtime and not torch.equal(layout_offsets, offsets):
            raise ValueError("fixed-width layout and packed offsets disagree")

        normed = layer._norm_input(x)
        linear = layer._linear_dim * layer._num_heads
        attention = layer._attention_dim * layer._num_heads
        projected = self._activate(
            layer,
            fixed_width_sequence_linear(
                normed,
                layer._uvqk,
                None,
                layout,
            ),
        )
        u, v, q, k = torch.split(
            projected,
            [linear, linear, attention, attention],
            dim=-1,
        )
        assert layer._gdr_gate is not None
        gates = fixed_width_sequence_linear(
            normed,
            layer._gdr_gate.weight.t(),
            layer._gdr_gate.bias,
            layout,
        )
        decay_logits, beta_logits = gates.chunk(2, dim=-1)

        assert layer._gdr_log_decay_scale is not None
        assert layer._gdr_decay_bias is not None
        event_gate: Optional[torch.Tensor]
        if self.prefill_backend == "model" and self.fused_binary_reads:
            identity_logits = torch.full_like(decay_logits, float("-inf"))
            mask = write_mask[:, None]
            decay_logits = torch.where(mask, decay_logits, identity_logits)
            beta_logits = torch.where(mask, beta_logits, identity_logits)
            event_gate = None
        else:
            event_gate = write_mask.to(normed.dtype)
        return u, GDRKernelInput(
            q=q.view(-1, layer._num_heads, layer._attention_dim),
            k=k.view(-1, layer._num_heads, layer._attention_dim),
            v=v.view(-1, layer._num_heads, layer._linear_dim),
            decay_logits=decay_logits,
            beta_logits=beta_logits,
            log_decay_scale=layer._gdr_log_decay_scale,
            decay_bias=layer._gdr_decay_bias,
            offsets=offsets,
            event_gate=event_gate,
            offsets_cpu=offsets_cpu,
        )

    def _project_write_step(
        self,
        layer: SequentialTransductionUnitJagged,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, GDRStepInput]:
        normed = layer._norm_input(x)
        projected = self._activate(layer, torch.mm(normed, layer._uvqk))
        linear = layer._linear_dim * layer._num_heads
        attention = layer._attention_dim * layer._num_heads
        u, v, q, k = torch.split(
            projected,
            [linear, linear, attention, attention],
            dim=-1,
        )
        assert layer._gdr_gate is not None
        decay_logits, beta_logits = layer._gdr_gate(normed).chunk(2, dim=-1)
        assert layer._gdr_log_decay_scale is not None
        assert layer._gdr_decay_bias is not None
        return u, GDRStepInput(
            q=q.view(-1, layer._num_heads, layer._attention_dim),
            k=k.view(-1, layer._num_heads, layer._attention_dim),
            v=v.view(-1, layer._num_heads, layer._linear_dim),
            decay_logits=decay_logits,
            beta_logits=beta_logits,
            log_decay_scale=layer._gdr_log_decay_scale,
            decay_bias=layer._gdr_decay_bias,
        )

    def _project_read(
        self,
        layer: SequentialTransductionUnitJagged,
        x: torch.Tensor,
        layer_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed = layer._norm_input(x)
        linear = layer._linear_dim * layer._num_heads
        attention = layer._attention_dim * layer._num_heads
        projected = self._activate(
            layer,
            torch.mm(normed, getattr(self, self._read_projection_names[layer_index])),
        )
        u, q = torch.split(projected, [linear, attention], dim=-1)
        return u, q.view(-1, layer._num_heads, layer._attention_dim)

    def _validate_cache(self, cache: TensorGDRStateCache) -> None:
        first = self.layers[0]
        expected = (
            len(self.layers),
            first._num_heads,
            first._attention_dim,
            first._linear_dim,
        )
        if cache.states.shape[1:] != expected:
            raise ValueError("cache layout does not match the HSTU-GDR stack")

    def prefill(
        self,
        selected: SelectedSequence,
        *,
        initial_states: Optional[torch.Tensor] = None,
        return_final_states: bool = True,
    ) -> DeltaRecExecutionOutput:
        if self.validate_runtime:
            selected.validate()
        batch = selected.offsets.numel() - 1
        lengths = selected.offsets[1:] - selected.offsets[:-1]
        if self.validate_runtime and bool((lengths == 0).any()):
            raise ValueError("FLA packed prefill requires at least one selected event per request")
        layers = self.layers
        first = layers[0]
        expected_states = (
            batch,
            len(layers),
            first._num_heads,
            first._attention_dim,
            first._linear_dim,
        )
        if initial_states is not None:
            if initial_states.shape != expected_states or initial_states.dtype != torch.float32:
                raise ValueError("initial_states must use the packed FP32 cache layout")
        x = self._preprocess(
            selected.values,
            selected.source_positions,
            selected.payloads,
        )
        final_states: list[torch.Tensor] = []
        write_tokens = selected.write_mask.sum()
        for layer_index, layer in enumerate(layers):
            assert layer._gdr_kernel is not None
            u, projected = self._project_mixed(
                layer,
                x,
                selected.write_mask,
                selected.offsets,
                selected.offsets_cpu,
            )
            layer_initial_state = (
                None if initial_states is None else initial_states[:, layer_index]
            )
            if self.prefill_backend == "model":
                kernel_output = layer._gdr_kernel(
                    projected,
                    layer_initial_state,
                    return_final_state=return_final_states,
                )
            else:
                from deltarec.adaptors.kuai.ops.triton.triton_delta_rec import (
                    triton_gdr_packed_prefill,
                )

                kernel_output = triton_gdr_packed_prefill(
                    projected,
                    layer_initial_state,
                    return_final_state=return_final_states,
                )
            x = self._postprocess(layer, x, u, kernel_output.context)
            if return_final_states:
                assert kernel_output.final_state is not None
                final_states.append(kernel_output.final_state.float())
        output = self.model._output_postproc(x)
        return DeltaRecExecutionOutput(
            embeddings=output,
            layer_states=(torch.stack(final_states, dim=1) if final_states else None),
            auxiliary={
                "selected_tokens": selected.write_mask.new_tensor(len(selected.values)),
                "write_tokens": write_tokens,
                "read_tokens": len(selected.values) - write_tokens,
            },
        )

    @torch.no_grad()
    def write(
        self,
        item_ids: torch.Tensor,
        source_positions: torch.Tensor,
        payloads: Mapping[str, torch.Tensor],
        cache: TensorGDRStateCache,
        cache_slots: torch.Tensor,
    ) -> DeltaRecExecutionOutput:
        self._validate_cache(cache)
        if self.validate_runtime:
            cache._validate_slots(cache_slots, unique=True)
        if item_ids.shape != cache_slots.shape or source_positions.shape != cache_slots.shape:
            raise ValueError("online write fields must align with cache slots")
        cache.activate_version(self.cache_version)
        slots = cache_slots.long()
        hits = cache.valid.index_select(0, slots)
        if self.strict_positions and bool(hits.any()):
            expected_positions = cache.next_positions.index_select(0, slots)
            if bool((source_positions[hits].long() != expected_positions[hits]).any()):
                raise ValueError("online write source position is not the next cache position")
        x = self._preprocess(item_ids, source_positions, payloads)
        layer_states: list[torch.Tensor] = []
        for layer_index, layer in enumerate(self.layers):
            u, projected = self._project_write_step(layer, x)
            state = cache.states[:, layer_index].index_select(0, slots)
            state = torch.where(hits[:, None, None, None], state, torch.zeros_like(state))
            if self.step_backend == "reference":
                context, next_state = reference_gdr_step(projected, state)
                cache.states[:, layer_index].index_copy_(0, slots, next_state)
            else:
                from deltarec.adaptors.kuai.ops.triton.triton_delta_rec import (
                    triton_gdr_cache_write,
                )

                context = triton_gdr_cache_write(
                    projected,
                    cache.states,
                    slots,
                    layer_index,
                    initialize_mask=~hits,
                )
            x = self._postprocess(layer, x, u, context)
            if self.step_backend == "reference" or self.validate_runtime:
                if self.step_backend == "triton":
                    next_state = cache.states[:, layer_index].index_select(0, slots)
                layer_states.append(next_state)
        cache.mark_written(
            slots,
            source_positions.long() + 1,
            self.cache_version,
        )
        return DeltaRecExecutionOutput(
            embeddings=self.model._output_postproc(x),
            layer_states=(torch.stack(layer_states, dim=1) if layer_states else None),
            auxiliary={"write_tokens": len(item_ids), "read_tokens": 0},
        )

    @torch.no_grad()
    def read(
        self,
        item_ids: torch.Tensor,
        source_positions: torch.Tensor,
        payloads: Mapping[str, torch.Tensor],
        cache: TensorGDRStateCache,
        cache_slots: torch.Tensor,
    ) -> DeltaRecExecutionOutput:
        self._validate_cache(cache)
        if self.validate_runtime:
            cache._validate_slots(cache_slots, unique=True)
        if cache.version != self.cache_version:
            raise ValueError("cannot read a cache under a different version")
        slots = cache_slots.long()
        if self.validate_runtime and not bool(cache.valid.index_select(0, slots).all()):
            raise ValueError("cached read requires a valid state for every request")
        x = self._preprocess(item_ids, source_positions, payloads)
        checksum = (
            cache.states.index_select(0, slots).clone()
            if self.validate_runtime
            else None
        )
        for layer_index, layer in enumerate(self.layers):
            u, q = self._project_read(layer, x, layer_index)
            if self.step_backend == "reference":
                context = reference_gdr_read(
                    q,
                    cache.states[:, layer_index].index_select(0, slots),
                )
            else:
                from deltarec.adaptors.kuai.ops.triton.triton_delta_rec import (
                    triton_gdr_cache_read,
                )

                context = triton_gdr_cache_read(
                    q,
                    cache.states,
                    slots,
                    layer_index,
                )
            x = self._postprocess(layer, x, u, context)
        if checksum is not None and not torch.equal(
            cache.states.index_select(0, slots), checksum
        ):
            raise RuntimeError("cached read mutated recurrent state")
        return DeltaRecExecutionOutput(
            embeddings=self.model._output_postproc(x),
            layer_states=None,
            auxiliary={"write_tokens": 0, "read_tokens": len(item_ids)},
        )

