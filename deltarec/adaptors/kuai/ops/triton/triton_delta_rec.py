# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""A100-oriented serving kernels for DeltaRec selection and recurrent cache."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from deltarec.adaptors.kuai.research.modeling.sequential.delta_rec import (
    CompactionOutput,
    CompiledWriteSelector,
    DeltaRecCompactor,
    DenseEventBatch,
    DualBudgetState,
    SelectionConstraints,
    ThresholdWritePolicy,
)
from deltarec.adaptors.kuai.research.modeling.sequential.delta_rec_cache import (
    GDRStepInput,
)
from deltarec.adaptors.kuai.research.modeling.sequential.selective_gdr import (
    GDRKernelInput,
    GDRKernelOutput,
    SelectedSequence,
    SelectionPlan,
)


@triton.jit
def _selector_threshold_kernel(
    Item,
    Rating,
    TimeGap,
    Position,
    Popularity,
    Repeat,
    Lengths,
    ItemTable,
    RatingTable,
    TimeGapTable,
    PositionTable,
    PopularityTable,
    RepeatTable,
    LayerNormWeight,
    LayerNormBias,
    HiddenWeight,
    HiddenBias,
    OutputWeight,
    OutputBias,
    UtilityScale,
    Threshold,
    ForceWrite,
    ForceRead,
    RequireOutput,
    Scores,
    SelectedMask,
    WriteMask,
    SelectedTileCounts,
    CandidateWriteTileCounts,
    CandidateTileCounts,
    T: tl.constexpr,
    EMBEDDING_DIM: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    BLOCK_EVENTS: tl.constexpr,
    HAS_CONSTRAINTS: tl.constexpr,
    STORE_SCORES: tl.constexpr,
):
    batch = tl.program_id(0)
    tile = tl.program_id(1)
    positions = tile * BLOCK_EVENTS + tl.arange(0, BLOCK_EVENTS)
    length = tl.load(Lengths + batch)
    valid = (positions < T) & (positions < length)
    dense_indices = batch * T + positions

    item = tl.load(Item + dense_indices, mask=valid, other=0).to(tl.int64)
    rating = tl.load(Rating + dense_indices, mask=valid, other=0).to(tl.int64)
    time_gap = tl.load(TimeGap + dense_indices, mask=valid, other=0).to(tl.int64)
    position = tl.load(Position + dense_indices, mask=valid, other=0).to(tl.int64)
    popularity = tl.load(Popularity + dense_indices, mask=valid, other=0).to(tl.int64)
    repeat = tl.load(Repeat + dense_indices, mask=valid, other=0).to(tl.int64)

    dims = tl.arange(0, EMBEDDING_DIM)
    table_stride = EMBEDDING_DIM + 1
    interaction = tl.load(ItemTable + item[:, None] * table_stride + dims[None, :])
    interaction += tl.load(
        RatingTable + rating[:, None] * table_stride + dims[None, :]
    )
    interaction += tl.load(
        TimeGapTable + time_gap[:, None] * table_stride + dims[None, :]
    )
    interaction += tl.load(
        PositionTable + position[:, None] * table_stride + dims[None, :]
    )
    interaction += tl.load(
        PopularityTable + popularity[:, None] * table_stride + dims[None, :]
    )
    interaction += tl.load(
        RepeatTable + repeat[:, None] * table_stride + dims[None, :]
    )

    base = tl.load(ItemTable + item * table_stride + EMBEDDING_DIM)
    base += tl.load(RatingTable + rating * table_stride + EMBEDDING_DIM)
    base += tl.load(TimeGapTable + time_gap * table_stride + EMBEDDING_DIM)
    base += tl.load(PositionTable + position * table_stride + EMBEDDING_DIM)
    base += tl.load(PopularityTable + popularity * table_stride + EMBEDDING_DIM)
    base += tl.load(RepeatTable + repeat * table_stride + EMBEDDING_DIM)

    mean = tl.sum(interaction, axis=1) / EMBEDDING_DIM
    centered = interaction - mean[:, None]
    variance = tl.sum(centered * centered, axis=1) / EMBEDDING_DIM
    normalized = centered * tl.rsqrt(variance[:, None] + 1e-5)
    normalized = normalized * tl.load(LayerNormWeight + dims)[None, :]
    normalized += tl.load(LayerNormBias + dims)[None, :]

    hidden_dims = tl.arange(0, HIDDEN_DIM)
    hidden_weight = tl.load(
        HiddenWeight
        + dims[:, None] * HIDDEN_DIM
        + hidden_dims[None, :]
    )
    hidden = tl.dot(normalized, hidden_weight, allow_tf32=False)
    hidden += tl.load(HiddenBias + hidden_dims)[None, :]
    hidden = hidden * tl.sigmoid(hidden)
    interaction_score = tl.sum(
        hidden * tl.load(OutputWeight + hidden_dims)[None, :],
        axis=1,
    ) + tl.load(OutputBias)
    score = (base + interaction_score) * tl.load(UtilityScale)
    threshold = tl.load(Threshold)

    if HAS_CONSTRAINTS:
        force_write = tl.load(ForceWrite + dense_indices, mask=valid, other=0).to(tl.int1)
        force_read = tl.load(ForceRead + dense_indices, mask=valid, other=0).to(tl.int1)
        require_output = tl.load(
            RequireOutput + dense_indices, mask=valid, other=0
        ).to(tl.int1)
    else:
        force_write = tl.zeros((BLOCK_EVENTS,), dtype=tl.int1)
        force_read = tl.zeros((BLOCK_EVENTS,), dtype=tl.int1)
        require_output = tl.zeros((BLOCK_EVENTS,), dtype=tl.int1)

    predicted_write = score > threshold
    write = valid & (predicted_write | force_write) & ~force_read
    selected = write | (valid & (force_read | require_output))
    candidate = valid & ~(force_write | force_read)

    tl.store(SelectedMask + dense_indices, selected, mask=positions < T)
    tl.store(WriteMask + dense_indices, write, mask=positions < T)
    if STORE_SCORES:
        tl.store(Scores + dense_indices, score, mask=valid)
    tile_offset = batch * tl.num_programs(1) + tile
    tl.store(SelectedTileCounts + tile_offset, tl.sum(selected.to(tl.int32), axis=0))
    tl.store(
        CandidateWriteTileCounts + tile_offset,
        tl.sum((write & candidate).to(tl.int32), axis=0),
    )
    tl.store(CandidateTileCounts + tile_offset, tl.sum(candidate.to(tl.int32), axis=0))


@triton.jit
def _stable_compact_kernel(
    Item,
    Scores,
    SelectedMask,
    WriteMask,
    SourceOffsets,
    TileBases,
    ValuesOut,
    PackedIndicesOut,
    DenseIndicesOut,
    PositionsOut,
    WriteMaskOut,
    ScoresOut,
    T: tl.constexpr,
    BLOCK_EVENTS: tl.constexpr,
):
    batch = tl.program_id(0)
    tile = tl.program_id(1)
    positions = tile * BLOCK_EVENTS + tl.arange(0, BLOCK_EVENTS)
    dense_indices = batch * T + positions
    in_bounds = positions < T
    selected = tl.load(SelectedMask + dense_indices, mask=in_bounds, other=0).to(tl.int1)
    local_rank = tl.cumsum(selected.to(tl.int32), axis=0) - 1
    tile_offset = batch * tl.num_programs(1) + tile
    destinations = tl.load(TileBases + tile_offset) + local_rank
    source_start = tl.load(SourceOffsets + batch)
    mask = in_bounds & selected

    tl.store(ValuesOut + destinations, tl.load(Item + dense_indices, mask=mask), mask=mask)
    tl.store(PackedIndicesOut + destinations, source_start + positions, mask=mask)
    tl.store(DenseIndicesOut + destinations, dense_indices, mask=mask)
    tl.store(PositionsOut + destinations, positions, mask=mask)
    tl.store(
        WriteMaskOut + destinations,
        tl.load(WriteMask + dense_indices, mask=mask),
        mask=mask,
    )
    tl.store(
        ScoresOut + destinations,
        tl.load(Scores + dense_indices, mask=mask),
        mask=mask,
    )


class TritonThresholdCompactor(DeltaRecCompactor):
    """One learned score, one threshold, and stable chronological compaction.

    Fixed-size Triton tiles are scheduling units only. They do not impose a
    chunk budget and never run Top-K or sorting.
    """

    def __init__(
        self,
        selector: CompiledWriteSelector,
        policy: ThresholdWritePolicy,
        *,
        block_events: int = 16,
        validate: bool = False,
    ) -> None:
        super().__init__()
        if selector.embedding_dim != 16 or selector.hidden_dim != 32:
            raise ValueError("the A100 selector kernel currently requires dimensions 16/32")
        if block_events != 16:
            raise ValueError("the A100 selector kernel currently requires block_events=16")
        self.selector = selector
        self.policy = policy
        self.block_events = block_events
        self.validate = validate
        self.profile_stages = False
        self._last_stage_events = None
        self.register_buffer(
            "hidden_weight_t",
            selector.hidden_weight.detach().t().contiguous(),
        )
        self.register_buffer(
            "frozen_threshold",
            torch.tensor(float(policy.threshold), dtype=torch.float32),
        )

    @staticmethod
    def _require_cuda_contiguous(name: str, value: torch.Tensor) -> None:
        if not value.is_cuda or not value.is_contiguous():
            raise ValueError(f"{name} must be a contiguous CUDA tensor")

    @torch.no_grad()
    def compact(
        self,
        source: DenseEventBatch,
        *,
        constraints: Optional[SelectionConstraints] = None,
        policy_state: Optional[DualBudgetState] = None,
        return_scores: bool = False,
    ) -> CompactionOutput:
        if self.validate:
            source.validate(self.selector.schema)
        fields = source.categorical
        for name, value in fields.items():
            self._require_cuda_contiguous(name, value)
        self._require_cuda_contiguous("lengths", source.lengths)
        batch, time = source.shape
        if batch < 1 or time < 1:
            raise ValueError("the Triton compactor requires nonempty dense dimensions")
        if constraints is not None:
            if self.validate:
                constraints.validate(source.item_ids.shape)
            for name, value in (
                ("force_write_mask", constraints.force_write_mask),
                ("force_read_mask", constraints.force_read_mask),
                ("require_output_mask", constraints.require_output_mask),
            ):
                self._require_cuda_contiguous(name, value)

        device = source.item_ids.device
        tiles = triton.cdiv(time, self.block_events)
        selected_mask = torch.empty((batch, time), dtype=torch.bool, device=device)
        write_mask = torch.empty_like(selected_mask)
        scores_dense = torch.empty((batch, time), dtype=torch.float32, device=device)
        selected_tile_counts = torch.empty((batch, tiles), dtype=torch.int32, device=device)
        candidate_write_counts = torch.empty_like(selected_tile_counts)
        candidate_counts = torch.empty_like(selected_tile_counts)
        threshold = (
            self.frozen_threshold
            if policy_state is None
            else policy_state.threshold.to(device=device, dtype=torch.float32)
        )
        if threshold.ndim != 0:
            raise ValueError("policy threshold must be scalar")
        dummy = source.item_ids
        force_write = dummy if constraints is None else constraints.force_write_mask
        force_read = dummy if constraints is None else constraints.force_read_mask
        require_output = dummy if constraints is None else constraints.require_output_mask
        tables = self.selector.tables

        stage_events = None
        if self.profile_stages:
            stage_events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            stage_events[0].record()
        _selector_threshold_kernel[(batch, tiles)](
            source.item_ids,
            source.rating_buckets,
            source.time_gap_buckets,
            source.position_buckets,
            source.popularity_buckets,
            source.repeat_buckets,
            source.lengths,
            tables["item"],
            tables["rating"],
            tables["time_gap"],
            tables["position"],
            tables["popularity"],
            tables["repeat"],
            self.selector.layer_norm_weight,
            self.selector.layer_norm_bias,
            self.hidden_weight_t,
            self.selector.hidden_bias,
            self.selector.output_weight.reshape(-1),
            self.selector.output_bias,
            self.selector.utility_scale,
            threshold,
            force_write,
            force_read,
            require_output,
            scores_dense,
            selected_mask,
            write_mask,
            selected_tile_counts,
            candidate_write_counts,
            candidate_counts,
            T=time,
            EMBEDDING_DIM=16,
            HIDDEN_DIM=32,
            BLOCK_EVENTS=self.block_events,
            HAS_CONSTRAINTS=constraints is not None,
            STORE_SCORES=True,
            num_warps=4,
        )
        if stage_events is not None:
            stage_events[1].record()

        selected_counts = selected_tile_counts.sum(dim=1, dtype=torch.int64)
        selected_offsets = torch.cat(
            [selected_counts.new_zeros(1), selected_counts.cumsum(0)]
        )
        source_lengths = source.lengths.to(torch.int64)
        source_offsets = torch.cat(
            [source_lengths.new_zeros(1), source_lengths.cumsum(0)]
        )
        tile_prefix = selected_tile_counts.to(torch.int64).cumsum(dim=1)
        tile_bases = (
            selected_offsets[:-1, None]
            + tile_prefix
            - selected_tile_counts.to(torch.int64)
        ).contiguous()

        # This is the one intentional device-to-host synchronization. Its
        # result sizes exact output storage and is reused by FLA as CPU offsets.
        sync_payload = torch.cat(
            [
                selected_offsets,
                candidate_write_counts.sum().reshape(1).to(torch.int64),
                candidate_counts.sum().reshape(1).to(torch.int64),
            ]
        ).cpu()
        if stage_events is not None:
            stage_events[2].record()
        selected_offsets_cpu = sync_payload[: batch + 1].contiguous()
        selected_total = int(selected_offsets_cpu[-1])
        current_writes = int(sync_payload[batch + 1])
        current_decisions = int(sync_payload[batch + 2])

        values = torch.empty(selected_total, dtype=source.item_ids.dtype, device=device)
        packed_indices = torch.empty(selected_total, dtype=torch.int64, device=device)
        dense_indices = torch.empty_like(packed_indices)
        source_positions = torch.empty_like(packed_indices)
        packed_write_mask = torch.empty(selected_total, dtype=torch.bool, device=device)
        selected_scores = torch.empty(selected_total, dtype=torch.float32, device=device)
        if selected_total:
            _stable_compact_kernel[(batch, tiles)](
                source.item_ids,
                scores_dense,
                selected_mask,
                write_mask,
                source_offsets,
                tile_bases,
                values,
                packed_indices,
                dense_indices,
                source_positions,
                packed_write_mask,
                selected_scores,
                T=time,
                BLOCK_EVENTS=self.block_events,
                num_warps=1,
            )
        if stage_events is not None:
            stage_events[3].record()
            self._last_stage_events = stage_events

        payloads = {
            name: value.reshape(batch * time, *value.shape[2:]).index_select(
                0, dense_indices
            )
            for name, value in source.payloads.items()
        }
        next_threshold = threshold.detach().float()
        if self.policy.dual_step > 0 and current_decisions:
            observed_rate = current_writes / current_decisions
            next_threshold = torch.clamp(
                next_threshold
                + self.policy.dual_step
                * (observed_rate - self.policy.target_write_rate),
                min=0.0,
                max=self.policy.max_threshold,
            )
        next_state = DualBudgetState(
            threshold=next_threshold,
            decisions=(0 if policy_state is None else policy_state.decisions)
            + current_decisions,
            writes=(0 if policy_state is None else policy_state.writes) + current_writes,
        )
        plan = SelectionPlan(
            indices=packed_indices,
            offsets=selected_offsets,
            source_positions=source_positions,
            write_mask=packed_write_mask,
            scores=selected_scores,
            offsets_cpu=selected_offsets_cpu,
        )
        selected = SelectedSequence(
            values=values.long(),
            offsets=selected_offsets,
            payloads=payloads,
            source_indices=packed_indices,
            source_positions=source_positions,
            write_mask=packed_write_mask,
            offsets_cpu=selected_offsets_cpu,
        )
        packed_scores = None
        if return_scores:
            packed_scores = scores_dense[source.valid_mask]
        if self.validate:
            selected.validate()
        return CompactionOutput(
            selected=selected,
            plan=plan,
            policy_state=next_state,
            scores=packed_scores,
        )

    def last_stage_times_ms(self) -> dict[str, float]:
        """Resolve optional CUDA stage timings after the caller synchronizes."""

        if self._last_stage_events is None:
            return {}
        start, selector_done, allocation_done, compact_done = self._last_stage_events
        return {
            "selector_threshold_ms": start.elapsed_time(selector_done),
            "allocation_prefix_sync_ms": selector_done.elapsed_time(allocation_done),
            "stable_compaction_ms": allocation_done.elapsed_time(compact_done),
        }


@triton.jit
def _gdr_cache_write_kernel(
    Q,
    K,
    V,
    DecayLogits,
    BetaLogits,
    LogDecayScale,
    DecayBias,
    State,
    Slots,
    InitializeMask,
    Context,
    LayerIndex,
    stride_qb,
    stride_qh,
    stride_qk,
    stride_kb,
    stride_kh,
    stride_kk,
    stride_vb,
    stride_vh,
    stride_vv,
    stride_db,
    stride_dh,
    stride_bb,
    stride_bh,
    stride_sc,
    stride_sl,
    stride_sh,
    stride_sk,
    stride_sv,
    stride_cb,
    stride_ch,
    stride_cv,
    KEY_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_KEY: tl.constexpr,
    BLOCK_VALUE: tl.constexpr,
    EPS: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    value_tile = tl.program_id(2)
    key_offsets = tl.arange(0, BLOCK_KEY)
    value_offsets = value_tile * BLOCK_VALUE + tl.arange(0, BLOCK_VALUE)
    key_mask = key_offsets < KEY_DIM
    value_mask = value_offsets < VALUE_DIM

    q = tl.load(
        Q + batch * stride_qb + head * stride_qh + key_offsets * stride_qk,
        mask=key_mask,
        other=0.0,
    ).to(tl.float32)
    k = tl.load(
        K + batch * stride_kb + head * stride_kh + key_offsets * stride_kk,
        mask=key_mask,
        other=0.0,
    ).to(tl.float32)
    q *= tl.rsqrt(tl.sum(q * q, axis=0) + EPS)
    k *= tl.rsqrt(tl.sum(k * k, axis=0) + EPS)

    slot = tl.load(Slots + batch).to(tl.int64)
    state_ptrs = (
        State
        + slot * stride_sc
        + LayerIndex * stride_sl
        + head * stride_sh
        + key_offsets[:, None] * stride_sk
        + value_offsets[None, :] * stride_sv
    )
    state = tl.load(
        state_ptrs,
        mask=key_mask[:, None] & value_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    initialize = tl.load(InitializeMask + batch).to(tl.int1)
    state = tl.where(initialize, 0.0, state)

    decay_input = tl.load(
        DecayLogits + batch * stride_db + head * stride_dh
    ).to(tl.float32) + tl.load(DecayBias + head).to(tl.float32)
    softplus = tl.maximum(decay_input, 0.0) + tl.log(
        1.0 + tl.exp(-tl.abs(decay_input))
    )
    decay = tl.exp(-tl.exp(tl.load(LogDecayScale + head)) * softplus)
    beta = tl.sigmoid(
        tl.load(BetaLogits + batch * stride_bb + head * stride_bh).to(tl.float32)
    )
    decayed = decay * state
    prediction = tl.sum(decayed * k[:, None], axis=0)
    value = tl.load(
        V + batch * stride_vb + head * stride_vh + value_offsets * stride_vv,
        mask=value_mask,
        other=0.0,
    ).to(tl.float32)
    residual = value - prediction
    next_state = decayed + beta * k[:, None] * residual[None, :]
    context = tl.sum(q[:, None] * next_state, axis=0) * (KEY_DIM ** -0.5)
    tl.store(
        state_ptrs,
        next_state,
        mask=key_mask[:, None] & value_mask[None, :],
    )
    tl.store(
        Context
        + batch * stride_cb
        + head * stride_ch
        + value_offsets * stride_cv,
        context,
        mask=value_mask,
    )


@triton.jit
def _gdr_cache_read_kernel(
    Q,
    State,
    Slots,
    Context,
    LayerIndex,
    stride_qb,
    stride_qh,
    stride_qk,
    stride_sc,
    stride_sl,
    stride_sh,
    stride_sk,
    stride_sv,
    stride_cb,
    stride_ch,
    stride_cv,
    KEY_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_KEY: tl.constexpr,
    BLOCK_VALUE: tl.constexpr,
    EPS: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    value_tile = tl.program_id(2)
    key_offsets = tl.arange(0, BLOCK_KEY)
    value_offsets = value_tile * BLOCK_VALUE + tl.arange(0, BLOCK_VALUE)
    key_mask = key_offsets < KEY_DIM
    value_mask = value_offsets < VALUE_DIM
    q = tl.load(
        Q + batch * stride_qb + head * stride_qh + key_offsets * stride_qk,
        mask=key_mask,
        other=0.0,
    ).to(tl.float32)
    q *= tl.rsqrt(tl.sum(q * q, axis=0) + EPS)
    slot = tl.load(Slots + batch).to(tl.int64)
    state = tl.load(
        State
        + slot * stride_sc
        + LayerIndex * stride_sl
        + head * stride_sh
        + key_offsets[:, None] * stride_sk
        + value_offsets[None, :] * stride_sv,
        mask=key_mask[:, None] & value_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    context = tl.sum(q[:, None] * state, axis=0) * (KEY_DIM ** -0.5)
    tl.store(
        Context
        + batch * stride_cb
        + head * stride_ch
        + value_offsets * stride_cv,
        context,
        mask=value_mask,
    )


def _validate_cache_tensors(
    q: torch.Tensor,
    state: torch.Tensor,
    slots: torch.Tensor,
    layer_index: int,
) -> tuple[int, int, int, int]:
    if not q.is_cuda or not state.is_cuda or not slots.is_cuda:
        raise ValueError("Triton GDR cache execution requires CUDA tensors")
    if q.ndim != 3 or state.ndim != 5:
        raise ValueError("Q and cache state must have shapes [B,H,K] and [C,L,H,K,V]")
    batch, heads, key_dim = q.shape
    if state.shape[2:4] != (heads, key_dim):
        raise ValueError("Q and cache state layouts do not match")
    if state.dtype != torch.float32:
        raise ValueError("cached recurrent states must be FP32")
    if slots.shape != (batch,) or slots.dtype not in (torch.int32, torch.int64):
        raise ValueError("cache slots must be an integer vector with shape [B]")
    if not 0 <= layer_index < state.shape[1]:
        raise ValueError("layer_index is outside the cache")
    return batch, heads, key_dim, state.shape[-1]


@torch.no_grad()
def triton_gdr_cache_write(
    projected: GDRStepInput,
    state: torch.Tensor,
    slots: torch.Tensor,
    layer_index: int,
    *,
    initialize_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    block_value: int = 16,
) -> torch.Tensor:
    batch, heads, key_dim, value_dim = _validate_cache_tensors(
        projected.q, state, slots, layer_index
    )
    if projected.k.shape != projected.q.shape or projected.v.shape != (
        batch,
        heads,
        value_dim,
    ):
        raise ValueError("projected K/V shapes do not match the cache layout")
    if projected.decay_logits.shape != (batch, heads) or projected.beta_logits.shape != (
        batch,
        heads,
    ):
        raise ValueError("step gate tensors must have shape [B, H]")
    if projected.log_decay_scale.shape != (heads,) or projected.decay_bias.shape != (
        heads,
    ):
        raise ValueError("persistent GDR gates must have shape [H]")
    if projected.k.device != state.device or projected.v.device != state.device:
        raise ValueError("projected tensors and cache must share a CUDA device")
    if initialize_mask is None:
        initialize_mask = torch.zeros(batch, dtype=torch.bool, device=state.device)
    if initialize_mask.shape != (batch,) or initialize_mask.dtype != torch.bool:
        raise ValueError("initialize_mask must be boolean with shape [B]")
    context = torch.empty_like(projected.v)
    block_key = triton.next_power_of_2(key_dim)
    _gdr_cache_write_kernel[(batch, heads, triton.cdiv(value_dim, block_value))](
        projected.q,
        projected.k,
        projected.v,
        projected.decay_logits,
        projected.beta_logits,
        projected.log_decay_scale,
        projected.decay_bias,
        state,
        slots,
        initialize_mask,
        context,
        layer_index,
        *projected.q.stride(),
        *projected.k.stride(),
        *projected.v.stride(),
        *projected.decay_logits.stride(),
        *projected.beta_logits.stride(),
        *state.stride(),
        *context.stride(),
        KEY_DIM=key_dim,
        VALUE_DIM=value_dim,
        BLOCK_KEY=block_key,
        BLOCK_VALUE=block_value,
        EPS=eps,
        num_warps=4,
    )
    return context


@torch.no_grad()
def triton_gdr_cache_read(
    q: torch.Tensor,
    state: torch.Tensor,
    slots: torch.Tensor,
    layer_index: int,
    *,
    eps: float = 1e-6,
    block_value: int = 16,
) -> torch.Tensor:
    batch, heads, key_dim, value_dim = _validate_cache_tensors(
        q, state, slots, layer_index
    )
    context = torch.empty(
        batch,
        heads,
        value_dim,
        dtype=q.dtype,
        device=q.device,
    )
    block_key = triton.next_power_of_2(key_dim)
    _gdr_cache_read_kernel[(batch, heads, triton.cdiv(value_dim, block_value))](
        q,
        state,
        slots,
        context,
        layer_index,
        *q.stride(),
        *state.stride(),
        *context.stride(),
        KEY_DIM=key_dim,
        VALUE_DIM=value_dim,
        BLOCK_KEY=block_key,
        BLOCK_VALUE=block_value,
        EPS=eps,
        num_warps=4,
    )
    return context


@triton.jit
def _gdr_packed_prefill_kernel(
    Q,
    K,
    V,
    DecayLogits,
    BetaLogits,
    LogDecayScale,
    DecayBias,
    Offsets,
    EventGate,
    InitialState,
    Context,
    FinalState,
    stride_qt,
    stride_qh,
    stride_qk,
    stride_kt,
    stride_kh,
    stride_kk,
    stride_vt,
    stride_vh,
    stride_vv,
    stride_dt,
    stride_dh,
    stride_bt,
    stride_bh,
    stride_ib,
    stride_ih,
    stride_ik,
    stride_iv,
    stride_ct,
    stride_ch,
    stride_cv,
    stride_fb,
    stride_fh,
    stride_fk,
    stride_fv,
    KEY_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_KEY: tl.constexpr,
    BLOCK_VALUE: tl.constexpr,
    EPS: tl.constexpr,
    HAS_EVENT_GATE: tl.constexpr,
    HAS_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    request = tl.program_id(0)
    head = tl.program_id(1)
    value_tile = tl.program_id(2)
    key_offsets = tl.arange(0, BLOCK_KEY)
    value_offsets = value_tile * BLOCK_VALUE + tl.arange(0, BLOCK_VALUE)
    key_mask = key_offsets < KEY_DIM
    value_mask = value_offsets < VALUE_DIM
    state_mask = key_mask[:, None] & value_mask[None, :]

    if HAS_INITIAL_STATE:
        state = tl.load(
            InitialState
            + request * stride_ib
            + head * stride_ih
            + key_offsets[:, None] * stride_ik
            + value_offsets[None, :] * stride_iv,
            mask=state_mask,
            other=0.0,
        ).to(tl.float32)
    else:
        state = tl.zeros((BLOCK_KEY, BLOCK_VALUE), dtype=tl.float32)

    token = tl.load(Offsets + request).to(tl.int64)
    end = tl.load(Offsets + request + 1).to(tl.int64)
    while token < end:
        q = tl.load(
            Q + token * stride_qt + head * stride_qh + key_offsets * stride_qk,
            mask=key_mask,
            other=0.0,
        ).to(tl.float32)
        q *= tl.rsqrt(tl.sum(q * q, axis=0) + EPS)
        if HAS_EVENT_GATE:
            write = tl.load(EventGate + token) > 0.5
        else:
            write = True

        k = tl.load(
            K + token * stride_kt + head * stride_kh + key_offsets * stride_kk,
            mask=key_mask,
            other=0.0,
        ).to(tl.float32)
        k *= tl.rsqrt(tl.sum(k * k, axis=0) + EPS)
        decay_input = tl.load(
            DecayLogits + token * stride_dt + head * stride_dh
        ).to(tl.float32) + tl.load(DecayBias + head).to(tl.float32)
        softplus = tl.maximum(decay_input, 0.0) + tl.log(
            1.0 + tl.exp(-tl.abs(decay_input))
        )
        decay = tl.exp(-tl.exp(tl.load(LogDecayScale + head)) * softplus)
        beta = tl.sigmoid(
            tl.load(BetaLogits + token * stride_bt + head * stride_bh).to(tl.float32)
        )
        decayed = decay * state
        prediction = tl.sum(decayed * k[:, None], axis=0)
        value = tl.load(
            V + token * stride_vt + head * stride_vh + value_offsets * stride_vv,
            mask=value_mask,
            other=0.0,
        ).to(tl.float32)
        candidate = decayed + beta * k[:, None] * (value - prediction)[None, :]
        state = tl.where(write, candidate, state)
        context = tl.sum(q[:, None] * state, axis=0) * (KEY_DIM ** -0.5)
        tl.store(
            Context
            + token * stride_ct
            + head * stride_ch
            + value_offsets * stride_cv,
            context,
            mask=value_mask,
        )
        token += 1

    if STORE_FINAL_STATE:
        tl.store(
            FinalState
            + request * stride_fb
            + head * stride_fh
            + key_offsets[:, None] * stride_fk
            + value_offsets[None, :] * stride_fv,
            state,
            mask=state_mask,
        )


@torch.no_grad()
def triton_gdr_packed_prefill(
    projected: GDRKernelInput,
    initial_state: Optional[torch.Tensor] = None,
    *,
    return_final_state: bool = False,
    eps: float = 1e-6,
    block_value: int = 16,
) -> GDRKernelOutput:
    """Serving-only recurrent packed GDR specialized for small A100 batches."""

    q, k, v = projected.q, projected.k, projected.v
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("Triton packed GDR requires CUDA Q/K/V")
    if q.ndim != 3 or k.shape != q.shape or v.ndim != 3 or v.shape[:2] != q.shape[:2]:
        raise ValueError("packed Q/K/V shapes are incompatible")
    tokens, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if projected.decay_logits.shape != (tokens, heads) or projected.beta_logits.shape != (
        tokens,
        heads,
    ):
        raise ValueError("packed GDR gates must have shape [tokens, heads]")
    if projected.log_decay_scale.shape != (heads,) or projected.decay_bias.shape != (
        heads,
    ):
        raise ValueError("persistent GDR gates must have shape [heads]")
    offsets = projected.offsets
    if offsets.device != q.device or offsets.ndim != 1:
        raise ValueError("packed offsets must be a CUDA vector")
    batch = offsets.numel() - 1
    if initial_state is not None:
        if initial_state.shape != (batch, heads, key_dim, value_dim):
            raise ValueError("initial_state does not match packed GDR layout")
        if initial_state.dtype != torch.float32 or initial_state.device != q.device:
            raise ValueError("initial_state must be FP32 on the Q device")
    if projected.event_gate is not None and projected.event_gate.shape != (tokens,):
        raise ValueError("event_gate must align with packed tokens")
    context = torch.empty_like(v)
    final_state = (
        torch.empty(
            batch,
            heads,
            key_dim,
            value_dim,
            dtype=torch.float32,
            device=q.device,
        )
        if return_final_state
        else None
    )
    dummy = q
    initial = dummy if initial_state is None else initial_state
    event_gate = dummy if projected.event_gate is None else projected.event_gate
    final = dummy if final_state is None else final_state
    block_key = triton.next_power_of_2(key_dim)
    _gdr_packed_prefill_kernel[
        (batch, heads, triton.cdiv(value_dim, block_value))
    ](
        q,
        k,
        v,
        projected.decay_logits,
        projected.beta_logits,
        projected.log_decay_scale,
        projected.decay_bias,
        offsets,
        event_gate,
        initial,
        context,
        final,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *projected.decay_logits.stride(),
        *projected.beta_logits.stride(),
        *(initial_state.stride() if initial_state is not None else (0, 0, 0, 0)),
        *context.stride(),
        *(final_state.stride() if final_state is not None else (0, 0, 0, 0)),
        KEY_DIM=key_dim,
        VALUE_DIM=value_dim,
        BLOCK_KEY=block_key,
        BLOCK_VALUE=block_value,
        EPS=eps,
        HAS_EVENT_GATE=projected.event_gate is not None,
        HAS_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=return_final_state,
        num_warps=4,
    )
    return GDRKernelOutput(context=context, final_state=final_state)

