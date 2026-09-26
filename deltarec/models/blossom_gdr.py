"""BlossomRec backbone with the DeltaRec gated-delta-rule boundary.

The recurrent contract in this adapter is intentionally the same contract used
by the HSTU DeltaRec implementation: q/k are normalized in the kernel, state
updates are decay -> residual -> write -> read, state arithmetic is FP32, and
event gates interpolate the complete transition.  The only deliberately
different part is the BlossomRec shell around the recurrent boundary.

The selector owns a frozen copy of the item table because BlossomRec does not
expose the official HSTU weak-reference selector table.  The copy is bound
only from the training catalog and is never optimized with the recommender.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from deltarec.layers.group_selection import select_group_events
from deltarec.models.hstu_multitask import HSTUMultitaskHead
from deltarec.layers.gdr_kernels import (
    build_gdr_kernel,
)
from deltarec.layers.selective_gdr import (
    GDRKernelInput,
)


PRODUCTION_HASH_SEED = 20260818


class BlossomGDRLayer(nn.Module):
    """GDR recurrence with the original Blossom residual/FFN shell."""

    def __init__(
        self,
        original: nn.Module,
        width: int,
        heads: int,
        backend: str = "fla",
        *,
        seed: int = PRODUCTION_HASH_SEED,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("embedding width must divide the head count")
        if backend not in ("reference", "fla"):
            raise ValueError("Blossom GDR backend must be 'reference' or 'fla'")

        self.heads = int(heads)
        self.dim = int(width // heads)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        # Decay and beta gates are projected from the same normalized input as
        # the HSTU gate projection.  No bias is part of the registered GDR
        # contract.
        self.gates = nn.Linear(width, 2 * heads, bias=False)

        # Match the production HSTU initialization rather than the previous
        # Blossom-only zero/logspace initialization.  A per-layer seed keeps
        # the initialization deterministic without touching the global RNG.
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        decay_scale = torch.empty(heads).uniform_(0.0, 16.0, generator=generator)
        self.log_decay_scale = nn.Parameter(torch.log(decay_scale.clamp_min(1e-4)))
        dt = torch.exp(
            torch.rand(heads, generator=generator)
            * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp_min(1e-4)
        self.decay_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

        nn.init.normal_(self.qkv.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.gates.weight)

        # These modules are the one intentional BlossomRec backbone seam.
        attention = original.blossom_attention
        self.dense = attention.dense
        self.norm = attention.LayerNorm
        self.dropout = attention.out_dropout
        self.feed_forward = original.feed_forward
        self.kernel = build_gdr_kernel(backend)

    def finish(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Apply the unchanged Blossom output projection and residual shell."""

        y = self.dense(context.flatten(-2).to(dtype=x.dtype))
        y = self.norm(self.dropout(y) + x)
        return self.feed_forward(y)

    def project(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        projected = self.qkv(x).reshape(-1, self.heads, 3, self.dim)
        q, k, v = projected.unbind(dim=2)
        decay, beta = self.gates(x).chunk(2, dim=-1)
        return q, k, v, decay, beta

    def prefill(
        self,
        x: torch.Tensor,
        offsets: torch.Tensor,
        event_gate: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.prefill_projected(x, self.project(x), offsets, event_gate)

    def prefill_projected(
        self,
        x: torch.Tensor,
        projected_tensors: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        offsets: torch.Tensor,
        event_gate: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, v, decay, beta = projected_tensors
        projected = GDRKernelInput(
            q=q,
            k=k,
            v=v,
            decay_logits=decay,
            beta_logits=beta,
            log_decay_scale=self.log_decay_scale,
            decay_bias=self.decay_bias,
            offsets=offsets,
            event_gate=event_gate,
        )
        result = self.kernel(projected, return_final_state=True)
        context = result.context
        final_state = result.final_state.float()
        return self.finish(x, context), final_state

    def read(self, x: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Read a candidate query without changing the supplied state."""

        q = F.linear(x, self.qkv.weight)
        q = q.reshape(*x.shape[:-1], self.heads, 3, self.dim)[..., 0, :]
        q = q / torch.sqrt(q.square().sum(dim=-1, keepdim=True) + 1e-6)
        context = torch.einsum("bkhd,bkhdv->bkhv", q, states) * self.dim ** -0.5
        return self.finish(x, context)


class GroupCWISelector(nn.Module):
    """Official PC-selector feature contract evaluated per semantic group."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.input = nn.Linear(3 * width, width)
        self.output = nn.Linear(width, 1)

    def forward(
        self, history: torch.Tensor, prototypes: torch.Tensor
    ) -> torch.Tensor:
        # [B,L,D] x [G,D] -> [B,G,L,3D], matching the official
        # concat(event,candidate,event*candidate) PC selector features.
        event = history[:, None, :, :].expand(-1, prototypes.shape[0], -1, -1)
        group = prototypes[None, :, None, :].expand(
            history.shape[0], -1, history.shape[1], -1
        )
        features = torch.cat((event, group, event * group), dim=-1)
        hidden = F.silu(self.input(features))
        return self.output(hidden).squeeze(-1).float()


@dataclass(frozen=True)
class GroupHistory:
    states: tuple[torch.Tensor, ...]
    selected_counts: torch.Tensor
    source_lengths: torch.Tensor


class BlossomDeltaRec(nn.Module):
    """BlossomRec body with group-shared DeltaRec GDR states."""

    def __init__(
        self,
        upstream: nn.Module,
        item_to_group: torch.Tensor,
        group_count: int,
        *,
        backend: str = "fla",
        multitask: bool = False,
        retention_ratio: float = 0.25,
        seed: int = PRODUCTION_HASH_SEED,
    ) -> None:
        super().__init__()
        self.item_embedding = upstream.item_embedding
        self.embedding_dim = int(self.item_embedding.embedding_dim)
        self.input_norm = upstream.LayerNorm
        self.input_dropout = upstream.dropout
        self.group_count = int(group_count)
        # Direct/library use keeps input validation enabled.  Production
        # runners set this Python flag false after immutable binding checks so
        # the hot path does not read CUDA assertion scalars per batch.
        self.runtime_validation = True
        if retention_ratio not in (0.25, 0.50, 1.0):
            raise ValueError("retention_ratio must be one of 0.25, 0.50, or 1.0")
        self.retention_ratio = float(retention_ratio)
        if self.group_count < 1:
            raise ValueError("group_count must be positive")

        mapping = item_to_group.to(dtype=torch.long).clone()
        if mapping.numel() != self.item_embedding.num_embeddings:
            raise ValueError("category mapping must cover the exact item table")
        if bool((mapping < 0).any()) or bool((mapping >= self.group_count).any()):
            raise ValueError("category mapping contains an invalid group")
        self.register_buffer("item_to_group", mapping)

        self.layers = nn.ModuleList(
            [
                BlossomGDRLayer(
                    layer,
                    self.embedding_dim,
                    int(upstream.n_heads),
                    backend=backend,
                    seed=int(seed) + index,
                )
                for index, layer in enumerate(upstream.trm_encoder.layer)
            ]
        )

        # Selector state is frozen after binding.  The table is a runtime-only
        # snapshot: the training/checkpoint seam excludes it from shared and
        # node-local payloads and reconstructs it from the Full-GDR parent.
        # This keeps the feature space frozen without publishing a duplicate
        # multi-GB Kuai embedding in every artifact.
        self.register_buffer(
            "selector_embedding", torch.zeros_like(self.item_embedding.weight)
        )
        self.register_buffer("selector_rms", torch.ones(()))
        self.register_buffer("selector_bound", torch.tensor(False))
        self._selector_bound = False
        self.register_buffer(
            "prototypes",
            torch.zeros(self.group_count, self.embedding_dim),
        )
        self.selector = GroupCWISelector(self.embedding_dim)
        self.task_head = HSTUMultitaskHead(self.embedding_dim) if multitask else None

    @torch.no_grad()
    def bind_selector_space(
        self,
        training_catalog_ids: torch.Tensor,
        *,
        feature_table: Optional[torch.Tensor] = None,
    ) -> None:
        """Freeze the selector item space using training-catalog IDs only."""

        ids = training_catalog_ids.to(device=self.item_to_group.device, dtype=torch.long)
        if ids.ndim != 1:
            raise ValueError("training catalog IDs must be one-dimensional")
        if ids.numel() == 0 or bool((ids <= 0).any()) or bool(
            (ids >= self.item_to_group.numel()).any()
        ):
            raise ValueError("selector binding requires non-padding training items")

        table = self.item_embedding.weight.detach() if feature_table is None else feature_table
        table = table.to(device=self.item_embedding.weight.device,
                         dtype=self.item_embedding.weight.dtype)
        if table.shape != self.item_embedding.weight.shape:
            raise ValueError("selector feature table must match the item table shape")
        self.selector_embedding.copy_(table)
        rms = table[ids].float().square().mean().sqrt().clamp_min(1e-6)
        self.selector_rms.copy_(rms)
        groups = self.item_to_group.index_select(0, ids)
        sums = torch.zeros_like(self.prototypes)
        sums.index_add_(0, groups, table.index_select(0, ids))
        counts = torch.bincount(groups, minlength=self.group_count).clamp_min(1)
        self.prototypes.copy_(sums / counts[:, None].to(sums.dtype))
        self.selector_bound.fill_(True)
        self._selector_bound = True

    def selector_scores(self, histories: torch.Tensor) -> torch.Tensor:
        # Keep the serialized buffer for provenance, but avoid reading a CUDA
        # scalar on every sparse microbatch after binding.
        if not self._selector_bound:
            if bool(self.selector_bound.detach().cpu()):
                self._selector_bound = True
            else:
                raise RuntimeError("selector space has not been bound")
        history = self.selector_embedding[histories]
        scale = self.selector_rms.to(device=history.device, dtype=torch.float32)
        history = history.float() / scale
        prototypes = self.prototypes.float() / scale
        # Selector ordering is a feature-space operation, not the backbone's
        # mixed-precision hot path.  Keeping it in FP32 also makes a frozen
        # theta(T) snapshot invariant under an outer autocast context.
        with torch.autocast(device_type=history.device.type, enabled=False):
            return self.selector(history, prototypes).float()

    @staticmethod
    def _dense_layout(
        lengths: torch.Tensor,
        width: int,
        groups: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = lengths.shape[0]
        device = lengths.device
        positions = torch.arange(width, device=device)[None, None, :]
        valid = positions < lengths[:, None, None]
        source = (
            torch.arange(batch, device=device)[:, None, None] * width + positions
        ).expand(batch, groups, width)
        valid = valid.expand(batch, groups, width)
        packed = source[valid]
        counts = lengths[:, None].expand(batch, groups).long()
        flat_counts = counts.reshape(-1)
        offsets = torch.cat((flat_counts.new_zeros(1), flat_counts.cumsum(0)))
        return packed, offsets, counts, valid

    def prefill(
        self,
        histories: torch.Tensor,
        lengths: torch.Tensor,
        *,
        sparse: bool = True,
        event_gates: Optional[torch.Tensor] = None,
        single_group: bool = False,
        recent_floor: int = 32,
        validate_runtime: Optional[bool] = None,
    ) -> GroupHistory:
        if histories.ndim != 2 or lengths.ndim != 1 or lengths.shape[0] != histories.shape[0]:
            raise ValueError("histories and lengths must be [B,L] and [B]")
        width = histories.shape[1]
        batch = histories.shape[0]
        runtime_validation = (
            self.runtime_validation if validate_runtime is None else bool(validate_runtime)
        )
        if width < 1 or (runtime_validation and (
            bool((lengths < 1).any()) or bool((lengths > width).any())
        )):
            raise ValueError("nonempty histories required")
        if single_group and sparse:
            raise ValueError("single-group mode is only for dense Full-GDR execution")
        if single_group and event_gates is not None:
            raise ValueError("single-group Full-GDR execution does not accept CWI gates")

        if sparse:
            scores = self.selector_scores(histories)
            with torch.no_grad():
                selection = select_group_events(
                    scores,
                    lengths,
                    recent_floor=recent_floor,
                    retention_ratio=self.retention_ratio,
                    validate_runtime=runtime_validation,
                )
            packed_source_indices = selection.packed_source_indices
            group_offsets = selection.group_offsets
            counts = selection.counts
            packed_gates = None
            stream_groups = self.group_count
        else:
            stream_groups = 1 if single_group else self.group_count
            (
                packed_source_indices,
                group_offsets,
                counts,
                dense_valid,
            ) = self._dense_layout(lengths, width, stream_groups)
            if event_gates is not None:
                if event_gates.shape != (batch, self.group_count, width):
                    raise ValueError("CWI gates must be [B,G,valid_history_width]")
                if single_group:
                    raise ValueError("CWI gates require the dense multi-group path")
                packed_gates = event_gates[dense_valid]
            else:
                packed_gates = None

        embeddings = self.item_embedding(histories)
        hidden = self.input_dropout(self.input_norm(embeddings))
        source_hidden = hidden.reshape(batch * width, self.embedding_dim)
        states: list[torch.Tensor] = []
        packed_hidden: Optional[torch.Tensor] = None

        for layer_index, layer in enumerate(self.layers):
            if layer_index == 0:
                # The first projection is shared across group streams; only
                # the selected rows are physically gathered afterwards.
                projected = layer.project(source_hidden)
                projected = tuple(
                    tensor.index_select(0, packed_source_indices)
                    for tensor in projected
                )
                packed_hidden = source_hidden.index_select(0, packed_source_indices)
            else:
                assert packed_hidden is not None
                projected = layer.project(packed_hidden)
            assert packed_hidden is not None
            packed_hidden, state = layer.prefill_projected(
                packed_hidden,
                projected,
                group_offsets,
                packed_gates,
            )
            state = state.reshape(batch, stream_groups, *state.shape[1:])
            if single_group:
                state = state.expand(batch, self.group_count, *state.shape[2:])
            states.append(state)

        return GroupHistory(
            states=tuple(states),
            selected_counts=counts if not single_group else lengths[:, None].expand(batch, self.group_count),
            source_lengths=lengths,
        )

    def pair_embeddings(
        self,
        history: GroupHistory,
        candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if candidates.ndim != 2:
            raise ValueError("candidate IDs must be [B,K]")
        if candidates.shape[0] != history.source_lengths.shape[0]:
            raise ValueError("candidate batch does not match history")
        if self.runtime_validation and (
            bool((candidates <= 0).any())
            or bool((candidates >= self.item_to_group.numel()).any())
        ):
            raise ValueError("candidate ID outside catalog")
        groups = self.item_to_group[candidates]
        rows = torch.arange(candidates.shape[0], device=candidates.device)[:, None]
        rows = rows.expand_as(candidates)
        candidate_embedding = self.item_embedding(candidates)
        user_history_embedding = self.input_dropout(
            self.input_norm(candidate_embedding)
        )
        for layer, states in zip(self.layers, history.states):
            user_history_embedding = layer.read(
                user_history_embedding, states[rows, groups]
            )
        return user_history_embedding, candidate_embedding

    def read(self, history: GroupHistory, candidates: torch.Tensor) -> torch.Tensor:
        user_history_embedding, candidate_embedding = self.pair_embeddings(
            history, candidates
        )
        if self.task_head is not None:
            return self.task_head(user_history_embedding, candidate_embedding)
        return (
            F.normalize(user_history_embedding.float(), dim=-1, eps=1e-6)
            * F.normalize(candidate_embedding.float(), dim=-1, eps=1e-6)
        ).sum(dim=-1)

    def forward(
        self,
        histories: torch.Tensor,
        lengths: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        return self.read(self.prefill(histories, lengths), candidates)

    def catalog_loss(
        self,
        history: GroupHistory,
        targets: torch.Tensor,
        catalog_ids: torch.Tensor,
        *,
        chunk_size: int = 2048,
    ) -> torch.Tensor:
        """Exact training-catalog CE with candidate activations recomputed."""

        if self.task_head is not None:
            raise ValueError("catalog CE applies only to rating datasets")
        batch = targets.shape[0]
        normalizer = torch.full(
            (batch,), -torch.inf, device=targets.device, dtype=torch.float32
        )
        target_logits = torch.zeros_like(normalizer)
        found = torch.zeros(batch, dtype=torch.bool, device=targets.device)

        def score(candidates: torch.Tensor) -> torch.Tensor:
            return self.read(
                GroupHistory(
                    states=history.states,
                    selected_counts=history.selected_counts,
                    source_lengths=history.source_lengths,
                ),
                candidates,
            )

        for ids in catalog_ids.split(chunk_size):
            candidates = ids[None, :].expand(batch, -1)
            if self.training and torch.is_grad_enabled():
                logits = checkpoint(score, candidates, use_reentrant=False)
            else:
                logits = score(candidates)
            logits = logits.float()
            normalizer = torch.logaddexp(normalizer, logits.logsumexp(dim=-1))
            matches = ids[None, :] == targets[:, None]
            present = matches.any(dim=-1)
            values = logits.masked_fill(~matches, -torch.inf).max(dim=-1).values
            target_logits = torch.where(present, values, target_logits)
            found |= present
        if not bool(found.all()):
            raise ValueError("training target absent from training catalog")
        return (normalizer - target_logits).mean()


def cwi_distribution_loss(
    scores: torch.Tensor,
    importance: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Distill signed CWI utility as a per-group event distribution."""

    target = importance.detach().float().masked_fill(~valid, 0.0)
    count = valid.sum(dim=-1, keepdim=True).clamp_min(1)
    target = target - target.sum(dim=-1, keepdim=True) / count
    scale = target.square().sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    target = torch.softmax((target / scale).clamp(-10.0, 10.0), dim=-1)
    logp = F.log_softmax(
        scores.float().masked_fill(~valid, -torch.inf), dim=-1
    )
    return -(target * logp).sum(dim=-1).mean()


__all__ = [
    "BlossomDeltaRec",
    "BlossomGDRLayer",
    "GroupCWISelector",
    "GroupHistory",
    "cwi_distribution_loss",
]
