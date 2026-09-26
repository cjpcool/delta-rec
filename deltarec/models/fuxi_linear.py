"""FuXi-Linear DeltaRec adapter.

FuXi-Linear exposes an official sequence encoder rather than the HSTU state
transition API.  The adapter therefore keeps the published FuXi backbone
unchanged and places the DeltaRec boundary at its item-feature input: dense
Full-GDR uses the complete sequence, CWI uses differentiable per-group gates
on item contributions, and sparse execution compacts the selected events in
chronological order before calling the official encoder.  This is the
backbone-specific seam; selector/checkpoint semantics remain shared.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn
import torch.nn.functional as F

from deltarec.layers.group_selection import select_group_events
from deltarec.models.hstu_multitask import HSTUMultitaskHead, HSTURankingHead


class FuxiDeltaRecError(RuntimeError):
    pass


class FuxiGroupCWISelector(nn.Module):
    """Candidate/group-independent CWI MLP in the frozen item feature space."""

    def __init__(self, width: int, *, seed: int = 20260915) -> None:
        super().__init__()
        hidden = max(8, int(width))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.input = nn.Linear(3 * int(width), hidden)
            self.output = nn.Linear(hidden, 1)

    def forward(self, history: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        event = history[:, None, :, :].expand(-1, prototypes.shape[0], -1, -1)
        group = prototypes[None, :, None, :].expand(
            history.shape[0], -1, history.shape[1], -1
        )
        features = torch.cat((event, group, event * group), dim=-1)
        return self.output(F.silu(self.input(features))).squeeze(-1).float()


@dataclass(frozen=True)
class FuxiGroupHistory:
    """The final query for every user/group stream."""

    queries: torch.Tensor
    selected_counts: torch.Tensor
    source_lengths: torch.Tensor


class FuxiLinearDeltaRec(nn.Module):
    """Group-shared DeltaRec shell around the official FuXi-Linear model."""

    def __init__(
        self,
        backbone: nn.Module,
        item_to_group: torch.Tensor,
        group_count: int,
        *,
        retention_ratio: float = 0.25,
        recent_floor: int = 32,
        multitask: bool = False,
        seed: int = 20260915,
        backbone_input_length: Optional[int] = None,
        dense_dynamic_width: bool = False,
    ) -> None:
        super().__init__()
        if group_count < 1:
            raise ValueError("FuXi group_count must be positive")
        if retention_ratio not in (0.25, 0.50, 1.0):
            raise ValueError("retention_ratio must be one of 0.25, 0.50, or 1.0")
        if recent_floor < 0:
            raise ValueError("recent_floor must be nonnegative")
        self.backbone = backbone
        self.group_count = int(group_count)
        self.retention_ratio = float(retention_ratio)
        self.recent_floor = int(recent_floor)
        # Direct/library use keeps input validation enabled.  Production
        # runners set this Python flag false after immutable binding checks so
        # the hot path does not read CUDA assertion scalars per batch.
        self.runtime_validation = True
        self.dense_dynamic_width = bool(dense_dynamic_width)
        if backbone_input_length is not None and int(backbone_input_length) < 1:
            raise ValueError("FuXi backbone_input_length must be positive")
        self.backbone_input_length = (
            None if backbone_input_length is None else int(backbone_input_length)
        )
        self.embedding_dim = int(self.item_embedding.embedding_dim)
        mapping = item_to_group.to(dtype=torch.long).clone()
        if mapping.ndim != 1 or mapping.numel() != self.item_embedding.num_embeddings:
            raise ValueError("FuXi category mapping must cover the exact item table")
        if bool((mapping < 0).any()) or bool((mapping >= self.group_count).any()):
            raise ValueError("FuXi category mapping contains an invalid group")
        self.register_buffer("item_to_group", mapping)
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
        self.selector = FuxiGroupCWISelector(self.embedding_dim, seed=seed)
        self._head_seed = int(seed)
        self.task_head = HSTUMultitaskHead(self.embedding_dim) if multitask else None

    def initialize_ranking_head(self) -> None:
        """Attach the rating ranker only when Sparse finetuning starts."""

        if self.task_head is not None:
            return
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self._head_seed)
            head = HSTURankingHead(self.embedding_dim)
        self.task_head = head.to(device=self.item_embedding.weight.device)

    @property
    def item_embedding(self) -> nn.Embedding:
        """Expose the official local table without registering a duplicate alias."""

        embedding_module = getattr(self.backbone, "_embedding_module", None)
        if embedding_module is not None and hasattr(embedding_module, "_item_emb"):
            return embedding_module._item_emb
        value = getattr(self.backbone, "item_embedding", None)
        if isinstance(value, nn.Embedding):
            return value
        raise FuxiDeltaRecError("official FuXi model does not expose a local item embedding")

    @torch.no_grad()
    def bind_selector_space(
        self,
        training_catalog_ids: torch.Tensor,
        *,
        feature_table: Optional[torch.Tensor] = None,
    ) -> None:
        ids = training_catalog_ids.to(
            device=self.item_embedding.weight.device, dtype=torch.long
        )
        if ids.ndim != 1 or ids.numel() == 0:
            raise ValueError("FuXi selector binding requires a nonempty 1-D catalog")
        if bool((ids <= 0).any()) or bool((ids >= self.item_to_group.numel()).any()):
            raise ValueError("FuXi selector binding contains padding/out-of-range IDs")
        table = self.item_embedding.weight.detach() if feature_table is None else feature_table
        table = table.to(
            device=self.item_embedding.weight.device,
            dtype=self.item_embedding.weight.dtype,
        )
        if tuple(table.shape) != tuple(self.item_embedding.weight.shape):
            raise ValueError("FuXi selector feature table shape changed")
        self.selector_embedding.copy_(table)
        scale = table[ids].float().square().mean().sqrt().clamp_min(1e-6)
        self.selector_rms.copy_(scale)
        groups = self.item_to_group.index_select(0, ids)
        sums = torch.zeros_like(self.prototypes)
        sums.index_add_(0, groups, table.index_select(0, ids))
        counts = torch.bincount(groups, minlength=self.group_count).clamp_min(1)
        self.prototypes.copy_(sums / counts[:, None].to(sums.dtype))
        self.selector_bound.fill_(True)
        self._selector_bound = True

    def selector_scores(self, histories: torch.Tensor) -> torch.Tensor:
        # Keep the serialized buffer for provenance, but use a Python flag in
        # the hot sparse path so a CUDA scalar is not synchronized per batch.
        # A state-loaded adapter recovers the flag once, outside steady state.
        if not self._selector_bound:
            if bool(self.selector_bound.detach().cpu()):
                self._selector_bound = True
            else:
                raise RuntimeError("FuXi selector feature space has not been bound")
        scale = self.selector_rms.to(histories.device, dtype=torch.float32)
        history = self.selector_embedding[histories].float() / scale
        prototypes = self.prototypes.to(histories.device, dtype=torch.float32) / scale
        # Selector features are frozen theta(T) features; do not let an outer
        # backbone autocast context quantize their ordering scores.
        with torch.autocast(device_type=history.device.type, enabled=False):
            return self.selector(history, prototypes).float()

    def _encode(
        self,
        histories: torch.Tensor,
        lengths: torch.Tensor,
        *,
        timestamps: Optional[torch.Tensor] = None,
        event_gates: Optional[torch.Tensor] = None,
        pad_to_backbone_width: bool = True,
        preserve_output_slot: bool = False,
    ) -> torch.Tensor:
        if histories.ndim != 2 or lengths.ndim != 1 or lengths.shape[0] != histories.shape[0]:
            raise ValueError("FuXi histories/lengths must have shapes [B,L]/[B]")
        embeddings = self.backbone.get_item_embeddings(histories)
        if event_gates is not None:
            if event_gates.shape != histories.shape:
                raise ValueError("FuXi event gates must match [B,L] history shape")
            # This is the explicit FuXi backbone boundary.  The official
            # positional preprocessor and retention/channel stack remain
            # untouched; CWI gates only the contribution of each item feature.
            embeddings = embeddings * event_gates.to(dtype=embeddings.dtype).unsqueeze(-1)
        if timestamps is None:
            timestamps = torch.zeros_like(histories)
        # The published FuXi trainer appends max_output_length right-padding
        # slots before calling encode.  Its positional table and causal mask
        # are constructed for that total width, even though ``lengths`` only
        # counts real history events.  Keep the exact boundary contract here:
        # append zero IDs/timestamps/embeddings and leave lengths unchanged.
        if self.backbone_input_length is not None and histories.shape[1] > self.backbone_input_length:
            raise ValueError("FuXi history width exceeds the official encoder input width")
        target_width = self.backbone_input_length if pad_to_backbone_width else None
        if preserve_output_slot and not pad_to_backbone_width:
            # The temporal channel queries the *next* timestamp. Keep a zero
            # output slot, and at least three slots for its q[:, -2:-1] tail.
            target_width = max(3, histories.shape[1] + 1)
            if self.backbone_input_length is not None:
                target_width = min(target_width, self.backbone_input_length)
        if target_width is not None:
            if histories.shape[1] > target_width:
                raise ValueError(
                    "FuXi history width exceeds the official encoder input width"
                )
            if histories.shape[1] < target_width:
                pad = target_width - histories.shape[1]
                histories = F.pad(histories, (0, pad))
                embeddings = F.pad(embeddings, (0, 0, 0, pad))
                timestamps = F.pad(timestamps, (0, pad))
        elif histories.shape[1] == 1:
            # The official retention kernel needs at least two physical slots;
            # lengths still exclude this zero padding token.
            histories = F.pad(histories, (0, 1))
            embeddings = F.pad(embeddings, (0, 0, 0, 1))
            timestamps = F.pad(timestamps, (0, 1))
        # The pinned FuXi source constructs its causal mask and positional
        # channel at the configured maximum width.  Sparse execution is still
        # mathematically valid at a shorter prefix, but those two tensors must
        # expose matching prefix views for the duration of the call.  Directly
        # installing a view in ``_parameters`` preserves gradient flow to the
        # original full positional parameter, which is restored immediately.
        execution_width = histories.shape[1]
        chunk_size = getattr(self.backbone, "chunk_size", None)
        if chunk_size is not None:
            execution_width = (
                (execution_width + int(chunk_size) - 1) // int(chunk_size)
            ) * int(chunk_size)
        original_mask = getattr(self.backbone, "_attn_mask", None)
        positional_parameters: list[tuple[nn.Module, torch.Tensor]] = []
        if original_mask is not None and original_mask.shape[-1] != execution_width:
            self.backbone._buffers["_attn_mask"] = original_mask[
                :execution_width, :execution_width
            ]
        layers = getattr(getattr(self.backbone, "_fuxi", None), "_attention_layers", ())
        for layer in layers:
            channel = getattr(layer, "_channel_p", None)
            positional = getattr(channel, "_emb", None)
            if positional is not None and positional.shape[0] != execution_width:
                positional_parameters.append((channel, positional))
                channel._parameters["_emb"] = positional[:execution_width]
        try:
            return self.backbone.encode(
                # FBGEMM cumsum requires contiguous input, including expanded counts.
                past_lengths=lengths.contiguous(),
                past_ids=histories,
                past_embeddings=embeddings,
                past_payloads={"timestamps": timestamps},
            )
        finally:
            if original_mask is not None:
                self.backbone._buffers["_attn_mask"] = original_mask
            for channel, positional in positional_parameters:
                channel._parameters["_emb"] = positional

    @staticmethod
    def _compact(
        histories: torch.Tensor,
        indices: torch.Tensor,
        counts: torch.Tensor,
        timestamps: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, groups, width = indices.shape
        source = histories[:, None, :].expand(batch, groups, histories.shape[1])
        selected = source.gather(2, indices.clamp_min(0))
        if timestamps is None:
            selected_timestamps = torch.zeros_like(selected)
        else:
            time_source = timestamps[:, None, :].expand(batch, groups, timestamps.shape[1])
            selected_timestamps = time_source.gather(2, indices.clamp_min(0))
        return selected.reshape(batch * groups, width), counts.reshape(-1), selected_timestamps.reshape(batch * groups, width)

    def prefill(
        self,
        histories: torch.Tensor,
        lengths: torch.Tensor,
        *,
        sparse: bool = True,
        event_gates: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
        single_group: bool = False,
        recent_floor: Optional[int] = None,
        validate_runtime: Optional[bool] = None,
    ) -> FuxiGroupHistory:
        if histories.ndim != 2 or lengths.ndim != 1 or lengths.shape[0] != histories.shape[0]:
            raise ValueError("FuXi histories and lengths must be [B,L] and [B]")
        runtime_validation = (
            self.runtime_validation if validate_runtime is None else bool(validate_runtime)
        )
        if runtime_validation and (
            bool((lengths < 1).any()) or bool((lengths > histories.shape[1]).any())
        ):
            raise ValueError("FuXi histories must be nonempty valid prefixes")
        if timestamps is not None and timestamps.shape != histories.shape:
            raise ValueError("FuXi timestamps must match history shape")
        batch, width = histories.shape
        if single_group and sparse:
            raise ValueError("single_group is only valid for dense Full-GDR")
        if single_group and event_gates is not None:
            raise ValueError("single_group cannot receive CWI gates")
        if sparse:
            scores = self.selector_scores(histories)
            with torch.no_grad():
                selection = select_group_events(
                    scores,
                    lengths,
                    recent_floor=self.recent_floor if recent_floor is None else recent_floor,
                    retention_ratio=self.retention_ratio,
                    validate_runtime=runtime_validation,
                )
            ids, counts, selected_timestamps = self._compact(
                histories,
                selection.indices,
                selection.counts,
                timestamps,
            )
            queries = self._encode(
                ids,
                counts,
                timestamps=selected_timestamps,
                pad_to_backbone_width=False,
            )
            queries = queries.reshape(batch, self.group_count, self.embedding_dim)
            return FuxiGroupHistory(
                queries=queries,
                selected_counts=selection.counts,
                source_lengths=lengths,
            )
        groups = 1 if single_group else self.group_count
        if event_gates is not None:
            if event_gates.shape != (batch, self.group_count, width):
                raise ValueError("FuXi CWI gates must have shape [B,G,L]")
            ids = histories[:, None, :].expand(batch, self.group_count, width).reshape(batch * self.group_count, width)
            flat_lengths = lengths[:, None].expand(batch, self.group_count).reshape(-1)
            flat_times = (
                torch.zeros_like(histories) if timestamps is None else timestamps
            )[:, None, :].expand(batch, self.group_count, width).reshape(batch * self.group_count, width)
            queries = self._encode(
                ids,
                flat_lengths,
                timestamps=flat_times,
                event_gates=event_gates.reshape(batch * self.group_count, width),
            ).reshape(batch, self.group_count, self.embedding_dim)
        else:
            query = self._encode(
                histories, lengths, timestamps=timestamps,
                pad_to_backbone_width=not self.dense_dynamic_width,
                preserve_output_slot=self.dense_dynamic_width,
            )
            queries = query[:, None, :].expand(batch, self.group_count, self.embedding_dim)
        selected_counts = lengths[:, None].expand(batch, self.group_count)
        return FuxiGroupHistory(queries=queries, selected_counts=selected_counts, source_lengths=lengths)

    def pair_embeddings(
        self,
        history: FuxiGroupHistory,
        candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if candidates.ndim != 2 or candidates.shape[0] != history.source_lengths.shape[0]:
            raise ValueError("FuXi candidates must be [B,K]")
        if self.runtime_validation and (
            bool((candidates <= 0).any())
            or bool((candidates >= self.item_to_group.numel()).any())
        ):
            raise ValueError("FuXi candidate is outside the item table")
        groups = self.item_to_group[candidates]
        rows = torch.arange(candidates.shape[0], device=candidates.device)[:, None].expand_as(candidates)
        user_history_embedding = history.queries[rows, groups]
        candidate_embedding = self.item_embedding(candidates)
        return user_history_embedding, candidate_embedding

    def read(self, history: FuxiGroupHistory, candidates: torch.Tensor) -> torch.Tensor:
        user_history_embedding, candidate_embedding = self.pair_embeddings(
            history, candidates
        )
        if self.task_head is not None:
            return self.task_head(user_history_embedding, candidate_embedding)
        return (
            F.normalize(user_history_embedding.float(), dim=-1, eps=1e-6)
            * F.normalize(candidate_embedding.float(), dim=-1, eps=1e-6)
        ).sum(-1)

    def forward(self, histories: torch.Tensor, lengths: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        return self.read(self.prefill(histories, lengths), candidates)

    def catalog_loss(
        self,
        history: FuxiGroupHistory,
        targets: torch.Tensor,
        catalog_ids: torch.Tensor,
        *,
        chunk_size: int = 2048,
    ) -> torch.Tensor:
        if self.task_head is not None:
            raise ValueError("catalog loss is only defined for rating runs")
        batch = targets.shape[0]
        normalizer = torch.full((batch,), -torch.inf, device=targets.device)
        target_logits = torch.zeros_like(normalizer)
        found = torch.zeros(batch, dtype=torch.bool, device=targets.device)
        for ids in catalog_ids.split(chunk_size):
            candidates = ids[None, :].expand(batch, -1)
            logits = self.read(history, candidates).float()
            normalizer = torch.logaddexp(normalizer, logits.logsumexp(-1))
            match = ids[None, :] == targets[:, None]
            target_logits = torch.where(match.any(-1), logits.masked_fill(~match, -torch.inf).max(-1).values, target_logits)
            found |= match.any(-1)
        if not bool(found.all()):
            raise ValueError("FuXi training target absent from training catalog")
        return (normalizer - target_logits).mean()

    def base_model_state(self) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu().clone() for key, value in self.backbone.state_dict().items()}

    def load_base_model_state(self, state: Any) -> None:
        if not isinstance(state, dict):
            state = dict(state)
        if "backbone" in state and isinstance(state["backbone"], dict):
            state = state["backbone"]
        expected = set(self.backbone.state_dict())
        observed = set(state)
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        if missing or unexpected:
            raise FuxiDeltaRecError(
                f"FuXi checkpoint state coverage changed: missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        self.backbone.load_state_dict(state, strict=True)

    def freeze_selector(self) -> None:
        self.selector.requires_grad_(False)
        self.selector.eval()
        for parameter in self.selector.parameters():
            parameter.grad = None


__all__ = [
    "FuxiDeltaRecError",
    "FuxiGroupCWISelector",
    "FuxiGroupHistory",
    "FuxiLinearDeltaRec",
]
