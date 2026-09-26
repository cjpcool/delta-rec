from __future__ import annotations

from dataclasses import dataclass

from typing import Any, Mapping

import weakref

import torch

import torch.nn.functional as F

from torch import nn

class SparseSelectorContractError(RuntimeError):
    """Raised when an official sparse-selector dependency is substitutable."""

@dataclass(frozen=True)
class VerifiedBoundGrouping:
    dataset: str
    group_count: int
    item_to_category_group: torch.Tensor
    category_group_prototypes: torch.Tensor
    evidence: Mapping[str, Any]

@dataclass(frozen=True)
class CandidateSelection:
    scores: torch.Tensor
    write_mask: torch.Tensor
    budgets: torch.Tensor
    valid_history: torch.Tensor
    actual_write_ratio: torch.Tensor

@dataclass(frozen=True)
class GroupSelection:
    candidate_scores: torch.Tensor
    group_scores: torch.Tensor
    write_mask: torch.Tensor
    budgets: torch.Tensor
    valid_history: torch.Tensor
    actual_write_ratio: torch.Tensor
    candidate_group_ids: torch.Tensor
    group_sizes: torch.Tensor

def exact_budget_write_mask(
    scores: torch.Tensor,
    history_lengths: torch.Tensor,
    *,
    retention_ratio: float,
    recent_floor: int = 32,
) -> CandidateSelection:
    """Select an exact deterministic Top-B mask on every logical state."""

    if scores.ndim != 3 or not torch.is_floating_point(scores):
        raise ValueError("selector scores must be floating point [B,S,L]")
    batch, states, width = scores.shape
    if batch < 1 or states < 1 or width < 1:
        raise ValueError("selector scores must contain requests, states, and events")
    if (
        history_lengths.shape != (batch,)
        or history_lengths.dtype not in (torch.int32, torch.int64)
        or history_lengths.device != scores.device
        or bool((history_lengths < 1).any())
        or bool((history_lengths > width).any())
    ):
        raise ValueError("history_lengths must address nonempty prefixes of scores")
    if not 0.0 < float(retention_ratio) <= 1.0:
        raise ValueError("retention_ratio must lie in (0,1]")
    if isinstance(recent_floor, bool) or not isinstance(recent_floor, int) or recent_floor < 1:
        raise ValueError("recent_floor must be a positive integer")
    positions = torch.arange(width, device=scores.device)
    valid = positions[None, :] < history_lengths[:, None]
    if not bool(torch.isfinite(scores.masked_select(valid[:, None, :])).all()):
        raise ValueError("valid selector scores must be finite")
    # This is a protocol-visible numerical choice.  Keep it bit-for-bit aligned
    # with ``deltarec.headline_v1.budget.exact_budget``: convert lengths and the
    # registered ratio to FP32 *before* multiplication, then take ceil.
    lengths_f32 = history_lengths.to(torch.float32)
    ratio_f32 = torch.tensor(
        float(retention_ratio), dtype=torch.float32, device=scores.device
    )
    ratio_budget = torch.ceil(lengths_f32 * ratio_f32).to(torch.int64)
    floor_budget = torch.minimum(
        history_lengths.to(torch.int64),
        history_lengths.new_full((batch,), recent_floor, dtype=torch.int64),
    )
    per_request = torch.maximum(ratio_budget, floor_budget)
    budgets = per_request[:, None].expand(batch, states).contiguous()

    # Stable descending sort makes equal-score events retain the frozen
    # chronological/original-row order (the smaller event index wins).
    safe = scores.float().masked_fill(~valid[:, None, :], float("-inf"))
    order = torch.argsort(safe, dim=-1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    rank_values = torch.arange(width, device=scores.device, dtype=order.dtype)
    ranks.scatter_(-1, order, rank_values.view(1, 1, width).expand_as(order))
    mask = (ranks < budgets[:, :, None]) & valid[:, None, :]
    counts = mask.sum(dim=-1)
    if not torch.equal(counts, budgets):
        raise RuntimeError("exact-budget selector did not write exactly B events")
    actual_ratio = budgets.to(torch.float64) / history_lengths[:, None].to(
        torch.float64
    )
    return CandidateSelection(
        scores=scores.float(),
        write_mask=mask,
        budgets=budgets,
        valid_history=valid,
        actual_write_ratio=actual_ratio,
    )

class _SharedOfficialEmbeddingSelector(nn.Module):
    def __init__(
        self,
        shared_embedding: nn.Embedding,
        *,
        expected_num_items: int,
        expected_embedding_dim: int,
        freeze_embedding_snapshot: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(shared_embedding, nn.Embedding):
            raise TypeError("official selector embedding must be nn.Embedding")
        if (
            shared_embedding.num_embeddings != expected_num_items + 1
            or shared_embedding.embedding_dim != expected_embedding_dim
            or shared_embedding.padding_idx not in {None, 0}
        ):
            raise SparseSelectorContractError(
                "official selector embedding shape/padding contract changed"
            )
        if freeze_embedding_snapshot:
            snapshot = nn.Embedding.from_pretrained(
                shared_embedding.weight.detach().clone(),
                freeze=True,
                padding_idx=shared_embedding.padding_idx,
            )
            self.add_module("_selector_embedding_snapshot", snapshot)
            object.__setattr__(self, "_official_embedding_ref", None)
        else:
            object.__setattr__(
                self, "_official_embedding_ref", weakref.ref(shared_embedding)
            )
        self.num_items = int(expected_num_items)
        self.embedding_dim = int(expected_embedding_dim)

    @property
    def official_embedding(self) -> nn.Embedding:
        snapshot = getattr(self, "_selector_embedding_snapshot", None)
        if snapshot is not None:
            return snapshot
        reference = self._official_embedding_ref
        value = None if reference is None else reference()
        if value is None:
            raise RuntimeError("official HSTU embedding owner has been released")
        return value

    def lookup(self, item_ids: torch.Tensor) -> torch.Tensor:
        if item_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("selector item IDs must be integer")
        if item_ids.numel() and (
            int(item_ids.min()) < 0 or int(item_ids.max()) > self.num_items
        ):
            raise ValueError("selector item ID is outside the official table")
        return self.official_embedding(item_ids.to(torch.int64))

    @staticmethod
    def _validate_ids(
        history_item_ids: torch.Tensor,
        candidate_item_ids: torch.Tensor,
        history_lengths: torch.Tensor,
    ) -> tuple[int, int, int]:
        if history_item_ids.ndim != 2 or candidate_item_ids.ndim != 2:
            raise ValueError("selector requires history [B,L] and candidates [B,K]")
        batch, width = history_item_ids.shape
        if candidate_item_ids.shape[0] != batch or candidate_item_ids.shape[1] < 1:
            raise ValueError("selector candidate batch is malformed")
        if (
            history_lengths.shape != (batch,)
            or history_lengths.dtype not in (torch.int32, torch.int64)
            or history_lengths.device != history_item_ids.device
            or candidate_item_ids.device != history_item_ids.device
            or bool((history_lengths < 1).any())
            or bool((history_lengths > width).any())
        ):
            raise ValueError("selector history lengths are malformed")
        positions = torch.arange(width, device=history_item_ids.device)
        if bool(
            history_item_ids.masked_select(
                positions[None, :] >= history_lengths[:, None]
            ).ne(0).any()
        ):
            raise ValueError("selector histories must use right-zero padding")
        return batch, int(candidate_item_ids.shape[1]), width

class RatingPCSelector(_SharedOfficialEmbeddingSelector):
    """Candidate-aware CWI MLP over the exact official item embedding."""

    def __init__(
        self,
        shared_embedding: nn.Embedding,
        *,
        expected_num_items: int,
        expected_embedding_dim: int,
        seed: int,
        freeze_embedding_snapshot: bool = False,
    ) -> None:
        super().__init__(
            shared_embedding,
            expected_num_items=expected_num_items,
            expected_embedding_dim=expected_embedding_dim,
            freeze_embedding_snapshot=freeze_embedding_snapshot,
        )
        self.seed = int(seed)
        hidden = max(8, expected_embedding_dim)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.input = nn.Linear(3 * expected_embedding_dim, hidden)
            self.output = nn.Linear(hidden, 1)
        if shared_embedding.weight.device.type != "meta":
            self.input.to(device=shared_embedding.weight.device)
            self.output.to(device=shared_embedding.weight.device)

    def exact_scores_from_embeddings(
        self,
        history_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if (
            history_embeddings.ndim != 3
            or candidate_embeddings.ndim != 3
            or history_embeddings.shape[0] != candidate_embeddings.shape[0]
            or history_embeddings.shape[2] != self.embedding_dim
            or candidate_embeddings.shape[2] != self.embedding_dim
        ):
            raise ValueError("PC embeddings must be [B,L,D] and [B,K,D]")
        events = history_embeddings[:, None, :, :]
        candidates = candidate_embeddings[:, :, None, :]
        shape = (
            history_embeddings.shape[0],
            candidate_embeddings.shape[1],
            history_embeddings.shape[1],
            self.embedding_dim,
        )
        features = torch.cat(
            (
                events.expand(shape),
                candidates.expand(shape),
                events.expand(shape) * candidates.expand(shape),
            ),
            dim=-1,
        )
        parameter = self.input.weight
        features = features.to(device=parameter.device, dtype=parameter.dtype)
        return self.output(F.silu(self.input(features))).squeeze(-1).float()

    def score_ids(
        self,
        history_item_ids: torch.Tensor,
        candidate_item_ids: torch.Tensor,
        history_lengths: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_ids(history_item_ids, candidate_item_ids, history_lengths)
        return self.exact_scores_from_embeddings(
            self.lookup(history_item_ids), self.lookup(candidate_item_ids)
        )

    def select(
        self,
        history_item_ids: torch.Tensor,
        candidate_item_ids: torch.Tensor,
        history_lengths: torch.Tensor,
        *,
        retention_ratio: float,
        recent_floor: int = 32,
    ) -> CandidateSelection:
        return exact_budget_write_mask(
            self.score_ids(history_item_ids, candidate_item_ids, history_lengths),
            history_lengths,
            retention_ratio=retention_ratio,
            recent_floor=recent_floor,
        )

class RatingGCSelector(nn.Module):
    """Pool exact PC scores into fixed category groups and select one mask/group."""

    def __init__(
        self,
        pc_selector: RatingPCSelector,
        grouping: VerifiedBoundGrouping,
    ) -> None:
        super().__init__()
        if grouping.dataset not in {"ml-20m", "amazon-books"}:
            raise SparseSelectorContractError("GC grouping is not a rating artifact")
        if grouping.item_to_category_group.shape != (pc_selector.num_items + 1,):
            raise SparseSelectorContractError("GC mapping and official item table disagree")
        if grouping.category_group_prototypes.shape[1] != pc_selector.embedding_dim:
            raise SparseSelectorContractError(
                "GC prototypes and official selector width disagree"
            )
        self.pc_selector = pc_selector
        self.pc_selector.requires_grad_(False)
        self.group_count = int(grouping.group_count)
        owner_device = pc_selector.official_embedding.weight.device
        buffer_device = None if owner_device.type == "meta" else owner_device
        self.register_buffer(
            "item_to_category_group",
            grouping.item_to_category_group.detach()
            .to(device=buffer_device, dtype=torch.int64)
            .contiguous(),
        )
        self.register_buffer(
            "category_group_prototypes",
            grouping.category_group_prototypes.detach()
            .to(device=buffer_device, dtype=torch.float32)
            .contiguous(),
        )
        self.grouping_evidence = dict(grouping.evidence)

    def group_ids(self, candidate_item_ids: torch.Tensor) -> torch.Tensor:
        if candidate_item_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("GC candidate IDs must be integer")
        ids = candidate_item_ids.to(torch.int64)
        if self.item_to_category_group.device != ids.device:
            raise ValueError("GC mapping and candidate IDs must share a device")
        in_range = (ids >= 0) & (ids < self.item_to_category_group.numel())
        safe = ids.clamp(0, self.item_to_category_group.numel() - 1)
        mapped = self.item_to_category_group.index_select(0, safe.reshape(-1)).reshape_as(
            ids
        )
        known = in_range & (mapped >= 0) & (mapped < self.group_count)
        return torch.where(known, mapped, mapped.new_zeros(mapped.shape))

    def select(
        self,
        history_item_ids: torch.Tensor,
        candidate_item_ids: torch.Tensor,
        history_lengths: torch.Tensor,
        *,
        retention_ratio: float,
        recent_floor: int = 32,
    ) -> GroupSelection:
        candidate_scores = self.pc_selector.score_ids(
            history_item_ids, candidate_item_ids, history_lengths
        )
        groups = self.group_ids(candidate_item_ids).to(candidate_scores.device)
        batch, candidates, width = candidate_scores.shape
        group_sizes = torch.stack(
            [(groups == group).sum(dim=1) for group in range(self.group_count)],
            dim=1,
        )
        history_embeddings = self.pc_selector.lookup(history_item_ids)
        prototypes = self.category_group_prototypes.to(
            device=history_embeddings.device, dtype=history_embeddings.dtype
        )[None].expand(batch, -1, -1)
        fallback = self.pc_selector.exact_scores_from_embeddings(
            history_embeddings, prototypes
        )
        pooled: list[torch.Tensor] = []
        for group in range(self.group_count):
            members = groups == group
            values = candidate_scores.masked_fill(
                ~members[:, :, None], float("-inf")
            )
            denominator = group_sizes[:, group].clamp_min(1).to(
                candidate_scores.dtype
            )[:, None]
            score = torch.logsumexp(values, dim=1) - torch.log(denominator)
            empty = group_sizes[:, group] == 0
            score = torch.where(empty[:, None], fallback[:, group, :], score)
            pooled.append(score)
        group_scores = torch.stack(pooled, dim=1)
        selected = exact_budget_write_mask(
            group_scores,
            history_lengths,
            retention_ratio=retention_ratio,
            recent_floor=recent_floor,
        )
        return GroupSelection(
            candidate_scores=candidate_scores,
            group_scores=selected.scores,
            write_mask=selected.write_mask,
            budgets=selected.budgets,
            valid_history=selected.valid_history,
            actual_write_ratio=selected.actual_write_ratio,
            candidate_group_ids=groups,
            group_sizes=group_sizes,
        )
