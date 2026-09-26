# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""Deterministic, permutation-invariant candidate grouping for Subplan 5C.

The registered grouping policy performs farthest-point traversal over FP32
L2-normalized frozen candidate embeddings.  Item IDs resolve all geometric
ties, and the final semantic group order is the ascending anchor-item order.
Candidate positions are used only to address the input tensors; they are never
used as a grouping feature.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


FROZEN_EMBEDDING_CLUSTER_V1 = "frozen_embedding_cluster_v1"
CANONICAL_ID_BALANCED = "canonical_id_balanced"
FIXED_CATEGORY_PROTOTYPE_V1 = "fixed_category_prototype_v1"


@dataclass(frozen=True)
class GroupingOutput:
    """Canonical group assignment and group-major physical candidate order.

    ``packed_candidate_indices`` addresses ``candidate_ids.reshape(-1)``.  Its
    rows are delimited by ``group_offsets`` in user-major, then group-major,
    order.  Candidates inside a group are ordered by item ID.  This makes the
    physical layout reproducible without changing the semantic candidate axis.
    """

    candidate_to_group: torch.Tensor
    group_sizes: torch.Tensor
    packed_candidate_indices: torch.Tensor
    group_offsets: torch.Tensor
    anchor_indices: torch.Tensor
    anchor_ids: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.candidate_to_group.shape[0])

    @property
    def candidate_count(self) -> int:
        return int(self.candidate_to_group.shape[1])

    @property
    def group_count(self) -> int:
        return int(self.group_sizes.shape[1])

    def validate(self, candidate_ids: torch.Tensor | None = None) -> None:
        """Validate the dense assignment and its packed representation."""

        if (
            self.candidate_to_group.ndim != 2
            or self.candidate_to_group.dtype != torch.int64
        ):
            raise ValueError("candidate_to_group must be int64 with shape [B,K]")
        batch, candidates = self.candidate_to_group.shape
        if batch < 1 or candidates < 1:
            raise ValueError("grouping must contain at least one candidate")
        if self.group_sizes.ndim != 2 or self.group_sizes.shape[0] != batch:
            raise ValueError("group_sizes must have shape [B,G]")
        groups = self.group_sizes.shape[1]
        if groups < 1 or self.group_sizes.dtype != torch.int64:
            raise ValueError("group_sizes must describe at least one group")
        device = self.candidate_to_group.device
        tensors = (
            self.group_sizes,
            self.packed_candidate_indices,
            self.group_offsets,
            self.anchor_indices,
            self.anchor_ids,
        )
        if any(tensor.device != device for tensor in tensors):
            raise ValueError("all grouping tensors must share a device")
        if self.anchor_indices.shape != (batch, groups):
            raise ValueError("anchor_indices must have shape [B,G]")
        if self.anchor_ids.shape != (batch, groups):
            raise ValueError("anchor_ids must have shape [B,G]")
        if self.anchor_indices.dtype != torch.int64:
            raise ValueError("anchor_indices must be int64")
        if self.anchor_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("anchor_ids must be integer")
        if bool((self.group_sizes < 0).any()):
            raise ValueError("semantic group sizes must be nonnegative")
        if not torch.equal(
            self.group_sizes.sum(dim=1),
            self.group_sizes.new_full((batch,), candidates),
        ):
            raise ValueError("group sizes must sum to K for every user")
        if bool((self.candidate_to_group < 0).any()) or bool(
            (self.candidate_to_group >= groups).any()
        ):
            raise ValueError("candidate_to_group contains an invalid group")
        user_bases = (
            torch.arange(batch, device=device, dtype=torch.int64)[:, None] * groups
        )
        expected_sizes = torch.bincount(
            (self.candidate_to_group + user_bases).reshape(-1),
            minlength=batch * groups,
        ).reshape(batch, groups)
        if not torch.equal(expected_sizes, self.group_sizes):
            raise ValueError("group sizes disagree with candidate assignments")
        if (
            self.group_offsets.shape != (batch * groups + 1,)
            or self.group_offsets.dtype != torch.int64
        ):
            raise ValueError("group_offsets must be int64 with shape [B*G+1]")
        if int(self.group_offsets[0]) != 0 or int(self.group_offsets[-1]) != (
            batch * candidates
        ):
            raise ValueError("group_offsets must span all packed candidates")
        if not torch.equal(
            self.group_offsets[1:] - self.group_offsets[:-1],
            self.group_sizes.reshape(-1),
        ):
            raise ValueError("group offsets disagree with group sizes")
        if (
            self.packed_candidate_indices.shape != (batch * candidates,)
            or self.packed_candidate_indices.dtype != torch.int64
        ):
            raise ValueError("packed_candidate_indices must contain B*K int64 entries")
        if not torch.equal(
            torch.sort(self.packed_candidate_indices).values,
            torch.arange(batch * candidates, dtype=torch.int64, device=device),
        ):
            raise ValueError("packed candidates must be a permutation of [0,B*K)")
        occupied = self.group_sizes > 0
        if bool((self.anchor_indices.masked_select(occupied) < 0).any()) or bool(
            (self.anchor_indices.masked_select(occupied) >= candidates).any()
        ):
            raise ValueError("occupied groups require a valid anchor candidate")
        if bool((self.anchor_indices.masked_select(~occupied) != -1).any()) or bool(
            (self.anchor_ids.masked_select(~occupied) != -1).any()
        ):
            raise ValueError("empty groups must use the -1 anchor sentinel")
        safe_anchor_indices = self.anchor_indices.clamp_min(0)
        anchor_groups = self.candidate_to_group.gather(1, safe_anchor_indices)
        expected_anchor_groups = torch.arange(
            groups, dtype=torch.int64, device=device
        )[None, :].expand(batch, groups)
        if not torch.equal(
            anchor_groups.masked_select(occupied),
            expected_anchor_groups.masked_select(occupied),
        ):
            raise ValueError("every occupied-group anchor must remain in its group")

        packed_rows = torch.repeat_interleave(
            torch.arange(batch * groups, device=device, dtype=torch.int64),
            self.group_sizes.reshape(-1),
        )
        packed_users = torch.div(packed_rows, groups, rounding_mode="floor")
        packed_groups = packed_rows.remainder(groups)
        packed_positions = self.packed_candidate_indices.remainder(candidates)
        packed_source_users = torch.div(
            self.packed_candidate_indices, candidates, rounding_mode="floor"
        )
        if not torch.equal(packed_users, packed_source_users):
            raise ValueError("packed candidates cross a user boundary")
        if not torch.equal(
            self.candidate_to_group[packed_users, packed_positions],
            packed_groups,
        ):
            raise ValueError("packed candidates cross a group boundary")

        if candidate_ids is not None:
            if candidate_ids.shape != (batch, candidates) or candidate_ids.dtype not in (
                torch.int32,
                torch.int64,
            ):
                raise ValueError("candidate_ids must be integer with shape [B,K]")
            if candidate_ids.device != device:
                raise ValueError("candidate_ids and grouping must share a device")
            gathered_anchor_ids = candidate_ids.gather(1, safe_anchor_indices)
            if not torch.equal(
                gathered_anchor_ids.masked_select(occupied),
                self.anchor_ids.masked_select(occupied),
            ):
                raise ValueError("anchor IDs disagree with anchor indices")
            packed_ids = candidate_ids.reshape(-1).index_select(
                0, self.packed_candidate_indices
            )
            if len(packed_ids) > 1:
                same_group = packed_rows[1:] == packed_rows[:-1]
                if bool(
                    (
                        same_group
                        & (packed_ids[1:] < packed_ids[:-1])
                    ).any()
                ):
                    raise ValueError(
                        "candidates inside a group must be item-ID ordered"
                    )


def _canonical_candidate_order(
    item_ids: torch.Tensor,
    normalized_embeddings: torch.Tensor,
    *,
    check_malformed_duplicates: bool = True,
) -> torch.Tensor:
    """Return an item-ID-first order independent of candidate transport order.

    Duplicate item IDs normally share the same frozen item-table embedding.  A
    lexicographic normalized-embedding fallback makes malformed same-ID/different-
    embedding inputs deterministic as well.  Exact duplicate copies are
    indistinguishable and may retain either occurrence order without changing
    any semantic group contents.
    """

    order = torch.argsort(item_ids, stable=True)
    if not check_malformed_duplicates:
        # The production adapter obtains embeddings by looking up these exact
        # IDs in one frozen table, so equal IDs necessarily have equal vectors.
        # Skipping the duplicate probe keeps the registered timed path fully
        # asynchronous; validated/reference callers retain the defensive
        # malformed-input fallback below.
        return order
    sorted_ids = item_ids.index_select(0, order)
    if len(sorted_ids) < 2 or not bool((sorted_ids[1:] == sorted_ids[:-1]).any()):
        # Production candidate sets are normally distinct.  Avoid D additional
        # stable sorts in that registered fast path.
        return order

    # Only duplicate IDs need the embedding fallback.  Stable sorting by the
    # secondary keys first and item ID last implements a tensor lexsort.
    order = torch.arange(item_ids.numel(), dtype=torch.int64, device=item_ids.device)
    for dimension in range(normalized_embeddings.shape[1] - 1, -1, -1):
        local = torch.argsort(
            normalized_embeddings.index_select(0, order)[:, dimension],
            stable=True,
        )
        order = order.index_select(0, local)
    local = torch.argsort(item_ids.index_select(0, order), stable=True)
    return order.index_select(0, local)


def _group_one_user(
    item_ids: torch.Tensor,
    normalized_embeddings: torch.Tensor,
    group_count: int,
    *,
    check_malformed_duplicates: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return assignment, anchors, and group-major local candidate positions."""

    candidates = item_ids.numel()
    canonical = _canonical_candidate_order(
        item_ids,
        normalized_embeddings,
        check_malformed_duplicates=check_malformed_duplicates,
    )
    selected = torch.zeros(candidates, dtype=torch.bool, device=item_ids.device)

    first_anchor = canonical[0]
    fps_anchors = [first_anchor]
    selected[first_anchor] = True
    cosine = normalized_embeddings @ normalized_embeddings[first_anchor]
    nearest_distance = 1.0 - cosine

    for _ in range(1, group_count):
        # ``canonical`` is item-ID ordered, so argmax's first-index tie behavior
        # implements the registered item-ID tie break.
        canonical_distances = nearest_distance.index_select(0, canonical).masked_fill(
            selected.index_select(0, canonical),
            float("-inf"),
        )
        next_anchor = canonical[torch.argmax(canonical_distances)]
        fps_anchors.append(next_anchor)
        selected[next_anchor] = True
        distance = 1.0 - normalized_embeddings @ normalized_embeddings[next_anchor]
        nearest_distance = torch.minimum(nearest_distance, distance)

    # The traversal order chooses the anchor set; semantic group IDs are instead
    # ordered by anchor item ID as required by the frozen contract.
    anchor_indices = torch.stack(fps_anchors)
    anchor_order = torch.argsort(
        item_ids.index_select(0, anchor_indices), stable=True
    )
    anchor_indices = anchor_indices.index_select(0, anchor_order)
    similarities = normalized_embeddings @ normalized_embeddings.index_select(
        0, anchor_indices
    ).transpose(0, 1)
    assignment = torch.argmax(similarities, dim=1).to(torch.int64)
    assignment = assignment.scatter(
        0,
        anchor_indices,
        torch.arange(group_count, dtype=torch.int64, device=item_ids.device),
    )

    canonical_assignment = assignment.index_select(0, canonical)
    packed_local = canonical.index_select(
        0, torch.argsort(canonical_assignment, stable=True)
    )
    return assignment, anchor_indices, packed_local


def _balanced_group_one_user(
    item_ids: torch.Tensor,
    normalized_embeddings: torch.Tensor,
    group_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the registered canonical-ID balanced systems control.

    Candidates are sorted canonically by item ID and assigned round-robin.  The
    first ``G`` canonical candidates are the group anchors, so every effective
    group is nonempty and group sizes differ by at most one.
    """

    candidates = item_ids.numel()
    canonical = _canonical_candidate_order(item_ids, normalized_embeddings)
    canonical_groups = torch.arange(
        candidates, dtype=torch.int64, device=item_ids.device
    ).remainder(group_count)
    assignment = torch.empty(candidates, dtype=torch.int64, device=item_ids.device)
    assignment.scatter_(0, canonical, canonical_groups)
    anchor_indices = canonical[:group_count]
    packed_local = torch.cat(
        [canonical[canonical_groups == group] for group in range(group_count)]
    )
    return assignment, anchor_indices, packed_local


def _group_batch_frozen_embedding_cluster(
    item_ids: torch.Tensor,
    normalized_embeddings: torch.Tensor,
    group_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized registered FPS path for frozen-table candidate embeddings."""

    batch, candidates = item_ids.shape
    canonical = torch.argsort(item_ids, dim=1, stable=True)
    selected = torch.zeros(
        (batch, candidates), dtype=torch.bool, device=item_ids.device
    )
    first_anchor = canonical[:, 0]
    anchors = [first_anchor]
    selected.scatter_(1, first_anchor[:, None], True)

    def gather_embedding(indices: torch.Tensor) -> torch.Tensor:
        return normalized_embeddings.gather(
            1,
            indices[:, None, None].expand(
                batch, 1, normalized_embeddings.shape[-1]
            ),
        ).squeeze(1)

    anchor_embedding = gather_embedding(first_anchor)
    nearest_distance = 1.0 - torch.bmm(
        normalized_embeddings, anchor_embedding[:, :, None]
    ).squeeze(-1)
    for _ in range(1, group_count):
        canonical_distances = nearest_distance.gather(1, canonical).masked_fill(
            selected.gather(1, canonical), float("-inf")
        )
        next_canonical = torch.argmax(canonical_distances, dim=1)
        next_anchor = canonical.gather(1, next_canonical[:, None]).squeeze(1)
        anchors.append(next_anchor)
        selected.scatter_(1, next_anchor[:, None], True)
        anchor_embedding = gather_embedding(next_anchor)
        distance = 1.0 - torch.bmm(
            normalized_embeddings, anchor_embedding[:, :, None]
        ).squeeze(-1)
        nearest_distance = torch.minimum(nearest_distance, distance)

    traversal_anchors = torch.stack(anchors, dim=1)
    traversal_anchor_ids = item_ids.gather(1, traversal_anchors)
    anchor_order = torch.argsort(traversal_anchor_ids, dim=1, stable=True)
    anchor_indices = traversal_anchors.gather(1, anchor_order)
    anchor_embeddings = normalized_embeddings.gather(
        1,
        anchor_indices[:, :, None].expand(
            batch, group_count, normalized_embeddings.shape[-1]
        ),
    )
    similarities = torch.bmm(
        normalized_embeddings, anchor_embeddings.transpose(1, 2)
    )
    assignment = torch.argmax(similarities, dim=2).to(torch.int64)
    assignment.scatter_(
        1,
        anchor_indices,
        torch.arange(
            group_count, dtype=torch.int64, device=item_ids.device
        )[None, :].expand(batch, group_count),
    )
    canonical_assignment = assignment.gather(1, canonical)
    group_order = torch.argsort(canonical_assignment, dim=1, stable=True)
    packed_local = canonical.gather(1, group_order)
    return assignment, anchor_indices, packed_local


def _group_batch_canonical_balanced(
    item_ids: torch.Tensor,
    group_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized canonical-ID round-robin systems control."""

    batch, candidates = item_ids.shape
    canonical = torch.argsort(item_ids, dim=1, stable=True)
    canonical_groups = torch.arange(
        candidates, dtype=torch.int64, device=item_ids.device
    ).remainder(group_count)
    assignment = torch.empty(
        (batch, candidates), dtype=torch.int64, device=item_ids.device
    )
    assignment.scatter_(
        1, canonical, canonical_groups[None, :].expand(batch, candidates)
    )
    anchor_indices = canonical[:, :group_count]
    group_order = torch.argsort(
        canonical_groups[None, :].expand(batch, candidates),
        dim=1,
        stable=True,
    )
    packed_local = canonical.gather(1, group_order)
    return assignment, anchor_indices, packed_local


def _group_batch_fixed_category(
    item_ids: torch.Tensor,
    category_group_ids: torch.Tensor,
    group_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use frozen global category/supergroup IDs as candidate group semantics.

    Unlike request-local FPS, group labels remain the global IDs ``[0, G)``.
    A request need not contain a candidate from every global group; empty groups
    use ``-1`` anchor sentinels and remain materializable from their frozen
    selector prototypes.
    """

    batch, candidates = item_ids.shape
    assignment = category_group_ids.to(torch.int64)
    canonical = torch.argsort(item_ids, dim=1, stable=True)
    canonical_groups = assignment.gather(1, canonical)
    packed_local = canonical.gather(
        1, torch.argsort(canonical_groups, dim=1, stable=True)
    )
    anchor_indices = torch.full(
        (batch, group_count), -1, dtype=torch.int64, device=item_ids.device
    )
    # Canonical item order makes the first member the deterministic anchor.
    for group in range(group_count):
        members = canonical_groups == group
        occupied = members.any(dim=1)
        first_position = torch.argmax(members.to(torch.int64), dim=1)
        first_member = canonical.gather(1, first_position[:, None]).squeeze(1)
        anchor_indices[:, group] = torch.where(
            occupied, first_member, anchor_indices[:, group]
        )
    return assignment, anchor_indices, packed_local


def group_candidates(
    candidate_ids: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    candidate_group_count: int,
    *,
    policy: str = FROZEN_EMBEDDING_CLUSTER_V1,
    candidate_category_group_ids: torch.Tensor | None = None,
    validate_runtime: bool = True,
) -> GroupingOutput:
    """Group each user's unordered candidate set with frozen embedding FPS.

    The effective group count is ``min(candidate_group_count, K)``.  No group
    spans users, every chosen anchor is forced to remain in its own group, and
    group labels are canonicalized by ascending anchor item ID.
    """

    if candidate_ids.ndim != 2 or candidate_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("candidate_ids must be an integer tensor with shape [B,K]")
    batch, candidates = candidate_ids.shape
    if batch < 1 or candidates < 1:
        raise ValueError("candidate_ids must contain at least one candidate")
    if (
        candidate_embeddings.ndim != 3
        or candidate_embeddings.shape[:2] != (batch, candidates)
        or not torch.is_floating_point(candidate_embeddings)
    ):
        raise ValueError("candidate_embeddings must be floating point with shape [B,K,D]")
    if candidate_embeddings.shape[-1] < 1:
        raise ValueError("candidate embeddings must have a positive width")
    if candidate_embeddings.device != candidate_ids.device:
        raise ValueError("candidate IDs and embeddings must share a device")
    if isinstance(candidate_group_count, bool) or not isinstance(candidate_group_count, int):
        raise TypeError("candidate_group_count must be an integer")
    if candidate_group_count < 1:
        raise ValueError("candidate_group_count must be positive")
    if policy not in (
        FROZEN_EMBEDDING_CLUSTER_V1,
        CANONICAL_ID_BALANCED,
        FIXED_CATEGORY_PROTOTYPE_V1,
    ):
        raise ValueError(f"unsupported candidate grouping policy: {policy!r}")
    if not isinstance(validate_runtime, bool):
        raise TypeError("validate_runtime must be boolean")
    if policy == FIXED_CATEGORY_PROTOTYPE_V1:
        if candidate_category_group_ids is None:
            raise ValueError(
                "fixed-category grouping requires candidate_category_group_ids"
            )
        if candidate_category_group_ids.shape != (batch, candidates) or (
            candidate_category_group_ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                "candidate_category_group_ids must be integer with shape [B,K]"
            )
        if candidate_category_group_ids.device != candidate_ids.device:
            raise ValueError("candidate category groups must share the ID device")
        if bool((candidate_category_group_ids < 0).any()) or bool(
            (candidate_category_group_ids >= candidate_group_count).any()
        ):
            raise ValueError("candidate category group IDs must lie in [0,G)")
        groups = candidate_group_count
        normalized = candidate_embeddings.detach().to(torch.float32)
    else:
        if candidate_category_group_ids is not None:
            raise ValueError(
                "candidate_category_group_ids are valid only for fixed-category grouping"
            )
        embeddings_fp32 = candidate_embeddings.detach().to(torch.float32)
        if validate_runtime and not bool(torch.isfinite(embeddings_fp32).all()):
            raise ValueError("candidate embeddings must be finite")
        norms = torch.linalg.vector_norm(embeddings_fp32, dim=-1)
        if validate_runtime and bool((norms == 0).any()):
            raise ValueError(
                "candidate embeddings must be nonzero before L2 normalization"
            )
        normalized = F.normalize(embeddings_fp32, p=2.0, dim=-1)
        groups = min(candidate_group_count, candidates)
    with torch.no_grad():
        if policy == FROZEN_EMBEDDING_CLUSTER_V1:
            candidate_to_group, anchor_indices, packed_local = (
                _group_batch_frozen_embedding_cluster(
                    candidate_ids, normalized, groups
                )
            )
        elif policy == CANONICAL_ID_BALANCED:
            candidate_to_group, anchor_indices, packed_local = (
                _group_batch_canonical_balanced(candidate_ids, groups)
            )
        else:
            assert candidate_category_group_ids is not None
            candidate_to_group, anchor_indices, packed_local = (
                _group_batch_fixed_category(
                    candidate_ids, candidate_category_group_ids, groups
                )
            )

    safe_anchor_indices = anchor_indices.clamp_min(0)
    anchor_ids = candidate_ids.gather(1, safe_anchor_indices)
    anchor_ids = anchor_ids.masked_fill(anchor_indices < 0, -1)
    user_bases = (
        torch.arange(batch, device=candidate_ids.device, dtype=torch.int64)[:, None]
        * groups
    )
    group_sizes = torch.bincount(
        (candidate_to_group + user_bases).reshape(-1),
        minlength=batch * groups,
    ).reshape(batch, groups)
    packed_candidate_indices = (
        packed_local
        + torch.arange(
            batch, device=candidate_ids.device, dtype=torch.int64
        )[:, None]
        * candidates
    ).reshape(-1)
    flat_sizes = group_sizes.reshape(-1)
    group_offsets = torch.cat(
        (flat_sizes.new_zeros(1), torch.cumsum(flat_sizes, dim=0)), dim=0
    )
    output = GroupingOutput(
        candidate_to_group=candidate_to_group,
        group_sizes=group_sizes,
        packed_candidate_indices=packed_candidate_indices,
        group_offsets=group_offsets,
        anchor_indices=anchor_indices,
        anchor_ids=anchor_ids,
    )
    if validate_runtime:
        output.validate(candidate_ids)
    return output

