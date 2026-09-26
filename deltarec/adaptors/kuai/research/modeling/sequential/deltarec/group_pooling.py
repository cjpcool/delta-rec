# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""Candidate-to-group score pooling for Subplan 5C."""

from __future__ import annotations

import torch


GROUP_POOLS = frozenset(("logmeanexp", "mean", "max", "mask_vote"))


def _validate_pooling_inputs(
    candidate_scores: torch.Tensor,
    candidate_to_group: torch.Tensor,
    group_count: int | None,
    history_lengths: torch.Tensor | None,
    validate_runtime: bool,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    if candidate_scores.ndim != 3 or not torch.is_floating_point(candidate_scores):
        raise ValueError("candidate_scores must be floating point with shape [B,K,L]")
    batch, candidates, width = candidate_scores.shape
    if batch < 1 or candidates < 1 or width < 1:
        raise ValueError("candidate_scores must contain users, candidates, and events")
    if candidate_to_group.shape != (batch, candidates) or candidate_to_group.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("candidate_to_group must be integer with shape [B,K]")
    if candidate_to_group.device != candidate_scores.device:
        raise ValueError("scores and group assignments must share a device")
    if not isinstance(validate_runtime, bool):
        raise TypeError("validate_runtime must be boolean")
    if group_count is None:
        if bool((candidate_to_group < 0).any()):
            raise ValueError("candidate_to_group must be nonnegative")
        groups = int(candidate_to_group.max()) + 1
    else:
        if isinstance(group_count, bool) or not isinstance(group_count, int):
            raise TypeError("group_count must be an integer")
        groups = group_count
    if groups < 1:
        raise ValueError("group_count must be positive")
    if bool((candidate_to_group < 0).any()) or bool(
        (candidate_to_group >= groups).any()
    ):
        raise ValueError("candidate_to_group contains an invalid group")
    user_bases = (
        torch.arange(
            batch, device=candidate_scores.device, dtype=torch.int64
        )[:, None]
        * groups
    )
    group_sizes = torch.bincount(
        (candidate_to_group.to(torch.int64) + user_bases).reshape(-1),
        minlength=batch * groups,
    ).reshape(batch, groups)
    positions = torch.arange(width, device=candidate_scores.device)
    if history_lengths is None:
        valid = torch.ones(
            (batch, width), dtype=torch.bool, device=candidate_scores.device
        )
    else:
        if history_lengths.shape != (batch,) or history_lengths.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("history_lengths must be integer with shape [B]")
        if history_lengths.device != candidate_scores.device:
            raise ValueError("history lengths and scores must share a device")
        if bool((history_lengths < 1).any()) or bool((history_lengths > width).any()):
            raise ValueError("history_lengths must address nonempty valid prefixes")
        valid = positions[None, :] < history_lengths[:, None]
    return groups, group_sizes, valid


def pool_group_scores(
    candidate_scores: torch.Tensor,
    candidate_to_group: torch.Tensor,
    *,
    group_count: int | None = None,
    pool: str = "logmeanexp",
    selected_masks: torch.Tensor | None = None,
    history_lengths: torch.Tensor | None = None,
    validate_runtime: bool = True,
) -> torch.Tensor:
    """Pool exact per-candidate scores into one score row per semantic group.

    ``logmeanexp`` implements the registered size-normalized log-sum-exp:

    ``logsumexp(scores in group) - log(group_size)``.

    ``mask_vote`` consumes exact candidate selection masks and returns the
    fraction of group candidates selecting each event.  Padding is overwritten
    before every reduction and returned as ``-inf``, so arbitrary padding poison
    cannot influence membership or gradients.
    """

    if pool == "mask-vote":
        pool = "mask_vote"
    if pool not in GROUP_POOLS:
        raise ValueError(f"unsupported group pool: {pool!r}")
    groups, group_sizes, valid = _validate_pooling_inputs(
        candidate_scores,
        candidate_to_group,
        group_count,
        history_lengths,
        validate_runtime,
    )
    batch, candidates, width = candidate_scores.shape
    if pool == "mask_vote":
        if (
            selected_masks is None
            or selected_masks.shape != (batch, candidates, width)
            or selected_masks.dtype != torch.bool
        ):
            raise ValueError("mask_vote requires boolean selected_masks with shape [B,K,L]")
        if selected_masks.device != candidate_scores.device:
            raise ValueError("selected masks and scores must share a device")
    elif selected_masks is not None and (
        selected_masks.shape != (batch, candidates, width)
        or selected_masks.dtype != torch.bool
        or selected_masks.device != candidate_scores.device
    ):
        raise ValueError("selected_masks must be boolean with shape [B,K,L]")

    if validate_runtime and pool != "mask_vote" and not bool(
        torch.isfinite(candidate_scores.masked_select(valid[:, None, :])).all()
    ):
        raise ValueError("valid candidate scores must be finite")

    # Overwrite padding before reduction.  In particular this prevents NaN
    # padding from leaking through a multiply-by-zero implementation of mean.
    safe_scores = candidate_scores.masked_fill(~valid[:, None, :], 0.0)
    pooled_rows: list[torch.Tensor] = []
    for group in range(groups):
        members = candidate_to_group == group
        member_mask = members[:, :, None]
        empty = group_sizes[:, group] == 0
        denominator = (
            group_sizes[:, group].clamp_min(1).to(candidate_scores.dtype)[:, None]
        )
        if pool == "logmeanexp":
            values = safe_scores.masked_fill(~member_mask, float("-inf"))
            pooled = torch.logsumexp(values, dim=1) - torch.log(denominator)
        elif pool == "mean":
            values = safe_scores.masked_fill(~member_mask, 0.0)
            pooled = values.sum(dim=1) / denominator
        elif pool == "max":
            values = safe_scores.masked_fill(~member_mask, float("-inf"))
            pooled = values.max(dim=1).values
        else:
            assert selected_masks is not None
            votes = (selected_masks & member_mask & valid[:, None, :]).to(
                candidate_scores.dtype
            )
            pooled = votes.sum(dim=1) / denominator
        # Fixed global category groups may be absent from a particular slate.
        # A finite identity score makes generic selection deterministic; the
        # registered category-prototype adapter supplies prototype scores for
        # every group and therefore does not use this fallback.
        pooled = pooled.masked_fill(empty[:, None] & valid, 0.0)
        pooled_rows.append(pooled.masked_fill(~valid, float("-inf")))
    return torch.stack(pooled_rows, dim=1)


# A readable alias for callers that put the noun before the operation.
group_pool_scores = pool_group_scores

