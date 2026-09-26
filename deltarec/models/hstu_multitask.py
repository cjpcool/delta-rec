"""Shared HSTU-aligned KuaiRand prediction head and objective."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from deltarec.metrics.ranking import KUAI_TASKS
from deltarec.layers.pair_head_ops import init_mlp_weights_optional_bias
from deltarec.layers.pair_head_ops import _compute_loss
from deltarec.layers.pair_head_ops import SwishLayerNorm


class HSTUPairHead(nn.Module):
    """HSTU-style two-layer head over explicit history/item pair features."""

    def __init__(self, embedding_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.layers = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, hidden_dim),
            SwishLayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        ).apply(init_mlp_weights_optional_bias)

    def forward(
        self,
        user_history_embedding: torch.Tensor,
        candidate_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if user_history_embedding.shape != candidate_embedding.shape:
            raise ValueError("history and candidate embeddings must share shape [B,K,D]")
        if user_history_embedding.ndim != 3 or user_history_embedding.shape[-1] != self.embedding_dim:
            raise ValueError("history and candidate embeddings must have shape [B,K,D]")
        features = torch.cat(
            (user_history_embedding, candidate_embedding), dim=-1
        ).float()
        return self.layers(features.flatten(0, -2)).reshape(
            *features.shape[:-1], -1
        )


class HSTUMultitaskHead(HSTUPairHead):
    """KuaiRand's unchanged ``2D -> 512 -> 8`` prediction head."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__(embedding_dim, 512, len(KUAI_TASKS))


class HSTURankingHead(HSTUPairHead):
    """Rating-dataset ``2D -> 256 -> 1`` candidate ranker."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__(embedding_dim, 256, 1)

    def forward(
        self,
        user_history_embedding: torch.Tensor,
        candidate_embedding: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(user_history_embedding, candidate_embedding).squeeze(-1)


def hstu_multitask_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Use HSTU's weighted ``0.2 * sum_t(mean(BCE_t))`` implementation."""

    if logits.shape != labels.shape or logits.ndim != 3:
        raise ValueError("Kuai logits and labels must share shape [B,K,8]")
    if logits.shape[-1] != len(KUAI_TASKS):
        raise ValueError(f"Kuai objective requires {len(KUAI_TASKS)} task logits")
    if weights is None:
        weights = torch.ones_like(labels)
    if weights.shape != logits.shape or bool((weights < 0).any()):
        raise ValueError("Kuai supervision weights must match labels and be nonnegative")
    tasks = logits.shape[-1]
    return _compute_loss(
        task_offsets=[0, tasks, tasks],
        causal_multitask_weights=0.2,
        mt_logits=logits.float().reshape(-1, tasks).transpose(0, 1),
        mt_labels=labels.float().reshape(-1, tasks).transpose(0, 1),
        mt_weights=weights.float().reshape(-1, tasks).transpose(0, 1),
        has_multiple_task_types=False,
    ).sum()


def checkpoint_state_without_legacy_head(
    model: nn.Module,
    state: Mapping[str, Any],
) -> tuple[dict[str, Any], set[str]]:
    """Keep backbone state while replacing an absent or obsolete prediction head."""

    prefix = "task_head."
    target_head = {key for key in model.state_dict() if key.startswith(prefix)}
    source_head = {key for key in state if key.startswith(prefix)}
    if source_head == target_head:
        return dict(state), set()
    if source_head and source_head != {"task_head.weight", "task_head.bias"}:
        raise ValueError(
            "checkpoint has an unrecognized multitask head: "
            f"observed={sorted(source_head)}, expected={sorted(target_head)}"
        )
    return {
        key: value for key, value in state.items() if not key.startswith(prefix)
    }, target_head


__all__ = [
    "HSTUMultitaskHead",
    "HSTUPairHead",
    "HSTURankingHead",
    "checkpoint_state_without_legacy_head",
    "hstu_multitask_bce",
]
