from __future__ import annotations

from dataclasses import dataclass, field

from enum import Enum

from typing import Any, Mapping

class ModelMode(str, Enum):
    TRAIN = "train"
    EVAL = "eval"
    BENCHMARK = "benchmark"

@dataclass(frozen=True)
class RerankingBatch:
    """The frozen common-protocol input boundary.

    Values are intentionally tensor-framework agnostic. A bridge may accept
    torch tensors, NumPy arrays, or immutable Python fixtures, but it must not
    change row order or candidate order.
    """

    user_ids: Any
    history_item_ids: Any
    history_lengths: Any
    candidate_item_ids: Any
    target_indices: Any | None = None
    labels: Any | None = None
    timestamps: Any | None = None
    task_labels: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required = {
            "user_ids": self.user_ids,
            "history_item_ids": self.history_item_ids,
            "history_lengths": self.history_lengths,
            "candidate_item_ids": self.candidate_item_ids,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"RerankingBatch missing required fields: {missing}")

@dataclass(frozen=True)
class AdapterOutput:
    """Output in original batch/candidate order.

    `ranking_scores` is mandatory. Multitask bridges additionally populate
    `task_scores`; training calls expose named losses, including `total`.
    """

    ranking_scores: Any
    task_scores: Mapping[str, Any] = field(default_factory=dict)
    losses: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ranking_scores is None:
            raise ValueError("AdapterOutput.ranking_scores must not be None")
