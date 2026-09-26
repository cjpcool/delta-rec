from __future__ import annotations

import math

from dataclasses import dataclass

from typing import Mapping

import numpy as np

KUAI_TASKS = (
    "click",
    "like",
    "follow",
    "comment",
    "forward",
    "hate",
    "long_view",
    "profile_enter",
)

def _as_numpy(value: object, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()  # type: ignore[union-attr]
    return np.asarray(value, dtype=dtype)

@dataclass(frozen=True)
class RankingMetrics:
    hr: float
    ndcg: float
    per_request_hr: np.ndarray
    per_request_ndcg: np.ndarray
    retrieval_recall: float

def ranking_metrics(
    *,
    candidate_ids: object,
    scores: object,
    target_ids: object,
    k: int = 10,
) -> RankingMetrics:
    """Compute end-to-end HR/NDCG with retrieval misses contributing zero.

    Scores are ordered descending; exact score ties are resolved by ascending
    candidate ID so results do not depend on backend sort stability.
    """

    candidates = _as_numpy(candidate_ids, dtype=np.int64)
    values = _as_numpy(scores, dtype=np.float64)
    targets = _as_numpy(target_ids, dtype=np.int64)
    if candidates.ndim != 2 or values.shape != candidates.shape:
        raise ValueError("candidate_ids and scores must share shape [N,K]")
    if targets.shape != (candidates.shape[0],):
        raise ValueError("target_ids must have shape [N]")
    if k < 1 or k > candidates.shape[1]:
        raise ValueError("k must lie in [1, candidate_count]")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    if np.any(candidates <= 0):
        raise ValueError("candidate IDs must be positive; padding is not a candidate")
    hr = np.zeros(candidates.shape[0], dtype=np.float64)
    ndcg = np.zeros_like(hr)
    retrieved = np.zeros_like(hr)
    for row in range(candidates.shape[0]):
        if np.unique(candidates[row]).size != candidates.shape[1]:
            raise ValueError(f"candidate row {row} contains duplicate item IDs")
        target_positions = np.flatnonzero(candidates[row] == targets[row])
        if target_positions.size == 0:
            continue
        if target_positions.size != 1:
            raise ValueError(f"target occurs more than once in candidate row {row}")
        retrieved[row] = 1.0
        order = np.lexsort((candidates[row], -values[row]))
        rank = int(np.flatnonzero(order == target_positions[0])[0]) + 1
        if rank <= k:
            hr[row] = 1.0
            ndcg[row] = 1.0 / math.log2(rank + 1)
    return RankingMetrics(
        hr=float(hr.mean()),
        ndcg=float(ndcg.mean()),
        per_request_hr=hr,
        per_request_ndcg=ndcg,
        retrieval_recall=float(retrieved.mean()),
    )

def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Tie-aware Mann-Whitney binary AUC without a sklearn dependency."""

    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(labels.size, dtype=np.float64)
    start = 0
    while start < labels.size:
        end = start + 1
        while end < labels.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    positive_rank_sum = ranks[labels.astype(bool)].sum()
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2)
        / (positives * negatives)
    )

@dataclass(frozen=True)
class KuaiMetrics:
    macro_gauc: float
    multitask_loss: float
    task_gauc: Mapping[str, float]
    task_loss: Mapping[str, float]
    task_prevalence: Mapping[str, float]
    eligible_users: Mapping[str, int]
    eligible_examples: Mapping[str, int]

def kuai_metrics(
    *,
    logits: object,
    labels: object,
    user_ids: object,
    weights: object | None = None,
    causal_multitask_weight: float = 0.2,
) -> KuaiMetrics:
    """Compute eight-task weighted user-GAUC and implementation-aligned loss.

    Inputs may be ``[N,K,T]`` or flattened ``[M,T]``.  For three-dimensional
    inputs, ``user_ids`` is ``[N]`` and is expanded across the K candidates.
    A per-task GAUC weights each evaluable user by its number of positive-weight
    candidate examples.  Macro GAUC gives every task equal weight.
    """

    raw_logits = _as_numpy(logits, dtype=np.float64)
    raw_labels = _as_numpy(labels, dtype=np.float64)
    users = _as_numpy(user_ids, dtype=np.int64)
    if raw_logits.shape != raw_labels.shape or raw_logits.ndim not in (2, 3):
        raise ValueError("logits and labels must share shape [M,8] or [N,K,8]")
    if raw_logits.shape[-1] != len(KUAI_TASKS):
        raise ValueError(f"last dimension must contain {len(KUAI_TASKS)} tasks")
    if not np.isfinite(raw_logits).all() or not np.isfinite(raw_labels).all():
        raise ValueError("logits and labels must be finite")
    if np.any((raw_labels != 0) & (raw_labels != 1)):
        raise ValueError("Kuai labels must be binary")
    if raw_logits.ndim == 3:
        if users.shape != (raw_logits.shape[0],):
            raise ValueError("user_ids must have shape [N] for slate inputs")
        expanded_users = np.repeat(users, raw_logits.shape[1])
        flat_logits = raw_logits.reshape(-1, raw_logits.shape[-1])
        flat_labels = raw_labels.reshape(-1, raw_labels.shape[-1])
    else:
        if users.shape != (raw_logits.shape[0],):
            raise ValueError("user_ids must have shape [M] for flat inputs")
        expanded_users = users
        flat_logits = raw_logits
        flat_labels = raw_labels
    if weights is None:
        flat_weights = np.ones_like(flat_labels, dtype=np.float64)
    else:
        raw_weights = _as_numpy(weights, dtype=np.float64)
        if raw_weights.shape != raw_logits.shape:
            raise ValueError("weights must have the same shape as logits")
        if not np.isfinite(raw_weights).all() or np.any(raw_weights < 0):
            raise ValueError("weights must be finite and nonnegative")
        flat_weights = raw_weights.reshape(flat_labels.shape)
    if not math.isfinite(causal_multitask_weight) or causal_multitask_weight <= 0:
        raise ValueError("causal_multitask_weight must be finite and positive")

    task_gauc: dict[str, float] = {}
    task_loss: dict[str, float] = {}
    prevalence: dict[str, float] = {}
    eligible_users: dict[str, int] = {}
    eligible_examples: dict[str, int] = {}
    unique_users = np.unique(expanded_users)
    for task_index, task in enumerate(KUAI_TASKS):
        task_weights = flat_weights[:, task_index]
        valid = task_weights > 0
        denominator = task_weights.sum()
        if denominator <= 0:
            raise ValueError(f"task {task} has no positive supervision weight")
        task_labels = flat_labels[:, task_index]
        task_logits = flat_logits[:, task_index]
        # Stable BCEWithLogits: max(x,0)-x*y+log1p(exp(-abs(x))).
        per_example_loss = (
            np.maximum(task_logits, 0)
            - task_logits * task_labels
            + np.log1p(np.exp(-np.abs(task_logits)))
        )
        task_loss[task] = float(
            causal_multitask_weight
            * np.sum(per_example_loss * task_weights)
            / denominator
        )
        prevalence[task] = float(np.sum(task_labels * task_weights) / denominator)
        auc_values: list[float] = []
        auc_weights: list[int] = []
        for user in unique_users:
            mask = (expanded_users == user) & valid
            count = int(mask.sum())
            if count == 0:
                continue
            auc = _binary_auc(task_labels[mask], task_logits[mask])
            if math.isfinite(auc):
                auc_values.append(auc)
                auc_weights.append(count)
        if not auc_values:
            raise ValueError(f"task {task} has no evaluable user with both classes")
        task_gauc[task] = float(np.average(auc_values, weights=auc_weights))
        eligible_users[task] = len(auc_values)
        eligible_examples[task] = int(sum(auc_weights))
    return KuaiMetrics(
        macro_gauc=float(np.mean(list(task_gauc.values()))),
        multitask_loss=float(sum(task_loss.values())),
        task_gauc=task_gauc,
        task_loss=task_loss,
        task_prevalence=prevalence,
        eligible_users=eligible_users,
        eligible_examples=eligible_examples,
    )
