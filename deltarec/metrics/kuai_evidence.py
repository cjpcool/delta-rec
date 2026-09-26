from __future__ import annotations

import json
import numpy as np

import math

import os

from pathlib import Path

import tempfile

from typing import Any, Mapping, Sequence

from deltarec.data.exposure import kuai_exposure_identity

MetaBridgeError = ValueError

from deltarec.data.kuai_slates import COMMON_CANDIDATE_COUNT

PER_USER_SCHEMA = "deltarec-meta-per-user-effectiveness-v1"

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

KUAI_SLATE_EVIDENCE_SCHEMA = "deltarec-meta-kuai-slate-effectiveness-v1"

class AtomicJsonlWriter:
    """Write line evidence atomically and refuse an existing destination."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self._handle: Any | None = None
        self._temporary: str | None = None
        self.rows = 0

    def __enter__(self) -> "AtomicJsonlWriter":
        if self.path.exists():
            raise MetaBridgeError(f"refusing to overwrite evidence: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.", dir=self.path.parent
        )
        self._temporary = temporary
        self._handle = os.fdopen(descriptor, "w", encoding="utf-8")
        return self

    def write(self, row: Mapping[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("evidence writer is not open")
        self._handle.write(
            json.dumps(
                dict(row),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        )
        self.rows += 1

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        assert self._handle is not None and self._temporary is not None
        try:
            if exc_type is None:
                self._handle.flush()
                os.fsync(self._handle.fileno())
            self._handle.close()
            if exc_type is None:
                os.replace(self._temporary, self.path)
            else:
                os.unlink(self._temporary)
        finally:
            self._handle = None
            self._temporary = None

def _stable_bce_numerator(
    logits: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> float:
    import numpy as np

    losses = (
        np.maximum(logits, 0)
        - logits * labels
        + np.log1p(np.exp(-np.abs(logits)))
    )
    return float(np.sum(losses * weights))

def write_kuai_per_user_evidence(
    *,
    logits: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    user_ids: np.ndarray,
    dataset: str,
    split: str,
    path: Path,
) -> dict[str, Any]:
    """Persist sufficient per-user statistics to reconstruct GAUC and loss."""

    import numpy as np
    from deltarec.metrics import ranking as headline_metrics

    if tuple(headline_metrics.KUAI_TASKS) != KUAI_TASKS:
        raise MetaBridgeError("common Kuai task order changed")

    if logits.shape != labels.shape or logits.shape != weights.shape:
        raise ValueError("Kuai evidence tensors must share shape")
    if logits.ndim != 3 or logits.shape[-1] != len(KUAI_TASKS):
        raise ValueError("Kuai evidence tensors must have shape [N,K,8]")
    if user_ids.shape != (logits.shape[0],):
        raise ValueError("Kuai user_ids must have one value per slate")
    user_rows: list[dict[str, Any]] = []
    reconstructed_gauc: dict[str, list[tuple[float, int]]] = {
        task: [] for task in KUAI_TASKS
    }
    loss_numerators = {task: 0.0 for task in KUAI_TASKS}
    loss_denominators = {task: 0.0 for task in KUAI_TASKS}
    for user in np.unique(user_ids):
        slate_mask = user_ids == user
        user_logits = np.asarray(
            logits[slate_mask].reshape(-1, len(KUAI_TASKS)), dtype=np.float64
        )
        user_labels = np.asarray(
            labels[slate_mask].reshape(-1, len(KUAI_TASKS)), dtype=np.float64
        )
        user_weights = np.asarray(
            weights[slate_mask].reshape(-1, len(KUAI_TASKS)), dtype=np.float64
        )
        tasks: dict[str, Any] = {}
        for task_index, task in enumerate(KUAI_TASKS):
            task_weights = user_weights[:, task_index]
            valid = task_weights > 0
            task_logits = user_logits[:, task_index]
            task_labels = user_labels[:, task_index]
            count = int(valid.sum())
            denominator = float(task_weights.sum())
            numerator = _stable_bce_numerator(
                task_logits, task_labels, task_weights
            )
            auc = headline_metrics._binary_auc(
                task_labels[valid], task_logits[valid]
            )
            auc_value: float | None = float(auc) if math.isfinite(auc) else None
            if auc_value is not None:
                reconstructed_gauc[task].append((auc_value, count))
            loss_numerators[task] += numerator
            loss_denominators[task] += denominator
            tasks[task] = {
                "gauc": auc_value,
                "eligible_examples": count if auc_value is not None else 0,
                "positive_weight_sum": float(
                    np.sum(task_labels * task_weights)
                ),
                "weight_sum": denominator,
                "bce_numerator": numerator,
            }
        user_rows.append(
            {
                "schema": PER_USER_SCHEMA,
                "dataset": dataset,
                "split": split,
                "user_id": int(user),
                "slates": int(slate_mask.sum()),
                "candidate_examples": int(slate_mask.sum() * logits.shape[1]),
                "tasks": tasks,
            }
        )
    with AtomicJsonlWriter(path) as evidence:
        for row in user_rows:
            evidence.write(row)
    task_gauc = {
        task: float(
            np.average(
                [value for value, _ in observations],
                weights=[count for _, count in observations],
            )
        )
        for task, observations in reconstructed_gauc.items()
        if observations
    }
    task_loss = {
        task: 0.2 * loss_numerators[task] / loss_denominators[task]
        for task in KUAI_TASKS
    }
    return {
        "users": len(user_rows),
        "task_gauc": task_gauc,
        "task_loss": task_loss,
    }

def write_kuai_per_slate_evidence(
    *,
    logits: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    user_ids: np.ndarray,
    slate_ids: Sequence[str],
    candidate_item_ids: np.ndarray,
    candidate_timestamps: np.ndarray,
    candidate_play_time_ms: np.ndarray,
    candidate_duration_ms: np.ndarray,
    dataset: str,
    split: str,
    path: Path,
) -> dict[str, Any]:
    """Persist raw K=32 evidence required by the paired slate bootstrap.

    Per-user GAUC summaries are not bootstrap units: resampling a logged slate
    changes the candidate multiset from which each user's AUC is recomputed.
    Consequently the headline ``EffectivenessRun`` points at this lossless
    evidence rather than the convenient per-user appendix.
    """

    import numpy as np

    expected = (len(user_ids), COMMON_CANDIDATE_COUNT, len(KUAI_TASKS))
    if logits.shape != expected or labels.shape != expected or weights.shape != expected:
        raise ValueError(f"Kuai slate evidence tensors must have shape {expected}")
    if len(slate_ids) != expected[0] or candidate_item_ids.shape != expected[:2]:
        raise ValueError("Kuai slate identities/candidates do not align with logits")
    for name, values in (
        ("candidate_timestamps", candidate_timestamps),
        ("candidate_play_time_ms", candidate_play_time_ms),
        ("candidate_duration_ms", candidate_duration_ms),
    ):
        if values.shape != expected[:2]:
            raise ValueError(f"{name} must have shape {expected[:2]}")
    if len(set(slate_ids)) != len(slate_ids):
        raise ValueError("Kuai slate IDs must be unique")
    if not np.isfinite(logits).all() or not np.isfinite(weights).all():
        raise ValueError("Kuai logits/weights must be finite")
    if np.any((labels != 0) & (labels != 1)) or np.any(weights < 0):
        raise ValueError("Kuai labels must be binary and weights nonnegative")
    with AtomicJsonlWriter(path) as evidence:
        for unit_index, slate_id in enumerate(slate_ids):
            candidates = [int(value) for value in candidate_item_ids[unit_index]]
            if len(candidates) != COMMON_CANDIDATE_COUNT or min(candidates) <= 0:
                raise ValueError("Kuai evidence candidates must be positive K=32 exposures")
            identity = kuai_exposure_identity(
                slate_id=str(slate_id),
                user_id=int(user_ids[unit_index]),
                candidate_item_ids=candidates,
                candidate_timestamps=candidate_timestamps[unit_index],
                candidate_play_time_ms=candidate_play_time_ms[unit_index],
                candidate_duration_ms=candidate_duration_ms[unit_index],
                labels=np.asarray(labels[unit_index], dtype=np.int8),
            )
            evidence.write(
                {
                    "schema": KUAI_SLATE_EVIDENCE_SCHEMA,
                    "dataset": dataset,
                    "split": split,
                    "unit_index": unit_index,
                    "slate_id": str(slate_id),
                    "user_id": int(user_ids[unit_index]),
                    "candidate_item_ids": candidates,
                    **identity,
                    "task_order": list(KUAI_TASKS),
                    "logits": np.asarray(logits[unit_index], dtype=np.float32).tolist(),
                    "labels": np.asarray(labels[unit_index], dtype=np.int8).tolist(),
                    "weights": np.asarray(weights[unit_index], dtype=np.float32).tolist(),
                }
            )
    return {"slates": len(slate_ids), "candidate_examples": len(slate_ids) * 32}
