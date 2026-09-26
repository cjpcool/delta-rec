"""Canonical identities for position-preserving KuaiRand logged exposures.

Kuai slates are logged *exposures*, not catalog candidate sets.  The same item
may therefore occur more than once in a K=32 slate and each occurrence can
carry different timestamp, watch-time, duration, and labels.  This module
defines the lossless identity used by every evaluator and by the paired
statistical audit.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


KUAI_EXPOSURE_IDENTITY_SCHEMA = "deltarec-kuai-exposure-identity-v1"
KUAI_EXPOSURE_COUNT = 32
KUAI_TASK_COUNT = 8


def _integers(value: Any, *, name: str) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence")
    result: list[int] = []
    for entry in value:
        if isinstance(entry, bool):
            raise ValueError(f"{name} must contain integers")
        try:
            converted = int(entry)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must contain integers") from error
        if converted != entry:
            raise ValueError(f"{name} must contain exact integers")
        result.append(converted)
    return result


def _label_rows(value: Any) -> list[list[int]]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError("labels must have shape [32,8]")
    rows = [_integers(row, name="labels") for row in value]
    if len(rows) != KUAI_EXPOSURE_COUNT or any(
        len(row) != KUAI_TASK_COUNT for row in rows
    ):
        raise ValueError("labels must have shape [32,8]")
    if any(label not in (0, 1) for row in rows for label in row):
        raise ValueError("labels must be binary")
    return rows


def kuai_exposure_identity(
    *,
    slate_id: str,
    user_id: int,
    candidate_item_ids: Any,
    candidate_timestamps: Any,
    candidate_play_time_ms: Any,
    candidate_duration_ms: Any,
    labels: Any,
) -> dict[str, Any]:
    """Return canonical position fields plus a source-row-derived SHA-256.

    Candidate item IDs intentionally need not be unique.  Position is part of
    every exposure identity, so two occurrences of the same item remain two
    independent examples even when all other logged fields happen to tie.
    """

    if not isinstance(slate_id, str) or not slate_id:
        raise ValueError("slate_id must be a nonempty string")
    if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id < 0:
        raise ValueError("user_id must be a nonnegative integer")
    items = _integers(candidate_item_ids, name="candidate_item_ids")
    timestamps = _integers(candidate_timestamps, name="candidate_timestamps")
    plays = _integers(candidate_play_time_ms, name="candidate_play_time_ms")
    durations = _integers(candidate_duration_ms, name="candidate_duration_ms")
    label_rows = _label_rows(labels)
    vectors = (items, timestamps, plays, durations)
    if any(len(vector) != KUAI_EXPOSURE_COUNT for vector in vectors):
        raise ValueError("Kuai exposure fields must all have frozen K=32")
    if any(item <= 0 for item in items):
        raise ValueError("candidate item IDs must be positive")
    if any(timestamp < 0 for timestamp in timestamps):
        raise ValueError("candidate timestamps must be nonnegative")
    if any(left > right for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("candidate timestamps must be nondecreasing")
    if any(value < 0 for value in (*plays, *durations)):
        raise ValueError("candidate play time and duration must be nonnegative")

    positions = list(range(KUAI_EXPOSURE_COUNT))
    canonical = {
        "schema": KUAI_EXPOSURE_IDENTITY_SCHEMA,
        "slate_id": slate_id,
        "user_id": user_id,
        "exposures": [
            {
                "exposure_position": position,
                "item_id": items[position],
                "timestamp": timestamps[position],
                "play_time_ms": plays[position],
                "duration_ms": durations[position],
                "labels": label_rows[position],
            }
            for position in positions
        ],
    }
    digest = hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "exposure_identity_schema": KUAI_EXPOSURE_IDENTITY_SCHEMA,
        "exposure_positions": positions,
        "candidate_timestamps": timestamps,
        "candidate_play_time_ms": plays,
        "candidate_duration_ms": durations,
        "exposure_identity_sha256": digest,
    }


def validate_kuai_exposure_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute and exactly validate the identity fields of one evidence row."""

    required = {
        "slate_id",
        "user_id",
        "candidate_item_ids",
        "candidate_timestamps",
        "candidate_play_time_ms",
        "candidate_duration_ms",
        "labels",
        "exposure_positions",
        "exposure_identity_schema",
        "exposure_identity_sha256",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"Kuai exposure identity is missing fields: {missing}")
    observed = kuai_exposure_identity(
        slate_id=str(row["slate_id"]),
        user_id=int(row["user_id"]),
        candidate_item_ids=row["candidate_item_ids"],
        candidate_timestamps=row["candidate_timestamps"],
        candidate_play_time_ms=row["candidate_play_time_ms"],
        candidate_duration_ms=row["candidate_duration_ms"],
        labels=row["labels"],
    )
    for field, expected in observed.items():
        if row.get(field) != expected:
            raise ValueError(f"Kuai exposure identity field {field} is stale or forged")
    return observed


__all__ = [
    "KUAI_EXPOSURE_COUNT",
    "KUAI_EXPOSURE_IDENTITY_SCHEMA",
    "KUAI_TASK_COUNT",
    "kuai_exposure_identity",
    "validate_kuai_exposure_identity",
]
