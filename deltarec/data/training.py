from __future__ import annotations

import csv

from pathlib import Path

import random

import sys

from typing import Any, Iterable, Iterator, Mapping, Sequence

from deltarec.adaptors.recbole import COMMON_MAX_HISTORY_LENGTH, KUAI_TASKS, RecBoleBridgeError

RATING_COLUMNS = {
    "user_id",
    "sequence_item_ids",
    "sequence_ratings",
    "sequence_timestamps",
}

SLATE_COLUMNS = {
    "slate_id",
    "user_id",
    "history_item_ids",
    "history_timestamps",
    "candidate_item_ids",
    *(f"label_{task}" for task in KUAI_TASKS),
}

def _allow_long_csv_fields() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10

def validate_sequence_input(path: Path) -> Path:
    source = path.expanduser().resolve()
    if "test" in str(source).lower():
        raise RecBoleBridgeError("training entry refuses every test-named sequence")
    _allow_long_csv_fields()
    with source.open(newline="", encoding="utf-8") as handle:
        columns = set(csv.DictReader(handle).fieldnames or ())
    if columns != RATING_COLUMNS:
        raise RecBoleBridgeError(
            f"rating columns mismatch: expected={sorted(RATING_COLUMNS)} actual={sorted(columns)}"
        )
    return source

def validate_slate_input(path: Path, *, role: str) -> Path:
    source = path.expanduser().resolve()
    if role not in {"train", "validation"}:
        raise ValueError("slate role must be train or validation")
    if "test" in str(source).lower():
        raise RecBoleBridgeError("training entry refuses every test-named slate")
    _allow_long_csv_fields()
    with source.open(newline="", encoding="utf-8") as handle:
        columns = set(csv.DictReader(handle).fieldnames or ())
    if not SLATE_COLUMNS.issubset(columns):
        raise RecBoleBridgeError(
            f"slate columns mismatch: missing={sorted(SLATE_COLUMNS - columns)}"
        )
    return source

def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(entry) for entry in value.split(",") if entry)

def _rating_examples(path: Path) -> Iterator[Mapping[str, Any]]:
    source = validate_sequence_input(path)
    with source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            items = _parse_ints(row["sequence_item_ids"])
            # The final item is the frozen validation target.  Every earlier
            # next-item prefix is training supervision, matching RecBole's
            # sequential expansion without exposing validation/test targets.
            for target_index in range(1, len(items) - 1):
                history = items[max(0, target_index - COMMON_MAX_HISTORY_LENGTH) : target_index]
                if history:
                    yield {
                        "user_id": int(row["user_id"]),
                        "history": history,
                        "target": items[target_index],
                    }

def _slate_examples(path: Path, *, role: str) -> Iterator[Mapping[str, Any]]:
    source = validate_slate_input(path, role=role)
    with source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            history = _parse_ints(row["history_item_ids"])[-COMMON_MAX_HISTORY_LENGTH:]
            candidates = _parse_ints(row["candidate_item_ids"])
            labels = tuple(_parse_ints(row[f"label_{task}"]) for task in KUAI_TASKS)
            if not history:
                raise RecBoleBridgeError(f"{role} slate has empty history: {row['slate_id']}")
            if len(candidates) != 32 or any(len(task) != 32 for task in labels):
                raise RecBoleBridgeError(f"{role} slate does not have K=32: {row['slate_id']}")
            yield {
                "user_id": int(row["user_id"]),
                "history": history,
                "candidates": candidates,
                "labels": tuple(zip(*labels)),
            }

def _bounded_shuffle(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    buffer_size: int = 32768,
) -> Iterator[Mapping[str, Any]]:
    if buffer_size <= 0:
        raise ValueError("shuffle buffer must be positive")
    generator = random.Random(seed)
    buffer: list[Mapping[str, Any]] = []
    for row in rows:
        if len(buffer) < buffer_size:
            buffer.append(row)
            continue
        index = generator.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = row
    generator.shuffle(buffer)
    yield from buffer

def _collate_rating(
    torch: Any,
    rows: Sequence[Mapping[str, Any]],
    device: Any,
    *,
    history_width: int | None = None,
) -> Mapping[str, Any]:
    """Collate rating prefixes, optionally using the batch's CPU-known width.

    The official DatasetV2 path keeps the protocol width of 1024 by default.
    DeltaRec's in-memory trajectory path can pass the maximum Python row
    length instead, avoiding a device ``lengths.max().item()`` just to crop a
    tensor that was already padded on the GPU.
    """

    width = COMMON_MAX_HISTORY_LENGTH if history_width is None else int(history_width)
    if width < 1 or width > COMMON_MAX_HISTORY_LENGTH:
        raise ValueError(f"rating history width must be in [1, {COMMON_MAX_HISTORY_LENGTH}]")
    if not rows:
        raise ValueError("cannot collate an empty rating batch")
    if any(len(row["history"]) > width for row in rows):
        raise ValueError("rating history exceeds the requested collate width")
    batch = {
        "histories": torch.tensor(
            [list(row["history"]) + [0] * (width - len(row["history"])) for row in rows],
            dtype=torch.long,
        ),
        "lengths": torch.tensor([len(row["history"]) for row in rows], dtype=torch.long),
        "targets": torch.tensor([row["target"] for row in rows], dtype=torch.long),
        "users": torch.tensor([row["user_id"] for row in rows], dtype=torch.long),
    }
    return {name: value.to(device) for name, value in batch.items()}

def _collate_slates(torch: Any, rows: Sequence[Mapping[str, Any]], device: Any) -> Mapping[str, Any]:
    batch = {
        "histories": torch.tensor(
            [
                list(row["history"])
                + [0] * (COMMON_MAX_HISTORY_LENGTH - len(row["history"]))
                for row in rows
            ],
            dtype=torch.long,
        ),
        "lengths": torch.tensor([len(row["history"]) for row in rows], dtype=torch.long),
        "candidates": torch.tensor([row["candidates"] for row in rows], dtype=torch.long),
        "labels": torch.tensor([row["labels"] for row in rows], dtype=torch.float32),
        "users": torch.tensor([row["user_id"] for row in rows], dtype=torch.long),
    }
    return {name: value.to(device) for name, value in batch.items()}

def _max_catalog_id(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "item_id" not in set(reader.fieldnames or ()):
            raise RecBoleBridgeError("full catalog must contain item_id")
        maximum = max((int(row["item_id"]) for row in reader), default=0)
    if maximum <= 0:
        raise RecBoleBridgeError("full catalog is empty")
    return maximum
