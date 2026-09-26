"""Materialize the frozen headline-v1 temporal splits and Kuai slates."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from array import array
from bisect import bisect_left
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .preprocess import KUAI_TASK_BITS
from .data_sources import file_digest, write_json_atomic


SPLIT_SCHEMA = "deltarec-headline-v1-frozen-splits-v1"
HEADLINE_MAX_HISTORY_LENGTH = 1024
RATING_FIELDS = (
    "user_id",
    "sequence_item_ids",
    "sequence_ratings",
    "sequence_timestamps",
)
KUAI_TASKS = tuple(name.removeprefix("is_") for name in KUAI_TASK_BITS)


def _allow_long_sequence_fields() -> None:
    # Long-tail users can produce multi-megabyte canonical CSV fields.  Raise
    # the parser ceiling deterministically without relying on site defaults.
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _parse_list(value: str, cast: Callable[[str], Any]) -> list[Any]:
    if value == "":
        return []
    return [cast(part) for part in value.split(",")]


def _join(values: Iterable[Any]) -> str:
    return ",".join(str(value) for value in values)


class _AtomicCsv:
    def __init__(self, path: Path, fieldnames: Sequence[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.temporary = path.with_name(path.name + ".tmp")
        self.handle = self.temporary.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.handle, fieldnames=fieldnames, lineterminator="\n"
        )
        self.writer.writeheader()

    def __enter__(self) -> "_AtomicCsv":
        return self

    def __exit__(self, kind: object, value: object, traceback: object) -> None:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        if kind is None:
            os.replace(self.temporary, self.path)
        else:
            try:
                self.temporary.unlink()
            except FileNotFoundError:
                pass


def _rating_row(
    user_id: str,
    items: Sequence[int],
    ratings: Sequence[float],
    timestamps: Sequence[int],
) -> dict[str, str]:
    return {
        "user_id": user_id,
        "sequence_item_ids": _join(items),
        "sequence_ratings": _join(ratings),
        "sequence_timestamps": _join(timestamps),
    }


def _write_catalog(path: Path, item_ids: set[int]) -> None:
    if not item_ids or min(item_ids) <= 0:
        raise ValueError("training catalog must contain positive item IDs")
    with _AtomicCsv(path, ("item_id",)) as output:
        for item_id in sorted(item_ids):
            output.writer.writerow({"item_id": item_id})


def _artifact_record(paths: Mapping[str, Path]) -> dict[str, Any]:
    return {
        name: {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": file_digest(path),
        }
        for name, path in sorted(paths.items())
    }


def materialize_rating_splits(
    sequence_path: str | Path,
    output_dir: str | Path,
    *,
    dataset: str,
    max_history_length: int = 1024,
) -> dict[str, Any]:
    """Freeze strict leave-two-out files consumable by official DatasetV2.

    The training/validation file ends at the validation target.  Consequently
    the official upstream ``ignore_last_n=1`` training path sees only the
    training history, while its direct evaluation path targets validation.
    Test is a different file and must not be mounted until validation choices
    have been locked.
    """

    if dataset not in {"ml-20m", "amazon-books"}:
        raise ValueError("rating splits support ML-20M or Amazon Books")
    if max_history_length != HEADLINE_MAX_HISTORY_LENGTH:
        raise ValueError("headline-v1 freezes max_history_length=1024")
    source = Path(sequence_path)
    _allow_long_sequence_fields()
    destination = Path(output_dir)
    train_validation_path = destination / "train_validation_sequences.csv"
    test_path = destination / "test_sequences.csv"
    catalog_path = destination / "train_catalog.csv"
    full_catalog_path = destination / "full_catalog.csv"
    train_catalog: set[int] = set()
    full_catalog: set[int] = set()
    source_users = kept_users = dropped_users = 0
    maximum_validation_history = maximum_test_history = 0
    timestamp_tie_exclusions = 0

    with source.open(newline="", encoding="utf-8") as handle, _AtomicCsv(
        train_validation_path, RATING_FIELDS
    ) as train_validation, _AtomicCsv(test_path, RATING_FIELDS) as test:
        reader = csv.DictReader(handle)
        missing = set(RATING_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"sequence file missing fields {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            source_users += 1
            items = _parse_list(row["sequence_item_ids"], int)
            ratings = _parse_list(row["sequence_ratings"], float)
            timestamps = _parse_list(row["sequence_timestamps"], int)
            if not (len(items) == len(ratings) == len(timestamps)) or len(items) < 3:
                raise ValueError(f"invalid rating sequence at row {row_number}")
            if any(item <= 0 for item in items):
                raise ValueError(f"non-positive rating item ID at row {row_number}")
            if not all(math.isfinite(rating) for rating in ratings):
                raise ValueError(f"non-finite rating at row {row_number}")
            if any(left > right for left, right in zip(timestamps, timestamps[1:])):
                raise ValueError(f"nonmonotonic sequence at row {row_number}")
            # Full catalog is an addressability/item-map artifact, not an
            # optimization/evaluation catalog. Include every canonical item,
            # even when a user is later dropped for lacking strict history.
            full_catalog.update(items)
            validation_index = len(items) - 2
            test_index = len(items) - 1
            validation_end = bisect_left(
                timestamps, timestamps[validation_index], 0, validation_index
            )
            test_end = bisect_left(
                timestamps, timestamps[test_index], 0, test_index
            )
            if validation_end < 1 or test_end < 1:
                dropped_users += 1
                continue
            validation_start = max(0, validation_end - max_history_length)
            test_start = max(0, test_end - max_history_length)
            validation_items = items[validation_start:validation_end] + [
                items[validation_index]
            ]
            validation_ratings = ratings[validation_start:validation_end] + [
                ratings[validation_index]
            ]
            validation_times = timestamps[validation_start:validation_end] + [
                timestamps[validation_index]
            ]
            test_items = items[test_start:test_end] + [items[test_index]]
            test_ratings = ratings[test_start:test_end] + [ratings[test_index]]
            test_times = timestamps[test_start:test_end] + [timestamps[test_index]]
            train_validation.writer.writerow(
                _rating_row(
                    row["user_id"],
                    validation_items,
                    validation_ratings,
                    validation_times,
                )
            )
            test.writer.writerow(
                _rating_row(row["user_id"], test_items, test_ratings, test_times)
            )
            train_catalog.update(items[:validation_end])
            kept_users += 1
            maximum_validation_history = max(
                maximum_validation_history, len(validation_items) - 1
            )
            maximum_test_history = max(maximum_test_history, len(test_items) - 1)
            timestamp_tie_exclusions += (
                validation_index - validation_end + test_index - test_end
            )

    _write_catalog(catalog_path, train_catalog)
    _write_catalog(full_catalog_path, full_catalog)
    artifacts = {
        "train_validation_sequences": train_validation_path,
        "test_sequences": test_path,
        "train_catalog": catalog_path,
        "full_catalog": full_catalog_path,
    }
    record = {
        "schema": SPLIT_SCHEMA,
        "dataset": dataset,
        "source": {
            "filename": source.name,
            "size_bytes": source.stat().st_size,
            "sha256": file_digest(source),
        },
        "parameters": {
            "protocol": "strict_event_leave_two_out",
            "history_cutoff": "timestamp_strictly_less_than_target",
            "max_history_length": max_history_length,
            "test_visibility": "test_file_mounted_only_after_protocol_lock",
            "embedding_addressability_catalog": "full_catalog",
            "negative_sampling_catalog": "train_catalog",
            "retrieval_candidate_catalog": "train_catalog",
            "group_fit_catalog": "train_catalog",
        },
        "statistics": {
            "source_users": source_users,
            "kept_users": kept_users,
            "dropped_users_without_training_history": dropped_users,
            "training_catalog_items": len(train_catalog),
            "full_catalog_items": len(full_catalog),
            "maximum_validation_history": maximum_validation_history,
            "maximum_test_history": maximum_test_history,
            "timestamp_tie_exclusions": timestamp_tie_exclusions,
        },
        "artifacts": _artifact_record(artifacts),
    }
    record["manifest_content_sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    write_json_atomic(destination / "split_manifest.json", record)
    return record


KUAI_SLATE_FIELDS = (
    "slate_id",
    "user_id",
    "history_item_ids",
    "history_timestamps",
    "history_action_masks",
    "history_play_time_ms",
    "history_duration_ms",
    "candidate_item_ids",
    "candidate_timestamps",
    "candidate_play_time_ms",
    "candidate_duration_ms",
    *(f"label_{task}" for task in KUAI_TASKS),
)


def reconstruct_kuai_candidate_action_masks(row: Mapping[str, str]) -> list[int]:
    """Losslessly reconstruct official 1..128 action bitmasks from slate labels."""

    columns: list[tuple[int, list[int]]] = []
    expected_length: int | None = None
    for raw_name, bit in KUAI_TASK_BITS.items():
        task = raw_name.removeprefix("is_")
        field = f"label_{task}"
        if field not in row:
            raise ValueError(f"Kuai slate is missing {field}")
        labels = _parse_list(row[field], int)
        if any(label not in {0, 1} for label in labels):
            raise ValueError(f"Kuai {field} must be binary")
        if expected_length is None:
            expected_length = len(labels)
        elif len(labels) != expected_length:
            raise ValueError("Kuai candidate label columns are unaligned")
        columns.append((bit, labels))
    return [
        sum(bit * labels[index] for bit, labels in columns)
        for index in range(expected_length or 0)
    ]


def _kuai_timestamps(sequence_path: Path) -> array:
    _allow_long_sequence_fields()
    values = array("q")
    with sequence_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "sequence_timestamps" not in set(reader.fieldnames or ()):
            raise ValueError("Kuai sequence file lacks sequence_timestamps")
        for row in reader:
            values.extend(_parse_list(row["sequence_timestamps"], int))
    if len(values) < 10:
        raise ValueError("Kuai temporal split requires at least ten exposures")
    return values


def _temporal_boundaries(values: array) -> tuple[int, int, dict[str, int]]:
    ordered = np.frombuffer(values, dtype=np.int64)
    ordered.sort(kind="stable")
    validation_index = int(math.floor(0.80 * len(ordered)))
    test_index = int(math.floor(0.90 * len(ordered)))
    validation_start = int(ordered[validation_index])
    test_start = int(ordered[test_index])
    if validation_start >= test_start:
        raise ValueError("Kuai global 80/10/10 boundaries collapse under timestamp ties")
    train_count = int(np.searchsorted(ordered, validation_start, side="left"))
    validation_count = int(
        np.searchsorted(ordered, test_start, side="left") - train_count
    )
    return validation_start, test_start, {
        "train_exposures": train_count,
        "validation_exposures": validation_count,
        "test_exposures": len(ordered) - train_count - validation_count,
        "total_exposures": len(ordered),
    }


def materialize_kuai_splits(
    sequence_path: str | Path,
    output_dir: str | Path,
    *,
    max_history_length: int = 1024,
    candidate_count: int = 32,
) -> dict[str, Any]:
    """Freeze global 80/10/10 Kuai partitions and nonoverlapping K=32 slates."""

    if max_history_length != HEADLINE_MAX_HISTORY_LENGTH or candidate_count != 32:
        raise ValueError("headline Kuai requires max_history_length=1024 and K=32")
    source = Path(sequence_path)
    _allow_long_sequence_fields()
    destination = Path(output_dir)
    timestamps = _kuai_timestamps(source)
    validation_start, test_start, exposure_counts = _temporal_boundaries(timestamps)
    del timestamps
    paths = {
        "train_slates": destination / "train_slates.csv",
        "validation_slates": destination / "validation_slates.csv",
        "test_slates": destination / "test_slates.csv",
    }
    split_bounds = {
        "train": (None, validation_start),
        "validation": (validation_start, test_start),
        "test": (test_start, None),
    }
    slate_counts = {split: 0 for split in split_bounds}
    tail_exposures = {split: 0 for split in split_bounds}
    no_history_windows = {split: 0 for split in split_bounds}
    history_events_materialized = {split: 0 for split in split_bounds}
    maximum_history_lengths = {split: 0 for split in split_bounds}
    candidate_bitmask_roundtrip_checks = {split: 0 for split in split_bounds}
    train_catalog: set[int] = set()
    writers = {
        split: _AtomicCsv(paths[f"{split}_slates"], KUAI_SLATE_FIELDS)
        for split in split_bounds
    }
    entered: list[_AtomicCsv] = []
    try:
        for writer in writers.values():
            entered.append(writer.__enter__())
        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {
                "user_id",
                "sequence_item_ids",
                "sequence_timestamps",
                "sequence_action_masks",
                "sequence_play_time_ms",
                "sequence_duration_ms",
            }
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"Kuai sequence file missing {sorted(missing)}")
            for row_number, row in enumerate(reader, start=2):
                items = _parse_list(row["sequence_item_ids"], int)
                times = _parse_list(row["sequence_timestamps"], int)
                masks = _parse_list(row["sequence_action_masks"], int)
                plays = _parse_list(row["sequence_play_time_ms"], int)
                durations = _parse_list(row["sequence_duration_ms"], int)
                if not (
                    len(items) == len(times) == len(masks) == len(plays) == len(durations)
                ):
                    raise ValueError(f"unaligned Kuai sequence at row {row_number}")
                if any(item <= 0 for item in items):
                    raise ValueError(f"non-positive Kuai item ID at row {row_number}")
                if any(left > right for left, right in zip(times, times[1:])):
                    raise ValueError(f"nonmonotonic Kuai sequence at row {row_number}")
                if any(mask < 0 or mask > 255 for mask in masks):
                    raise ValueError(f"invalid Kuai action mask at row {row_number}")
                if any(value < 0 for value in plays) or any(
                    value < 0 for value in durations
                ):
                    raise ValueError(f"negative Kuai duration at row {row_number}")
                train_end = bisect_left(times, validation_start)
                train_catalog.update(items[:train_end])
                for split, (lower, upper) in split_bounds.items():
                    begin = 0 if lower is None else bisect_left(times, lower)
                    end = len(times) if upper is None else bisect_left(times, upper)
                    usable = ((end - begin) // candidate_count) * candidate_count
                    tail_exposures[split] += end - begin - usable
                    for window_start in range(begin, begin + usable, candidate_count):
                        window_end = window_start + candidate_count
                        earliest = times[window_start]
                        history_end = bisect_left(times, earliest, 0, window_start)
                        if history_end == 0:
                            no_history_windows[split] += 1
                            continue
                        history_start = max(0, history_end - max_history_length)
                        history_length = history_end - history_start
                        slate_id = f"{split}-{slate_counts[split]:09d}"
                        output: dict[str, Any] = {
                            "slate_id": slate_id,
                            "user_id": row["user_id"],
                            "history_item_ids": _join(items[history_start:history_end]),
                            "history_timestamps": _join(times[history_start:history_end]),
                            "history_action_masks": _join(
                                masks[history_start:history_end]
                            ),
                            "history_play_time_ms": _join(
                                plays[history_start:history_end]
                            ),
                            "history_duration_ms": _join(
                                durations[history_start:history_end]
                            ),
                            "candidate_item_ids": _join(items[window_start:window_end]),
                            "candidate_timestamps": _join(times[window_start:window_end]),
                            "candidate_play_time_ms": _join(plays[window_start:window_end]),
                            "candidate_duration_ms": _join(
                                durations[window_start:window_end]
                            ),
                        }
                        for raw_name, bit in KUAI_TASK_BITS.items():
                            task = raw_name.removeprefix("is_")
                            output[f"label_{task}"] = _join(
                                1 if mask & bit else 0
                                for mask in masks[window_start:window_end]
                            )
                        reconstructed_masks = reconstruct_kuai_candidate_action_masks(
                            output
                        )
                        expected_masks = masks[window_start:window_end]
                        if reconstructed_masks != expected_masks:
                            raise RuntimeError(
                                "candidate task labels do not reconstruct source action masks"
                            )
                        writers[split].writer.writerow(output)
                        slate_counts[split] += 1
                        history_events_materialized[split] += history_length
                        maximum_history_lengths[split] = max(
                            maximum_history_lengths[split], history_length
                        )
                        candidate_bitmask_roundtrip_checks[split] += candidate_count
    except BaseException as error:
        for writer in reversed(entered):
            writer.__exit__(type(error), error, error.__traceback__)
        raise
    else:
        for writer in reversed(entered):
            writer.__exit__(None, None, None)

    catalog_path = destination / "train_catalog.csv"
    _write_catalog(catalog_path, train_catalog)
    artifacts = {**paths, "train_catalog": catalog_path}
    total = exposure_counts["total_exposures"]
    record = {
        "schema": SPLIT_SCHEMA,
        "dataset": "kuairand-1k",
        "source": {
            "filename": source.name,
            "size_bytes": source.stat().st_size,
            "sha256": file_digest(source),
        },
        "parameters": {
            "protocol": "global_timestamp_80_10_10",
            "same_timestamp_isolation": True,
            "candidate_count": candidate_count,
            "window_stride": candidate_count,
            "tail_policy": "drop",
            "history_cutoff": "strictly_before_earliest_candidate_timestamp",
            "max_history_length": max_history_length,
            "slate_fields": list(KUAI_SLATE_FIELDS),
            "action_mask_bits": dict(KUAI_TASK_BITS),
            "candidate_label_bit_order": [
                f"label_{name.removeprefix('is_')}" for name in KUAI_TASK_BITS
            ],
            "candidate_labels_losslessly_reconstruct_action_masks": True,
        },
        "boundaries": {
            "validation_start_timestamp": validation_start,
            "test_start_timestamp": test_start,
        },
        "statistics": {
            **exposure_counts,
            "realized_exposure_ratios": {
                "train": exposure_counts["train_exposures"] / total,
                "validation": exposure_counts["validation_exposures"] / total,
                "test": exposure_counts["test_exposures"] / total,
            },
            "slates": slate_counts,
            "tail_exposures_dropped": tail_exposures,
            "zero_history_windows_dropped": no_history_windows,
            "history_events_materialized": history_events_materialized,
            "maximum_history_lengths": maximum_history_lengths,
            "candidate_bitmask_roundtrip_checks": candidate_bitmask_roundtrip_checks,
            "training_catalog_items": len(train_catalog),
        },
        "artifacts": _artifact_record(artifacts),
    }
    record["manifest_content_sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    write_json_atomic(destination / "split_manifest.json", record)
    return record


__all__ = [
    "HEADLINE_MAX_HISTORY_LENGTH",
    "SPLIT_SCHEMA",
    "materialize_rating_splits",
    "materialize_kuai_splits",
    "reconstruct_kuai_candidate_action_masks",
]
