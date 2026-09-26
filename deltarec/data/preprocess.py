"""Deterministic, streaming preprocessors for the three headline datasets."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .data_sources import canonical_json_digest, file_digest, write_json_atomic


PREPROCESS_SCHEMA = "deltarec-headline-v1-preprocess-v1"
SEQUENCE_SCHEMA = "deltarec-headline-v1-sequence-csv-v1"
KUAI_TASK_BITS: dict[str, int] = {
    "is_click": 1,
    "is_like": 2,
    "is_follow": 4,
    "is_comment": 8,
    "is_forward": 16,
    "is_hate": 32,
    "long_view": 64,
    "is_profile_enter": 128,
}
KUAI_CONTEXT_FEATURES = (
    "user_active_degree",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
)


@dataclass(frozen=True)
class SequenceRow:
    user_id: int | str
    item_ids: tuple[int, ...]
    timestamps: tuple[int, ...]
    values: tuple[float, ...] | None = None
    action_masks: tuple[int, ...] | None = None
    play_time_ms: tuple[int, ...] | None = None
    duration_ms: tuple[int, ...] | None = None


def _join(values: Sequence[int | float]) -> str:
    return ",".join(str(value) for value in values)


def _temporary_output(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(path.name + ".tmp")


def _validate_sequence(row: SequenceRow) -> None:
    length = len(row.item_ids)
    if len(row.timestamps) != length:
        raise ValueError(f"unaligned timestamps for user {row.user_id!r}")
    for optional in (row.values, row.action_masks, row.play_time_ms, row.duration_ms):
        if optional is not None and len(optional) != length:
            raise ValueError(f"unaligned sequence field for user {row.user_id!r}")
    if any(item <= 0 for item in row.item_ids):
        raise ValueError("canonical item IDs must be positive; zero is padding")
    if any(left > right for left, right in zip(row.timestamps, row.timestamps[1:])):
        raise ValueError(f"nonmonotonic timestamps for user {row.user_id!r}")


class _SequenceCsvWriter:
    def __init__(self, output_path: Path, *, kind: str) -> None:
        if kind not in {"ratings", "kuairand"}:
            raise ValueError(f"unknown sequence kind {kind!r}")
        self.output_path = output_path
        self.temporary_path = _temporary_output(output_path)
        self.handle = self.temporary_path.open("w", newline="", encoding="utf-8")
        if kind == "ratings":
            fields = [
                "user_id",
                "sequence_item_ids",
                "sequence_ratings",
                "sequence_timestamps",
            ]
        else:
            fields = [
                "user_id",
                "sequence_item_ids",
                "sequence_timestamps",
                "sequence_action_masks",
                "sequence_play_time_ms",
                "sequence_duration_ms",
            ]
        self.kind = kind
        self.writer = csv.DictWriter(self.handle, fieldnames=fields, lineterminator="\n")
        self.writer.writeheader()

    def write(self, row: SequenceRow) -> None:
        _validate_sequence(row)
        if self.kind == "ratings":
            if row.values is None:
                raise ValueError("ratings sequence requires values")
            output = {
                "user_id": row.user_id,
                "sequence_item_ids": _join(row.item_ids),
                "sequence_ratings": _join(row.values),
                "sequence_timestamps": _join(row.timestamps),
            }
        else:
            if row.action_masks is None or row.play_time_ms is None or row.duration_ms is None:
                raise ValueError("KuaiRand sequence requires action/play/duration fields")
            output = {
                "user_id": row.user_id,
                "sequence_item_ids": _join(row.item_ids),
                "sequence_timestamps": _join(row.timestamps),
                "sequence_action_masks": _join(row.action_masks),
                "sequence_play_time_ms": _join(row.play_time_ms),
                "sequence_duration_ms": _join(row.duration_ms),
            }
        self.writer.writerow(output)

    def close(self, *, commit: bool) -> None:
        if not self.handle.closed:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()
        if commit:
            os.replace(self.temporary_path, self.output_path)
        else:
            try:
                self.temporary_path.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self) -> "_SequenceCsvWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close(commit=exc_type is None)


def _parse_int(value: str, *, field: str, row_number: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid integer {field} at source row {row_number}") from error


def _write_preprocess_record(
    output_dir: Path,
    *,
    dataset: str,
    sources: Mapping[str, Path],
    artifacts: Mapping[str, Path],
    parameters: Mapping[str, Any],
    statistics: Mapping[str, Any],
) -> dict[str, Any]:
    record = {
        "schema": PREPROCESS_SCHEMA,
        "sequence_schema": SEQUENCE_SCHEMA,
        "dataset": dataset,
        "sources": {
            name: {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": file_digest(path),
            }
            for name, path in sorted(sources.items())
        },
        "artifacts": {
            name: {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": file_digest(path),
            }
            for name, path in sorted(artifacts.items())
        },
        "parameters": dict(parameters),
        "statistics": dict(statistics),
        "code": {
            "preprocess.py": file_digest(Path(__file__)),
            "splits.py": file_digest(Path(__file__).with_name("splits.py")),
        },
    }
    record["manifest_content_sha256"] = canonical_json_digest(record)
    write_json_atomic(output_dir / "preprocess_manifest.json", record)
    return record


def preprocess_movielens_20m(
    ratings_path: str | Path,
    output_dir: str | Path,
    *,
    minimum_sequence_length: int = 3,
) -> dict[str, Any]:
    """Stream official ML-20M ratings into canonical chronological sequences.

    GroupLens stores each user's rows contiguously.  We sort every user by
    ``(timestamp, source_row_ordinal)`` and fail if a closed user reappears,
    avoiding an unbounded 20M-row in-memory regrouping while still making the
    ordering contract explicit and auditable.
    """

    if minimum_sequence_length < 3:
        raise ValueError("minimum_sequence_length must be at least 3")
    source = Path(ratings_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    sequence_path = destination / "sequences.csv"
    rows_read = 0
    interactions_written = 0
    users_written = 0
    reordered_users = 0
    unique_items: set[int] = set()
    closed_users: set[int] = set()
    current_user: int | None = None
    current: list[tuple[int, int, int, float]] = []

    def flush(writer: _SequenceCsvWriter) -> None:
        nonlocal interactions_written, users_written, reordered_users, current
        if current_user is None:
            return
        ordered = sorted(current, key=lambda entry: (entry[0], entry[1]))
        if ordered != current:
            reordered_users += 1
        if len(ordered) >= minimum_sequence_length:
            writer.write(
                SequenceRow(
                    user_id=current_user,
                    item_ids=tuple(entry[2] for entry in ordered),
                    timestamps=tuple(entry[0] for entry in ordered),
                    values=tuple(entry[3] for entry in ordered),
                )
            )
            users_written += 1
            interactions_written += len(ordered)
        current = []

    with source.open(newline="", encoding="utf-8") as handle, _SequenceCsvWriter(
        sequence_path, kind="ratings"
    ) as writer:
        reader = csv.DictReader(handle)
        required = {"userId", "movieId", "rating", "timestamp"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"ML-20M ratings missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            rows_read += 1
            user_id = _parse_int(row["userId"], field="userId", row_number=row_number)
            item_id = _parse_int(row["movieId"], field="movieId", row_number=row_number)
            timestamp = _parse_int(
                row["timestamp"], field="timestamp", row_number=row_number
            )
            try:
                rating = float(row["rating"])
            except ValueError as error:
                raise ValueError(f"invalid rating at source row {row_number}") from error
            if user_id <= 0 or item_id <= 0:
                raise ValueError(f"non-positive ID at source row {row_number}")
            if current_user is None:
                current_user = user_id
            elif user_id != current_user:
                flush(writer)
                closed_users.add(current_user)
                if user_id in closed_users:
                    raise ValueError(
                        "ML-20M source is not grouped contiguously by user; use a "
                        "canonical external sort before preprocessing"
                    )
                current_user = user_id
            current.append((timestamp, row_number, item_id, rating))
            unique_items.add(item_id)
        flush(writer)

    item_map_path = destination / "item_id_map.csv"
    with item_map_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["raw_movie_id", "model_id"])
        writer.writerows((item, item) for item in sorted(unique_items))

    statistics = {
        "source_rows": rows_read,
        "eligible_users": users_written,
        "interactions": interactions_written,
        "unique_items": len(unique_items),
        "reordered_users": reordered_users,
        "max_item_id": max(unique_items) if unique_items else 0,
    }
    return _write_preprocess_record(
        destination,
        dataset="ml-20m",
        sources={"ratings": source},
        artifacts={"sequences": sequence_path, "item_id_map": item_map_path},
        parameters={
            "minimum_sequence_length": minimum_sequence_length,
            "item_id_policy": "preserve_positive_movielens_id",
            "sort_key": ["timestamp", "source_row_ordinal"],
            "split": "event_leave_two_out_with_strict_timestamp_histories",
        },
        statistics=statistics,
    )


def _connect_disposable_sqlite(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-131072")  # 128 MiB
    return connection


def _write_id_map(
    path: Path, rows: Iterable[tuple[str | int, int]], *, raw_field: str
) -> int:
    temporary = _temporary_output(path)
    count = 0
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([raw_field, "model_id"])
        for raw_id, model_id in rows:
            writer.writerow([raw_id, model_id])
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return count


def preprocess_amazon_books(
    ratings_path: str | Path,
    output_dir: str | Path,
    *,
    minimum_user_count: int = 5,
    minimum_item_count: int = 5,
    minimum_sequence_length: int = 5,
) -> dict[str, Any]:
    """Build the deterministic maximal Amazon user/item k-core.

    User and ASIN vertices below their thresholds are peeled repeatedly until
    a fixed point.  ``minimum_sequence_length`` participates in the user
    threshold so dropping short sequences cannot leave sub-threshold items in
    the emitted graph.  Surviving raw strings receive binary-lexicographic
    categorical IDs, shifted to start at one for items so zero remains
    padding.  SQLite supplies bounded-memory filtering and external sorting
    for the 22.5M-row source.
    """

    if min(minimum_user_count, minimum_item_count, minimum_sequence_length) < 1:
        raise ValueError("Amazon filtering thresholds must be positive")
    source = Path(ratings_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    sequence_path = destination / "sequences.csv"
    item_map_path = destination / "item_id_map.csv"
    user_map_path = destination / "user_id_map.csv"

    with tempfile.TemporaryDirectory(prefix="amazon-books-sort-", dir=destination) as tmp:
        database_path = Path(tmp) / "sort.sqlite3"
        connection = _connect_disposable_sqlite(database_path)
        try:
            connection.execute(
                "CREATE TABLE raw (row_ordinal INTEGER PRIMARY KEY, user_raw TEXT NOT NULL, "
                "item_raw TEXT NOT NULL, rating REAL NOT NULL, timestamp INTEGER NOT NULL)"
            )
            batch: list[tuple[int, str, str, float, int]] = []
            with source.open(newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                for row_ordinal, row in enumerate(reader):
                    if len(row) != 4:
                        raise ValueError(
                            f"Amazon ratings row {row_ordinal + 1} has {len(row)} columns"
                        )
                    user_raw, item_raw, rating_text, timestamp_text = row
                    if not user_raw or not item_raw:
                        raise ValueError(f"empty Amazon ID at row {row_ordinal + 1}")
                    try:
                        rating = float(rating_text)
                        timestamp = int(timestamp_text)
                    except ValueError as error:
                        raise ValueError(
                            f"invalid Amazon numeric value at row {row_ordinal + 1}"
                        ) from error
                    batch.append((row_ordinal, user_raw, item_raw, rating, timestamp))
                    if len(batch) >= 50_000:
                        connection.executemany("INSERT INTO raw VALUES (?,?,?,?,?)", batch)
                        batch.clear()
                if batch:
                    connection.executemany("INSERT INTO raw VALUES (?,?,?,?,?)", batch)
            source_rows = int(connection.execute("SELECT COUNT(*) FROM raw").fetchone()[0])
            source_users = int(
                connection.execute("SELECT COUNT(DISTINCT user_raw) FROM raw").fetchone()[0]
            )
            source_items = int(
                connection.execute("SELECT COUNT(DISTINCT item_raw) FROM raw").fetchone()[0]
            )
            connection.execute("CREATE TABLE filtered AS SELECT * FROM raw")
            connection.execute("CREATE INDEX filtered_user ON filtered(user_raw)")
            connection.execute("CREATE INDEX filtered_item ON filtered(item_raw)")
            effective_user_count = max(minimum_user_count, minimum_sequence_length)
            pruning_trace: list[dict[str, int]] = []
            while True:
                connection.execute("DROP TABLE IF EXISTS low_degree_user")
                connection.execute(
                    "CREATE TEMP TABLE low_degree_user AS "
                    "SELECT user_raw FROM filtered GROUP BY user_raw HAVING COUNT(*) < ?",
                    (effective_user_count,),
                )
                removed_users = int(
                    connection.execute("SELECT COUNT(*) FROM low_degree_user").fetchone()[0]
                )
                before_users = int(
                    connection.execute("SELECT COUNT(*) FROM filtered").fetchone()[0]
                )
                if removed_users:
                    connection.execute(
                        "DELETE FROM filtered WHERE user_raw IN "
                        "(SELECT user_raw FROM low_degree_user)"
                    )
                after_users = int(
                    connection.execute("SELECT COUNT(*) FROM filtered").fetchone()[0]
                )

                connection.execute("DROP TABLE IF EXISTS low_degree_item")
                connection.execute(
                    "CREATE TEMP TABLE low_degree_item AS "
                    "SELECT item_raw FROM filtered GROUP BY item_raw HAVING COUNT(*) < ?",
                    (minimum_item_count,),
                )
                removed_items = int(
                    connection.execute("SELECT COUNT(*) FROM low_degree_item").fetchone()[0]
                )
                before_items = after_users
                if removed_items:
                    connection.execute(
                        "DELETE FROM filtered WHERE item_raw IN "
                        "(SELECT item_raw FROM low_degree_item)"
                    )
                after_items = int(
                    connection.execute("SELECT COUNT(*) FROM filtered").fetchone()[0]
                )
                pruning_trace.append(
                    {
                        "iteration": len(pruning_trace) + 1,
                        "removed_users": removed_users,
                        "removed_items": removed_items,
                        "removed_user_interactions": before_users - after_users,
                        "removed_item_interactions": before_items - after_items,
                        "remaining_interactions": after_items,
                    }
                )
                if removed_users == 0 and removed_items == 0:
                    break

            filtered_rows = int(
                connection.execute("SELECT COUNT(*) FROM filtered").fetchone()[0]
            )
            if filtered_rows == 0:
                raise RuntimeError("Amazon iterative k-core is empty")
            core_users = int(
                connection.execute(
                    "SELECT COUNT(DISTINCT user_raw) FROM filtered"
                ).fetchone()[0]
            )
            core_items = int(
                connection.execute(
                    "SELECT COUNT(DISTINCT item_raw) FROM filtered"
                ).fetchone()[0]
            )
            minimum_observed_user_count = int(
                connection.execute(
                    "SELECT MIN(degree) FROM "
                    "(SELECT COUNT(*) AS degree FROM filtered GROUP BY user_raw)"
                ).fetchone()[0]
            )
            minimum_observed_item_count = int(
                connection.execute(
                    "SELECT MIN(degree) FROM "
                    "(SELECT COUNT(*) AS degree FROM filtered GROUP BY item_raw)"
                ).fetchone()[0]
            )
            if (
                minimum_observed_user_count < effective_user_count
                or minimum_observed_item_count < minimum_item_count
            ):
                raise RuntimeError("Amazon iterative k-core did not reach a fixed point")
            connection.execute(
                "CREATE TABLE user_map (user_raw TEXT PRIMARY KEY, model_id INTEGER UNIQUE)"
            )
            connection.execute(
                "CREATE TABLE item_map (item_raw TEXT PRIMARY KEY, model_id INTEGER UNIQUE)"
            )
            user_values = connection.execute(
                "SELECT DISTINCT user_raw FROM filtered ORDER BY user_raw COLLATE BINARY"
            )
            connection.executemany(
                "INSERT INTO user_map VALUES (?,?)",
                ((raw, model_id) for model_id, (raw,) in enumerate(user_values)),
            )
            item_values = connection.execute(
                "SELECT DISTINCT item_raw FROM filtered ORDER BY item_raw COLLATE BINARY"
            )
            connection.executemany(
                "INSERT INTO item_map VALUES (?,?)",
                ((raw, model_id) for model_id, (raw,) in enumerate(item_values, start=1)),
            )
            _write_id_map(
                user_map_path,
                connection.execute("SELECT user_raw, model_id FROM user_map ORDER BY model_id"),
                raw_field="raw_user_id",
            )
            unique_items = _write_id_map(
                item_map_path,
                connection.execute("SELECT item_raw, model_id FROM item_map ORDER BY model_id"),
                raw_field="raw_item_id",
            )
            query = connection.execute(
                "SELECT um.model_id, im.model_id, f.rating, f.timestamp, f.row_ordinal "
                "FROM filtered f JOIN user_map um USING(user_raw) "
                "JOIN item_map im USING(item_raw) "
                "ORDER BY um.model_id, f.timestamp, f.row_ordinal"
            )
            users_written = 0
            interactions_written = 0
            current_user: int | None = None
            current: list[tuple[int, float, int]] = []

            def flush(writer: _SequenceCsvWriter) -> None:
                nonlocal current, users_written, interactions_written
                if current_user is not None and len(current) >= minimum_sequence_length:
                    writer.write(
                        SequenceRow(
                            user_id=current_user,
                            item_ids=tuple(value[0] for value in current),
                            values=tuple(value[1] for value in current),
                            timestamps=tuple(value[2] for value in current),
                        )
                    )
                    users_written += 1
                    interactions_written += len(current)
                current = []

            with _SequenceCsvWriter(sequence_path, kind="ratings") as writer:
                for user_id, item_id, rating, timestamp, _ in query:
                    if current_user is None:
                        current_user = int(user_id)
                    elif user_id != current_user:
                        flush(writer)
                        current_user = int(user_id)
                    current.append((int(item_id), float(rating), int(timestamp)))
                flush(writer)
        finally:
            connection.close()

    statistics = {
        "source_rows": source_rows,
        "source_users": source_users,
        "source_items": source_items,
        "filtered_interactions": filtered_rows,
        "mapped_items": unique_items,
        "eligible_users": users_written,
        "interactions": interactions_written,
        "k_core_users": core_users,
        "k_core_items": core_items,
        "minimum_observed_user_count": minimum_observed_user_count,
        "minimum_observed_item_count": minimum_observed_item_count,
        "k_core_pruning_iterations": len(pruning_trace) - 1,
        "k_core_pruning_trace": pruning_trace,
    }
    return _write_preprocess_record(
        destination,
        dataset="amazon-books",
        sources={"ratings": source},
        artifacts={
            "sequences": sequence_path,
            "item_id_map": item_map_path,
            "user_id_map": user_map_path,
        },
        parameters={
            "minimum_user_count": minimum_user_count,
            "minimum_item_count": minimum_item_count,
            "minimum_sequence_length": minimum_sequence_length,
            "effective_user_count": max(
                minimum_user_count, minimum_sequence_length
            ),
            "filter": "deterministic_iterative_bipartite_k_core_to_fixed_point",
            "id_policy": "utf8_binary_lexicographic_categories_shifted_plus_one",
            "sort_key": ["timestamp", "source_row_ordinal"],
            "split": "event_leave_two_out_with_strict_timestamp_histories",
        },
        statistics=statistics,
    )


def preprocess_kuairand_1k(
    log_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    user_features_path: str | Path | None = None,
    minimum_sequence_length: int = 3,
    require_user_in_all_logs: bool = True,
    item_id_shift: int = 1,
) -> dict[str, Any]:
    """Canonicalize KuaiRand standard logs and published eight-task bitmasks."""

    if len(log_paths) < 1:
        raise ValueError("at least one KuaiRand log is required")
    if minimum_sequence_length < 1:
        raise ValueError("minimum_sequence_length must be positive")
    if item_id_shift < 1:
        raise ValueError("item_id_shift must reserve zero for padding")
    sources = [Path(path) for path in log_paths]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    sequence_path = destination / "sequences.csv"
    item_map_path = destination / "item_id_map.csv"
    user_features_output_path = destination / "user_features.csv"
    source_rows = 0

    with tempfile.TemporaryDirectory(prefix="kuairand-sort-", dir=destination) as tmp:
        connection = _connect_disposable_sqlite(Path(tmp) / "sort.sqlite3")
        try:
            connection.execute(
                "CREATE TABLE events (file_rank INTEGER NOT NULL, row_ordinal INTEGER NOT NULL, "
                "user_id INTEGER NOT NULL, video_id INTEGER NOT NULL, time_ms INTEGER NOT NULL, "
                "action_mask INTEGER NOT NULL, play_time_ms INTEGER NOT NULL, "
                "duration_ms INTEGER NOT NULL, PRIMARY KEY(file_rank,row_ordinal))"
            )
            for file_rank, path in enumerate(sources):
                batch: list[tuple[int, int, int, int, int, int, int, int]] = []
                with path.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    required = {
                        "user_id",
                        "video_id",
                        "time_ms",
                        "play_time_ms",
                        "duration_ms",
                        *KUAI_TASK_BITS,
                    }
                    missing = required - set(reader.fieldnames or ())
                    if missing:
                        raise ValueError(f"KuaiRand log {path} missing {sorted(missing)}")
                    for row_ordinal, row in enumerate(reader):
                        source_rows += 1
                        user_id = _parse_int(
                            row["user_id"], field="user_id", row_number=row_ordinal + 2
                        )
                        video_id = _parse_int(
                            row["video_id"], field="video_id", row_number=row_ordinal + 2
                        )
                        time_ms = _parse_int(
                            row["time_ms"], field="time_ms", row_number=row_ordinal + 2
                        )
                        play_time_ms = _parse_int(
                            row["play_time_ms"],
                            field="play_time_ms",
                            row_number=row_ordinal + 2,
                        )
                        duration_ms = _parse_int(
                            row["duration_ms"],
                            field="duration_ms",
                            row_number=row_ordinal + 2,
                        )
                        if video_id < 0:
                            raise ValueError("KuaiRand video_id must be nonnegative")
                        action_mask = 0
                        for task, bit in KUAI_TASK_BITS.items():
                            action = _parse_int(
                                row[task], field=task, row_number=row_ordinal + 2
                            )
                            if action not in (0, 1):
                                raise ValueError(f"KuaiRand task {task} must be binary")
                            action_mask |= bit if action else 0
                        batch.append(
                            (
                                file_rank,
                                row_ordinal,
                                user_id,
                                video_id,
                                time_ms,
                                action_mask,
                                play_time_ms,
                                duration_ms,
                            )
                        )
                        if len(batch) >= 50_000:
                            connection.executemany("INSERT INTO events VALUES (?,?,?,?,?,?,?,?)", batch)
                            batch.clear()
                    if batch:
                        connection.executemany("INSERT INTO events VALUES (?,?,?,?,?,?,?,?)", batch)
            connection.execute("CREATE INDEX events_user ON events(user_id)")
            if require_user_in_all_logs:
                connection.execute(
                    "CREATE TABLE eligible_user AS SELECT user_id FROM events GROUP BY user_id "
                    "HAVING COUNT(DISTINCT file_rank) = ?",
                    (len(sources),),
                )
            else:
                connection.execute(
                    "CREATE TABLE eligible_user AS SELECT DISTINCT user_id FROM events"
                )
            connection.execute("CREATE UNIQUE INDEX eligible_user_id ON eligible_user(user_id)")
            contextual_feature_categories: dict[str, dict[str, int]] | None = None
            if user_features_path is not None:
                contextual_feature_categories = {
                    feature: {} for feature in KUAI_CONTEXT_FEATURES
                }
                connection.execute(
                    "CREATE TABLE user_feature (user_id INTEGER PRIMARY KEY, "
                    + ", ".join(f"{feature} INTEGER NOT NULL" for feature in KUAI_CONTEXT_FEATURES)
                    + ")"
                )
                feature_rows: list[tuple[int, ...]] = []
                with Path(user_features_path).open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    required = {"user_id", *KUAI_CONTEXT_FEATURES}
                    missing = required - set(reader.fieldnames or ())
                    if missing:
                        raise ValueError(
                            f"KuaiRand user features missing {sorted(missing)}"
                        )
                    seen_feature_users: set[int] = set()
                    for row_number, row in enumerate(reader, start=2):
                        user_id = _parse_int(
                            row["user_id"], field="user_id", row_number=row_number
                        )
                        if user_id in seen_feature_users:
                            raise ValueError(f"duplicate KuaiRand user feature row {user_id}")
                        seen_feature_users.add(user_id)
                        encoded: list[int] = []
                        for feature in KUAI_CONTEXT_FEATURES:
                            raw_value = row[feature]
                            mapping = contextual_feature_categories[feature]
                            if raw_value not in mapping:
                                # Match the published pandas row.unique()
                                # encounter-order mapping and reserve zero.
                                mapping[raw_value] = len(mapping) + 1
                            encoded.append(mapping[raw_value])
                        feature_rows.append((user_id, *encoded))
                connection.executemany(
                    "INSERT INTO user_feature VALUES (" + ",".join("?" for _ in range(6)) + ")",
                    feature_rows,
                )
                # The published final pandas merge is inner, so users missing
                # static context must not remain in the sequence artifact.
                connection.execute(
                    "DELETE FROM eligible_user WHERE user_id NOT IN "
                    "(SELECT user_id FROM user_feature)"
                )
                feature_tmp = _temporary_output(user_features_output_path)
                with feature_tmp.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle, lineterminator="\n")
                    writer.writerow(["user_id", *KUAI_CONTEXT_FEATURES])
                    writer.writerows(
                        connection.execute(
                            "SELECT f.user_id, "
                            + ", ".join(f"f.{feature}" for feature in KUAI_CONTEXT_FEATURES)
                            + " FROM user_feature f JOIN eligible_user e USING(user_id) "
                            "ORDER BY f.user_id"
                        )
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(feature_tmp, user_features_output_path)
            unique_videos = _write_id_map(
                item_map_path,
                (
                    (video_id, int(video_id) + item_id_shift)
                    for (video_id,) in connection.execute(
                        "SELECT DISTINCT e.video_id FROM events e JOIN eligible_user u "
                        "USING(user_id) ORDER BY e.video_id"
                    )
                ),
                raw_field="raw_video_id",
            )
            query = connection.execute(
                "SELECT e.user_id, e.video_id, e.time_ms, e.action_mask, "
                "e.play_time_ms, e.duration_ms FROM events e "
                "JOIN eligible_user u USING(user_id) "
                "ORDER BY e.user_id, e.time_ms, e.file_rank, e.row_ordinal"
            )
            users_written = 0
            interactions_written = 0
            current_user: int | None = None
            current: list[tuple[int, int, int, int, int]] = []

            def flush(writer: _SequenceCsvWriter) -> None:
                nonlocal current, users_written, interactions_written
                if current_user is not None and len(current) >= minimum_sequence_length:
                    writer.write(
                        SequenceRow(
                            user_id=current_user,
                            item_ids=tuple(row[0] + item_id_shift for row in current),
                            timestamps=tuple(row[1] for row in current),
                            action_masks=tuple(row[2] for row in current),
                            play_time_ms=tuple(row[3] for row in current),
                            duration_ms=tuple(row[4] for row in current),
                        )
                    )
                    users_written += 1
                    interactions_written += len(current)
                current = []

            with _SequenceCsvWriter(sequence_path, kind="kuairand") as writer:
                for user_id, video_id, time_ms, mask, play_ms, duration_ms in query:
                    if current_user is None:
                        current_user = int(user_id)
                    elif user_id != current_user:
                        flush(writer)
                        current_user = int(user_id)
                    current.append(
                        (
                            int(video_id),
                            int(time_ms),
                            int(mask),
                            int(play_ms),
                            int(duration_ms),
                        )
                    )
                flush(writer)
        finally:
            connection.close()

    statistics = {
        "source_rows": source_rows,
        "eligible_users": users_written,
        "interactions": interactions_written,
        "unique_videos": unique_videos,
    }
    artifact_paths = {"sequences": sequence_path, "item_id_map": item_map_path}
    source_paths = {f"standard_log_{index}": path for index, path in enumerate(sources)}
    if user_features_path is not None:
        artifact_paths["user_features"] = user_features_output_path
        source_paths["user_features"] = Path(user_features_path)
    return _write_preprocess_record(
        destination,
        dataset="kuairand-1k",
        sources=source_paths,
        artifacts=artifact_paths,
        parameters={
            "minimum_sequence_length": minimum_sequence_length,
            "require_user_in_all_logs": require_user_in_all_logs,
            "item_id_shift": item_id_shift,
            "action_bits": KUAI_TASK_BITS,
            "context_features": list(KUAI_CONTEXT_FEATURES),
            "context_encoding": (
                "published_source_encounter_order_plus_one"
                if user_features_path is not None
                else "not_materialized"
            ),
            "sort_key": ["time_ms", "source_file_rank", "source_row_ordinal"],
            "split": "global_logged_future_windows_with_strict_timestamp_histories",
        },
        statistics=statistics,
    )


def _parse_list(value: str, converter: Any) -> tuple[Any, ...]:
    if value == "":
        return ()
    return tuple(converter(part) for part in value.split(","))


def iter_sequence_rows(path: str | Path) -> Iterator[SequenceRow]:
    """Read either canonical ratings or KuaiRand sequence CSVs."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        base = {"user_id", "sequence_item_ids", "sequence_timestamps"}
        if not base.issubset(fields):
            raise ValueError(f"sequence CSV is missing {sorted(base - fields)}")
        is_ratings = "sequence_ratings" in fields
        is_kuai = "sequence_action_masks" in fields
        if is_ratings == is_kuai:
            raise ValueError("sequence CSV must contain exactly one supported value schema")
        for row in reader:
            user_text = row["user_id"]
            try:
                user_id: int | str = int(user_text)
            except ValueError:
                user_id = user_text
            common = {
                "user_id": user_id,
                "item_ids": _parse_list(row["sequence_item_ids"], int),
                "timestamps": _parse_list(row["sequence_timestamps"], int),
            }
            if is_ratings:
                result = SequenceRow(
                    **common, values=_parse_list(row["sequence_ratings"], float)
                )
            else:
                result = SequenceRow(
                    **common,
                    action_masks=_parse_list(row["sequence_action_masks"], int),
                    play_time_ms=_parse_list(row["sequence_play_time_ms"], int),
                    duration_ms=_parse_list(row["sequence_duration_ms"], int),
                )
            _validate_sequence(result)
            yield result


def decode_kuairand_action_mask(mask: int) -> dict[str, int]:
    if isinstance(mask, bool) or not isinstance(mask, int) or mask < 0 or mask > 255:
        raise ValueError("KuaiRand action mask must be an integer in [0,255]")
    return {task: int(bool(mask & bit)) for task, bit in KUAI_TASK_BITS.items()}
