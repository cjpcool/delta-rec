from __future__ import annotations

import csv

from dataclasses import dataclass, field

import os

from pathlib import Path

from typing import Any, Mapping

MetaBridgeError = ValueError

COMMON_HISTORY_LENGTH = 1024

COMMON_CANDIDATE_COUNT = 32

KUAI_CONTEXTUAL_COUNT = 6

COMMON_TOTAL_SEQUENCE_LENGTH = (
    COMMON_HISTORY_LENGTH + COMMON_CANDIDATE_COUNT + KUAI_CONTEXTUAL_COUNT
)

KUAI_TASK_BITS: tuple[tuple[str, int], ...] = (
    ("click", 1),
    ("like", 2),
    ("follow", 4),
    ("comment", 8),
    ("forward", 16),
    ("hate", 32),
    ("long_view", 64),
    ("profile_enter", 128),
)

KUAI_TASK_NAMES = tuple(
    "long_view" if task == "long_view" else f"is_{task}"
    for task, _ in KUAI_TASK_BITS
)

KUAI_CONTEXT_FIELDS = (
    "user_id",
    "user_active_degree",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
)

KUAI_UIH_KEYS = (
    *KUAI_CONTEXT_FIELDS,
    "video_id",
    "action_timestamp",
    "action_weight",
    "watch_time",
)

KUAI_CANDIDATE_KEYS = (
    "item_video_id",
    "item_action_weight",
    "item_target_watchtime",
    "item_query_time",
)

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
    *(f"label_{task}" for task, _ in KUAI_TASK_BITS),
)

KUAI_USER_FEATURE_FIELDS = KUAI_CONTEXT_FIELDS

def _parse_int_list(value: str, *, field_name: str) -> tuple[int, ...]:
    if not value:
        return ()
    try:
        return tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise MetaBridgeError(f"{field_name} contains a non-integer value") from error

@dataclass(frozen=True)
class ParsedKuaiSlate:
    slate_id: str
    user_id: int
    history_item_ids: tuple[int, ...]
    history_timestamps: tuple[int, ...]
    history_action_masks: tuple[int, ...]
    history_play_time_ms: tuple[int, ...]
    history_duration_ms: tuple[int, ...]
    candidate_item_ids: tuple[int, ...]
    candidate_timestamps: tuple[int, ...]
    candidate_action_masks: tuple[int, ...]
    candidate_play_time_ms: tuple[int, ...]
    candidate_duration_ms: tuple[int, ...]

def parse_kuai_slate_row(row: Mapping[str, str]) -> ParsedKuaiSlate:
    missing = set(KUAI_SLATE_FIELDS) - set(row)
    if missing:
        raise MetaBridgeError(f"Kuai slate missing fields: {sorted(missing)}")
    history = _parse_int_list(row["history_item_ids"], field_name="history_item_ids")
    history_times = _parse_int_list(
        row["history_timestamps"], field_name="history_timestamps"
    )
    history_masks = _parse_int_list(
        row["history_action_masks"], field_name="history_action_masks"
    )
    history_plays = _parse_int_list(
        row["history_play_time_ms"], field_name="history_play_time_ms"
    )
    history_durations = _parse_int_list(
        row["history_duration_ms"], field_name="history_duration_ms"
    )
    candidates = _parse_int_list(
        row["candidate_item_ids"], field_name="candidate_item_ids"
    )
    candidate_times = _parse_int_list(
        row["candidate_timestamps"], field_name="candidate_timestamps"
    )
    candidate_plays = _parse_int_list(
        row["candidate_play_time_ms"], field_name="candidate_play_time_ms"
    )
    candidate_durations = _parse_int_list(
        row["candidate_duration_ms"], field_name="candidate_duration_ms"
    )
    labels: list[tuple[int, ...]] = []
    for task, _ in KUAI_TASK_BITS:
        values = _parse_int_list(row[f"label_{task}"], field_name=f"label_{task}")
        if any(value not in (0, 1) for value in values):
            raise MetaBridgeError(f"label_{task} must be binary")
        labels.append(values)
    candidate_masks = tuple(
        sum(bit * labels[column][index] for column, (_, bit) in enumerate(KUAI_TASK_BITS))
        for index in range(len(candidates))
    )
    history_lengths = {
        len(history),
        len(history_times),
        len(history_masks),
        len(history_plays),
        len(history_durations),
    }
    candidate_lengths = {
        len(candidates),
        len(candidate_times),
        len(candidate_plays),
        len(candidate_durations),
        len(candidate_masks),
        *(len(values) for values in labels),
    }
    if len(history_lengths) != 1 or not history or len(history) > COMMON_HISTORY_LENGTH:
        raise MetaBridgeError("Kuai history fields must align with length in [1, 1024]")
    if candidate_lengths != {COMMON_CANDIDATE_COUNT}:
        raise MetaBridgeError("Kuai candidate fields must all have frozen K=32")
    if any(item <= 0 for item in (*history, *candidates)):
        raise MetaBridgeError("Kuai mapped item IDs must be positive")
    if any(mask < 0 or mask > 255 for mask in (*history_masks, *candidate_masks)):
        raise MetaBridgeError("Kuai action masks must fit the official eight bits")
    if any(value < 0 for value in (*history_plays, *history_durations, *candidate_plays, *candidate_durations)):
        raise MetaBridgeError("Kuai play/duration fields must be nonnegative")
    if any(left > right for left, right in zip(history_times, history_times[1:])):
        raise MetaBridgeError("Kuai history timestamps must be nondecreasing")
    if any(left > right for left, right in zip(candidate_times, candidate_times[1:])):
        raise MetaBridgeError("Kuai candidate timestamps must be nondecreasing")
    if max(history_times) >= min(candidate_times):
        raise MetaBridgeError("Kuai history must be strictly earlier than every candidate")
    try:
        user_id = int(row["user_id"])
    except ValueError as error:
        raise MetaBridgeError("Kuai user_id must be an integer") from error
    if user_id < 0:
        raise MetaBridgeError("Kuai user_id must be nonnegative")
    return ParsedKuaiSlate(
        slate_id=row["slate_id"],
        user_id=user_id,
        history_item_ids=history,
        history_timestamps=history_times,
        history_action_masks=history_masks,
        history_play_time_ms=history_plays,
        history_duration_ms=history_durations,
        candidate_item_ids=candidates,
        candidate_timestamps=candidate_times,
        candidate_action_masks=candidate_masks,
        candidate_play_time_ms=candidate_plays,
        candidate_duration_ms=candidate_durations,
    )

def load_kuai_user_features(path: Path) -> dict[int, tuple[int, ...]]:
    resolved = path.expanduser().resolve()
    with resolved.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != KUAI_USER_FEATURE_FIELDS:
            raise MetaBridgeError(
                "Kuai user feature header must exactly match official contextual fields"
            )
        output: dict[int, tuple[int, ...]] = {}
        limits = (10_000_000, 8, 9, 9, 8, 8)
        for row_number, row in enumerate(reader, 2):
            values = tuple(int(row[field]) for field in KUAI_CONTEXT_FIELDS)
            user_id = values[0]
            if user_id in output:
                raise MetaBridgeError(f"duplicate Kuai user at row {row_number}")
            if any(value < 0 or value >= limit for value, limit in zip(values, limits)):
                raise MetaBridgeError(
                    f"Kuai context value exceeds official embedding range at row {row_number}"
                )
            output[user_id] = values
    if not output:
        raise MetaBridgeError("Kuai user feature file is empty")
    return output

class HeadlineKuaiSlateDataset:
    """Random-access, bounded-memory reader for one frozen split file."""

    def __init__(self, slate_csv: Path, user_features_csv: Path) -> None:
        self.path = slate_csv.expanduser().resolve()
        if not self.path.is_file():
            raise MetaBridgeError(f"Kuai slate file does not exist: {self.path}")
        self.user_features = load_kuai_user_features(user_features_csv)
        self.offsets: list[int] = []
        with self.path.open("rb") as handle:
            header_line = handle.readline()
            try:
                header = next(csv.reader([header_line.decode("utf-8")]))
            except (UnicodeDecodeError, csv.Error) as error:
                raise MetaBridgeError(f"invalid Kuai slate header: {self.path}") from error
            if tuple(header) != KUAI_SLATE_FIELDS:
                missing = sorted(set(KUAI_SLATE_FIELDS) - set(header))
                raise MetaBridgeError(
                    "Kuai slate header is not the lossless headline schema; "
                    f"missing={missing}. Regenerate the frozen slates."
                )
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                self.offsets.append(offset)
        if not self.offsets:
            raise MetaBridgeError(f"Kuai split contains no slates: {self.path}")
        self._handle: Any | None = None
        self._handle_pid: int | None = None

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handle"] = None
        state["_handle_pid"] = None
        return state

    def close(self) -> None:
        if getattr(self, "_handle", None) is not None:
            self._handle.close()
            self._handle = None
            self._handle_pid = None

    def __del__(self) -> None:
        self.close()

    def _row(self, index: int) -> dict[str, str]:
        if index < 0:
            index += len(self.offsets)
        if not 0 <= index < len(self.offsets):
            raise IndexError(index)
        process = os.getpid()
        if self._handle is None or self._handle_pid != process:
            if self._handle is not None:
                self._handle.close()
            self._handle = self.path.open("rb")
            self._handle_pid = process
        self._handle.seek(self.offsets[index])
        raw = self._handle.readline().decode("utf-8")
        values = next(csv.reader([raw]))
        if len(values) != len(KUAI_SLATE_FIELDS):
            raise MetaBridgeError(f"malformed Kuai slate row at index {index}")
        return dict(zip(KUAI_SLATE_FIELDS, values))

    def parsed(self, index: int) -> ParsedKuaiSlate:
        slate = parse_kuai_slate_row(self._row(index))
        if slate.user_id not in self.user_features:
            raise MetaBridgeError(
                f"slate {slate.slate_id} has no frozen contextual user features"
            )
        return slate

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        import torch
        from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

        slate = self.parsed(index)
        context = self.user_features[slate.user_id]
        history_length = len(slate.history_item_ids)
        uih_lengths = torch.tensor(
            [1] * len(KUAI_CONTEXT_FIELDS)
            + [history_length]
            * (len(KUAI_UIH_KEYS) - len(KUAI_CONTEXT_FIELDS)),
            dtype=torch.long,
        )
        uih_values = torch.tensor(
            [
                *context,
                *slate.history_item_ids,
                *slate.history_timestamps,
                *slate.history_action_masks,
                *slate.history_play_time_ms,
            ],
            dtype=torch.long,
        )
        query_time = max(slate.history_timestamps)
        candidate_lengths = torch.full(
            (len(KUAI_CANDIDATE_KEYS),), COMMON_CANDIDATE_COUNT, dtype=torch.long
        )
        candidate_values = torch.tensor(
            [
                *slate.candidate_item_ids,
                *slate.candidate_action_masks,
                *slate.candidate_play_time_ms,
                *([query_time] * COMMON_CANDIDATE_COUNT),
            ],
            dtype=torch.long,
        )
        return (
            KeyedJaggedTensor(
                keys=list(KUAI_UIH_KEYS), lengths=uih_lengths, values=uih_values
            ),
            KeyedJaggedTensor(
                keys=list(KUAI_CANDIDATE_KEYS),
                lengths=candidate_lengths,
                values=candidate_values,
            ),
        )
