# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""Persistent per-candidate GDR prefix-state cache.

Each row is keyed by the complete serving identity
``(dataset, user, history_version, candidate, model, selector, budget)``.
Candidate rows for a user/history generation are staged together and become
visible through one published pointer.  A lookup may request any subset of
that generation in any transport order, but every requested candidate must
have an independently keyed row.  This preserves true per-candidate reuse
while preventing partially written or mixed-generation rows from being served.

The public and artifact state layout is ``[N,K,layers,H,Dk,Dv]`` and recurrent
state storage is always finite FP32.
"""

from __future__ import annotations

import hashlib
import json
import os
import string
import tempfile
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .group_cache import CacheMissError
from .selection import validate_retention_ratio


PC_STATE_CACHE_SCHEMA = "deltarec-pc-prefix-state-cache-v1"
PC_STATE_LAYOUT = "N,K,layers,H,Dk,Dv"


def _json_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _require_sha256(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in string.hexdigits for character in value)
    ):
        raise ValueError(f"{name} must be a 64-character hexadecimal digest")
    return value.lower()


def _require_positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256(
        f"{tuple(tensor.shape)}:{tensor.dtype}".encode("utf-8")
    )
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def canonical_candidate_set_sha256(candidate_ids: Sequence[int]) -> str:
    """Hash one duplicate-free candidate set independently of input order."""

    values = [int(value) for value in candidate_ids]
    if any(value < 0 for value in values):
        raise ValueError("candidate IDs must be nonnegative")
    if len(values) < 1:
        raise ValueError("candidate set must be nonempty")
    if len(values) != len(set(values)):
        raise ValueError("per-candidate cache requires unique candidate IDs")
    return _json_digest({"candidate_ids": sorted(values)})


@dataclass(frozen=True)
class PCStateCacheVersion:
    """Independent versions for every semantic part of a PC state key."""

    model_version: str
    selector_version: str
    budget_version: str
    schema_version: str

    def __post_init__(self) -> None:
        for name in (
            "model_version",
            "selector_version",
            "budget_version",
            "schema_version",
        ):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name))

    @property
    def fingerprint(self) -> str:
        return _json_digest(self.as_record())

    def as_record(self) -> dict[str, str]:
        return {
            "model_version": self.model_version,
            "selector_version": self.selector_version,
            "budget_version": self.budget_version,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_record(cls, value: Any) -> "PCStateCacheVersion":
        names = {
            "model_version",
            "selector_version",
            "budget_version",
            "schema_version",
        }
        if not isinstance(value, Mapping) or set(value) != names:
            raise ValueError("PC state cache has an invalid version record")
        return cls(**{name: str(value[name]) for name in names})


def build_pc_state_cache_version(
    *,
    model_sha256: str,
    selector_artifacts: Mapping[str, str],
    budget_policy: str,
    write_ratio: float,
    recent_floor: int,
    layer_count: int,
    num_heads: int,
    key_dim: int,
    value_dim: int,
    contextual_seq_len: int,
    projection_dtype: str = "float32",
    recurrence_backend: str = "reference",
) -> PCStateCacheVersion:
    """Bind PC states to exact model, selector, budget, and tensor semantics."""

    model_version = _require_sha256(model_sha256, "model_sha256")
    if not selector_artifacts or any(
        not isinstance(name, str)
        or not name
        or not isinstance(value, str)
        or not value
        for name, value in selector_artifacts.items()
    ):
        raise ValueError("selector_artifacts must be a nonempty string mapping")
    if not isinstance(budget_policy, str) or not budget_policy:
        raise ValueError("budget_policy must be a nonempty string")
    write_ratio = validate_retention_ratio(write_ratio)
    if (
        isinstance(recent_floor, bool)
        or not isinstance(recent_floor, int)
        or recent_floor < 0
    ):
        raise ValueError("recent_floor must be a nonnegative integer")
    if (
        isinstance(contextual_seq_len, bool)
        or not isinstance(contextual_seq_len, int)
        or contextual_seq_len < 0
    ):
        raise ValueError("contextual_seq_len must be a nonnegative integer")
    if not isinstance(projection_dtype, str) or not projection_dtype:
        raise ValueError("projection_dtype must be a nonempty string")
    if not isinstance(recurrence_backend, str) or not recurrence_backend:
        raise ValueError("recurrence_backend must be a nonempty string")
    dimensions = {
        "layer_count": _require_positive_integer(layer_count, "layer_count"),
        "num_heads": _require_positive_integer(num_heads, "num_heads"),
        "key_dim": _require_positive_integer(key_dim, "key_dim"),
        "value_dim": _require_positive_integer(value_dim, "value_dim"),
    }
    return PCStateCacheVersion(
        model_version=model_version,
        selector_version=_json_digest(
            {
                "contract": PC_STATE_CACHE_SCHEMA,
                "selector_artifacts": dict(sorted(selector_artifacts.items())),
            }
        ),
        budget_version=_json_digest(
            {
                "contract": PC_STATE_CACHE_SCHEMA,
                "budget_policy": budget_policy,
                "write_ratio": write_ratio,
                "recent_floor": int(recent_floor),
            }
        ),
        schema_version=_json_digest(
            {
                "contract": PC_STATE_CACHE_SCHEMA,
                "layout": PC_STATE_LAYOUT,
                "dtype": "float32",
                "projection_dtype": projection_dtype,
                "recurrence_backend": recurrence_backend,
                "contextual_seq_len": int(contextual_seq_len),
                **dimensions,
            }
        ),
    )


@dataclass(frozen=True)
class PCStateKey:
    dataset_id: str
    user_id: int
    history_version: int
    candidate_id: int
    model_version: str
    selector_version: str
    budget_version: str
    schema_version: str


@dataclass(frozen=True)
class _PublishedPCGeneration:
    history_version: int
    candidate_ids: tuple[int, ...]
    candidate_set_sha256: str


@dataclass(frozen=True)
class PCStateLookup:
    """Candidate-order-aligned FP32 states for cache-hit serving."""

    states: torch.Tensor
    candidate_ids: torch.Tensor
    published_history_versions: tuple[int, ...]
    candidate_set_sha256: tuple[str, ...]
    published_candidate_set_sha256: tuple[str, ...]
    materialization_bytes: int
    transferred_bytes: int

    @property
    def hit_count(self) -> int:
        return int(self.states.shape[0] * self.states.shape[1])

    @property
    def initial_states(self) -> torch.Tensor:
        return self.states


def _artifact_sha256(payload: Mapping[str, Any]) -> str:
    tensor_names = (
        "row_user_ids",
        "row_history_versions",
        "row_candidate_ids",
        "row_states",
        "published_user_ids",
        "published_history_versions",
        "published_candidate_offsets",
        "published_candidate_ids",
    )
    tensors: dict[str, str] = {}
    for name in tensor_names:
        value = payload.get(name)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"PC state cache artifact is missing tensor {name}")
        tensors[name] = _tensor_sha256(value)
    record = {
        "schema": payload.get("schema"),
        "dataset_id": payload.get("dataset_id"),
        "version": payload.get("version"),
        "state_shape": payload.get("state_shape"),
        "state_layout": payload.get("state_layout"),
        "state_dtype": payload.get("state_dtype"),
        "components": payload.get("components"),
        "published_candidate_set_sha256": payload.get(
            "published_candidate_set_sha256"
        ),
        "tensors": tensors,
    }
    return _json_digest(record)


def _to_integer_list(value: torch.Tensor | Sequence[int], name: str) -> list[int]:
    if isinstance(value, torch.Tensor):
        if value.ndim != 1 or value.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must be an integer vector")
        result = value.detach().to(device="cpu", dtype=torch.int64).tolist()
    else:
        result = list(value)
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in result):
        raise ValueError(f"{name} must contain nonnegative integers")
    return [int(item) for item in result]


class PCStateCache:
    """Atomic generations whose physical and logical keys are candidate-specific."""

    def __init__(
        self,
        *,
        dataset_id: str,
        version: PCStateCacheVersion,
        state_shape: Sequence[int],
        components: Mapping[str, Any],
        storage_device: torch.device | str = "cpu",
        pin_memory: bool = False,
    ) -> None:
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError("dataset_id must be a nonempty string")
        if not isinstance(version, PCStateCacheVersion):
            raise TypeError("version must be a PCStateCacheVersion")
        shape = tuple(int(value) for value in state_shape)
        if len(shape) != 4 or any(value < 1 for value in shape):
            raise ValueError("state_shape must be [layers,H,Dk,Dv]")
        target = torch.device(storage_device)
        if pin_memory and target.type != "cpu":
            raise ValueError("pin_memory is valid only for CPU storage")
        if pin_memory and not torch.cuda.is_available():
            raise RuntimeError("pinned PC cache requires a CUDA runtime")
        # Artifact metadata must be deterministically serializable because it
        # is covered by the checksum.
        try:
            json.dumps(components, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError("components must be JSON serializable") from error

        self.dataset_id = dataset_id
        self.version = version
        self.state_shape = shape
        self.components = dict(components)
        self.storage_device = target
        self.pin_memory = bool(pin_memory)
        self._rows: dict[PCStateKey, torch.Tensor] = {}
        self._published: dict[
            tuple[str, int, str, str, str, str], _PublishedPCGeneration
        ] = {}
        self._lock = threading.RLock()

    def _owner(self, user_id: int) -> tuple[str, int, str, str, str, str]:
        return (
            self.dataset_id,
            int(user_id),
            self.version.model_version,
            self.version.selector_version,
            self.version.budget_version,
            self.version.schema_version,
        )

    def _key(self, user_id: int, history_version: int, candidate_id: int) -> PCStateKey:
        return PCStateKey(
            dataset_id=self.dataset_id,
            user_id=int(user_id),
            history_version=int(history_version),
            candidate_id=int(candidate_id),
            model_version=self.version.model_version,
            selector_version=self.version.selector_version,
            budget_version=self.version.budget_version,
            schema_version=self.version.schema_version,
        )

    def _place(self, value: torch.Tensor) -> torch.Tensor:
        cpu = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if self.storage_device.type == "cpu":
            return cpu.pin_memory() if self.pin_memory else cpu.clone()
        return cpu.to(device=self.storage_device)

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._published)

    @property
    def state_row_count(self) -> int:
        with self._lock:
            return len(self._rows)

    @property
    def state_bytes_per_candidate(self) -> int:
        elements = 1
        for value in self.state_shape:
            elements *= value
        return elements * torch.empty((), dtype=torch.float32).element_size()

    def publish_user_states(
        self,
        *,
        user_id: int,
        history_version: int,
        candidate_ids: torch.Tensor | Sequence[int],
        states: torch.Tensor,
        expected_previous_version: int | None = None,
    ) -> None:
        """Stage all K rows and atomically publish one user generation."""

        users = _to_integer_list([user_id], "user_id")
        versions = _to_integer_list([history_version], "history_version")
        candidates = _to_integer_list(candidate_ids, "candidate_ids")
        candidate_hash = canonical_candidate_set_sha256(candidates)
        if states.shape != (len(candidates), *self.state_shape):
            raise ValueError("states must have shape [K,layers,H,Dk,Dv]")
        if states.dtype != torch.float32 or not bool(torch.isfinite(states).all()):
            raise ValueError("published recurrent states must be finite FP32")
        staged = {
            candidate: self._place(states[index])
            for index, candidate in enumerate(candidates)
        }
        user = users[0]
        history = versions[0]
        owner = self._owner(user)
        new_keys = [self._key(user, history, candidate) for candidate in candidates]
        published = _PublishedPCGeneration(
            history_version=history,
            candidate_ids=tuple(sorted(candidates)),
            candidate_set_sha256=candidate_hash,
        )

        with self._lock:
            previous = self._published.get(owner)
            previous_version = None if previous is None else previous.history_version
            if expected_previous_version is not None and previous_version != int(
                expected_previous_version
            ):
                raise CacheMissError("published PC history version changed before commit")
            if previous_version is not None and history <= previous_version:
                raise ValueError("history_version must advance monotonically")
            if any(key in self._rows for key in new_keys):
                raise ValueError("PC state generation collides with existing rows")
            try:
                for key in new_keys:
                    self._rows[key] = staged[key.candidate_id]
                self._published[owner] = published
            except Exception:
                for key in new_keys:
                    self._rows.pop(key, None)
                if previous is None:
                    self._published.pop(owner, None)
                else:
                    self._published[owner] = previous
                raise
            if previous is not None:
                for candidate in previous.candidate_ids:
                    self._rows.pop(
                        self._key(user, previous.history_version, candidate), None
                    )

    def publish_batch(
        self,
        *,
        user_ids: torch.Tensor | Sequence[int],
        history_versions: torch.Tensor | Sequence[int],
        candidate_ids: torch.Tensor,
        states: torch.Tensor,
    ) -> None:
        users = _to_integer_list(user_ids, "user_ids")
        versions = _to_integer_list(history_versions, "history_versions")
        if candidate_ids.ndim != 2 or candidate_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("candidate_ids must be integer [N,K]")
        if len(users) != len(versions) or candidate_ids.shape[0] != len(users):
            raise ValueError("batch cache keys must have equal row counts")
        if states.shape != (
            len(users),
            candidate_ids.shape[1],
            *self.state_shape,
        ):
            raise ValueError("states must have shape [N,K,layers,H,Dk,Dv]")
        for row, (user, history) in enumerate(zip(users, versions)):
            self.publish_user_states(
                user_id=user,
                history_version=history,
                candidate_ids=candidate_ids[row],
                states=states[row],
            )

    def published_history_version(self, user_id: int) -> int:
        user = _to_integer_list([user_id], "user_id")[0]
        with self._lock:
            generation = self._published.get(self._owner(user))
            if generation is None:
                raise CacheMissError("user has no atomically published PC state")
            return generation.history_version

    def lookup(
        self,
        *,
        dataset_id: str,
        user_ids: torch.Tensor | Sequence[int],
        history_versions: torch.Tensor | Sequence[int],
        candidate_ids: torch.Tensor,
        version: PCStateCacheVersion,
        target_device: torch.device | str,
        non_blocking: bool = True,
    ) -> PCStateLookup:
        if dataset_id != self.dataset_id:
            raise CacheMissError("PC state cache dataset mismatch")
        if version != self.version:
            raise CacheMissError("PC model/selector/budget/schema version mismatch")
        if candidate_ids.ndim != 2 or candidate_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("candidate_ids must be integer [N,K]")
        batch, candidate_count = candidate_ids.shape
        if batch < 1 or candidate_count < 1:
            raise ValueError("candidate_ids must contain nonempty N and K")
        users = _to_integer_list(user_ids, "user_ids")
        versions = _to_integer_list(history_versions, "history_versions")
        if len(users) != batch or len(versions) != batch:
            raise ValueError("cache keys must contain one row per request")
        ids_cpu = candidate_ids.detach().to(device="cpu", dtype=torch.int64)
        rows: list[torch.Tensor] = []
        request_hashes: list[str] = []
        published_hashes: list[str] = []

        with self._lock:
            for row, (user, history) in enumerate(zip(users, versions)):
                requested = [int(value) for value in ids_cpu[row].tolist()]
                requested_hash = canonical_candidate_set_sha256(requested)
                generation = self._published.get(self._owner(user))
                if generation is None:
                    raise CacheMissError("user has no atomically published PC state")
                if history != generation.history_version:
                    raise CacheMissError("requested PC history version is stale or unpublished")
                if (
                    canonical_candidate_set_sha256(generation.candidate_ids)
                    != generation.candidate_set_sha256
                ):
                    raise CacheMissError("published PC candidate-set hash mismatch")
                for candidate in requested:
                    if candidate not in generation.candidate_ids:
                        raise CacheMissError("requested PC candidate is not published")
                    state = self._rows.get(self._key(user, history, candidate))
                    if state is None:
                        raise CacheMissError("atomically published PC candidate row is missing")
                    if (
                        state.shape != self.state_shape
                        or state.dtype != torch.float32
                        or not bool(torch.isfinite(state).all())
                    ):
                        raise CacheMissError("published PC candidate row is invalid")
                    rows.append(state)
                request_hashes.append(requested_hash)
                published_hashes.append(generation.candidate_set_sha256)

        resident = torch.stack(rows).reshape(batch, candidate_count, *self.state_shape)
        target = torch.device(target_device)
        moving = resident.device != target
        resident = resident.to(
            device=target,
            non_blocking=bool(
                non_blocking
                and resident.device.type == "cpu"
                and target.type == "cuda"
                and resident.is_pinned()
            ),
        )
        materialization_bytes = batch * candidate_count * self.state_bytes_per_candidate
        return PCStateLookup(
            states=resident,
            candidate_ids=ids_cpu.to(target),
            published_history_versions=tuple(versions),
            candidate_set_sha256=tuple(request_hashes),
            published_candidate_set_sha256=tuple(published_hashes),
            materialization_bytes=materialization_bytes,
            transferred_bytes=materialization_bytes if moving else 0,
        )

    def artifact_state(self) -> dict[str, Any]:
        with self._lock:
            keys = sorted(
                self._rows,
                key=lambda key: (key.user_id, key.history_version, key.candidate_id),
            )
            published = sorted(
                ((owner[1], generation) for owner, generation in self._published.items()),
                key=lambda value: value[0],
            )
            candidate_offsets = [0]
            published_candidates: list[int] = []
            published_hashes: list[str] = []
            for _, generation in published:
                published_candidates.extend(generation.candidate_ids)
                candidate_offsets.append(len(published_candidates))
                published_hashes.append(generation.candidate_set_sha256)
            payload: dict[str, Any] = {
                "schema": PC_STATE_CACHE_SCHEMA,
                "dataset_id": self.dataset_id,
                "version": self.version.as_record(),
                "state_shape": list(self.state_shape),
                "state_layout": PC_STATE_LAYOUT,
                "state_dtype": "float32",
                "components": dict(self.components),
                "row_user_ids": torch.tensor(
                    [key.user_id for key in keys], dtype=torch.int64
                ),
                "row_history_versions": torch.tensor(
                    [key.history_version for key in keys], dtype=torch.int64
                ),
                "row_candidate_ids": torch.tensor(
                    [key.candidate_id for key in keys], dtype=torch.int64
                ),
                "row_states": (
                    torch.stack([self._rows[key].detach().cpu() for key in keys])
                    if keys
                    else torch.empty((0, *self.state_shape), dtype=torch.float32)
                ),
                "published_user_ids": torch.tensor(
                    [user for user, _ in published], dtype=torch.int64
                ),
                "published_history_versions": torch.tensor(
                    [generation.history_version for _, generation in published],
                    dtype=torch.int64,
                ),
                "published_candidate_offsets": torch.tensor(
                    candidate_offsets, dtype=torch.int64
                ),
                "published_candidate_ids": torch.tensor(
                    published_candidates, dtype=torch.int64
                ),
                "published_candidate_set_sha256": published_hashes,
            }
        payload["artifact_sha256"] = _artifact_sha256(payload)
        return payload

    def save(self, path: str | Path) -> Path:
        """Persist through a same-directory temporary followed by atomic replace."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.artifact_state()
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_path)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination

    @classmethod
    def from_artifact_state(
        cls,
        payload: Mapping[str, Any],
        *,
        storage_device: torch.device | str = "cpu",
        pin_memory: bool = False,
        expected_dataset_id: str | None = None,
        expected_version: PCStateCacheVersion | None = None,
    ) -> "PCStateCache":
        if payload.get("schema") != PC_STATE_CACHE_SCHEMA:
            raise ValueError("PC state cache schema mismatch")
        if payload.get("artifact_sha256") != _artifact_sha256(payload):
            raise ValueError("PC state cache artifact checksum mismatch")
        if payload.get("state_layout") != PC_STATE_LAYOUT:
            raise ValueError("PC state cache layout mismatch")
        if payload.get("state_dtype") != "float32":
            raise ValueError("PC state cache dtype metadata mismatch")
        dataset_id = payload.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError("PC state cache dataset is invalid")
        version = PCStateCacheVersion.from_record(payload.get("version"))
        if expected_dataset_id is not None and dataset_id != expected_dataset_id:
            raise CacheMissError("PC state cache dataset mismatch")
        if expected_version is not None and version != expected_version:
            raise CacheMissError("PC state cache version mismatch")
        components = payload.get("components")
        if not isinstance(components, Mapping):
            raise ValueError("PC state cache components must be a mapping")
        cache = cls(
            dataset_id=dataset_id,
            version=version,
            state_shape=payload.get("state_shape", ()),
            components=components,
            storage_device=storage_device,
            pin_memory=pin_memory,
        )

        tensor_names = (
            "row_user_ids",
            "row_history_versions",
            "row_candidate_ids",
            "published_user_ids",
            "published_history_versions",
            "published_candidate_offsets",
            "published_candidate_ids",
        )
        for name in tensor_names:
            value = payload[name]
            if value.ndim != 1 or value.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"PC state cache {name} must be an integer vector")
        row_users = payload["row_user_ids"].tolist()
        row_histories = payload["row_history_versions"].tolist()
        row_candidates = payload["row_candidate_ids"].tolist()
        row_states = payload["row_states"]
        if not (
            len(row_users)
            == len(row_histories)
            == len(row_candidates)
            == len(row_states)
        ):
            raise ValueError("PC state cache row key/value counts disagree")
        if row_states.shape != (len(row_users), *cache.state_shape):
            raise ValueError("PC state cache row-state layout mismatch")
        if row_states.dtype != torch.float32 or not bool(torch.isfinite(row_states).all()):
            raise ValueError("PC state cache artifact states must be finite FP32")

        published_users = payload["published_user_ids"].tolist()
        published_histories = payload["published_history_versions"].tolist()
        offsets = payload["published_candidate_offsets"].tolist()
        published_candidates = payload["published_candidate_ids"].tolist()
        published_hashes = payload.get("published_candidate_set_sha256")
        if (
            not isinstance(published_hashes, list)
            or len(published_users) != len(published_histories)
            or len(published_users) != len(published_hashes)
            or len(offsets) != len(published_users) + 1
            or not offsets
            or offsets[0] != 0
            or offsets[-1] != len(published_candidates)
            or any(left >= right for left, right in zip(offsets, offsets[1:]))
        ):
            raise ValueError("PC state cache published manifest is malformed")

        declared: set[PCStateKey] = set()
        for index, (user, history, candidate_hash) in enumerate(
            zip(published_users, published_histories, published_hashes)
        ):
            if int(user) < 0 or int(history) < 0:
                raise ValueError("PC state cache has an invalid published pointer")
            candidates = [
                int(value)
                for value in published_candidates[offsets[index] : offsets[index + 1]]
            ]
            expected_hash = canonical_candidate_set_sha256(candidates)
            if not isinstance(candidate_hash, str) or candidate_hash != expected_hash:
                raise ValueError("PC state cache candidate-set hash mismatch")
            owner = cache._owner(int(user))
            if owner in cache._published:
                raise ValueError("PC state cache has duplicate published users")
            generation = _PublishedPCGeneration(
                history_version=int(history),
                candidate_ids=tuple(sorted(candidates)),
                candidate_set_sha256=expected_hash,
            )
            cache._published[owner] = generation
            declared.update(
                cache._key(int(user), int(history), candidate)
                for candidate in generation.candidate_ids
            )

        observed: set[PCStateKey] = set()
        for index, (user, history, candidate) in enumerate(
            zip(row_users, row_histories, row_candidates)
        ):
            if int(user) < 0 or int(history) < 0 or int(candidate) < 0:
                raise ValueError("PC state cache artifact contains an invalid row key")
            key = cache._key(int(user), int(history), int(candidate))
            if key in observed:
                raise ValueError("PC state cache artifact contains duplicate rows")
            observed.add(key)
            cache._rows[key] = cache._place(row_states[index])
        if observed != declared:
            raise ValueError("PC state cache has partial or orphan candidate rows")
        return cache

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> "PCStateCache":
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise ValueError("PC state cache artifact must be a mapping")
        return cls.from_artifact_state(payload, **kwargs)


__all__ = [
    "PC_STATE_CACHE_SCHEMA",
    "PC_STATE_LAYOUT",
    "PCStateCache",
    "PCStateCacheVersion",
    "PCStateKey",
    "PCStateLookup",
    "build_pc_state_cache_version",
    "canonical_candidate_set_sha256",
]

