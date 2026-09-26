# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""Candidate-set-independent global-group GDR state cache.

The request-local Subplan 5C caches are intentionally keyed by a complete
candidate set.  This module implements the fixed-category production contract:
one FP32 recurrent prefix state per
``(dataset, user, history_version, global_group, model_version)``.  Publication
is atomic per user/history version and lookup materializes only the occupied
``(user, global_group)`` rows needed by the current candidate batch.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import tempfile
import threading
from typing import Any
import string

import torch

from ..delta_rec_cache import DeltaRecCacheVersion
from .group_cache import CacheMissError
from .selection import validate_retention_ratio


GLOBAL_GROUP_CACHE_SCHEMA = "subplan5c-global-group-state-cache-v1"


def _json_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256(
        f"{tuple(tensor.shape)}:{tensor.dtype}".encode("utf-8")
    )
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_global_group_model_version(
    *,
    dataset_id: str,
    state_cache_version: DeltaRecCacheVersion,
    selector_artifacts: Mapping[str, str],
    grouping_artifacts: Mapping[str, str],
    candidate_group_count: int,
    grouping_policy: str,
    retention_ratio: float,
    recent_floor: int,
    state_layout: str = "layers,H,Dk,Dv",
    state_dtype: str = "float32",
) -> str:
    """Digest every component that can change a cached global-group state."""

    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("dataset_id must be a nonempty string")
    if not isinstance(state_cache_version, DeltaRecCacheVersion):
        raise TypeError("state_cache_version must be a DeltaRecCacheVersion")
    if candidate_group_count < 1:
        raise ValueError("candidate_group_count must be positive")
    if grouping_policy != "fixed_category_prototype_v1":
        raise ValueError("global-group cache requires fixed_category_prototype_v1")
    retention_ratio = validate_retention_ratio(retention_ratio)
    if recent_floor < 0:
        raise ValueError("recent_floor must be nonnegative")
    components = {
        "schema": GLOBAL_GROUP_CACHE_SCHEMA,
        "dataset_id": dataset_id,
        "state_cache_version": state_cache_version.fingerprint,
        "selector_artifacts": dict(sorted(selector_artifacts.items())),
        "grouping_artifacts": dict(sorted(grouping_artifacts.items())),
        "candidate_group_count": int(candidate_group_count),
        "grouping_policy": grouping_policy,
        "retention_ratio": retention_ratio,
        "recent_floor": int(recent_floor),
        "state_layout": state_layout,
        "state_dtype": state_dtype,
    }
    return _json_digest(components)


@dataclass(frozen=True)
class GlobalGroupStateKey:
    dataset_id: str
    user_id: int
    history_version: int
    global_group_id: int
    model_version: str


@dataclass(frozen=True)
class GlobalGroupStateLookup:
    """Occupied state rows and the packed candidate plan for one batch."""

    states: torch.Tensor
    candidate_group_ids: torch.Tensor
    candidate_to_state_row: torch.Tensor
    packed_candidate_indices: torch.Tensor
    offsets: torch.Tensor
    occupied_user_rows: torch.Tensor
    occupied_group_ids: torch.Tensor
    published_history_versions: tuple[int, ...]
    materialization_bytes: int
    transferred_bytes: int

    @property
    def occupied_state_count(self) -> int:
        return int(self.states.shape[0])


def _artifact_sha256(payload: Mapping[str, Any]) -> str:
    tensor_names = (
        "item_to_category_group",
        "row_user_ids",
        "row_history_versions",
        "row_group_ids",
        "row_states",
        "published_user_ids",
        "published_history_versions",
    )
    record = {
        "schema": payload.get("schema"),
        "dataset_id": payload.get("dataset_id"),
        "model_version": payload.get("model_version"),
        "selection_cache_version": payload.get("selection_cache_version"),
        "state_cache_version": payload.get("state_cache_version"),
        "unknown_category_group": payload.get("unknown_category_group"),
        "group_count": payload.get("group_count"),
        "state_shape": payload.get("state_shape"),
        "components": payload.get("components"),
        "tensors": {
            name: _tensor_sha256(payload[name]) for name in tensor_names
        },
    }
    return _json_digest(record)


class GlobalGroupStateCache:
    """Mutable write-through table with atomic per-user version publication."""

    def __init__(
        self,
        *,
        dataset_id: str,
        model_version: str,
        selection_cache_version: DeltaRecCacheVersion,
        state_cache_version: DeltaRecCacheVersion,
        item_to_category_group: torch.Tensor,
        unknown_category_group: int,
        group_count: int,
        state_shape: Sequence[int],
        components: Mapping[str, Any],
        storage_device: torch.device | str = "cpu",
        pin_memory: bool = False,
    ) -> None:
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError("dataset_id must be a nonempty string")
        if (
            not isinstance(model_version, str)
            or len(model_version) != 64
            or any(character not in string.hexdigits for character in model_version)
        ):
            raise ValueError("model_version must be a SHA-256 digest")
        if not isinstance(selection_cache_version, DeltaRecCacheVersion):
            raise TypeError("selection_cache_version must be a DeltaRecCacheVersion")
        if not isinstance(state_cache_version, DeltaRecCacheVersion):
            raise TypeError("state_cache_version must be a DeltaRecCacheVersion")
        if group_count < 1:
            raise ValueError("group_count must be positive")
        if not 0 <= unknown_category_group < group_count:
            raise ValueError("unknown_category_group must lie in [0,G)")
        if item_to_category_group.ndim != 1 or item_to_category_group.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("item_to_category_group must be an integer vector")
        if item_to_category_group.numel() < 1:
            raise ValueError("item_to_category_group must be nonempty")
        mapping = item_to_category_group.detach().to(device="cpu", dtype=torch.int64)
        if bool((mapping < -1).any()) or bool((mapping >= group_count).any()):
            raise ValueError("item-to-group entries must be -1 or lie in [0,G)")
        shape = tuple(int(value) for value in state_shape)
        if len(shape) != 4 or any(value < 1 for value in shape):
            raise ValueError("state_shape must be [layers,H,Dk,Dv]")
        target = torch.device(storage_device)
        if pin_memory and target.type != "cpu":
            raise ValueError("pin_memory is valid only for CPU storage")
        if pin_memory and not torch.cuda.is_available():
            raise RuntimeError("pinned global-group cache requires a CUDA runtime")
        if not isinstance(components, Mapping):
            raise TypeError("components must be a mapping")
        component_ratio = validate_retention_ratio(
            components.get("retention_ratio")
        )
        component_floor = components.get("recent_floor")
        if isinstance(component_floor, bool) or not isinstance(component_floor, int):
            raise ValueError("components.recent_floor must be an integer")
        if component_floor < 0:
            raise ValueError("components.recent_floor must be nonnegative")

        self.dataset_id = dataset_id
        self.model_version = model_version.lower()
        self.selection_cache_version = selection_cache_version
        self.state_cache_version = state_cache_version
        self.item_to_category_group = mapping.contiguous()
        self.unknown_category_group = int(unknown_category_group)
        self.group_count = int(group_count)
        self.state_shape = shape
        self.components = {
            **dict(components),
            "retention_ratio": component_ratio,
            "recent_floor": int(component_floor),
        }
        self.storage_device = target
        self.pin_memory = bool(pin_memory)
        self._rows: dict[GlobalGroupStateKey, torch.Tensor] = {}
        self._published: dict[tuple[str, int, str], int] = {}
        self._lock = threading.RLock()

    def _place(self, value: torch.Tensor) -> torch.Tensor:
        cpu = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if self.storage_device.type == "cpu":
            return cpu.pin_memory() if self.pin_memory else cpu.clone()
        return cpu.to(device=self.storage_device)

    def resolve_candidate_groups(self, candidate_ids: torch.Tensor) -> torch.Tensor:
        if candidate_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("candidate_ids must be integer")
        ids = candidate_ids.to(torch.int64)
        mapping = self.item_to_category_group.to(device=ids.device)
        in_range = (ids >= 0) & (ids < mapping.numel())
        safe = ids.clamp(0, max(0, mapping.numel() - 1))
        mapped = mapping.index_select(0, safe.reshape(-1)).reshape_as(ids)
        known = in_range & (mapped >= 0)
        return torch.where(
            known,
            mapped,
            mapped.new_full(mapped.shape, self.unknown_category_group),
        )

    def publish_user_states(
        self,
        *,
        user_id: int,
        history_version: int,
        states: torch.Tensor,
        expected_previous_version: int | None = None,
    ) -> None:
        """Eagerly write all G states, then atomically publish the new version."""

        if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id < 0:
            raise ValueError("user_id must be a nonnegative integer")
        if (
            isinstance(history_version, bool)
            or not isinstance(history_version, int)
            or history_version < 0
        ):
            raise ValueError("history_version must be a nonnegative integer")
        if states.shape != (self.group_count, *self.state_shape):
            raise ValueError("states must have shape [G,layers,H,Dk,Dv]")
        if states.dtype != torch.float32 or not bool(torch.isfinite(states).all()):
            raise ValueError("published recurrent states must be finite FP32")
        staged = [self._place(states[group]) for group in range(self.group_count)]
        owner = (self.dataset_id, int(user_id), self.model_version)
        new_keys = [
            GlobalGroupStateKey(
                self.dataset_id,
                int(user_id),
                int(history_version),
                group,
                self.model_version,
            )
            for group in range(self.group_count)
        ]
        with self._lock:
            previous = self._published.get(owner)
            if expected_previous_version is not None and previous != int(
                expected_previous_version
            ):
                raise CacheMissError("published history version changed before commit")
            if previous is not None and history_version <= previous:
                raise ValueError("history_version must advance monotonically")
            try:
                for key, state in zip(new_keys, staged):
                    self._rows[key] = state
                self._published[owner] = int(history_version)
            except Exception:
                for key in new_keys:
                    self._rows.pop(key, None)
                if previous is None:
                    self._published.pop(owner, None)
                else:
                    self._published[owner] = previous
                raise
            if previous is not None:
                for group in range(self.group_count):
                    self._rows.pop(
                        GlobalGroupStateKey(
                            self.dataset_id,
                            int(user_id),
                            previous,
                            group,
                            self.model_version,
                        ),
                        None,
                    )

    def publish_batch(
        self,
        *,
        user_ids: torch.Tensor | Sequence[int],
        history_versions: torch.Tensor | Sequence[int],
        states: torch.Tensor,
    ) -> None:
        users = (
            user_ids.detach().cpu().to(torch.int64).tolist()
            if isinstance(user_ids, torch.Tensor)
            else list(user_ids)
        )
        versions = (
            history_versions.detach().cpu().to(torch.int64).tolist()
            if isinstance(history_versions, torch.Tensor)
            else list(history_versions)
        )
        if states.ndim != 6 or states.shape[:2] != (
            len(users),
            self.group_count,
        ):
            raise ValueError("batch states must have shape [B,G,layers,H,Dk,Dv]")
        if len(users) != len(versions):
            raise ValueError("user_ids and history_versions must have equal length")
        for row, (user, version) in enumerate(zip(users, versions)):
            self.publish_user_states(
                user_id=int(user),
                history_version=int(version),
                states=states[row],
            )

    def published_history_version(self, user_id: int) -> int:
        with self._lock:
            try:
                return self._published[(self.dataset_id, int(user_id), self.model_version)]
            except KeyError as error:
                raise CacheMissError("user has no atomically published state version") from error

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._published)

    @property
    def state_row_count(self) -> int:
        with self._lock:
            return len(self._rows)

    @property
    def state_bytes_per_group(self) -> int:
        elements = 1
        for value in self.state_shape:
            elements *= value
        return elements * torch.empty((), dtype=torch.float32).element_size()

    def lookup(
        self,
        *,
        user_ids: torch.Tensor | Sequence[int],
        history_versions: torch.Tensor | Sequence[int],
        candidate_ids: torch.Tensor,
        model_version: str,
        target_device: torch.device | str,
        non_blocking: bool = True,
    ) -> GlobalGroupStateLookup:
        if model_version != self.model_version:
            raise CacheMissError("global-group model version mismatch")
        if candidate_ids.ndim != 2 or candidate_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("candidate_ids must be integer [B,K]")
        batch, candidates = candidate_ids.shape
        if batch < 1 or candidates < 1:
            raise ValueError("candidate_ids must contain a nonempty B and K")
        users = (
            user_ids.detach().cpu().to(torch.int64).tolist()
            if isinstance(user_ids, torch.Tensor)
            else list(user_ids)
        )
        versions = (
            history_versions.detach().cpu().to(torch.int64).tolist()
            if isinstance(history_versions, torch.Tensor)
            else list(history_versions)
        )
        if len(users) != batch or len(versions) != batch:
            raise ValueError("cache keys must contain one row per request")
        group_ids_cpu = self.resolve_candidate_groups(
            candidate_ids.detach().to(device="cpu")
        )
        candidate_ids_cpu = candidate_ids.detach().to(device="cpu", dtype=torch.int64)
        state_rows: list[torch.Tensor] = []
        occupied_users: list[int] = []
        occupied_groups: list[int] = []
        candidate_to_state = torch.empty((batch, candidates), dtype=torch.int64)
        packed_candidates: list[int] = []
        offsets = [0]

        with self._lock:
            for user_row, (user, version) in enumerate(zip(users, versions)):
                owner = (self.dataset_id, int(user), self.model_version)
                published = self._published.get(owner)
                if published is None:
                    raise CacheMissError("user has no atomically published state version")
                if int(version) != published:
                    raise CacheMissError("requested history version is stale or unpublished")
                row_groups = sorted(set(int(v) for v in group_ids_cpu[user_row].tolist()))
                for group in row_groups:
                    key = GlobalGroupStateKey(
                        self.dataset_id,
                        int(user),
                        published,
                        group,
                        self.model_version,
                    )
                    state = self._rows.get(key)
                    if state is None:
                        raise CacheMissError("atomically published group state is missing")
                    state_row = len(state_rows)
                    state_rows.append(state)
                    occupied_users.append(user_row)
                    occupied_groups.append(group)
                    slots = [
                        slot
                        for slot in range(candidates)
                        if int(group_ids_cpu[user_row, slot]) == group
                    ]
                    slots.sort(key=lambda slot: (int(candidate_ids_cpu[user_row, slot]), slot))
                    for slot in slots:
                        candidate_to_state[user_row, slot] = state_row
                        packed_candidates.append(user_row * candidates + slot)
                    offsets.append(len(packed_candidates))

        target = torch.device(target_device)
        states = torch.stack(state_rows)
        moving = states.device != target
        states = states.to(
            device=target,
            non_blocking=bool(
                non_blocking
                and states.device.type == "cpu"
                and target.type == "cuda"
                and states.is_pinned()
            ),
        )
        state_bytes = len(state_rows) * self.state_bytes_per_group
        return GlobalGroupStateLookup(
            states=states,
            candidate_group_ids=group_ids_cpu.to(target),
            candidate_to_state_row=candidate_to_state.to(target),
            packed_candidate_indices=torch.tensor(
                packed_candidates, dtype=torch.int64, device=target
            ),
            offsets=torch.tensor(offsets, dtype=torch.int64, device=target),
            occupied_user_rows=torch.tensor(
                occupied_users, dtype=torch.int64, device=target
            ),
            occupied_group_ids=torch.tensor(
                occupied_groups, dtype=torch.int64, device=target
            ),
            published_history_versions=tuple(int(v) for v in versions),
            materialization_bytes=state_bytes,
            transferred_bytes=state_bytes if moving else 0,
        )

    def artifact_state(self) -> dict[str, Any]:
        with self._lock:
            keys = sorted(
                self._rows,
                key=lambda key: (key.user_id, key.history_version, key.global_group_id),
            )
            published = sorted((owner[1], version) for owner, version in self._published.items())
            payload: dict[str, Any] = {
                "schema": GLOBAL_GROUP_CACHE_SCHEMA,
                "dataset_id": self.dataset_id,
                "model_version": self.model_version,
                "state_cache_version": {
                    "backbone": self.state_cache_version.backbone,
                    "selector": self.state_cache_version.selector,
                    "schema": self.state_cache_version.schema,
                    "policy": self.state_cache_version.policy,
                },
                "selection_cache_version": {
                    "backbone": self.selection_cache_version.backbone,
                    "selector": self.selection_cache_version.selector,
                    "schema": self.selection_cache_version.schema,
                    "policy": self.selection_cache_version.policy,
                },
                "unknown_category_group": self.unknown_category_group,
                "group_count": self.group_count,
                "state_shape": list(self.state_shape),
                "components": dict(self.components),
                "item_to_category_group": self.item_to_category_group.clone(),
                "row_user_ids": torch.tensor([key.user_id for key in keys], dtype=torch.int64),
                "row_history_versions": torch.tensor(
                    [key.history_version for key in keys], dtype=torch.int64
                ),
                "row_group_ids": torch.tensor(
                    [key.global_group_id for key in keys], dtype=torch.int64
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
                    [version for _, version in published], dtype=torch.int64
                ),
            }
        payload["artifact_sha256"] = _artifact_sha256(payload)
        return payload

    def save(self, path: str | Path) -> Path:
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
        expected_model_version: str | None = None,
    ) -> "GlobalGroupStateCache":
        if payload.get("schema") != GLOBAL_GROUP_CACHE_SCHEMA:
            raise ValueError("global-group cache schema mismatch")
        if payload.get("artifact_sha256") != _artifact_sha256(payload):
            raise ValueError("global-group cache artifact checksum mismatch")
        dataset_id = str(payload["dataset_id"])
        model_version = str(payload["model_version"])
        if expected_dataset_id is not None and dataset_id != expected_dataset_id:
            raise CacheMissError("global-group dataset mismatch")
        if expected_model_version is not None and model_version != expected_model_version:
            raise CacheMissError("global-group model version mismatch")
        version_record = payload["state_cache_version"]
        if not isinstance(version_record, Mapping):
            raise ValueError("invalid state-cache version record")
        state_cache_version = DeltaRecCacheVersion(
            backbone=str(version_record["backbone"]),
            selector=str(version_record["selector"]),
            schema=str(version_record["schema"]),
            policy=str(version_record["policy"]),
        )
        selection_record = payload["selection_cache_version"]
        if not isinstance(selection_record, Mapping):
            raise ValueError("invalid selection-cache version record")
        selection_cache_version = DeltaRecCacheVersion(
            backbone=str(selection_record["backbone"]),
            selector=str(selection_record["selector"]),
            schema=str(selection_record["schema"]),
            policy=str(selection_record["policy"]),
        )
        cache = cls(
            dataset_id=dataset_id,
            model_version=model_version,
            selection_cache_version=selection_cache_version,
            state_cache_version=state_cache_version,
            item_to_category_group=payload["item_to_category_group"],
            unknown_category_group=int(payload["unknown_category_group"]),
            group_count=int(payload["group_count"]),
            state_shape=payload["state_shape"],
            components=payload["components"],
            storage_device=storage_device,
            pin_memory=pin_memory,
        )
        row_users = payload["row_user_ids"].tolist()
        row_versions = payload["row_history_versions"].tolist()
        row_groups = payload["row_group_ids"].tolist()
        row_states = payload["row_states"]
        if not (
            len(row_users) == len(row_versions) == len(row_groups) == len(row_states)
        ):
            raise ValueError("global-group cache row key/value counts disagree")
        if row_states.shape != (len(row_users), *cache.state_shape):
            raise ValueError("global-group cache row-state layout mismatch")
        if row_states.dtype != torch.float32 or not bool(torch.isfinite(row_states).all()):
            raise ValueError("global-group cache artifact states must be finite FP32")
        for index, (user, version, group) in enumerate(
            zip(row_users, row_versions, row_groups)
        ):
            if int(user) < 0 or int(version) < 0 or not 0 <= int(group) < cache.group_count:
                raise ValueError("global-group cache artifact contains an invalid row key")
            key = GlobalGroupStateKey(
                dataset_id, int(user), int(version), int(group), model_version
            )
            if key in cache._rows:
                raise ValueError("global-group cache artifact contains duplicate rows")
            cache._rows[key] = cache._place(row_states[index])
        published_users = payload["published_user_ids"].tolist()
        published_versions = payload["published_history_versions"].tolist()
        if len(published_users) != len(published_versions):
            raise ValueError("published key/value counts disagree")
        for user, version in zip(published_users, published_versions):
            owner = (dataset_id, int(user), model_version)
            if int(user) < 0 or int(version) < 0 or owner in cache._published:
                raise ValueError("global-group cache has an invalid published pointer")
            cache._published[owner] = int(version)
            for group in range(cache.group_count):
                key = GlobalGroupStateKey(
                    dataset_id, int(user), int(version), group, model_version
                )
                if key not in cache._rows:
                    raise ValueError("published version has a missing global-group row")
        return cache

    @classmethod
    def load(
        cls,
        path: str | Path,
        **kwargs: Any,
    ) -> "GlobalGroupStateCache":
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise ValueError("global-group cache artifact must be a mapping")
        return cls.from_artifact_state(payload, **kwargs)


__all__ = [
    "GLOBAL_GROUP_CACHE_SCHEMA",
    "GlobalGroupStateCache",
    "GlobalGroupStateKey",
    "GlobalGroupStateLookup",
    "build_global_group_model_version",
]

