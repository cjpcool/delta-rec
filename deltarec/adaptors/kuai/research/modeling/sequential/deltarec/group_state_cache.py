# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""Persistent recurrent-state lookup for Subplan 5C serving.

``GroupSelectionLookupCache`` remains the owner of the exact request key and
canonical candidate grouping.  This module composes that immutable table with
the FP32 recurrent state produced after each group's contextual and selected
history prefix.  A hit can therefore serve candidate-only, read-only GDR
events without fetching or projecting the full history again.

The state rows are deliberately aligned one-for-one with selection-cache rows.
They are keyed by the *complete* canonical candidate set, not by independent
``(user, candidate)`` pairs: 5C grouping and pooled history selection depend on
the other candidates in the set.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import string
import tempfile
from typing import Any

import torch

from ..delta_rec_cache import DeltaRecCacheVersion
from .group_cache import (
    CacheMissError,
    GroupSelectionLookupCache,
)
from .grouping import GroupingOutput


_STATE_CACHE_CONTRACT = "subplan5c-group-prefix-state-lookup-v1"
_STATE_CACHE_ARTIFACT_SCHEMA_VERSION = 1
_STATE_LAYOUT = "N,G,layers,H,Dk,Dv"


def _json_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _version_record(version: DeltaRecCacheVersion) -> dict[str, str]:
    return {
        "backbone": version.backbone,
        "selector": version.selector,
        "schema": version.schema,
        "policy": version.policy,
    }


def _parse_version_record(value: Any) -> DeltaRecCacheVersion:
    if not isinstance(value, Mapping) or set(value) != {
        "backbone",
        "selector",
        "schema",
        "policy",
    }:
        raise ValueError("group state cache artifact has an invalid version record")
    fields: dict[str, str] = {}
    for name in ("backbone", "selector", "schema", "policy"):
        field = value[name]
        if not isinstance(field, str) or not field:
            raise ValueError(
                f"group state cache version field {name} must be nonempty"
            )
        fields[name] = field
    return DeltaRecCacheVersion(**fields)


def build_group_state_cache_version(
    *,
    model_sha256: str,
    selection_version: DeltaRecCacheVersion,
    layer_count: int,
    group_count: int,
    num_heads: int,
    key_dim: int,
    value_dim: int,
    contextual_seq_len: int,
) -> DeltaRecCacheVersion:
    """Bind cached prefix states to model, selection, and tensor semantics.

    Unlike the positions-only selection cache, a recurrent-state artifact is
    invalid after *any* effective backbone/GDR parameter change.  Callers must
    supply the SHA-256 of the final serving checkpoint rather than an
    initialization-only hash.
    """

    if not isinstance(model_sha256, str) or len(model_sha256) != 64 or any(
        character not in string.hexdigits for character in model_sha256
    ):
        raise ValueError("model_sha256 must be a 64-character hexadecimal digest")
    if not isinstance(selection_version, DeltaRecCacheVersion):
        raise TypeError("selection_version must be a DeltaRecCacheVersion")
    dimensions = {
        "layer_count": _require_positive_integer(layer_count, "layer_count"),
        "group_count": _require_positive_integer(group_count, "group_count"),
        "num_heads": _require_positive_integer(num_heads, "num_heads"),
        "key_dim": _require_positive_integer(key_dim, "key_dim"),
        "value_dim": _require_positive_integer(value_dim, "value_dim"),
    }
    if (
        isinstance(contextual_seq_len, bool)
        or not isinstance(contextual_seq_len, int)
        or contextual_seq_len < 0
    ):
        raise ValueError("contextual_seq_len must be a nonnegative integer")

    return DeltaRecCacheVersion(
        backbone=_json_digest(
            {
                "contract": _STATE_CACHE_CONTRACT,
                "model_sha256": model_sha256.lower(),
            }
        ),
        # Keeping the complete selection fingerprint directly inspectable also
        # lets the cache constructor reject accidental composition with a
        # different positions table.
        selector=selection_version.fingerprint,
        schema=_json_digest(
            {
                "contract": _STATE_CACHE_CONTRACT,
                "layout": _STATE_LAYOUT,
                "dtype": "float32",
                "contextual_seq_len": int(contextual_seq_len),
                **dimensions,
            }
        ),
        policy=_json_digest(
            {
                "contract": _STATE_CACHE_CONTRACT,
                "selection_version": _version_record(selection_version),
            }
        ),
    )


def _state_artifact_sha256(payload: Mapping[str, Any]) -> str:
    """Hash the nested selection artifact identity and exact state bytes."""

    selection_payload = payload.get("selection_cache")
    if not isinstance(selection_payload, Mapping):
        raise ValueError("group state cache selection artifact must be a mapping")
    selection_sha256 = selection_payload.get("artifact_sha256")
    if not isinstance(selection_sha256, str) or not selection_sha256:
        raise ValueError("nested selection cache artifact is missing its checksum")
    states = payload.get("states")
    if not isinstance(states, torch.Tensor):
        raise ValueError("group state cache artifact states must be a tensor")
    states_cpu = states.detach().to(device="cpu").contiguous()
    metadata = {
        "schema_version": payload.get("schema_version"),
        "cache_contract": payload.get("cache_contract"),
        "version": payload.get("version"),
        "selection_artifact_sha256": selection_sha256,
        "states_dtype": str(states_cpu.dtype),
        "states_shape": list(states_cpu.shape),
    }
    digest = hashlib.sha256(
        json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    )
    digest.update(states_cpu.numpy().tobytes())
    return digest.hexdigest()


def _logical_to_layer_major(states: torch.Tensor) -> torch.Tensor:
    """Return contiguous ``[layers,N,G,H,Dk,Dv]`` physical storage."""

    layer_major = states.permute(2, 0, 1, 3, 4, 5)
    if layer_major.is_contiguous():
        return layer_major
    if states.device.type == "cpu" and states.is_pinned():
        output = torch.empty(
            layer_major.shape,
            dtype=layer_major.dtype,
            device="cpu",
            pin_memory=True,
        )
        output.copy_(layer_major)
        return output
    return layer_major.contiguous()


def _layer_major_to_logical(states: torch.Tensor) -> torch.Tensor:
    """Expose ``[N,G,layers,H,Dk,Dv]`` as a zero-copy public view."""

    return states.permute(1, 2, 0, 3, 4, 5)


@dataclass(frozen=True)
class GroupStateCacheLookup:
    """Exact grouping plus FP32 per-layer states for candidate-only reads."""

    grouping: GroupingOutput
    states: torch.Tensor
    layer_major_states: torch.Tensor
    hit_count: int
    materialization_bytes: int
    transferred_bytes: int
    selection_materialization_bytes: int
    state_materialization_bytes: int
    cache_rows: tuple[int, ...]
    storage_device: str

    @property
    def initial_states(self) -> torch.Tensor:
        """Alias documenting how the online GDR executor consumes ``states``."""

        return self.states


class GroupStateLookupCache:
    """Persistent exact-key table of group-prefix recurrent states.

    Public and artifact state layout is ``[N,G,layers,H,Dk,Dv]``.  Residency
    is physically layer-major ``[layers,N,G,H,Dk,Dv]`` so a lookup gathers the
    entry dimension once and the executor's layer-major view is contiguous
    without a transpose copy.  States remain FP32 in every residency mode.
    The composed selection cache owns key resolution, collision checks,
    candidate permutation, and grouping materialization.
    """

    def __init__(
        self,
        *,
        selection_cache: GroupSelectionLookupCache,
        states: torch.Tensor,
        version: DeltaRecCacheVersion,
    ) -> None:
        if not isinstance(selection_cache, GroupSelectionLookupCache):
            raise TypeError("selection_cache must be a GroupSelectionLookupCache")
        if not isinstance(states, torch.Tensor):
            raise TypeError("states must be a torch.Tensor")
        if not isinstance(version, DeltaRecCacheVersion):
            raise TypeError("version must be a DeltaRecCacheVersion")
        if states.ndim != 6:
            raise ValueError("states must have shape [N,G,layers,H,Dk,Dv]")
        expected_prefix = (selection_cache.entry_count, selection_cache.group_count)
        if states.shape[:2] != expected_prefix:
            raise ValueError(
                "state entries/groups must align one-for-one with the selection cache"
            )
        if any(int(size) < 1 for size in states.shape[2:]):
            raise ValueError("every recurrent state dimension must be positive")
        if states.dtype != torch.float32:
            raise ValueError("cached recurrent states must use FP32 storage")
        if states.device != selection_cache.storage_device:
            raise ValueError("state and selection payloads must share a storage device")
        if selection_cache.storage_device.type == "cpu" and (
            states.is_pinned() != selection_cache.is_pinned
        ):
            raise ValueError(
                "state and selection payloads must use the same pinned-host policy"
            )
        if not bool(torch.isfinite(states).all()):
            raise ValueError("cached recurrent states must be finite")
        if version.selector != selection_cache.version.fingerprint:
            raise ValueError(
                "state cache version is not bound to the composed selection cache"
            )

        states_layer_major = _logical_to_layer_major(states)
        if not states_layer_major.is_contiguous():
            raise ValueError("layer-major recurrent state storage must be contiguous")
        if selection_cache.storage_device.type == "cpu" and (
            states_layer_major.is_pinned() != selection_cache.is_pinned
        ):
            raise ValueError(
                "layer-major states lost the selection cache's pinned-host policy"
            )

        self.selection_cache = selection_cache
        self._states_layer_major = states_layer_major
        self.version = version

    @classmethod
    def from_outputs(
        cls,
        *,
        selection_cache: GroupSelectionLookupCache,
        states: torch.Tensor,
        version: DeltaRecCacheVersion,
        storage_device: torch.device | str = "cpu",
        pin_memory: bool = False,
        validate_runtime: bool = True,
    ) -> "GroupStateLookupCache":
        """Build a row-aligned state table from audited offline GDR outputs."""

        if not isinstance(selection_cache, GroupSelectionLookupCache):
            raise TypeError("selection_cache must be a GroupSelectionLookupCache")
        if not isinstance(states, torch.Tensor):
            raise TypeError("states must be a torch.Tensor")
        if states.dtype != torch.float32:
            raise ValueError("offline recurrent states must be FP32")
        if states.ndim != 6 or states.shape[:2] != (
            selection_cache.entry_count,
            selection_cache.group_count,
        ):
            raise ValueError(
                "offline states must have shape [N,G,layers,H,Dk,Dv] aligned "
                "with selection-cache rows"
            )
        if any(int(size) < 1 for size in states.shape[2:]):
            raise ValueError("every recurrent state dimension must be positive")
        if not isinstance(version, DeltaRecCacheVersion):
            raise TypeError("version must be a DeltaRecCacheVersion")
        if not isinstance(pin_memory, bool) or not isinstance(validate_runtime, bool):
            raise TypeError("pin_memory and validate_runtime must be boolean")
        if version.selector != selection_cache.version.fingerprint:
            raise ValueError(
                "state cache version is not bound to the supplied selection cache"
            )

        target = torch.device(storage_device)
        if pin_memory and target.type != "cpu":
            raise ValueError("pin_memory is valid only for CPU-resident tables")
        if pin_memory and not torch.cuda.is_available():
            raise RuntimeError("pinned state-cache storage requires a CUDA runtime")

        states_cpu = states.detach().to(device="cpu").contiguous()
        if validate_runtime and not bool(torch.isfinite(states_cpu).all()):
            raise ValueError("offline recurrent states must be finite")
        states_layer_major_cpu = _logical_to_layer_major(states_cpu)
        placed_selection = GroupSelectionLookupCache.from_artifact_state(
            selection_cache.artifact_state(),
            storage_device=target,
            pin_memory=pin_memory,
            expected_version=selection_cache.version,
        )
        if target.type == "cpu":
            placed_states_layer_major = (
                states_layer_major_cpu.pin_memory()
                if pin_memory
                else states_layer_major_cpu.clone()
            )
        else:
            placed_states_layer_major = states_layer_major_cpu.to(device=target)
        return cls(
            selection_cache=placed_selection,
            states=_layer_major_to_logical(placed_states_layer_major),
            version=version,
        )

    @property
    def states(self) -> torch.Tensor:
        """Logical ``[N,G,layers,H,Dk,Dv]`` zero-copy state view."""

        return _layer_major_to_logical(self._states_layer_major)

    @property
    def layer_major_states(self) -> torch.Tensor:
        """Contiguous physical ``[layers,N,G,H,Dk,Dv]`` state storage."""

        return self._states_layer_major

    @property
    def entry_count(self) -> int:
        return int(self._states_layer_major.shape[1])

    @property
    def group_count(self) -> int:
        return int(self._states_layer_major.shape[2])

    @property
    def layer_count(self) -> int:
        return int(self._states_layer_major.shape[0])

    @property
    def num_heads(self) -> int:
        return int(self._states_layer_major.shape[3])

    @property
    def key_dim(self) -> int:
        return int(self._states_layer_major.shape[4])

    @property
    def value_dim(self) -> int:
        return int(self._states_layer_major.shape[5])

    @property
    def candidate_count(self) -> int:
        return self.selection_cache.candidate_count

    @property
    def storage_device(self) -> torch.device:
        return self._states_layer_major.device

    @property
    def is_pinned(self) -> bool:
        return bool(
            self.storage_device.type == "cpu"
            and self._states_layer_major.is_pinned()
        )

    @property
    def state_payload_bytes(self) -> int:
        return (
            self._states_layer_major.numel()
            * self._states_layer_major.element_size()
        )

    @property
    def selection_payload_bytes(self) -> int:
        return self.selection_cache.payload_bytes

    @property
    def payload_bytes(self) -> int:
        """Tensor bytes for states and exact-key/grouping metadata."""

        return self.selection_payload_bytes + self.state_payload_bytes

    @property
    def state_bytes_per_entry(self) -> int:
        return (
            self._states_layer_major.numel()
            // self.entry_count
            * self._states_layer_major.element_size()
        )

    @property
    def bytes_per_entry(self) -> int:
        return self.payload_bytes // self.entry_count

    def artifact_state(self) -> dict[str, Any]:
        """Return a checksummed CPU-portable state and selection artifact."""

        payload: dict[str, Any] = {
            "schema_version": _STATE_CACHE_ARTIFACT_SCHEMA_VERSION,
            "cache_contract": _STATE_CACHE_CONTRACT,
            "version": _version_record(self.version),
            "selection_cache": self.selection_cache.artifact_state(),
            "states": self.states.detach().to(device="cpu").contiguous().clone(),
        }
        payload["artifact_sha256"] = _state_artifact_sha256(payload)
        return payload

    def save(self, path: str | Path) -> Path:
        """Atomically persist an offline-built table for a serving process."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
            torch.save(self.artifact_state(), temporary)
            os.replace(temporary, target)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return target

    @classmethod
    def from_artifact_state(
        cls,
        payload: Mapping[str, Any],
        *,
        storage_device: torch.device | str = "cpu",
        pin_memory: bool = False,
        expected_version: DeltaRecCacheVersion | None = None,
    ) -> "GroupStateLookupCache":
        """Validate and place a persisted state table without model execution."""

        if not isinstance(payload, Mapping):
            raise ValueError("group state cache artifact must be a mapping")
        if payload.get("schema_version") != _STATE_CACHE_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported group state cache artifact schema")
        if payload.get("cache_contract") != _STATE_CACHE_CONTRACT:
            raise ValueError("group state cache artifact contract mismatch")
        recorded_digest = payload.get("artifact_sha256")
        if not isinstance(recorded_digest, str) or not recorded_digest:
            raise ValueError("group state cache artifact is missing its checksum")
        if _state_artifact_sha256(payload) != recorded_digest:
            raise ValueError("group state cache artifact checksum mismatch")

        version = _parse_version_record(payload.get("version"))
        if expected_version is not None and version != expected_version:
            raise CacheMissError("persisted group state cache version mismatch")
        states = payload.get("states")
        if not isinstance(states, torch.Tensor):
            raise ValueError("group state cache artifact states must be a tensor")
        states_cpu = states.detach().to(device="cpu").contiguous()
        if states_cpu.dtype != torch.float32 or states_cpu.ndim != 6:
            raise ValueError(
                "persisted states must be FP32 [N,G,layers,H,Dk,Dv]"
            )
        if any(int(size) < 1 for size in states_cpu.shape):
            raise ValueError("persisted state dimensions must all be positive")
        if not bool(torch.isfinite(states_cpu).all()):
            raise ValueError("persisted recurrent states must be finite")

        selection_payload = payload.get("selection_cache")
        if not isinstance(selection_payload, Mapping):
            raise ValueError("group state cache is missing its selection artifact")
        selection_cache = GroupSelectionLookupCache.from_artifact_state(
            selection_payload,
            storage_device=storage_device,
            pin_memory=pin_memory,
        )
        target = torch.device(storage_device)
        if pin_memory and target.type != "cpu":
            raise ValueError("pin_memory is valid only for CPU-resident tables")
        if pin_memory and not torch.cuda.is_available():
            raise RuntimeError("pinned state-cache storage requires a CUDA runtime")
        states_layer_major_cpu = _logical_to_layer_major(states_cpu)
        if target.type == "cpu":
            placed_states_layer_major = (
                states_layer_major_cpu.pin_memory()
                if pin_memory
                else states_layer_major_cpu.clone()
            )
        else:
            placed_states_layer_major = states_layer_major_cpu.to(device=target)
        return cls(
            selection_cache=selection_cache,
            states=_layer_major_to_logical(placed_states_layer_major),
            version=version,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        storage_device: torch.device | str = "cpu",
        pin_memory: bool = False,
        expected_version: DeltaRecCacheVersion | None = None,
    ) -> "GroupStateLookupCache":
        """Load a training-produced state table into serving residency."""

        payload = torch.load(
            Path(path),
            map_location="cpu",
            weights_only=True,
        )
        return cls.from_artifact_state(
            payload,
            storage_device=storage_device,
            pin_memory=pin_memory,
            expected_version=expected_version,
        )

    def lookup(
        self,
        *,
        user_ids: torch.Tensor | Sequence[int],
        history_versions: torch.Tensor | Sequence[int | str],
        candidate_ids: torch.Tensor,
        history_lengths: torch.Tensor,
        version: DeltaRecCacheVersion,
        target_device: torch.device | str,
        validate_runtime: bool = True,
        non_blocking: bool = True,
    ) -> GroupStateCacheLookup:
        """Resolve an exact batch hit and materialize grouping plus states."""

        if not isinstance(version, DeltaRecCacheVersion):
            raise TypeError("version must be a DeltaRecCacheVersion")
        if version != self.version:
            raise CacheMissError("group state cache version mismatch")
        if not isinstance(validate_runtime, bool) or not isinstance(non_blocking, bool):
            raise TypeError("validate_runtime and non_blocking must be boolean")

        selection_lookup = self.selection_cache.lookup_grouping(
            user_ids=user_ids,
            history_versions=history_versions,
            candidate_ids=candidate_ids,
            history_lengths=history_lengths,
            version=self.selection_cache.version,
            target_device=target_device,
            validate_runtime=validate_runtime,
            non_blocking=non_blocking,
        )
        batch = selection_lookup.hit_count
        target = torch.device(target_device)
        row_index = torch.tensor(
            selection_lookup.cache_rows,
            dtype=torch.int64,
            device=self.storage_device,
        )
        moving_devices = self.storage_device != target
        if (
            self.storage_device.type == "cpu"
            and target.type == "cuda"
            and self.is_pinned
        ):
            selected_states_layer_major = torch.empty(
                (
                    self.layer_count,
                    batch,
                    self.group_count,
                    self.num_heads,
                    self.key_dim,
                    self.value_dim,
                ),
                dtype=torch.float32,
                device="cpu",
                pin_memory=True,
            )
            torch.index_select(
                self._states_layer_major,
                1,
                row_index,
                out=selected_states_layer_major,
            )
        else:
            selected_states_layer_major = self._states_layer_major.index_select(
                1, row_index
            )
        if moving_devices:
            async_copy = bool(
                non_blocking
                and self.storage_device.type == "cpu"
                and target.type == "cuda"
                and selected_states_layer_major.is_pinned()
            )
            selected_states_layer_major = selected_states_layer_major.to(
                device=target,
                non_blocking=async_copy,
            )
        if validate_runtime:
            if selected_states_layer_major.shape != (
                self.layer_count,
                batch,
                self.group_count,
                self.num_heads,
                self.key_dim,
                self.value_dim,
            ):
                raise CacheMissError("materialized group states have an invalid layout")
            if selected_states_layer_major.dtype != torch.float32 or not bool(
                torch.isfinite(selected_states_layer_major).all()
            ):
                raise CacheMissError("materialized group states are invalid")
        selected_states = _layer_major_to_logical(selected_states_layer_major)

        state_materialization_bytes = batch * self.state_bytes_per_entry
        materialization_bytes = (
            selection_lookup.materialization_bytes + state_materialization_bytes
        )
        transferred_bytes = selection_lookup.transferred_bytes + (
            state_materialization_bytes if moving_devices else 0
        )
        return GroupStateCacheLookup(
            grouping=selection_lookup.grouping,
            states=selected_states,
            layer_major_states=selected_states_layer_major,
            hit_count=batch,
            materialization_bytes=materialization_bytes,
            transferred_bytes=transferred_bytes,
            selection_materialization_bytes=selection_lookup.materialization_bytes,
            state_materialization_bytes=state_materialization_bytes,
            cache_rows=selection_lookup.cache_rows,
            storage_device=str(self.storage_device),
        )


__all__ = [
    "GroupStateCacheLookup",
    "GroupStateLookupCache",
    "build_group_state_cache_version",
]

