# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""Versioned serving lookup for Subplan 5C grouping and history selection.

The expensive grouping, selector, pooling, and top-k work belongs to the
offline/training side of this cache.  A serving lookup materializes only the
small group layout and chronological history positions needed by
``pack_group_streams``.  Candidate transport order is deliberately not part of
the key: the cached canonical assignment is scattered back to the request's
current order on every hit.

This cache stores *positions*, not flattened dense-history indices.  The latter
depend on the other requests in the current serving batch and are therefore
rebuilt using that batch's current dense-history width.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any

import torch

from ..delta_rec_cache import DeltaRecCacheVersion
from .group_selection import GroupSelectionOutput
from .grouping import GroupingOutput
from .selection import exact_budget, validate_retention_ratio


_CACHE_CONTRACT = "headline-group-selection-lookup-v2"
_CACHE_ARTIFACT_SCHEMA_VERSION = 2
_CACHE_ARTIFACT_TENSORS = (
    "canonical_candidate_ids",
    "canonical_group_ids",
    "anchor_ids",
    "selected_positions",
    "history_lengths",
)


class CacheMissError(KeyError):
    """A fail-closed miss or contract mismatch in the serving lookup table."""


def _json_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_group_selection_cache_version(
    *,
    selector_artifacts: Mapping[str, str],
    grouping_artifacts: Mapping[str, str] | None = None,
    candidate_group_count: int,
    grouping_policy: str,
    group_pool: str,
    retention_ratio: float,
    recent_floor: int,
    contextual_seq_len: int,
) -> DeltaRecCacheVersion:
    """Build the cache contract from selection-affecting model semantics.

    Execution chunk size and GDR weights are intentionally absent: neither can
    change candidate grouping or retained source positions.  A change to any
    selector artifact or semantic selection knob produces a different version.
    """

    if not isinstance(selector_artifacts, Mapping):
        raise TypeError("selector_artifacts must be a string mapping")
    artifacts: dict[str, str] = {}
    for name, digest in selector_artifacts.items():
        if not isinstance(name, str) or not name:
            raise ValueError("selector artifact names must be nonempty strings")
        if not isinstance(digest, str) or not digest:
            raise ValueError("selector artifact digests must be nonempty strings")
        artifacts[name] = digest
    frozen_grouping_artifacts: dict[str, str] = {}
    if grouping_artifacts is not None:
        if not isinstance(grouping_artifacts, Mapping):
            raise TypeError("grouping_artifacts must be a string mapping")
        for name, digest in grouping_artifacts.items():
            if not isinstance(name, str) or not name:
                raise ValueError("grouping artifact names must be nonempty strings")
            if not isinstance(digest, str) or not digest:
                raise ValueError("grouping artifact digests must be nonempty strings")
            frozen_grouping_artifacts[name] = digest
    if (
        isinstance(candidate_group_count, bool)
        or not isinstance(candidate_group_count, int)
        or candidate_group_count < 1
    ):
        raise ValueError("candidate_group_count must be a positive integer")
    if not isinstance(grouping_policy, str) or not grouping_policy:
        raise ValueError("grouping_policy must be a nonempty string")
    if not isinstance(group_pool, str) or not group_pool:
        raise ValueError("group_pool must be a nonempty string")
    retention_ratio = validate_retention_ratio(retention_ratio)
    for name, value in (
        ("recent_floor", recent_floor),
        ("contextual_seq_len", contextual_seq_len),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")

    policy_payload: dict[str, Any] = {
        "candidate_group_count": candidate_group_count,
        "grouping_policy": grouping_policy,
        "group_pool": group_pool,
        "retention_ratio": retention_ratio,
        "recent_floor": recent_floor,
    }
    # Preserve byte-identical versions for the two pre-existing policies while
    # binding the new fixed-category semantics to both frozen tensors.
    if frozen_grouping_artifacts:
        policy_payload["grouping_artifacts"] = frozen_grouping_artifacts

    return DeltaRecCacheVersion(
        # This table contains no recurrent state, so it does not bind to GDR
        # parameters.  The explicit constant prevents it being confused with a
        # state cache using the same selector artifacts.
        backbone=_json_digest(
            {
                "contract": _CACHE_CONTRACT,
                "payload": "grouping-and-chronological-source-positions",
                "recurrent_state": "not-stored",
            }
        ),
        selector=_json_digest(artifacts),
        schema=_json_digest(
            {
                "contract": _CACHE_CONTRACT,
                "contextual_seq_len": contextual_seq_len,
                "candidate_key": "sorted-unique-int64-multiset",
                "position_dtype": "int32",
            }
        ),
        policy=_json_digest(policy_payload),
    )


def _require_integer_matrix(value: torch.Tensor, name: str) -> None:
    if value.ndim != 2 or value.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be an integer tensor with shape [B,K]")
    if value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError(f"{name} must contain a nonempty batch and candidate set")


def _integer_values(
    values: torch.Tensor | Sequence[int],
    *,
    name: str,
    expected: int,
) -> list[int]:
    if isinstance(values, torch.Tensor):
        if values.ndim != 1 or values.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must be a one-dimensional integer tensor")
        materialized = values.detach().to(device="cpu", dtype=torch.int64).tolist()
    else:
        if isinstance(values, (str, bytes)):
            raise TypeError(f"{name} must be an integer sequence")
        materialized = list(values)
    if len(materialized) != expected:
        raise ValueError(f"{name} must have length {expected}")
    output: list[int] = []
    for value in materialized:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} entries must be integers")
        output.append(int(value))
    return output


def _history_version_values(
    values: torch.Tensor | Sequence[int | str],
    *,
    expected: int,
) -> list[tuple[str, int | str]]:
    if isinstance(values, torch.Tensor):
        raw: Sequence[int | str] = _integer_values(
            values,
            name="history_versions",
            expected=expected,
        )
    else:
        if isinstance(values, (str, bytes)):
            raise TypeError("history_versions must be a sequence")
        raw = list(values)
        if len(raw) != expected:
            raise ValueError(f"history_versions must have length {expected}")
    output: list[tuple[str, int | str]] = []
    for value in raw:
        if isinstance(value, bool):
            raise TypeError("history version entries cannot be boolean")
        if isinstance(value, int):
            output.append(("int", int(value)))
        elif isinstance(value, str) and value:
            output.append(("str", value))
        else:
            raise TypeError(
                "history version entries must be integers or nonempty strings"
            )
    return output


def _candidate_fingerprint(sorted_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    digest.update(_CACHE_CONTRACT.encode("ascii"))
    digest.update(len(sorted_ids).to_bytes(8, "little", signed=False))
    for item_id in sorted_ids:
        digest.update(int(item_id).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _canonicalize_candidates(
    candidate_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, ...]], list[str]]:
    """Canonicalize on CPU for stable request hashing and collision checks."""

    _require_integer_matrix(candidate_ids, "candidate_ids")
    ids_cpu = candidate_ids.detach().to(
        device="cpu", dtype=torch.int64
    ).contiguous()
    canonical_order = torch.argsort(ids_cpu, dim=1, stable=True)
    canonical_ids = ids_cpu.gather(1, canonical_order)
    if candidate_ids.shape[1] > 1:
        duplicate_rows = (canonical_ids[:, 1:] == canonical_ids[:, :-1]).any(dim=1)
        if bool(duplicate_rows.any()):
            rows = torch.nonzero(duplicate_rows, as_tuple=False).flatten().tolist()
            message = (
                "candidate cache requires unique item IDs; duplicate IDs in rows "
                f"{rows}"
            )
            raise ValueError(message)
    rows = [tuple(int(value) for value in row) for row in canonical_ids.tolist()]
    fingerprints = [_candidate_fingerprint(row) for row in rows]
    return ids_cpu, canonical_order, rows, fingerprints


def _tensor_bytes(value: torch.Tensor) -> int:
    return value.numel() * value.element_size()


def _cache_artifact_sha256(payload: Mapping[str, Any]) -> str:
    """Hash portable metadata and exact tensor bytes for persisted tables."""

    metadata = {
        "schema_version": payload.get("schema_version"),
        "cache_contract": payload.get("cache_contract"),
        "version": payload.get("version"),
        "retention_ratio": payload.get("retention_ratio"),
        "recent_floor": payload.get("recent_floor"),
        "keys": payload.get("keys"),
    }
    digest = hashlib.sha256(
        json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    )
    for name in _CACHE_ARTIFACT_TENSORS:
        value = payload.get(name)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"cache artifact field {name} must be a tensor")
        cpu = value.detach().to(device="cpu").contiguous()
        digest.update(name.encode("ascii"))
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(json.dumps(list(cpu.shape), separators=(",", ":")).encode())
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class GroupGroupingCacheLookup:
    """Exact cache hit materializing grouping metadata but no selection plan."""

    grouping: GroupingOutput
    hit_count: int
    history_lengths: torch.Tensor
    materialization_bytes: int
    transferred_bytes: int
    cache_rows: tuple[int, ...]
    storage_device: str


@dataclass(frozen=True)
class GroupSelectionCacheLookup:
    """Fully materialized cache hit ready for ``pack_group_streams``."""

    grouping: GroupingOutput
    selection: GroupSelectionOutput
    hit_count: int
    history_lengths: torch.Tensor
    materialization_bytes: int
    transferred_bytes: int
    cache_rows: tuple[int, ...]
    storage_device: str


class GroupSelectionLookupCache:
    """Fixed-shape lookup table for exact Subplan 5C serving semantics.

    Tensor payload can live in regular CPU memory, pinned CPU memory, or on a
    GPU.  The Python key index remains host resident because serving request
    identifiers and candidate IDs are small control-plane metadata.
    """

    def __init__(
        self,
        *,
        version: DeltaRecCacheVersion,
        retention_ratio: float,
        recent_floor: int,
        canonical_candidate_ids: torch.Tensor,
        canonical_group_ids: torch.Tensor,
        anchor_ids: torch.Tensor,
        selected_positions: torch.Tensor,
        history_lengths: torch.Tensor,
        keys: Sequence[tuple[int, tuple[str, int | str], str, str]],
        canonical_candidate_rows: Sequence[tuple[int, ...]],
        anchor_position_rows: Sequence[tuple[int, ...]],
        history_length_rows: Sequence[int],
    ) -> None:
        retention_ratio = validate_retention_ratio(retention_ratio)
        if isinstance(recent_floor, bool) or not isinstance(recent_floor, int):
            raise TypeError("recent_floor must be an integer")
        if recent_floor < 0:
            raise ValueError("recent_floor must be nonnegative")
        entries, candidates = canonical_candidate_ids.shape
        if entries < 1 or candidates < 1:
            raise ValueError("cache tensors must contain at least one entry")
        if canonical_candidate_ids.dtype != torch.int64:
            raise ValueError("canonical candidate storage must be int64")
        if canonical_group_ids.shape != (entries, candidates) or (
            canonical_group_ids.dtype not in (torch.uint8, torch.int32)
        ):
            raise ValueError("canonical group storage has an invalid layout")
        if anchor_ids.ndim != 2 or anchor_ids.shape[0] != entries or (
            anchor_ids.dtype != torch.int64
        ):
            raise ValueError("anchor storage must be int64 with shape [N,G]")
        groups = int(anchor_ids.shape[1])
        if groups < 1:
            raise ValueError("cache must describe at least one semantic group")
        if selected_positions.ndim != 3 or selected_positions.shape[:2] != (
            entries,
            groups,
        ) or selected_positions.dtype != torch.int32:
            raise ValueError("selected position storage must be int32 [N,G,S]")
        if selected_positions.shape[2] < 1:
            raise ValueError("cache selected-position width must be positive")
        if history_lengths.shape != (entries,) or history_lengths.dtype != torch.int32:
            raise ValueError("history length storage must be int32 [N]")
        storage_device = canonical_candidate_ids.device
        tensors = (
            canonical_group_ids,
            anchor_ids,
            selected_positions,
            history_lengths,
        )
        if any(tensor.device != storage_device for tensor in tensors):
            raise ValueError("all cache payload tensors must share a device")
        if (
            len(keys) != entries
            or len(canonical_candidate_rows) != entries
            or len(anchor_position_rows) != entries
            or len(history_length_rows) != entries
        ):
            raise ValueError("cache metadata must align with tensor entries")
        if any(len(row) != groups for row in anchor_position_rows):
            raise ValueError("anchor-position metadata must have G entries per row")
        key_to_row: dict[tuple[int, tuple[str, int | str], str, str], int] = {}
        for row, key in enumerate(keys):
            if key in key_to_row:
                raise ValueError(
                    "duplicate exact cache key at rows "
                    f"{key_to_row[key]}, {row}"
                )
            key_to_row[key] = row

        self.version = version
        self.retention_ratio = retention_ratio
        self.recent_floor = int(recent_floor)
        self.canonical_candidate_ids = canonical_candidate_ids
        self.canonical_group_ids = canonical_group_ids
        self.anchor_ids = anchor_ids
        self.selected_positions = selected_positions
        self.history_lengths = history_lengths
        self._keys = tuple(keys)
        self._key_to_row = key_to_row
        self._canonical_candidate_rows = tuple(canonical_candidate_rows)
        self._anchor_position_rows = tuple(anchor_position_rows)
        self._history_length_rows = tuple(int(value) for value in history_length_rows)

    @classmethod
    def from_outputs(
        cls,
        *,
        user_ids: torch.Tensor | Sequence[int],
        history_versions: torch.Tensor | Sequence[int | str],
        candidate_ids: torch.Tensor,
        history_lengths: torch.Tensor,
        grouping: GroupingOutput,
        selection: GroupSelectionOutput,
        version: DeltaRecCacheVersion,
        retention_ratio: float,
        recent_floor: int,
        storage_device: torch.device | str = "cpu",
        pin_memory: bool = False,
        validate_runtime: bool = True,
    ) -> "GroupSelectionLookupCache":
        """Materialize an offline table from audited cold-path outputs."""

        _require_integer_matrix(candidate_ids, "candidate_ids")
        batch, candidates = candidate_ids.shape
        if history_lengths.shape != (batch,) or history_lengths.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("history_lengths must be an integer tensor with shape [B]")
        if candidate_ids.device != history_lengths.device:
            raise ValueError("candidate IDs and history lengths must share a device")
        if grouping.candidate_to_group.device != candidate_ids.device or (
            selection.indices.device != candidate_ids.device
        ):
            raise ValueError(
                "cold grouping, selection, and request tensors must share a device"
            )
        if grouping.candidate_to_group.shape != (batch, candidates):
            raise ValueError("grouping shape does not match cached candidates")
        groups = grouping.group_count
        if selection.counts.shape != (batch, groups):
            raise ValueError("selection shape does not match cached groups")
        if selection.indices.ndim != 3 or selection.indices.shape[:2] != (
            batch,
            groups,
        ):
            raise ValueError("selection indices must have shape [B,G,S]")
        if selection.indices.shape[2] < 1:
            raise ValueError("cache construction requires nonempty selections")
        if not isinstance(version, DeltaRecCacheVersion):
            raise TypeError("version must be a DeltaRecCacheVersion")
        if not isinstance(pin_memory, bool) or not isinstance(validate_runtime, bool):
            raise TypeError("pin_memory and validate_runtime must be boolean")

        history_cpu = history_lengths.detach().to(device="cpu", dtype=torch.int64)
        if bool((history_cpu < 1).any()):
            raise ValueError("cache construction requires nonempty histories")
        if bool((history_cpu > torch.iinfo(torch.int32).max).any()):
            raise ValueError("history lengths exceed compact int32 cache storage")
        users = _integer_values(user_ids, name="user_ids", expected=batch)
        histories = _history_version_values(history_versions, expected=batch)
        _, canonical_order, canonical_rows, fingerprints = _canonicalize_candidates(
            candidate_ids
        )

        # Move the cold outputs to CPU once.  This is deliberately offline work
        # and keeps cache construction independent of the requested residency.
        assignment_cpu = grouping.candidate_to_group.detach().to(
            device="cpu", dtype=torch.int64
        )
        canonical_groups = assignment_cpu.gather(1, canonical_order)
        if bool((canonical_groups < 0).any()) or bool(
            (canonical_groups >= groups).any()
        ):
            raise ValueError("cold grouping contains an invalid group ID")
        group_dtype = torch.uint8 if groups <= 256 else torch.int32
        canonical_groups = canonical_groups.to(group_dtype)
        anchors_cpu = grouping.anchor_ids.detach().to(
            device="cpu", dtype=torch.int64
        ).contiguous()
        anchor_position_rows: list[tuple[int, ...]] = []
        for canonical, anchors in zip(canonical_rows, anchors_cpu.tolist()):
            position_by_id = {
                item_id: position for position, item_id in enumerate(canonical)
            }
            try:
                anchor_position_rows.append(
                    tuple(
                        -1 if int(anchor) == -1 else position_by_id[int(anchor)]
                        for anchor in anchors
                    )
                )
            except KeyError as error:
                raise ValueError(
                    "cold grouping anchor does not belong to its candidate set"
                ) from error
        positions_cpu = selection.indices.detach().to(
            device="cpu", dtype=torch.int64
        ).contiguous()
        if bool((positions_cpu > torch.iinfo(torch.int32).max).any()):
            raise ValueError("selected positions exceed compact int32 cache storage")

        retention_ratio = validate_retention_ratio(retention_ratio)
        budgets_cpu = exact_budget(
            history_cpu,
            retention_ratio=retention_ratio,
            recent_floor=recent_floor,
            validate_runtime=True,
        )
        expected_counts = budgets_cpu[:, None].expand(batch, groups)
        counts_cpu = selection.counts.detach().to(device="cpu", dtype=torch.int64)
        if not torch.equal(counts_cpu, expected_counts):
            raise ValueError("cold selection counts do not match the concrete ratio budget")
        slots = (
            torch.arange(positions_cpu.shape[2], dtype=torch.int64)[None, None, :]
            < expected_counts[..., None]
        )
        valid_positions = positions_cpu.masked_select(slots)
        if bool((valid_positions < 0).any()):
            raise ValueError("cold selection contains a negative valid position")
        row_limits = history_cpu[:, None].expand(batch, groups).reshape(-1)
        packed_limits = torch.repeat_interleave(row_limits, expected_counts.reshape(-1))
        if bool((valid_positions >= packed_limits).any()):
            raise ValueError("cold selection addresses beyond a valid history")
        if bool((positions_cpu.masked_select(~slots) != -1).any()):
            raise ValueError("cold selection padding must use -1")
        if positions_cpu.shape[2] > 1:
            adjacent_valid = slots[..., 1:]
            if bool(
                (
                    adjacent_valid
                    & (positions_cpu[..., 1:] <= positions_cpu[..., :-1])
                ).any()
            ):
                raise ValueError("cached selected positions must be chronological")
        positions_cpu = positions_cpu.to(torch.int32)

        if validate_runtime:
            grouping.validate(candidate_ids)
            selection.validate(int(history_cpu.max()))

        keys = [
            (user, history, fingerprint, version.fingerprint)
            for user, history, fingerprint in zip(users, histories, fingerprints)
        ]
        target = torch.device(storage_device)
        if pin_memory and target.type != "cpu":
            raise ValueError("pin_memory is valid only for CPU-resident tables")
        if pin_memory and not torch.cuda.is_available():
            raise RuntimeError(
                "pinned cache storage requires an available CUDA runtime"
            )

        def place(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
            output = value.to(dtype=dtype).contiguous()
            if target.type == "cpu":
                output = output.clone()
                return output.pin_memory() if pin_memory else output
            return output.to(device=target)

        return cls(
            version=version,
            retention_ratio=retention_ratio,
            recent_floor=recent_floor,
            canonical_candidate_ids=place(
                torch.tensor(canonical_rows, dtype=torch.int64), torch.int64
            ),
            canonical_group_ids=place(canonical_groups, group_dtype),
            anchor_ids=place(anchors_cpu, torch.int64),
            selected_positions=place(positions_cpu, torch.int32),
            history_lengths=place(history_cpu, torch.int32),
            keys=keys,
            canonical_candidate_rows=canonical_rows,
            anchor_position_rows=anchor_position_rows,
            history_length_rows=history_cpu.tolist(),
        )

    @property
    def entry_count(self) -> int:
        return int(self.canonical_candidate_ids.shape[0])

    @property
    def candidate_count(self) -> int:
        return int(self.canonical_candidate_ids.shape[1])

    @property
    def group_count(self) -> int:
        return int(self.anchor_ids.shape[1])

    @property
    def storage_device(self) -> torch.device:
        return self.canonical_candidate_ids.device

    @property
    def is_pinned(self) -> bool:
        return bool(
            self.storage_device.type == "cpu"
            and self.canonical_candidate_ids.is_pinned()
        )

    @property
    def payload_bytes(self) -> int:
        """Exact tensor payload size (Python key metadata is excluded)."""

        return sum(
            _tensor_bytes(value)
            for value in (
                self.canonical_candidate_ids,
                self.canonical_group_ids,
                self.anchor_ids,
                self.selected_positions,
                self.history_lengths,
            )
        )

    @property
    def bytes_per_entry(self) -> int:
        return self.payload_bytes // self.entry_count

    def artifact_state(self) -> dict[str, Any]:
        """Return a checksummed, CPU-portable training/serving artifact.

        Tensor residency is deliberately normalized to CPU.  Serving chooses
        pinned-host or CUDA placement only while loading, so one training
        artifact can back either deployment policy.
        """

        payload: dict[str, Any] = {
            "schema_version": _CACHE_ARTIFACT_SCHEMA_VERSION,
            "cache_contract": _CACHE_CONTRACT,
            "retention_ratio": self.retention_ratio,
            "recent_floor": self.recent_floor,
            "version": {
                "backbone": self.version.backbone,
                "selector": self.version.selector,
                "schema": self.version.schema,
                "policy": self.version.policy,
            },
            "keys": [
                [
                    int(user),
                    [str(history[0]), history[1]],
                    str(candidate_fingerprint),
                    str(version_fingerprint),
                ]
                for user, history, candidate_fingerprint, version_fingerprint in self._keys
            ],
        }
        for name in _CACHE_ARTIFACT_TENSORS:
            payload[name] = (
                getattr(self, name)
                .detach()
                .to(device="cpu")
                .contiguous()
                .clone()
            )
        payload["artifact_sha256"] = _cache_artifact_sha256(payload)
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
    ) -> "GroupSelectionLookupCache":
        """Validate and place a persisted table without rerunning a selector."""

        if not isinstance(payload, Mapping):
            raise ValueError("group cache artifact must be a mapping")
        if payload.get("schema_version") != _CACHE_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported group cache artifact schema")
        if payload.get("cache_contract") != _CACHE_CONTRACT:
            raise ValueError("group cache artifact contract mismatch")
        recorded_digest = payload.get("artifact_sha256")
        if not isinstance(recorded_digest, str) or not recorded_digest:
            raise ValueError("group cache artifact is missing its checksum")
        if _cache_artifact_sha256(payload) != recorded_digest:
            raise ValueError("group cache artifact checksum mismatch")

        raw_version = payload.get("version")
        if not isinstance(raw_version, Mapping) or set(raw_version) != {
            "backbone",
            "selector",
            "schema",
            "policy",
        }:
            raise ValueError("group cache artifact has an invalid version record")
        version = DeltaRecCacheVersion(
            backbone=str(raw_version["backbone"]),
            selector=str(raw_version["selector"]),
            schema=str(raw_version["schema"]),
            policy=str(raw_version["policy"]),
        )
        if expected_version is not None and version != expected_version:
            raise CacheMissError("persisted group cache version mismatch")
        retention_ratio = validate_retention_ratio(payload.get("retention_ratio"))
        recent_floor = payload.get("recent_floor")
        if isinstance(recent_floor, bool) or not isinstance(recent_floor, int):
            raise ValueError("persisted recent_floor must be an integer")
        if recent_floor < 0:
            raise ValueError("persisted recent_floor must be nonnegative")

        tensors: dict[str, torch.Tensor] = {}
        for name in _CACHE_ARTIFACT_TENSORS:
            value = payload.get(name)
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"group cache artifact field {name} is not a tensor")
            tensors[name] = value.detach().to(device="cpu").contiguous()
        candidate_ids = tensors["canonical_candidate_ids"]
        candidate_groups = tensors["canonical_group_ids"]
        anchors = tensors["anchor_ids"]
        positions = tensors["selected_positions"]
        history_lengths = tensors["history_lengths"]
        if candidate_ids.ndim != 2 or candidate_ids.dtype != torch.int64:
            raise ValueError("persisted canonical candidates must be int64 [N,K]")
        entries, candidates = candidate_ids.shape
        if entries < 1 or candidates < 1:
            raise ValueError("persisted group cache must be nonempty")
        if candidates > 1 and bool(
            (candidate_ids[:, 1:] <= candidate_ids[:, :-1]).any()
        ):
            raise ValueError("persisted canonical candidates must be sorted and unique")
        if anchors.ndim != 2 or anchors.shape[0] != entries:
            raise ValueError("persisted anchors must have shape [N,G]")
        groups = int(anchors.shape[1])
        if candidate_groups.shape != (entries, candidates):
            raise ValueError("persisted candidate groups do not align with candidates")
        group_ids_i64 = candidate_groups.to(torch.int64)
        if bool((group_ids_i64 < 0).any()) or bool((group_ids_i64 >= groups).any()):
            raise ValueError("persisted candidate groups contain an invalid group ID")
        row_bases = torch.arange(entries, dtype=torch.int64)[:, None] * groups
        group_sizes = torch.bincount(
            (group_ids_i64 + row_bases).reshape(-1),
            minlength=entries * groups,
        ).reshape(entries, groups)
        if bool((group_sizes < 0).any()):
            raise ValueError("persisted candidate group sizes must be nonnegative")
        occupied = group_sizes > 0
        if bool((anchors.masked_select(~occupied) != -1).any()):
            raise ValueError("persisted empty groups must use -1 anchors")
        if bool((anchors.masked_select(occupied) < 0).any()):
            raise ValueError("persisted occupied groups require valid anchors")
        if history_lengths.shape != (entries,) or history_lengths.dtype != torch.int32:
            raise ValueError("persisted history lengths must be int32 [N]")
        if bool((history_lengths < 1).any()):
            raise ValueError("persisted histories must be nonempty")
        if positions.ndim != 3 or positions.shape[:2] != (entries, groups) or (
            positions.dtype != torch.int32
        ):
            raise ValueError("persisted selected positions must be int32 [N,G,S]")

        raw_keys = payload.get("keys")
        if not isinstance(raw_keys, Sequence) or isinstance(raw_keys, (str, bytes)):
            raise ValueError("group cache artifact keys must be a sequence")
        if len(raw_keys) != entries:
            raise ValueError("group cache artifact keys do not align with entries")
        canonical_rows = [
            tuple(int(item_id) for item_id in row)
            for row in candidate_ids.tolist()
        ]
        keys: list[tuple[int, tuple[str, int | str], str, str]] = []
        for row, (raw_key, canonical) in enumerate(zip(raw_keys, canonical_rows)):
            if not isinstance(raw_key, Sequence) or len(raw_key) != 4:
                raise ValueError(f"invalid persisted cache key at row {row}")
            raw_history = raw_key[1]
            if not isinstance(raw_history, Sequence) or len(raw_history) != 2:
                raise ValueError(f"invalid persisted history version at row {row}")
            history_kind = raw_history[0]
            history_value = raw_history[1]
            if history_kind == "int" and isinstance(history_value, int) and not isinstance(
                history_value, bool
            ):
                typed_history: tuple[str, int | str] = ("int", int(history_value))
            elif history_kind == "str" and isinstance(history_value, str) and history_value:
                typed_history = ("str", history_value)
            else:
                raise ValueError(f"invalid persisted history version at row {row}")
            candidate_fingerprint = str(raw_key[2])
            version_fingerprint = str(raw_key[3])
            if candidate_fingerprint != _candidate_fingerprint(canonical):
                raise ValueError(f"persisted candidate fingerprint mismatch at row {row}")
            if version_fingerprint != version.fingerprint:
                raise ValueError(f"persisted version fingerprint mismatch at row {row}")
            user = raw_key[0]
            if isinstance(user, bool) or not isinstance(user, int):
                raise ValueError(f"invalid persisted user ID at row {row}")
            keys.append(
                (
                    int(user),
                    typed_history,
                    candidate_fingerprint,
                    version_fingerprint,
                )
            )

        anchor_position_rows: list[tuple[int, ...]] = []
        for row, (canonical, row_anchors) in enumerate(
            zip(canonical_rows, anchors.tolist())
        ):
            position_by_id = {
                item_id: position for position, item_id in enumerate(canonical)
            }
            try:
                anchor_positions = tuple(
                    -1 if int(anchor) == -1 else position_by_id[int(anchor)]
                    for anchor in row_anchors
                )
            except KeyError as error:
                raise ValueError(
                    f"persisted anchor is outside candidate row {row}"
                ) from error
            for group, position in enumerate(anchor_positions):
                if position < 0:
                    if int(group_sizes[row, group]) != 0:
                        raise ValueError(
                            f"persisted occupied group lacks anchor at row {row}, "
                            f"group {group}"
                        )
                    continue
                if int(group_ids_i64[row, position]) != group:
                    raise ValueError(
                        f"persisted anchor/group mismatch at row {row}, group {group}"
                    )
            anchor_position_rows.append(anchor_positions)

        budgets = exact_budget(
            history_lengths.to(torch.int64),
            retention_ratio=retention_ratio,
            recent_floor=recent_floor,
        )
        if positions.shape[2] < int(budgets.max()):
            raise ValueError("persisted selected-position width is too small")
        slots = (
            torch.arange(positions.shape[2], dtype=torch.int64)[None, None, :]
            < budgets[:, None, None]
        ).expand_as(positions)
        valid_positions = positions.to(torch.int64).masked_select(slots)
        limits = torch.repeat_interleave(
            history_lengths.to(torch.int64)[:, None]
            .expand(entries, groups)
            .reshape(-1),
            budgets[:, None].expand(entries, groups).reshape(-1),
        )
        if bool((valid_positions < 0).any()) or bool((valid_positions >= limits).any()):
            raise ValueError("persisted selected position is outside its history")
        if bool((positions.masked_select(~slots) != -1).any()):
            raise ValueError("persisted selected-position padding must be -1")
        if positions.shape[2] > 1 and bool(
            (
                slots[..., 1:]
                & (positions[..., 1:] <= positions[..., :-1])
            ).any()
        ):
            raise ValueError("persisted selected positions must be chronological")

        target = torch.device(storage_device)
        if pin_memory and target.type != "cpu":
            raise ValueError("pin_memory is valid only for CPU-resident tables")
        if pin_memory and not torch.cuda.is_available():
            raise RuntimeError("pinned cache storage requires an available CUDA runtime")

        def place(value: torch.Tensor) -> torch.Tensor:
            output = value.contiguous().clone()
            if target.type == "cpu":
                return output.pin_memory() if pin_memory else output
            return output.to(device=target)

        return cls(
            version=version,
            retention_ratio=retention_ratio,
            recent_floor=recent_floor,
            canonical_candidate_ids=place(candidate_ids),
            canonical_group_ids=place(candidate_groups),
            anchor_ids=place(anchors),
            selected_positions=place(positions),
            history_lengths=place(history_lengths),
            keys=keys,
            canonical_candidate_rows=canonical_rows,
            anchor_position_rows=anchor_position_rows,
            history_length_rows=history_lengths.tolist(),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        storage_device: torch.device | str = "cpu",
        pin_memory: bool = False,
        expected_version: DeltaRecCacheVersion | None = None,
    ) -> "GroupSelectionLookupCache":
        """Load a training-produced artifact directly into serving residency."""

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

    def lookup_grouping(
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
    ) -> GroupGroupingCacheLookup:
        """Resolve an exact hit while skipping selected-position materialization.

        Recurrent-state serving already contains the prefix result and only
        needs the canonical candidate grouping.  Keeping this as a separate
        entry point prevents state-cache latency and byte accounting from
        silently including the positions-only payload used by cold-prefix
        packing.
        """

        _require_integer_matrix(candidate_ids, "candidate_ids")
        batch, candidates = candidate_ids.shape
        if candidates != self.candidate_count:
            raise CacheMissError(
                "candidate count "
                f"{candidates} does not match cached K={self.candidate_count}"
            )
        if history_lengths.shape != (batch,) or history_lengths.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("history_lengths must be an integer tensor with shape [B]")
        if not isinstance(version, DeltaRecCacheVersion):
            raise TypeError("version must be a DeltaRecCacheVersion")
        if version != self.version:
            raise CacheMissError("group selection cache version mismatch")
        if not isinstance(validate_runtime, bool) or not isinstance(non_blocking, bool):
            raise TypeError("validate_runtime and non_blocking must be boolean")

        users = _integer_values(user_ids, name="user_ids", expected=batch)
        histories = _history_version_values(history_versions, expected=batch)
        ids_cpu, canonical_order_cpu, canonical_rows, fingerprints = (
            _canonicalize_candidates(candidate_ids)
        )
        current_lengths_cpu: torch.Tensor | None = None
        if validate_runtime or history_lengths.device.type == "cpu":
            current_lengths_cpu = history_lengths.detach().to(
                device="cpu", dtype=torch.int64
            ).contiguous()
            if bool((current_lengths_cpu < 1).any()):
                raise CacheMissError("serving cache requires nonempty histories")

        rows: list[int] = []
        version_fingerprint = version.fingerprint
        for batch_row, (user, history, fingerprint, canonical) in enumerate(
            zip(users, histories, fingerprints, canonical_rows)
        ):
            key = (user, history, fingerprint, version_fingerprint)
            cache_row = self._key_to_row.get(key)
            if cache_row is None:
                raise CacheMissError(f"cache miss for serving batch row {batch_row}")
            if canonical != self._canonical_candidate_rows[cache_row]:
                raise CacheMissError(
                    f"candidate fingerprint collision at serving batch row {batch_row}"
                )
            if current_lengths_cpu is not None and (
                int(current_lengths_cpu[batch_row])
                != self._history_length_rows[cache_row]
            ):
                raise CacheMissError(
                    f"history length changed at serving batch row {batch_row}"
                )
            rows.append(cache_row)

        if current_lengths_cpu is None:
            current_lengths_cpu = torch.tensor(
                [self._history_length_rows[row] for row in rows],
                dtype=torch.int64,
                device="cpu",
            )

        target = torch.device(target_device)
        row_index = torch.tensor(rows, dtype=torch.int64, device=self.storage_device)
        moving_devices = self.storage_device != target

        def materialize(value: torch.Tensor) -> torch.Tensor:
            if (
                self.storage_device.type == "cpu"
                and target.type == "cuda"
                and self.is_pinned
            ):
                selected = torch.empty(
                    (batch, *value.shape[1:]),
                    dtype=value.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                torch.index_select(value, 0, row_index, out=selected)
            else:
                selected = value.index_select(0, row_index)
            if not moving_devices:
                return selected
            async_copy = bool(
                non_blocking
                and self.storage_device.type == "cpu"
                and target.type == "cuda"
                and selected.is_pinned()
            )
            return selected.to(device=target, non_blocking=async_copy)

        canonical_groups = materialize(self.canonical_group_ids).to(torch.int64)
        anchor_ids = materialize(self.anchor_ids)
        canonical_order = canonical_order_cpu.to(device=target, dtype=torch.int64)
        candidate_to_group = torch.empty(
            (batch, candidates), dtype=torch.int64, device=target
        )
        candidate_to_group.scatter_(1, canonical_order, canonical_groups)
        groups = self.group_count
        group_sizes = torch.zeros((batch, groups), dtype=torch.int64, device=target)
        group_sizes.scatter_add_(
            1,
            candidate_to_group,
            torch.ones_like(candidate_to_group),
        )
        canonical_group_order = torch.argsort(
            canonical_groups, dim=1, stable=True
        )
        packed_local = canonical_order.gather(1, canonical_group_order)
        packed_candidate_indices = (
            packed_local
            + torch.arange(batch, dtype=torch.int64, device=target)[:, None]
            * candidates
        ).reshape(-1)
        flat_sizes = group_sizes.reshape(-1)
        group_offsets = torch.cat(
            (flat_sizes.new_zeros(1), torch.cumsum(flat_sizes, dim=0)), dim=0
        )
        anchor_canonical_positions = torch.tensor(
            [self._anchor_position_rows[row] for row in rows],
            dtype=torch.int64,
            device=target,
        )
        anchor_indices = canonical_order.gather(
            1, anchor_canonical_positions.clamp_min(0)
        ).masked_fill(anchor_canonical_positions < 0, -1)
        grouping = GroupingOutput(
            candidate_to_group=candidate_to_group,
            group_sizes=group_sizes,
            packed_candidate_indices=packed_candidate_indices,
            group_offsets=group_offsets,
            anchor_indices=anchor_indices,
            anchor_ids=anchor_ids,
        )
        history_lengths_target = history_lengths.detach().to(
            device=target, dtype=torch.int64
        )
        if validate_runtime:
            request_candidate_ids = ids_cpu.to(device=target, dtype=torch.int64)
            grouping.validate(request_candidate_ids)

        materialization_bytes = batch * (
            self.candidate_count * self.canonical_group_ids.element_size()
            + self.group_count * self.anchor_ids.element_size()
        )
        return GroupGroupingCacheLookup(
            grouping=grouping,
            hit_count=batch,
            history_lengths=history_lengths_target,
            materialization_bytes=materialization_bytes,
            transferred_bytes=materialization_bytes if moving_devices else 0,
            cache_rows=tuple(rows),
            storage_device=str(self.storage_device),
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
    ) -> GroupSelectionCacheLookup:
        """Return a complete batch hit or raise :class:`CacheMissError`.

        Partial hits are intentionally never mixed with stale or recomputed
        rows.  A caller that wants a cold fallback can catch ``CacheMissError``
        and route the entire request batch through the registered cold path.
        """

        _require_integer_matrix(candidate_ids, "candidate_ids")
        batch, candidates = candidate_ids.shape
        if candidates != self.candidate_count:
            raise CacheMissError(
                "candidate count "
                f"{candidates} does not match cached K={self.candidate_count}"
            )
        if history_lengths.shape != (batch,) or history_lengths.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("history_lengths must be an integer tensor with shape [B]")
        if not isinstance(version, DeltaRecCacheVersion):
            raise TypeError("version must be a DeltaRecCacheVersion")
        if version != self.version:
            raise CacheMissError("group selection cache version mismatch")
        if not isinstance(validate_runtime, bool) or not isinstance(non_blocking, bool):
            raise TypeError("validate_runtime and non_blocking must be boolean")

        users = _integer_values(user_ids, name="user_ids", expected=batch)
        histories = _history_version_values(history_versions, expected=batch)
        ids_cpu, canonical_order_cpu, canonical_rows, fingerprints = (
            _canonicalize_candidates(candidate_ids)
        )
        # Production callers bind history length to ``history_version``.  The
        # registered fast path trusts that control-plane contract and avoids a
        # GPU-to-host synchronization solely to re-read length metadata.  Full
        # runtime validation (and every CPU call) still checks the live tensor
        # against the table entry below.
        current_lengths_cpu: torch.Tensor | None = None
        if validate_runtime or history_lengths.device.type == "cpu":
            current_lengths_cpu = history_lengths.detach().to(
                device="cpu", dtype=torch.int64
            ).contiguous()
            if bool((current_lengths_cpu < 1).any()):
                raise CacheMissError("serving cache requires nonempty histories")

        rows: list[int] = []
        version_fingerprint = version.fingerprint
        for batch_row, (user, history, fingerprint, canonical) in enumerate(
            zip(users, histories, fingerprints, canonical_rows)
        ):
            key = (user, history, fingerprint, version_fingerprint)
            cache_row = self._key_to_row.get(key)
            if cache_row is None:
                raise CacheMissError(f"cache miss for serving batch row {batch_row}")
            # SHA-256 selects the row; exact canonical comparison makes even a
            # hypothetical collision fail closed.
            if canonical != self._canonical_candidate_rows[cache_row]:
                raise CacheMissError(
                    f"candidate fingerprint collision at serving batch row {batch_row}"
                )
            if current_lengths_cpu is not None and (
                int(current_lengths_cpu[batch_row])
                != self._history_length_rows[cache_row]
            ):
                raise CacheMissError(
                    f"history length changed at serving batch row {batch_row}"
                )
            rows.append(cache_row)

        if current_lengths_cpu is None:
            current_lengths_cpu = torch.tensor(
                [self._history_length_rows[row] for row in rows],
                dtype=torch.int64,
                device="cpu",
            )

        target = torch.device(target_device)
        row_index = torch.tensor(rows, dtype=torch.int64, device=self.storage_device)
        moving_devices = self.storage_device != target

        def materialize(value: torch.Tensor) -> torch.Tensor:
            # ``Tensor.index_select`` does not promise to preserve pinned host
            # allocation.  Gather into an explicitly pinned output so the
            # subsequent H2D copy can actually be asynchronous.
            if (
                self.storage_device.type == "cpu"
                and target.type == "cuda"
                and self.is_pinned
            ):
                selected = torch.empty(
                    (batch, *value.shape[1:]),
                    dtype=value.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                torch.index_select(value, 0, row_index, out=selected)
            else:
                selected = value.index_select(0, row_index)
            if not moving_devices:
                return selected
            async_copy = bool(
                non_blocking
                and self.storage_device.type == "cpu"
                and target.type == "cuda"
                and selected.is_pinned()
            )
            return selected.to(device=target, non_blocking=async_copy)

        canonical_groups = materialize(self.canonical_group_ids).to(torch.int64)
        anchor_ids = materialize(self.anchor_ids)
        selected_positions = materialize(self.selected_positions)
        cached_lengths = (
            materialize(self.history_lengths).to(torch.int64)
            if validate_runtime
            else None
        )
        canonical_order = canonical_order_cpu.to(device=target, dtype=torch.int64)
        request_candidate_ids = (
            ids_cpu.to(device=target, dtype=torch.int64)
            if validate_runtime
            else None
        )
        history_lengths_target = history_lengths.detach().to(
            device=target, dtype=torch.int64
        )

        groups = self.group_count
        candidate_to_group = torch.empty(
            (batch, candidates), dtype=torch.int64, device=target
        )
        candidate_to_group.scatter_(1, canonical_order, canonical_groups)
        group_sizes = torch.zeros(
            (batch, groups), dtype=torch.int64, device=target
        )
        group_sizes.scatter_add_(
            1,
            candidate_to_group,
            torch.ones_like(candidate_to_group),
        )
        canonical_group_order = torch.argsort(
            canonical_groups, dim=1, stable=True
        )
        packed_local = canonical_order.gather(1, canonical_group_order)
        packed_candidate_indices = (
            packed_local
            + torch.arange(batch, dtype=torch.int64, device=target)[:, None]
            * candidates
        ).reshape(-1)
        flat_sizes = group_sizes.reshape(-1)
        group_offsets = torch.cat(
            (flat_sizes.new_zeros(1), torch.cumsum(flat_sizes, dim=0)), dim=0
        )

        anchor_canonical_positions = torch.tensor(
            [self._anchor_position_rows[row] for row in rows],
            dtype=torch.int64,
            device=target,
        )
        anchor_indices = canonical_order.gather(
            1, anchor_canonical_positions.clamp_min(0)
        ).masked_fill(anchor_canonical_positions < 0, -1)
        grouping = GroupingOutput(
            candidate_to_group=candidate_to_group,
            group_sizes=group_sizes,
            packed_candidate_indices=packed_candidate_indices,
            group_offsets=group_offsets,
            anchor_indices=anchor_indices,
            anchor_ids=anchor_ids,
        )

        budgets_cpu = exact_budget(
            current_lengths_cpu,
            retention_ratio=self.retention_ratio,
            recent_floor=self.recent_floor,
            validate_runtime=validate_runtime,
        )
        max_budget = int(budgets_cpu.max())
        if max_budget > selected_positions.shape[2]:
            raise CacheMissError("cached selected-position width is too small")
        budgets = budgets_cpu.to(device=target, dtype=torch.int64)
        counts = budgets[:, None].expand(batch, groups).contiguous()
        indices = selected_positions[:, :, :max_budget].to(torch.int64)
        flat_counts = counts.reshape(-1)
        selection_offsets = torch.cat(
            (flat_counts.new_zeros(1), torch.cumsum(flat_counts, dim=0)), dim=0
        )
        selected_token_count = int(budgets_cpu.sum()) * groups
        logical_rows = torch.repeat_interleave(
            torch.arange(batch * groups, device=target, dtype=torch.int64),
            flat_counts,
            output_size=selected_token_count,
        )
        within_rows = torch.arange(
            selected_token_count, device=target, dtype=torch.int64
        ) - selection_offsets[:-1].index_select(
            0, logical_rows
        )
        source_positions = indices.reshape(-1).index_select(
            0, logical_rows * max_budget + within_rows
        )
        source_users = torch.div(logical_rows, groups, rounding_mode="floor")
        dense_history_width = int(current_lengths_cpu.max())
        packed_source_indices = source_users * dense_history_width + source_positions
        selection = GroupSelectionOutput(
            indices=indices,
            counts=counts,
            budgets=budgets,
            packed_source_indices=packed_source_indices,
            source_positions=source_positions,
            group_offsets=selection_offsets,
            dense_mask=None,
        )

        if validate_runtime:
            assert cached_lengths is not None and request_candidate_ids is not None
            if not torch.equal(cached_lengths, history_lengths_target):
                raise CacheMissError(
                    "materialized cache history lengths are corrupted"
                )
            grouping.validate(request_candidate_ids)
            selection.validate(dense_history_width)

        materialization_bytes = batch * (
            self.candidate_count * self.canonical_group_ids.element_size()
            + self.group_count * self.anchor_ids.element_size()
            + self.group_count
            * self.selected_positions.shape[2]
            * self.selected_positions.element_size()
            + (
                self.history_lengths.element_size()
                if validate_runtime
                else 0
            )
        )
        return GroupSelectionCacheLookup(
            grouping=grouping,
            selection=selection,
            hit_count=batch,
            history_lengths=history_lengths_target,
            materialization_bytes=materialization_bytes,
            transferred_bytes=materialization_bytes if moving_devices else 0,
            cache_rows=tuple(rows),
            storage_device=str(self.storage_device),
        )


__all__ = [
    "CacheMissError",
    "GroupGroupingCacheLookup",
    "GroupSelectionCacheLookup",
    "GroupSelectionLookupCache",
    "build_group_selection_cache_version",
]

