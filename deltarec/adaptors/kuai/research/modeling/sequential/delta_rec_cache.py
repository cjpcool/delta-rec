# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""Versioned recurrent-state cache and reference execution for DeltaRec."""

from __future__ import annotations

import abc
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from deltarec.adaptors.kuai.research.modeling.sequential.selective_gdr import (
    SelectedSequence,
)


@dataclass(frozen=True)
class DeltaRecCacheVersion:
    backbone: str
    selector: str
    schema: str
    policy: str

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not value:
                raise ValueError(f"cache version field {name} must be nonempty")

    @staticmethod
    def _digest(payload: Mapping[str, str]) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def fingerprint(self) -> str:
        return self._digest(self.__dict__)

    @property
    def selection_fingerprint(self) -> str:
        return self._digest(
            {
                "selector": self.selector,
                "schema": self.schema,
                "policy": self.policy,
            }
        )

    def rebuild_source(self, previous: "DeltaRecCacheVersion") -> str:
        """Return the minimum correct source for rebuilding this version."""

        return (
            "ledger"
            if self.selection_fingerprint == previous.selection_fingerprint
            else "full_history"
        )


@dataclass(frozen=True)
class RetainedEvent:
    item_id: int
    source_position: int
    fields: Mapping[str, int | float] = field(default_factory=dict)
    payloads: Mapping[str, int | float] = field(default_factory=dict)


@dataclass(frozen=True)
class RetainedEventRecord:
    selection_fingerprint: str
    events: tuple[RetainedEvent, ...]


class RetainedEventLedger:
    """Small in-memory reference ledger used for same-policy cache rebuilds."""

    def __init__(self) -> None:
        self._records: dict[int, RetainedEventRecord] = {}

    def replace(
        self,
        user_id: int,
        events: Sequence[RetainedEvent],
        version: DeltaRecCacheVersion,
    ) -> None:
        self._records[int(user_id)] = RetainedEventRecord(
            selection_fingerprint=version.selection_fingerprint,
            events=tuple(events),
        )

    def append(
        self,
        user_id: int,
        event: RetainedEvent,
        version: DeltaRecCacheVersion,
    ) -> None:
        current = self._records.get(int(user_id))
        if current is None or current.selection_fingerprint != version.selection_fingerprint:
            events: tuple[RetainedEvent, ...] = ()
        else:
            events = current.events
        self._records[int(user_id)] = RetainedEventRecord(
            selection_fingerprint=version.selection_fingerprint,
            events=events + (event,),
        )

    def get(
        self,
        user_id: int,
        version: DeltaRecCacheVersion,
    ) -> Optional[tuple[RetainedEvent, ...]]:
        record = self._records.get(int(user_id))
        if record is None or record.selection_fingerprint != version.selection_fingerprint:
            return None
        return record.events

    def invalidate(self, user_id: Optional[int] = None) -> None:
        if user_id is None:
            self._records.clear()
        else:
            self._records.pop(int(user_id), None)


@dataclass(frozen=True)
class GDRCacheLookup:
    slots: torch.Tensor
    hit_mask: torch.Tensor
    states: torch.Tensor
    next_positions: torch.Tensor


class TensorGDRStateCache:
    """Preallocated GPU cache with one fixed-size FP32 state per layer."""

    def __init__(
        self,
        *,
        capacity: int,
        num_layers: int,
        num_heads: int,
        key_dim: int,
        value_dim: int,
        device: torch.device | str,
        version: Optional[DeltaRecCacheVersion] = None,
    ) -> None:
        for name, value in (
            ("capacity", capacity),
            ("num_layers", num_layers),
            ("num_heads", num_heads),
            ("key_dim", key_dim),
            ("value_dim", value_dim),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")
        self.states = torch.zeros(
            capacity,
            num_layers,
            num_heads,
            key_dim,
            value_dim,
            dtype=torch.float32,
            device=device,
        )
        self.valid = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.next_positions = torch.zeros(capacity, dtype=torch.int64, device=device)
        self.version = version

    @property
    def capacity(self) -> int:
        return self.states.shape[0]

    @property
    def bytes_per_entry(self) -> int:
        return self.states[0].numel() * self.states.element_size()

    def _validate_slots(self, slots: torch.Tensor, *, unique: bool = False) -> None:
        if slots.ndim != 1 or slots.dtype not in (torch.int32, torch.int64):
            raise ValueError("cache slots must be an integer vector")
        if slots.device != self.states.device:
            raise ValueError("cache slots and state storage must share a device")
        if slots.numel() and (int(slots.min()) < 0 or int(slots.max()) >= self.capacity):
            raise ValueError("cache slot is out of range")
        if unique and slots.numel() != torch.unique(slots).numel():
            raise ValueError("an online cache batch cannot contain duplicate slots")

    def activate_version(self, version: DeltaRecCacheVersion) -> bool:
        """Activate a model version, invalidating stale states lazily."""

        if self.version == version:
            return False
        self.valid.zero_()
        self.next_positions.zero_()
        self.version = version
        return True

    def lookup(
        self,
        slots: torch.Tensor,
        version: DeltaRecCacheVersion,
    ) -> GDRCacheLookup:
        self._validate_slots(slots)
        if self.version != version:
            hit_mask = torch.zeros_like(slots, dtype=torch.bool)
        else:
            hit_mask = self.valid.index_select(0, slots.long())
        return GDRCacheLookup(
            slots=slots,
            hit_mask=hit_mask,
            states=self.states.index_select(0, slots.long()),
            next_positions=self.next_positions.index_select(0, slots.long()),
        )

    def commit_prefill(
        self,
        slots: torch.Tensor,
        states: torch.Tensor,
        next_positions: torch.Tensor,
        version: DeltaRecCacheVersion,
    ) -> None:
        self._validate_slots(slots, unique=True)
        if states.shape != (len(slots), *self.states.shape[1:]):
            raise ValueError("prefill states do not match cache layout")
        if states.dtype != torch.float32:
            raise ValueError("cached recurrent states must be FP32")
        if next_positions.shape != slots.shape or next_positions.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("next_positions must be an integer vector")
        self.activate_version(version)
        indices = slots.long()
        self.states.index_copy_(0, indices, states)
        self.next_positions.index_copy_(0, indices, next_positions.long())
        self.valid.index_fill_(0, indices, True)

    def mark_written(
        self,
        slots: torch.Tensor,
        next_positions: torch.Tensor,
        version: DeltaRecCacheVersion,
    ) -> None:
        self._validate_slots(slots, unique=True)
        if self.version != version:
            raise ValueError("cannot update a cache under a different version")
        if next_positions.shape != slots.shape:
            raise ValueError("next_positions must align with slots")
        indices = slots.long()
        self.next_positions.index_copy_(0, indices, next_positions.long())
        self.valid.index_fill_(0, indices, True)

    def advance(
        self,
        slots: torch.Tensor,
        next_positions: torch.Tensor,
        version: DeltaRecCacheVersion,
        *,
        expected_positions: Optional[torch.Tensor] = None,
    ) -> None:
        """Advance causal position metadata without running the main model."""

        self._validate_slots(slots, unique=True)
        self.activate_version(version)
        if next_positions.shape != slots.shape:
            raise ValueError("next_positions must align with slots")
        indices = slots.long()
        hits = self.valid.index_select(0, indices)
        if expected_positions is not None:
            if expected_positions.shape != slots.shape:
                raise ValueError("expected_positions must align with slots")
            current = self.next_positions.index_select(0, indices)
            if bool((hits & (current != expected_positions.long())).any()):
                raise ValueError("skip source position is not the next cache position")
        misses = indices[~hits]
        if misses.numel():
            self.states.index_fill_(0, misses, 0.0)
        self.next_positions.index_copy_(0, indices, next_positions.long())
        self.valid.index_fill_(0, indices, True)

    def invalidate(self, slots: Optional[torch.Tensor] = None) -> None:
        if slots is None:
            self.valid.zero_()
            self.next_positions.zero_()
            return
        self._validate_slots(slots)
        indices = slots.long()
        self.valid.index_fill_(0, indices, False)
        self.next_positions.index_fill_(0, indices, 0)


@dataclass(frozen=True)
class GDRStepInput:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    decay_logits: torch.Tensor
    beta_logits: torch.Tensor
    log_decay_scale: torch.Tensor
    decay_bias: torch.Tensor

    def validate(self, state: torch.Tensor) -> tuple[int, int, int, int]:
        if self.q.ndim != 3 or self.k.shape != self.q.shape or self.v.ndim != 3:
            raise ValueError("step Q/K/V must have shape [batch, heads, dim]")
        batch, heads, key_dim = self.q.shape
        if self.v.shape[:2] != (batch, heads):
            raise ValueError("step V must share batch and head dimensions")
        value_dim = self.v.shape[-1]
        if state.shape != (batch, heads, key_dim, value_dim):
            raise ValueError("step state has an incompatible shape")
        for name, gate in (
            ("decay_logits", self.decay_logits),
            ("beta_logits", self.beta_logits),
        ):
            if gate.shape != (batch, heads):
                raise ValueError(f"{name} must have shape [batch, heads]")
        for name, value in (
            ("log_decay_scale", self.log_decay_scale),
            ("decay_bias", self.decay_bias),
        ):
            if value.shape != (heads,):
                raise ValueError(f"{name} must have shape [heads]")
        return batch, heads, key_dim, value_dim


def reference_gdr_step(
    projected: GDRStepInput,
    state: torch.Tensor,
    *,
    write_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference inclusive read/write step with an FP32 recurrent state."""

    batch, _, key_dim, _ = projected.validate(state)
    if state.dtype != torch.float32:
        raise ValueError("reference GDR state must be FP32")
    if write_mask is None:
        write_mask = torch.ones(batch, dtype=torch.bool, device=state.device)
    if write_mask.shape != (batch,) or write_mask.dtype != torch.bool:
        raise ValueError("write_mask must be boolean with shape [batch]")
    q = projected.q.float()
    k = projected.k.float()
    q = q / torch.sqrt(torch.sum(q * q, dim=-1, keepdim=True) + eps)
    k = k / torch.sqrt(torch.sum(k * k, dim=-1, keepdim=True) + eps)
    v = projected.v.float()
    log_decay = -torch.exp(projected.log_decay_scale.float())[None, :] * F.softplus(
        projected.decay_logits.float() + projected.decay_bias.float()[None, :]
    )
    decay = torch.exp(log_decay)
    beta = torch.sigmoid(projected.beta_logits.float())
    decayed = decay[:, :, None, None] * state
    prediction = torch.einsum("bhkv,bhk->bhv", decayed, k)
    residual = v - prediction
    candidate = decayed + beta[:, :, None, None] * torch.einsum(
        "bhk,bhv->bhkv", k, residual
    )
    next_state = torch.where(write_mask[:, None, None, None], candidate, state)
    context = key_dim**-0.5 * torch.einsum("bhk,bhkv->bhv", q, next_state)
    return context.to(projected.v.dtype), next_state


def reference_gdr_read(
    q: torch.Tensor,
    state: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    if q.ndim != 3 or state.ndim != 4 or q.shape[:2] != state.shape[:2]:
        raise ValueError("read Q/state shapes are incompatible")
    if q.shape[-1] != state.shape[-2] or state.dtype != torch.float32:
        raise ValueError("read state must be FP32 with the Q key dimension")
    normalized_q = q.float()
    normalized_q = normalized_q / torch.sqrt(
        torch.sum(normalized_q * normalized_q, dim=-1, keepdim=True) + eps
    )
    context = q.shape[-1] ** -0.5 * torch.einsum(
        "bhk,bhkv->bhv", normalized_q, state
    )
    return context.to(q.dtype)


@dataclass(frozen=True)
class DeltaRecExecutionOutput:
    embeddings: torch.Tensor
    layer_states: Optional[torch.Tensor] = None
    auxiliary: Mapping[str, Any] = field(default_factory=dict)


class DeltaRecBackboneAdapter(nn.Module, abc.ABC):
    """Backbone-owned projection/output path around shared DeltaRec state."""

    @abc.abstractmethod
    def prefill(
        self,
        selected: SelectedSequence,
        *,
        initial_states: Optional[torch.Tensor] = None,
        return_final_states: bool = True,
    ) -> DeltaRecExecutionOutput:
        pass

    @abc.abstractmethod
    def write(
        self,
        item_ids: torch.Tensor,
        source_positions: torch.Tensor,
        payloads: Mapping[str, torch.Tensor],
        cache: TensorGDRStateCache,
        cache_slots: torch.Tensor,
    ) -> DeltaRecExecutionOutput:
        pass

    @abc.abstractmethod
    def read(
        self,
        item_ids: torch.Tensor,
        source_positions: torch.Tensor,
        payloads: Mapping[str, torch.Tensor],
        cache: TensorGDRStateCache,
        cache_slots: torch.Tensor,
    ) -> DeltaRecExecutionOutput:
        pass

    def skip(
        self,
        source_positions: torch.Tensor,
        cache: TensorGDRStateCache,
        cache_slots: torch.Tensor,
    ) -> None:
        """Advance request metadata while performing no embedding or GDR work."""

        version = getattr(self, "cache_version", None)
        if not isinstance(version, DeltaRecCacheVersion):
            raise ValueError("backbone adapter does not define a cache version")
        cache.advance(
            cache_slots,
            source_positions.long() + 1,
            version,
            expected_positions=source_positions,
        )

