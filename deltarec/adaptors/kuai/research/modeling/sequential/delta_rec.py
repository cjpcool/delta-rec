# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from dataclasses import dataclass, field

from typing import Mapping

import torch

@dataclass(frozen=True)
class SelectorFeatureSchema:
    num_items: int
    rating_buckets: int = 8
    time_gap_buckets: int = 32
    position_buckets: int = 16
    popularity_buckets: int = 32
    repeat_buckets: int = 16
    numerical_features: int = 4

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")

    @property
    def cardinalities(self) -> dict[str, int]:
        return {
            "item": self.num_items + 1,
            "rating": self.rating_buckets,
            "time_gap": self.time_gap_buckets,
            "position": self.position_buckets,
            "popularity": self.popularity_buckets,
            "repeat": self.repeat_buckets,
        }

@dataclass(frozen=True)
class SelectorFeatures:
    item_ids: torch.Tensor
    rating_buckets: torch.Tensor
    time_gap_buckets: torch.Tensor
    position_buckets: torch.Tensor
    popularity_buckets: torch.Tensor
    repeat_buckets: torch.Tensor
    numerical: torch.Tensor
    offsets: torch.Tensor

    @property
    def categorical(self) -> dict[str, torch.Tensor]:
        return {
            "item": self.item_ids,
            "rating": self.rating_buckets,
            "time_gap": self.time_gap_buckets,
            "position": self.position_buckets,
            "popularity": self.popularity_buckets,
            "repeat": self.repeat_buckets,
        }

    def validate(self, schema: SelectorFeatureSchema) -> None:
        tokens = len(self.item_ids)
        if self.offsets.ndim != 1 or self.offsets.numel() < 1:
            raise ValueError("offsets must have shape [batch + 1]")
        if self.offsets.dtype not in (torch.int32, torch.int64):
            raise ValueError("offsets must be an integer tensor")
        if int(self.offsets[0]) != 0 or int(self.offsets[-1]) != tokens:
            raise ValueError("offsets must span every selector event")
        if bool((self.offsets[1:] < self.offsets[:-1]).any()):
            raise ValueError("offsets must be nondecreasing")
        for name, values in self.categorical.items():
            if values.shape != (tokens,) or values.dtype != torch.int64:
                raise ValueError(f"{name} must be int64 with shape [tokens]")
            cardinality = schema.cardinalities[name]
            if tokens and (
                int(values.min()) < 0 or int(values.max()) >= cardinality
            ):
                raise ValueError(f"{name} contains an out-of-range bucket")
        if self.numerical.shape != (tokens, schema.numerical_features):
            raise ValueError(
                "numerical must have shape [tokens, schema.numerical_features]"
            )
        if not torch.is_floating_point(self.numerical):
            raise ValueError("numerical features must be floating point")
        if not bool(torch.isfinite(self.numerical).all()):
            raise ValueError("numerical features must be finite")

    def to(self, device: torch.device | str) -> "SelectorFeatures":
        return SelectorFeatures(
            item_ids=self.item_ids.to(device),
            rating_buckets=self.rating_buckets.to(device),
            time_gap_buckets=self.time_gap_buckets.to(device),
            position_buckets=self.position_buckets.to(device),
            popularity_buckets=self.popularity_buckets.to(device),
            repeat_buckets=self.repeat_buckets.to(device),
            numerical=self.numerical.to(device),
            offsets=self.offsets.to(device),
        )

@dataclass(frozen=True)
class DenseEventBatch:
    """Low-cost event fields before main-model embedding and projection.

    Every tensor is dense over ``[batch, time]``. Only the prefix described by
    ``lengths`` is valid. Payloads may add trailing dimensions, but their first
    two dimensions must match the event fields.
    """

    item_ids: torch.Tensor
    rating_buckets: torch.Tensor
    time_gap_buckets: torch.Tensor
    position_buckets: torch.Tensor
    popularity_buckets: torch.Tensor
    repeat_buckets: torch.Tensor
    lengths: torch.Tensor
    payloads: Mapping[str, torch.Tensor] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.item_ids.shape)  # pyre-ignore[7]

    def validate(self, schema: SelectorFeatureSchema) -> None:
        if self.item_ids.ndim != 2:
            raise ValueError("dense event fields must have shape [batch, time]")
        batch, time = self.item_ids.shape
        if self.lengths.shape != (batch,) or self.lengths.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("lengths must be an integer tensor with shape [batch]")
        if bool((self.lengths < 0).any()) or bool((self.lengths > time).any()):
            raise ValueError("lengths must lie in [0, time]")
        for name, values in self.categorical.items():
            if values.shape != (batch, time) or values.dtype not in (
                torch.uint8,
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                raise ValueError(f"{name} must be an integer [batch, time] tensor")
            cardinality = schema.cardinalities[name]
            valid = self.valid_mask
            if bool(valid.any()):
                valid_values = values[valid]
                if int(valid_values.min()) < 0 or int(valid_values.max()) >= cardinality:
                    raise ValueError(f"{name} contains an out-of-range bucket")
        for name, payload in self.payloads.items():
            if payload.ndim < 2 or payload.shape[:2] != (batch, time):
                raise ValueError(f"payload {name!r} must begin with [batch, time]")

    @property
    def categorical(self) -> dict[str, torch.Tensor]:
        return {
            "item": self.item_ids,
            "rating": self.rating_buckets,
            "time_gap": self.time_gap_buckets,
            "position": self.position_buckets,
            "popularity": self.popularity_buckets,
            "repeat": self.repeat_buckets,
        }

    @property
    def valid_mask(self) -> torch.Tensor:
        time = self.item_ids.shape[1]
        return torch.arange(time, device=self.item_ids.device)[None, :] < self.lengths[:, None]

    def packed_features(self, schema: SelectorFeatureSchema) -> SelectorFeatures:
        """Reference packing used outside the accelerated CUDA path."""

        self.validate(schema)
        valid = self.valid_mask
        denominators = self.item_ids.new_tensor(
            [
                max(schema.time_gap_buckets - 1, 1),
                max(schema.position_buckets - 1, 1),
                max(schema.popularity_buckets - 1, 1),
                max(schema.repeat_buckets - 1, 1),
            ],
            dtype=torch.float32,
        )
        numerical = torch.stack(
            [
                self.time_gap_buckets,
                self.position_buckets,
                self.popularity_buckets,
                self.repeat_buckets,
            ],
            dim=-1,
        ).float() / denominators
        offsets = torch.cat(
            [self.lengths.new_zeros(1), self.lengths.cumsum(0)]
        )
        packed = SelectorFeatures(
            item_ids=self.item_ids[valid].long(),
            rating_buckets=self.rating_buckets[valid].long(),
            time_gap_buckets=self.time_gap_buckets[valid].long(),
            position_buckets=self.position_buckets[valid].long(),
            popularity_buckets=self.popularity_buckets[valid].long(),
            repeat_buckets=self.repeat_buckets[valid].long(),
            numerical=numerical[valid],
            offsets=offsets,
        )
        packed.validate(schema)
        return packed

@dataclass(frozen=True)
class EventRoles:
    mandatory_mask: torch.Tensor
    read_only_mask: torch.Tensor

    @classmethod
    def all_optional(cls, tokens: int, device: torch.device) -> "EventRoles":
        mask = torch.zeros(tokens, dtype=torch.bool, device=device)
        return cls(mandatory_mask=mask, read_only_mask=mask.clone())

    def validate(self, tokens: int) -> None:
        for name, value in (
            ("mandatory_mask", self.mandatory_mask),
            ("read_only_mask", self.read_only_mask),
        ):
            if value.shape != (tokens,) or value.dtype != torch.bool:
                raise ValueError(f"{name} must be boolean with shape [tokens]")
        if bool((self.read_only_mask & ~self.mandatory_mask).any()):
            raise ValueError("read_only events must be mandatory")

@dataclass(frozen=True)
class SelectionConstraints:
    """Hard execution constraints around the learned threshold decision.

    ``require_output_mask`` is useful for loss/query anchors: the event is
    physically present, but it writes only when the selector chooses it.
    """

    force_write_mask: torch.Tensor
    force_read_mask: torch.Tensor
    require_output_mask: torch.Tensor

    @classmethod
    def none(cls, shape: torch.Size | tuple[int, ...], device: torch.device) -> "SelectionConstraints":
        empty = torch.zeros(shape, dtype=torch.bool, device=device)
        return cls(empty, empty.clone(), empty.clone())

    @classmethod
    def from_event_roles(cls, roles: EventRoles) -> "SelectionConstraints":
        return cls(
            force_write_mask=roles.mandatory_mask & ~roles.read_only_mask,
            force_read_mask=roles.read_only_mask,
            require_output_mask=torch.zeros_like(roles.mandatory_mask),
        )

    def validate(self, shape: torch.Size | tuple[int, ...]) -> None:
        for name, value in (
            ("force_write_mask", self.force_write_mask),
            ("force_read_mask", self.force_read_mask),
            ("require_output_mask", self.require_output_mask),
        ):
            if value.shape != shape or value.dtype != torch.bool:
                raise ValueError(f"{name} must be boolean with shape {tuple(shape)}")
        if bool((self.force_write_mask & self.force_read_mask).any()):
            raise ValueError("an event cannot be both force-write and force-read")

    def decisions(self, scores: torch.Tensor, threshold: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.validate(scores.shape)
        predicted_write = scores > threshold
        write = (predicted_write | self.force_write_mask) & ~self.force_read_mask
        selected = write | self.force_read_mask | self.require_output_mask
        return selected, write

