# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""Production DLRMv3 adapter for group-shared candidate-conditioned GDR.

The adapter keeps candidate grouping as model semantics and selector chunking as
an execution-only choice.  It computes the frozen PC-MLP score for every
``(user, candidate, history-event)`` tuple, pools those scores within semantic
candidate groups, and submits all ``B * G`` streams in one packed recurrence
call per layer.  Candidate tokens are appended as read-only events and the
returned queries are restored to the caller's original ``[B, K]`` order.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Callable, Optional, TypeVar

import torch

from deltarec.adaptors.kuai.modules.dlrm_candidate_symmetric import (
    FrozenCandidateSelector,
    _jagged_embeddings_to_dense,
)
from deltarec.adaptors.kuai.modules.dlrm_delta_rec import (
    PRODUCTION_HASH_SEED,
    DLRMv3GDRSTULayer,
)
from deltarec.adaptors.kuai.modules.stu import STU, STUStack
from deltarec.adaptors.kuai.research.modeling.sequential.delta_rec_cache import (
    DeltaRecCacheVersion,
)
from deltarec.adaptors.kuai.research.modeling.sequential.deltarec.group_executor import (
    GroupSharedDeltaRecConfig,
    GroupSharedGDRExecutor,
)
from deltarec.adaptors.kuai.research.modeling.sequential.deltarec.group_cache import (
    GroupSelectionLookupCache,
    build_group_selection_cache_version,
)
from deltarec.adaptors.kuai.research.modeling.sequential.deltarec.group_state_cache import (
    GroupStateLookupCache,
    build_group_state_cache_version,
)
from deltarec.adaptors.kuai.research.modeling.sequential.deltarec.global_group_state_cache import (
    GlobalGroupStateCache,
    build_global_group_model_version,
)
from deltarec.adaptors.kuai.research.modeling.sequential.deltarec.group_packing import (
    PackedGroupSequence,
    pack_global_group_queries,
    pack_group_queries,
    pack_group_streams,
)
from deltarec.adaptors.kuai.research.modeling.sequential.deltarec.group_pooling import (
    pool_group_scores,
)
from deltarec.adaptors.kuai.research.modeling.sequential.deltarec.group_selection import (
    GroupSelectionOutput,
    select_group_events,
)
from deltarec.adaptors.kuai.research.modeling.sequential.deltarec.grouping import (
    GroupingOutput,
    group_candidates,
)
from deltarec.adaptors.kuai.research.modeling.sequential.deltarec.selection import (
    select_candidate_events,
)


_T = TypeVar("_T")


def _module_state_sha256(module: torch.nn.Module) -> str:
    """Hash the exact frozen serving weights without dtype conversion."""

    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        contiguous = value.detach().to(device="cpu").contiguous()
        digest.update(
            f"{name}:{tuple(contiguous.shape)}:{contiguous.dtype}".encode("utf-8")
        )
        digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    """Hash an immutable grouping-table tensor as part of cache semantics."""

    contiguous = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(f"{tuple(contiguous.shape)}:{contiguous.dtype}".encode("utf-8"))
    digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _jagged_to_dense(
    values: torch.Tensor,
    lengths: torch.Tensor,
    max_length: int,
    *,
    fill_value: int = 0,
) -> torch.Tensor:
    """Convert a flat jagged integer field to a padded dense matrix."""

    if values.ndim != 1 or lengths.ndim != 1:
        raise ValueError("jagged values and lengths must be vectors")
    if values.dtype not in (torch.int32, torch.int64) or lengths.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("jagged values and lengths must be integer tensors")
    if max_length < 1:
        raise ValueError("dense jagged width must be positive")
    if int(lengths.sum()) != values.numel():
        raise ValueError("jagged values and lengths disagree")
    batch = int(lengths.numel())
    output = values.new_full((batch, max_length), fill_value)
    if values.numel() == 0:
        return output
    rows = torch.repeat_interleave(
        torch.arange(batch, device=values.device, dtype=torch.int64),
        lengths.to(torch.int64),
    )
    offsets = torch.cat(
        (lengths.new_zeros(1), lengths.to(torch.int64).cumsum(dim=0)), dim=0
    )
    positions = torch.arange(
        values.numel(), device=values.device, dtype=torch.int64
    ) - offsets[:-1].to(torch.int64).index_select(0, rows)
    output[rows, positions] = values
    return output


class DLRMv3GroupSharedSTUStack(STU):
    """DLRMv3 STU boundary with one recurrent state per candidate group.

    ``execution_chunk_size`` is consumed only while evaluating the frozen
    selector.  Grouping, pooling, selection, packing, and recurrence always see
    the complete candidate set.  Consequently a fixed ``candidate_group_count``
    has identical semantics for every selector scheduling chunk.
    """

    def __init__(
        self,
        original: STUStack,
        *,
        selector: Optional[FrozenCandidateSelector] = None,
        selection_cache_version: Optional[DeltaRecCacheVersion] = None,
        config: Optional[GroupSharedDeltaRecConfig] = None,
        seed: int = PRODUCTION_HASH_SEED,
        kernel_backend: str = "fla",
        contextual_seq_len: int = 0,
        item_to_category_group: Optional[torch.Tensor] = None,
        category_group_prototypes: Optional[torch.Tensor] = None,
        unknown_category_group: int = 0,
    ) -> None:
        super().__init__(is_inference=original.is_inference)
        if kernel_backend not in ("reference", "fla", "triton"):
            raise ValueError("kernel_backend must be 'reference', 'fla', or 'triton'")
        if selector is not None and selector.seed != seed:
            raise ValueError("selector hash seed and production backend seed differ")
        if selector is None and selection_cache_version is None:
            raise ValueError(
                "a selector or an explicit serving selection-cache version is required"
            )
        if selection_cache_version is not None and not isinstance(
            selection_cache_version, DeltaRecCacheVersion
        ):
            raise TypeError("selection_cache_version must be a DeltaRecCacheVersion")
        if (
            isinstance(contextual_seq_len, bool)
            or not isinstance(contextual_seq_len, int)
            or contextual_seq_len < 0
        ):
            raise ValueError("contextual_seq_len must be a nonnegative integer")

        self.selector = selector
        self.config = config or GroupSharedDeltaRecConfig()
        self.contextual_seq_len = int(contextual_seq_len)
        fixed_category = self.config.grouping_policy == "fixed_category_prototype_v1"
        if isinstance(unknown_category_group, bool) or not isinstance(
            unknown_category_group, int
        ):
            raise TypeError("unknown_category_group must be an integer")
        if not 0 <= unknown_category_group < self.config.candidate_group_count:
            raise ValueError("unknown_category_group must lie in [0,G)")
        self.unknown_category_group = int(unknown_category_group)
        if fixed_category and selector is not None:
            if item_to_category_group is None or category_group_prototypes is None:
                raise ValueError(
                    "fixed_category_prototype_v1 requires an item-to-group table "
                    "and one frozen selector prototype per group"
                )
            if item_to_category_group.ndim != 1 or item_to_category_group.dtype not in (
                torch.int32,
                torch.int64,
            ):
                raise ValueError("item_to_category_group must be an integer vector")
            if category_group_prototypes.ndim != 2 or not torch.is_floating_point(
                category_group_prototypes
            ):
                raise ValueError(
                    "category_group_prototypes must be floating point [G,D]"
                )
            if category_group_prototypes.shape[0] != self.config.candidate_group_count:
                raise ValueError("category prototype count must equal configured G")
            table_i64 = item_to_category_group.detach().to(torch.int64)
            if bool((table_i64 < -1).any()) or bool(
                (table_i64 >= self.config.candidate_group_count).any()
            ):
                raise ValueError("item-to-group values must be -1 or lie in [0,G)")
            if not bool(torch.isfinite(category_group_prototypes).all()):
                raise ValueError("category group prototypes must be finite")
        elif not fixed_category and (
            item_to_category_group is not None or category_group_prototypes is not None
        ):
            raise ValueError(
                "category grouping artifacts require fixed_category_prototype_v1"
            )
        grouping_table = (
            item_to_category_group.detach().to(torch.int64).contiguous()
            if item_to_category_group is not None
            else torch.empty(0, dtype=torch.int64)
        )
        grouping_prototypes = (
            category_group_prototypes.detach().to(torch.float32).contiguous()
            if category_group_prototypes is not None
            else torch.empty((0, 0), dtype=torch.float32)
        )
        self.register_buffer(
            "item_to_category_group", grouping_table, persistent=fixed_category
        )
        self.register_buffer(
            "category_group_prototypes", grouping_prototypes, persistent=fixed_category
        )
        layers = torch.nn.ModuleList(
            [
                DLRMv3GDRSTULayer(
                    layer,
                    seed=seed + index,
                    kernel_backend=kernel_backend,
                    assume_binary_event_gate=True,
                )
                for index, layer in enumerate(original._stu_layers)
            ]
        )
        self.packed_executor = GroupSharedGDRExecutor(layers, config=self.config)

        base_parameter = next(original.parameters(), None)
        if base_parameter is not None and base_parameter.device.type != "meta":
            self.to(base_parameter.device)

        initialization = hashlib.sha256(b"dlrmv3-group-shared-gdr-v1")
        for index, layer in enumerate(self.layers):
            for name in ("gdr_log_decay_scale", "gdr_decay_bias", "gdr_gate_weight"):
                value = getattr(layer, name).detach().cpu().contiguous()
                initialization.update(f"{index}:{name}:{tuple(value.shape)}".encode())
                initialization.update(value.numpy().tobytes())
        self.gdr_initialization_hash = initialization.hexdigest()
        self.selector_artifact_sha256 = (
            dict(selector.artifact_sha256) if selector is not None else {}
        )
        self.grouping_artifact_sha256 = (
            {
                "item_to_category_group": _tensor_sha256(
                    self.item_to_category_group
                ),
                "category_group_prototypes": _tensor_sha256(
                    self.category_group_prototypes
                ),
                "unknown_category_group": hashlib.sha256(
                    str(self.unknown_category_group).encode("ascii")
                ).hexdigest(),
            }
            if fixed_category and selector is not None
            else {}
        )
        if selector is not None:
            derived_cache_version = build_group_selection_cache_version(
                selector_artifacts=self.selector_artifact_sha256,
                grouping_artifacts=self.grouping_artifact_sha256,
                candidate_group_count=self.config.candidate_group_count,
                grouping_policy=self.config.grouping_policy,
                group_pool=self.config.group_pool,
                retention_ratio=self.config.retention_ratio,
                recent_floor=self.config.recent_floor,
                contextual_seq_len=self.contextual_seq_len,
            )
            if (
                selection_cache_version is not None
                and selection_cache_version != derived_cache_version
            ):
                raise ValueError(
                    "explicit cache version disagrees with selector/config semantics"
                )
            self.selection_cache_version = derived_cache_version
        else:
            assert selection_cache_version is not None
            self.selection_cache_version = selection_cache_version
        self.serving_model_sha256 = _module_state_sha256(self.packed_executor)
        first_layer = self.layers[0].base
        self.state_cache_version = build_group_state_cache_version(
            model_sha256=self.serving_model_sha256,
            selection_version=self.selection_cache_version,
            layer_count=len(self.layers),
            group_count=self.config.candidate_group_count,
            num_heads=first_layer._num_heads,
            key_dim=first_layer._attention_dim,
            value_dim=first_layer._hidden_dim,
            contextual_seq_len=self.contextual_seq_len,
        )
        self.gdr_backend = kernel_backend
        self.profile_stages = False

        self.last_grouping: Optional[GroupingOutput] = None
        self.last_selection: Optional[GroupSelectionOutput] = None
        self.last_packed: Optional[PackedGroupSequence] = None
        self.last_final_states: Optional[torch.Tensor] = None
        self.last_diagnostics: dict[str, Any] = {}
        self.last_cache_build_diagnostics: dict[str, Any] = {}
        self.last_state_cache_build_diagnostics: dict[str, Any] = {}

    @property
    def layers(self) -> torch.nn.ModuleList:
        """Expose the wrapped production layers without double-registering them."""

        return self.packed_executor.layers

    @classmethod
    def for_cached_serving(
        cls,
        original: STUStack,
        *,
        selection_cache: GroupSelectionLookupCache,
        config: Optional[GroupSharedDeltaRecConfig] = None,
        seed: int = PRODUCTION_HASH_SEED,
        kernel_backend: str = "fla",
        contextual_seq_len: int = 0,
    ) -> "DLRMv3GroupSharedSTUStack":
        """Construct a hit-only runtime without loading selector weights/tables.

        The deployment manifest is responsible for pairing ``config`` with the
        persisted table contract.  The cache's version is still checked on
        every lookup, and cold/table-build APIs fail explicitly because this
        runtime contains no selector.
        """

        if not isinstance(selection_cache, GroupSelectionLookupCache):
            raise TypeError("selection_cache must be a GroupSelectionLookupCache")
        effective_config = config or GroupSharedDeltaRecConfig(
            retention_ratio=selection_cache.retention_ratio,
            recent_floor=selection_cache.recent_floor,
        )
        if (
            effective_config.retention_ratio != selection_cache.retention_ratio
            or effective_config.recent_floor != selection_cache.recent_floor
        ):
            raise ValueError("selection cache and runtime disagree on ratio/recent floor")
        return cls(
            original,
            selector=None,
            selection_cache_version=selection_cache.version,
            config=effective_config,
            seed=seed,
            kernel_backend=kernel_backend,
            contextual_seq_len=contextual_seq_len,
        )

    @classmethod
    def for_state_cached_serving(
        cls,
        original: STUStack,
        *,
        state_cache: GroupStateLookupCache,
        config: Optional[GroupSharedDeltaRecConfig] = None,
        seed: int = PRODUCTION_HASH_SEED,
        kernel_backend: str = "fla",
        contextual_seq_len: int = 0,
    ) -> "DLRMv3GroupSharedSTUStack":
        """Construct a candidate-only runtime backed by frozen prefix states."""

        if not isinstance(state_cache, GroupStateLookupCache):
            raise TypeError("state_cache must be a GroupStateLookupCache")
        selection_cache = state_cache.selection_cache
        effective_config = config or GroupSharedDeltaRecConfig(
            retention_ratio=selection_cache.retention_ratio,
            recent_floor=selection_cache.recent_floor,
        )
        if (
            effective_config.retention_ratio != selection_cache.retention_ratio
            or effective_config.recent_floor != selection_cache.recent_floor
        ):
            raise ValueError("state cache and runtime disagree on ratio/recent floor")
        runtime = cls(
            original,
            selector=None,
            selection_cache_version=state_cache.selection_cache.version,
            config=effective_config,
            seed=seed,
            kernel_backend=kernel_backend,
            contextual_seq_len=contextual_seq_len,
        )
        if runtime.state_cache_version != state_cache.version:
            raise ValueError(
                "state cache does not match the frozen serving model/configuration"
            )
        return runtime

    @classmethod
    def for_global_group_cached_serving(
        cls,
        original: STUStack,
        *,
        state_cache: GlobalGroupStateCache,
        config: Optional[GroupSharedDeltaRecConfig] = None,
        seed: int = PRODUCTION_HASH_SEED,
        kernel_backend: str = "fla",
        contextual_seq_len: int = 0,
    ) -> "DLRMv3GroupSharedSTUStack":
        """Construct candidate-only serving from global fixed-category states."""

        if not isinstance(state_cache, GlobalGroupStateCache):
            raise TypeError("state_cache must be a GlobalGroupStateCache")
        effective_config = config or GroupSharedDeltaRecConfig(
            candidate_group_count=state_cache.group_count,
            grouping_policy="fixed_category_prototype_v1",
        )
        if effective_config.grouping_policy != "fixed_category_prototype_v1":
            raise ValueError("global-group serving requires fixed-category grouping")
        if effective_config.candidate_group_count != state_cache.group_count:
            raise ValueError("global-group cache and runtime disagree on G")
        if (
            effective_config.retention_ratio
            != float(state_cache.components["retention_ratio"])
            or effective_config.recent_floor
            != int(state_cache.components["recent_floor"])
        ):
            raise ValueError("global-group cache and runtime disagree on ratio/recent floor")
        runtime = cls(
            original,
            selector=None,
            selection_cache_version=state_cache.selection_cache_version,
            config=effective_config,
            seed=seed,
            kernel_backend=kernel_backend,
            contextual_seq_len=contextual_seq_len,
            item_to_category_group=state_cache.item_to_category_group,
            unknown_category_group=state_cache.unknown_category_group,
        )
        if runtime.state_cache_version != state_cache.state_cache_version:
            raise ValueError(
                "global-group cache does not match the frozen serving model/configuration"
            )
        return runtime

    def _timed(
        self,
        operation: Callable[[], _T],
        *,
        synchronize: Callable[[], None],
    ) -> tuple[_T, float]:
        synchronize()
        started = time.perf_counter()
        value = operation()
        synchronize()
        return value, (time.perf_counter() - started) * 1000.0

    def _selector_scores(
        self,
        history_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        history_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Evaluate exact O(B*K*L) PC-MLP scores with bounded scheduling."""

        if self.selector is None:
            raise RuntimeError("cached-serving-only runtime has no selector")
        candidates = int(candidate_embeddings.shape[1])
        chunk_size = self.config.execution_chunk_size
        # Reuse one embedding-table lookup for both grouping and every selector
        # chunk.  The exact selector still materializes/evaluates all B*K*L
        # pairs; only its temporary feature allocation is chunked.
        if bool(getattr(self.selector, "requires_shared_embeddings", False)):
            score_embeddings = getattr(self.selector, "score_embeddings", None)
            if not callable(score_embeddings):
                raise TypeError("shared GC selector lacks score_embeddings")
            scores = score_embeddings(
                history_embeddings,
                candidate_embeddings,
                history_lengths,
                dtype=torch.float32,
                candidate_chunk_size=chunk_size,
            )
        else:
            scores = self.selector.exact_selector.exact_packed_scores(
                history_embeddings,
                candidate_embeddings,
                history_lengths,
                candidate_chunk_size=chunk_size,
            )
        return scores, (candidates + chunk_size - 1) // chunk_size

    def _fixed_category_group_ids(
        self, candidate_ids: torch.Tensor
    ) -> torch.Tensor:
        """Map every catalog ID to a frozen global group with unknown fallback."""

        if self.config.grouping_policy != "fixed_category_prototype_v1":
            raise RuntimeError("fixed category lookup requested by another policy")
        if self.item_to_category_group.numel() == 0:
            raise RuntimeError("fixed-category runtime has no item-to-group table")
        ids = candidate_ids.to(torch.int64)
        in_range = (ids >= 0) & (ids < self.item_to_category_group.numel())
        safe_ids = ids.clamp(0, self.item_to_category_group.numel() - 1)
        mapped = self.item_to_category_group.index_select(0, safe_ids.reshape(-1)).reshape_as(
            ids
        )
        known = in_range & (mapped >= 0)
        return torch.where(
            known,
            mapped,
            mapped.new_full(mapped.shape, self.unknown_category_group),
        )

    def _fixed_category_scores(
        self,
        history_embeddings: torch.Tensor,
        history_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Score history against frozen group prototypes, independent of a slate."""

        if self.category_group_prototypes.numel() == 0:
            raise RuntimeError("fixed-category runtime has no group prototypes")
        if self.category_group_prototypes.shape[1] != history_embeddings.shape[-1]:
            raise ValueError(
                "category prototype width must match selector embedding width"
            )
        prototypes = self.category_group_prototypes.to(
            device=history_embeddings.device, dtype=history_embeddings.dtype
        )[None, :, :].expand(history_embeddings.shape[0], -1, -1)
        return self._selector_scores(
            history_embeddings,
            prototypes,
            history_lengths,
        )

    def _validate_inputs(
        self,
        *,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        num_targets: torch.Tensor,
        history_item_ids: torch.Tensor,
        candidate_item_ids: torch.Tensor,
    ) -> tuple[int, int, torch.Tensor]:
        if x.ndim != 2 or not torch.is_floating_point(x):
            raise ValueError("x must be floating point with shape [tokens,D]")
        if x_lengths.ndim != 1 or x_lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError("x_lengths must be an integer vector")
        batch = int(x_lengths.numel())
        if batch < 1:
            raise ValueError("group-shared execution requires a nonempty batch")
        if num_targets.shape != (batch,) or num_targets.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("num_targets must be an integer vector with shape [B]")
        if x_offsets.shape != (batch + 1,) or x_offsets.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("x_offsets must be an integer vector with shape [B+1]")
        if history_item_ids.ndim != 1 or candidate_item_ids.ndim != 1:
            raise ValueError("production item IDs must be flat jagged vectors")
        if history_item_ids.dtype not in (torch.int32, torch.int64) or (
            candidate_item_ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("production item IDs must be integer tensors")
        fields = (
            x_lengths,
            x_offsets,
            num_targets,
            history_item_ids,
            candidate_item_ids,
        )
        if any(field.device != x.device for field in fields):
            raise ValueError("production stream tensors and item IDs must share a device")
        if int(x_offsets[0]) != 0 or int(x_offsets[-1]) != len(x):
            raise ValueError("x_offsets must span every source token")
        if not torch.equal(
            x_offsets[1:] - x_offsets[:-1], x_lengths.to(x_offsets.dtype)
        ):
            raise ValueError("x offsets and lengths disagree")

        candidates = int(num_targets.max())
        if candidates < 1:
            raise ValueError("group-shared execution requires at least one candidate")
        if bool((num_targets != candidates).any()):
            raise ValueError(
                "group-shared production execution currently requires one uniform, "
                "nonzero candidate count per request"
            )
        history_lengths = (
            x_lengths.to(torch.int64)
            - self.contextual_seq_len
            - num_targets.to(torch.int64)
        )
        if bool((history_lengths < 1).any()):
            raise ValueError("group-shared production streams require nonempty histories")
        if int(history_lengths.sum()) != history_item_ids.numel():
            raise ValueError("history IDs do not match the production stream lengths")
        if int(num_targets.sum()) != candidate_item_ids.numel():
            raise ValueError("candidate IDs do not match the production stream lengths")
        return batch, candidates, history_lengths

    @staticmethod
    def _validate_cache_key_fields(
        *,
        user_ids: torch.Tensor,
        history_versions: torch.Tensor,
        batch: int,
    ) -> None:
        for name, value in (
            ("user_ids", user_ids),
            ("history_versions", history_versions),
        ):
            if value.shape != (batch,) or value.dtype not in (
                torch.int32,
                torch.int64,
            ):
                raise ValueError(f"{name} must be an integer vector with shape [B]")

    def _validate_cached_inputs(
        self,
        *,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        num_targets: torch.Tensor,
        user_ids: torch.Tensor,
        history_versions: torch.Tensor,
        candidate_item_ids: torch.Tensor,
    ) -> tuple[int, int, torch.Tensor]:
        """Validate the key-ready serving boundary without requiring history IDs."""

        if x.ndim != 2 or not torch.is_floating_point(x):
            raise ValueError("x must be floating point with shape [tokens,D]")
        if x_lengths.ndim != 1 or x_lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError("x_lengths must be an integer vector")
        batch = int(x_lengths.numel())
        if batch < 1:
            raise ValueError("cached group-shared execution requires a nonempty batch")
        if num_targets.shape != (batch,) or num_targets.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("num_targets must be an integer vector with shape [B]")
        if x_offsets.shape != (batch + 1,) or x_offsets.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("x_offsets must be an integer vector with shape [B+1]")
        if any(field.device != x.device for field in (x_lengths, x_offsets, num_targets)):
            raise ValueError("source layout tensors must share the x device")
        if candidate_item_ids.ndim != 1 or candidate_item_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("candidate_item_ids must be a flat integer vector")
        if candidate_item_ids.numel() < batch or candidate_item_ids.numel() % batch:
            raise ValueError("candidate IDs do not match the production target counts")
        candidates = candidate_item_ids.numel() // batch
        if self.config.validate_runtime:
            if int(x_offsets[0]) != 0 or int(x_offsets[-1]) != len(x):
                raise ValueError("x_offsets must span every source token")
            if not torch.equal(
                x_offsets[1:] - x_offsets[:-1], x_lengths.to(x_offsets.dtype)
            ):
                raise ValueError("x offsets and lengths disagree")
            if bool((num_targets != candidates).any()):
                raise ValueError(
                    "cached group-shared execution requires one uniform, nonzero "
                    "candidate count per request"
                )
        self._validate_cache_key_fields(
            user_ids=user_ids,
            history_versions=history_versions,
            batch=batch,
        )
        history_lengths = (
            x_lengths.to(torch.int64)
            - self.contextual_seq_len
            - num_targets.to(torch.int64)
        )
        if self.config.validate_runtime and bool((history_lengths < 1).any()):
            raise ValueError("cached group-shared streams require nonempty histories")
        return batch, candidates, history_lengths

    def forward_group_shared(
        self,
        *,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        num_targets: torch.Tensor,
        history_item_ids: torch.Tensor,
        candidate_item_ids: torch.Tensor,
        history_selector_embeddings: Optional[torch.Tensor] = None,
        candidate_selector_embeddings: Optional[torch.Tensor] = None,
        return_final_states: bool = True,
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        GroupingOutput,
        GroupSelectionOutput,
        dict[str, Any],
    ]:
        """Run group-shared packed GDR and return candidate-aligned queries."""

        if self.selector is None:
            raise RuntimeError(
                "cached-serving-only runtime cannot execute the cold selector path"
            )
        profile_synchronized = bool(x.is_cuda and self.profile_stages)

        def synchronize() -> None:
            if profile_synchronized:
                torch.cuda.synchronize(x.device)

        synchronize()
        total_started = time.perf_counter()
        (validated, input_preparation_ms) = self._timed(
            lambda: self._validate_inputs(
                x=x,
                x_lengths=x_lengths,
                x_offsets=x_offsets,
                num_targets=num_targets,
                history_item_ids=history_item_ids,
                candidate_item_ids=candidate_item_ids,
            ),
            synchronize=synchronize,
        )
        batch, candidates, history_lengths = validated
        max_history = int(history_lengths.max())
        history_ids, dense_history_ms = self._timed(
            lambda: _jagged_to_dense(
                history_item_ids,
                history_lengths,
                max_history,
            ),
            synchronize=synchronize,
        )
        candidate_ids = candidate_item_ids.reshape(batch, candidates)
        fixed_category = self.config.grouping_policy == "fixed_category_prototype_v1"
        requires_shared = bool(
            getattr(self.selector, "requires_shared_embeddings", False)
        )
        if requires_shared and (
            history_selector_embeddings is None
            or candidate_selector_embeddings is None
        ):
            raise RuntimeError(
                "the Kuai GC selector requires exact shared embedding tensors"
            )
        if (history_selector_embeddings is None) != (
            candidate_selector_embeddings is None
        ):
            raise ValueError(
                "history and candidate selector embeddings must be supplied together"
            )

        def selector_embeddings() -> tuple[torch.Tensor, torch.Tensor]:
            if history_selector_embeddings is None:
                return (
                    self.selector.lookup(history_ids, dtype=torch.float32),
                    (
                        self.category_group_prototypes.new_zeros(
                            (batch, candidates, self.category_group_prototypes.shape[1])
                        )
                        if fixed_category
                        else self.selector.lookup(candidate_ids, dtype=torch.float32)
                    ),
                )
            if history_selector_embeddings.device != x.device or (
                candidate_selector_embeddings is None
                or candidate_selector_embeddings.device != x.device
            ):
                raise ValueError(
                    "selector embeddings and production stream must share a device"
                )
            return (
                _jagged_embeddings_to_dense(
                    history_selector_embeddings,
                    history_lengths,
                    max_history,
                ),
                _jagged_embeddings_to_dense(
                    candidate_selector_embeddings,
                    num_targets,
                    candidates,
                ),
            )

        embeddings, embedding_lookup_ms = self._timed(
            selector_embeddings,
            synchronize=synchronize,
        )
        history_embeddings, candidate_embeddings = embeddings
        candidate_category_group_ids = (
            self._fixed_category_group_ids(candidate_ids) if fixed_category else None
        )
        grouping, grouping_ms = self._timed(
            lambda: group_candidates(
                candidate_ids,
                candidate_embeddings,
                self.config.candidate_group_count,
                policy=self.config.grouping_policy,
                candidate_category_group_ids=candidate_category_group_ids,
                validate_runtime=self.config.validate_runtime,
            ),
            synchronize=synchronize,
        )
        selector_result, selector_ms = self._timed(
            lambda: (
                self._fixed_category_scores(history_embeddings, history_lengths)
                if fixed_category
                else self._selector_scores(
                    history_embeddings,
                    candidate_embeddings,
                    history_lengths,
                )
            ),
            synchronize=synchronize,
        )
        candidate_scores, selector_chunk_count = selector_result

        def pool_scores() -> torch.Tensor:
            selected_masks = None
            if self.config.group_pool == "mask_vote":
                candidate_selection = select_candidate_events(
                    candidate_scores,
                    history_lengths,
                    recent_floor=self.config.recent_floor,
                    retention_ratio=self.config.retention_ratio,
                    return_dense_mask=True,
                    validate_runtime=self.config.validate_runtime,
                )
                selected_masks = candidate_selection.dense_mask
            return pool_group_scores(
                candidate_scores,
                grouping.candidate_to_group,
                group_count=grouping.group_count,
                pool=self.config.group_pool,
                selected_masks=selected_masks,
                history_lengths=history_lengths,
                validate_runtime=self.config.validate_runtime,
            )

        if fixed_category:
            group_scores = candidate_scores
            pooling_ms = 0.0
        else:
            group_scores, pooling_ms = self._timed(
                pool_scores,
                synchronize=synchronize,
            )
        selection, selection_ms = self._timed(
            lambda: select_group_events(
                group_scores,
                history_lengths,
                recent_floor=self.config.recent_floor,
                retention_ratio=self.config.retention_ratio,
                return_dense_mask=False,
                validate_runtime=self.config.validate_runtime,
            ),
            synchronize=synchronize,
        )
        packed, packing_ms = self._timed(
            lambda: pack_group_streams(
                x=x,
                x_lengths=x_lengths,
                x_offsets=x_offsets,
                num_targets=num_targets,
                grouping=grouping,
                selection=selection,
                contextual_seq_len=self.contextual_seq_len,
                validate_runtime=self.config.validate_runtime,
            ),
            synchronize=synchronize,
        )

        for layer in self.layers:
            layer.profile_stages = profile_synchronized
        executor_output, executor_ms = self._timed(
            lambda: self.packed_executor(
                source_x=x,
                packed=packed,
                return_final_states=return_final_states,
            ),
            synchronize=synchronize,
        )
        if executor_output.physical_gdr_calls != len(self.layers):
            raise RuntimeError("group executor did not issue exactly one GDR call per layer")

        layer_projection_ms: Optional[float] = None
        recurrent_ms: Optional[float] = None
        candidate_output_ms: Optional[float] = None
        if profile_synchronized:
            layer_times = [layer.last_stage_times_ms() for layer in self.layers]
            layer_projection_ms = sum(
                stage.get("projection_ms", 0.0) for stage in layer_times
            )
            recurrent_ms = sum(stage.get("prefill_ms", 0.0) for stage in layer_times)
            candidate_output_ms = sum(
                stage.get("output_ms", 0.0) for stage in layer_times
            )

        queries = executor_output.queries
        final_states = executor_output.final_states
        if queries.shape[:2] != (batch, candidates):
            raise RuntimeError("group executor did not restore the original candidate axis")
        if final_states is not None and (
            final_states.shape[:3]
            != (batch, grouping.group_count, len(self.layers))
            or final_states.dtype != torch.float32
        ):
            raise RuntimeError("group executor returned an invalid FP32 state layout")

        synchronize()
        total_ms = (time.perf_counter() - total_started) * 1000.0
        timings_valid = profile_synchronized or not x.is_cuda
        stage_times_ms: dict[str, Optional[float]] = {
            "input_preparation_ms": (
                input_preparation_ms + dense_history_ms if timings_valid else None
            ),
            "embedding_lookup_ms": embedding_lookup_ms if timings_valid else None,
            "grouping_ms": grouping_ms if timings_valid else None,
            "selector_ms": selector_ms if timings_valid else None,
            "pooling_ms": pooling_ms if timings_valid else None,
            "selection_ms": selection_ms if timings_valid else None,
            "packing_ms": packing_ms if timings_valid else None,
            "executor_ms": executor_ms if timings_valid else None,
            # Layer zero's full-source projection happens inside executor_ms and
            # outside DLRMv3GDRSTULayer's internal projection event pair.
            "layer_internal_projection_ms": layer_projection_ms,
            "recurrent_ms": recurrent_ms,
            "candidate_output_ms": candidate_output_ms,
            "total_ms": total_ms if timings_valid else None,
        }
        diagnostics: dict[str, Any] = {
            "batch_size": batch,
            "candidate_count": candidates,
            "candidate_group_count": self.config.candidate_group_count,
            "effective_group_count": grouping.group_count,
            "grouping_policy": self.config.grouping_policy,
            "group_pool": self.config.group_pool,
            "retention_ratio": self.config.retention_ratio,
            "recent_floor": self.config.recent_floor,
            "execution_chunk_size": self.config.execution_chunk_size,
            "selector_chunk_count": selector_chunk_count,
            "selector_complexity": "O(B*K*L)",
            "selector_score_events": int(history_lengths.sum()) * candidates,
            "input_history_tokens": (
                len(x)
                - batch * self.contextual_seq_len
                - candidate_item_ids.numel()
            ),
            "selected_group_history_tokens": selection.source_positions.numel(),
            "packed_tokens_per_layer": packed.packed_tokens,
            "write_tokens_per_layer": packed.write_tokens,
            "read_tokens_per_layer": packed.read_tokens,
            "logical_encodes": packed.sequence_count,
            "logical_states": packed.sequence_count * len(self.layers),
            "physical_gdr_calls": executor_output.physical_gdr_calls,
            "all_groups_single_packed_submission": True,
            "query_event_gate": 0,
            "candidate_order": "original_input_order",
            "state_layout": "B,G,layers,H,Dk,Dv",
            "state_dtype": "float32",
            "configured_projection_dtype": self.config.projection_dtype,
            "actual_input_projection_dtype": str(x.dtype).replace("torch.", "", 1),
            "projection_linear_mode": self.config.projection_mode,
            "offset_shape": list(packed.offsets.shape),
            "group_size_min": int(grouping.group_sizes.min()),
            "group_size_max": int(grouping.group_sizes.max()),
            "return_final_states": return_final_states,
            "component_timings_synchronized": profile_synchronized,
            "projection_timing_scope": (
                "layer_internal_events_only; layer0 full-source projection is "
                "included in executor_ms"
            ),
            "stage_times_ms": stage_times_ms,
            "selector_artifacts": dict(self.selector_artifact_sha256),
            "gdr_initialization_sha256": self.gdr_initialization_hash,
            "actual_layer_backends": [layer.kernel_backend for layer in self.layers],
            **dict(executor_output.auxiliary),
        }
        # Flat timing aliases retain the production adapter's existing
        # component-diagnostics convention while stage_times_ms is easier to
        # serialize as a cohesive record.
        diagnostics.update(stage_times_ms)

        self.last_grouping = grouping
        self.last_selection = selection
        self.last_packed = packed
        self.last_final_states = final_states
        self.last_diagnostics = diagnostics
        return queries, final_states, grouping, selection, dict(diagnostics)

    @torch.no_grad()
    def materialize_state_cache(
        self,
        *,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        num_targets: torch.Tensor,
        history_item_ids: torch.Tensor,
        candidate_item_ids: torch.Tensor,
        user_ids: torch.Tensor,
        history_versions: torch.Tensor,
        storage_device: torch.device | str = "cpu",
        pin_memory: bool = False,
    ) -> GroupStateLookupCache:
        """Freeze selected-history GDR prefix states for candidate-only serving.

        This is a post-training/offline operation.  It executes the exact cold
        5C path once, including history prefill, then persists the FP32 state
        for every ``(user, group, layer)``.  Candidate events use a zero write
        gate, so the returned final states are exactly the prefix states and do
        not depend on candidate transport order.
        """

        if self.selector is None:
            raise RuntimeError(
                "cached-serving-only runtime cannot materialize a state table"
            )
        if self.training:
            raise RuntimeError("state-cache materialization requires eval mode")
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
        started = time.perf_counter()
        queries, states, grouping, selection, cold_diagnostics = (
            self.forward_group_shared(
                x=x,
                x_lengths=x_lengths,
                x_offsets=x_offsets,
                num_targets=num_targets,
                history_item_ids=history_item_ids,
                candidate_item_ids=candidate_item_ids,
                return_final_states=True,
            )
        )
        del queries
        if states is None or states.dtype != torch.float32:
            raise RuntimeError("offline 5C prefill did not produce FP32 group states")
        batch = int(x_lengths.numel())
        candidates = candidate_item_ids.numel() // batch
        candidate_ids = candidate_item_ids.reshape(batch, candidates)
        history_lengths = (
            x_lengths.to(torch.int64)
            - self.contextual_seq_len
            - num_targets.to(torch.int64)
        )
        selection_cache = GroupSelectionLookupCache.from_outputs(
            user_ids=user_ids,
            history_versions=history_versions,
            candidate_ids=candidate_ids,
            history_lengths=history_lengths,
            grouping=grouping,
            selection=selection,
            version=self.selection_cache_version,
            retention_ratio=self.config.retention_ratio,
            recent_floor=self.config.recent_floor,
            storage_device=storage_device,
            pin_memory=pin_memory,
            validate_runtime=self.config.validate_runtime,
        )
        state_cache = GroupStateLookupCache.from_outputs(
            selection_cache=selection_cache,
            states=states,
            version=self.state_cache_version,
            storage_device=storage_device,
            pin_memory=pin_memory,
            validate_runtime=self.config.validate_runtime,
        )
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
        build_ms = (time.perf_counter() - started) * 1000.0
        self.last_state_cache_build_diagnostics = {
            "excluded_from_serving_latency": True,
            "operation": "group_prefix_state_table_materialization",
            "build_ms": build_ms,
            "entry_count": state_cache.entry_count,
            "payload_bytes": state_cache.payload_bytes,
            "bytes_per_entry": state_cache.bytes_per_entry,
            "state_payload_bytes": state_cache.state_payload_bytes,
            "storage_device": str(state_cache.storage_device),
            "pin_memory": bool(pin_memory),
            "candidate_count": candidates,
            "candidate_group_count": grouping.group_count,
            "layer_count": len(self.layers),
            "state_dtype": "float32",
            "selector_executed_offline": True,
            "history_prefill_executed_offline": True,
            "cold_physical_gdr_calls": cold_diagnostics["physical_gdr_calls"],
            "state_cache_version": self.state_cache_version.fingerprint,
            "model_sha256": self.serving_model_sha256,
            "cache_key": (
                "user_id,history_version,canonical_candidate_set_sha256,"
                "state_cache_contract_version"
            ),
        }
        return state_cache

    @torch.no_grad()
    def materialize_global_group_state_cache(
        self,
        *,
        dataset_id: str,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        num_targets: torch.Tensor,
        history_item_ids: torch.Tensor,
        candidate_item_ids: torch.Tensor,
        user_ids: torch.Tensor,
        history_versions: torch.Tensor,
        storage_device: torch.device | str = "cpu",
        pin_memory: bool = False,
    ) -> GlobalGroupStateCache:
        """Materialize and atomically publish all fixed global-group states."""

        if self.config.grouping_policy != "fixed_category_prototype_v1":
            raise RuntimeError(
                "global-group cache materialization requires fixed-category grouping"
            )
        if self.selector is None:
            raise RuntimeError("candidate-only runtime cannot materialize global states")
        if self.training:
            raise RuntimeError("global-group cache materialization requires eval mode")
        if self.item_to_category_group.numel() == 0 or (
            self.category_group_prototypes.numel() == 0
        ):
            raise RuntimeError("fixed-category grouping artifacts are unavailable")
        started = time.perf_counter()
        queries, states, _, _, diagnostics = self.forward_group_shared(
            x=x,
            x_lengths=x_lengths,
            x_offsets=x_offsets,
            num_targets=num_targets,
            history_item_ids=history_item_ids,
            candidate_item_ids=candidate_item_ids,
            return_final_states=True,
        )
        del queries
        if states is None or states.dtype != torch.float32:
            raise RuntimeError("fixed-category prefill did not return FP32 group states")
        if states.shape[1] != self.config.candidate_group_count:
            raise RuntimeError("fixed-category prefill omitted a global group")
        model_version = build_global_group_model_version(
            dataset_id=dataset_id,
            state_cache_version=self.state_cache_version,
            selector_artifacts=self.selector_artifact_sha256,
            grouping_artifacts=self.grouping_artifact_sha256,
            candidate_group_count=self.config.candidate_group_count,
            grouping_policy=self.config.grouping_policy,
            retention_ratio=self.config.retention_ratio,
            recent_floor=self.config.recent_floor,
        )
        components = {
            "serving_model_sha256": self.serving_model_sha256,
            "selector_artifacts": dict(self.selector_artifact_sha256),
            "grouping_artifacts": dict(self.grouping_artifact_sha256),
            "grouping_policy": self.config.grouping_policy,
            "group_pool": self.config.group_pool,
            "retention_ratio": self.config.retention_ratio,
            "recent_floor": self.config.recent_floor,
            "state_layout": "layers,H,Dk,Dv",
            "state_dtype": "float32",
        }
        cache = GlobalGroupStateCache(
            dataset_id=dataset_id,
            model_version=model_version,
            selection_cache_version=self.selection_cache_version,
            state_cache_version=self.state_cache_version,
            item_to_category_group=self.item_to_category_group,
            unknown_category_group=self.unknown_category_group,
            group_count=self.config.candidate_group_count,
            state_shape=states.shape[2:],
            components=components,
            storage_device=storage_device,
            pin_memory=pin_memory,
        )
        cache.publish_batch(
            user_ids=user_ids,
            history_versions=history_versions,
            states=states,
        )
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
        self.last_state_cache_build_diagnostics = {
            "excluded_from_serving_latency": True,
            "operation": "global_group_state_table_materialization",
            "build_ms": (time.perf_counter() - started) * 1000.0,
            "entry_count": cache.entry_count,
            "state_row_count": cache.state_row_count,
            "state_bytes_per_group": cache.state_bytes_per_group,
            "group_count": cache.group_count,
            "storage_device": str(cache.storage_device),
            "pin_memory": bool(pin_memory),
            "model_version": cache.model_version,
            "cache_key": (
                "dataset_id,user_id,history_version,global_group_id,model_version"
            ),
            "candidate_id_in_key": False,
            "candidate_set_hash_in_key": False,
            "eager_all_group_materialization": True,
            "atomic_history_version_publication": True,
            "cold_physical_gdr_calls": diagnostics["physical_gdr_calls"],
        }
        return cache

    @torch.no_grad()
    def materialize_selection_cache(
        self,
        *,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        num_targets: torch.Tensor,
        history_item_ids: torch.Tensor,
        candidate_item_ids: torch.Tensor,
        user_ids: torch.Tensor,
        history_versions: torch.Tensor,
        storage_device: torch.device | str = "cpu",
        pin_memory: bool = False,
    ) -> GroupSelectionLookupCache:
        """Build the exact selected-history table outside the serving path.

        This is an offline/training operation.  It intentionally executes the
        frozen selector, grouping, pooling, and selection, but never packs a
        recurrent stream and never runs GDR.  ``cached_forward`` consumes only
        the resulting immutable lookup table.
        """

        if self.selector is None:
            raise RuntimeError(
                "cached-serving-only runtime cannot materialize a selection table"
            )
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
        started = time.perf_counter()
        batch, candidates, history_lengths = self._validate_inputs(
            x=x,
            x_lengths=x_lengths,
            x_offsets=x_offsets,
            num_targets=num_targets,
            history_item_ids=history_item_ids,
            candidate_item_ids=candidate_item_ids,
        )
        self._validate_cache_key_fields(
            user_ids=user_ids,
            history_versions=history_versions,
            batch=batch,
        )
        max_history = int(history_lengths.max())
        history_ids = _jagged_to_dense(
            history_item_ids,
            history_lengths,
            max_history,
        )
        candidate_ids = candidate_item_ids.reshape(batch, candidates)
        history_embeddings = self.selector.lookup(history_ids, dtype=torch.float32)
        fixed_category = self.config.grouping_policy == "fixed_category_prototype_v1"
        candidate_embeddings = (
            self.category_group_prototypes.new_zeros(
                (batch, candidates, self.category_group_prototypes.shape[1])
            )
            if fixed_category
            else self.selector.lookup(candidate_ids, dtype=torch.float32)
        )
        candidate_category_group_ids = (
            self._fixed_category_group_ids(candidate_ids) if fixed_category else None
        )
        grouping = group_candidates(
            candidate_ids,
            candidate_embeddings,
            self.config.candidate_group_count,
            policy=self.config.grouping_policy,
            candidate_category_group_ids=candidate_category_group_ids,
            validate_runtime=self.config.validate_runtime,
        )
        candidate_scores, selector_chunk_count = (
            self._fixed_category_scores(history_embeddings, history_lengths)
            if fixed_category
            else self._selector_scores(
                history_embeddings,
                candidate_embeddings,
                history_lengths,
            )
        )
        selected_masks = None
        if self.config.group_pool == "mask_vote" and not fixed_category:
            candidate_selection = select_candidate_events(
                candidate_scores,
                history_lengths,
                recent_floor=self.config.recent_floor,
                retention_ratio=self.config.retention_ratio,
                return_dense_mask=True,
                validate_runtime=self.config.validate_runtime,
            )
            selected_masks = candidate_selection.dense_mask
        group_scores = (
            candidate_scores
            if fixed_category
            else pool_group_scores(
                candidate_scores,
                grouping.candidate_to_group,
                group_count=grouping.group_count,
                pool=self.config.group_pool,
                selected_masks=selected_masks,
                history_lengths=history_lengths,
                validate_runtime=self.config.validate_runtime,
            )
        )
        selection = select_group_events(
            group_scores,
            history_lengths,
            recent_floor=self.config.recent_floor,
            retention_ratio=self.config.retention_ratio,
            return_dense_mask=False,
            validate_runtime=self.config.validate_runtime,
        )
        cache = GroupSelectionLookupCache.from_outputs(
            user_ids=user_ids,
            history_versions=history_versions,
            candidate_ids=candidate_ids,
            history_lengths=history_lengths,
            grouping=grouping,
            selection=selection,
            version=self.selection_cache_version,
            retention_ratio=self.config.retention_ratio,
            recent_floor=self.config.recent_floor,
            storage_device=storage_device,
            pin_memory=pin_memory,
            validate_runtime=self.config.validate_runtime,
        )
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
        build_ms = (time.perf_counter() - started) * 1000.0
        self.last_cache_build_diagnostics = {
            "excluded_from_serving_latency": True,
            "operation": "group_selection_table_materialization",
            "build_ms": build_ms,
            "entry_count": cache.entry_count,
            "payload_bytes": cache.payload_bytes,
            "bytes_per_entry": cache.bytes_per_entry,
            "storage_device": str(cache.storage_device),
            "pin_memory": bool(pin_memory),
            "candidate_count": candidates,
            "candidate_group_count": grouping.group_count,
            "selector_chunk_count": selector_chunk_count,
            "selector_score_events": int(history_lengths.sum()) * candidates,
            "selected_group_history_tokens": selection.selected_tokens,
            "cache_version": self.selection_cache_version.fingerprint,
            "cache_key": (
                "user_id,history_version,canonical_candidate_set_sha256,"
                "cache_contract_version"
            ),
        }
        return cache

    def global_cached_state_forward(
        self,
        *,
        candidate_x: torch.Tensor,
        candidate_item_ids: torch.Tensor,
        user_ids: torch.Tensor,
        history_versions: torch.Tensor,
        state_cache: GlobalGroupStateCache,
        return_final_states: bool = True,
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        dict[str, Any],
    ]:
        """Serve only occupied global groups from candidate-independent keys."""

        if self.config.grouping_policy != "fixed_category_prototype_v1":
            raise RuntimeError("global-group serving requires fixed-category grouping")
        if not isinstance(state_cache, GlobalGroupStateCache):
            raise TypeError("state_cache must be a GlobalGroupStateCache")
        if state_cache.state_cache_version != self.state_cache_version:
            raise RuntimeError("global-group state cache version mismatch")
        if candidate_x.ndim != 3 or not torch.is_floating_point(candidate_x):
            raise ValueError("candidate_x must be floating point [B,K,D]")
        batch, candidates, _ = candidate_x.shape
        if candidate_item_ids.ndim == 1:
            if candidate_item_ids.numel() != batch * candidates:
                raise ValueError("candidate IDs do not match candidate_x")
            candidate_ids = candidate_item_ids.reshape(batch, candidates)
        elif candidate_item_ids.shape == (batch, candidates):
            candidate_ids = candidate_item_ids
        else:
            raise ValueError("candidate_item_ids must have shape [B*K] or [B,K]")
        if candidate_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("candidate_item_ids must be integer")
        self._validate_cache_key_fields(
            user_ids=user_ids,
            history_versions=history_versions,
            batch=batch,
        )
        profile_synchronized = bool(candidate_x.is_cuda and self.profile_stages)

        def synchronize() -> None:
            if profile_synchronized:
                torch.cuda.synchronize(candidate_x.device)

        synchronize()
        started = time.perf_counter()
        lookup, lookup_ms = self._timed(
            lambda: state_cache.lookup(
                user_ids=user_ids,
                history_versions=history_versions,
                candidate_ids=candidate_ids,
                model_version=state_cache.model_version,
                target_device=candidate_x.device,
                non_blocking=True,
            ),
            synchronize=synchronize,
        )
        packed, packing_ms = self._timed(
            lambda: pack_global_group_queries(
                candidate_x,
                lookup,
                validate_runtime=self.config.validate_runtime,
            ),
            synchronize=synchronize,
        )
        for layer in self.layers:
            layer.profile_stages = profile_synchronized
        executor_output, executor_ms = self._timed(
            lambda: self.packed_executor.forward_from_global_states(
                candidate_x=candidate_x,
                packed=packed,
                states=lookup.states,
                return_final_states=return_final_states,
            ),
            synchronize=synchronize,
        )
        synchronize()
        total_ms = (time.perf_counter() - started) * 1000.0
        queries = executor_output.queries
        final_states = executor_output.final_states
        if queries.shape[:2] != (batch, candidates):
            raise RuntimeError("global-group executor did not restore candidate order")
        if final_states is not None and (
            final_states.shape != lookup.states.shape
            or final_states.dtype != torch.float32
        ):
            raise RuntimeError("global-group executor returned invalid states")
        timings_valid = profile_synchronized or not candidate_x.is_cuda
        diagnostics: dict[str, Any] = {
            "cache_hit": True,
            "cache_miss_count": 0,
            "cache_key": (
                "dataset_id,user_id,history_version,global_group_id,model_version"
            ),
            "dataset_id": state_cache.dataset_id,
            "model_version": state_cache.model_version,
            "candidate_id_in_key": False,
            "candidate_set_hash_in_key": False,
            "batch_size": batch,
            "candidate_count": candidates,
            "configured_global_group_count": state_cache.group_count,
            "occupied_group_state_count": lookup.occupied_state_count,
            "state_cache_entry_count": state_cache.entry_count,
            "state_cache_row_count": state_cache.state_row_count,
            "state_bytes_per_group": state_cache.state_bytes_per_group,
            "cache_materialization_bytes": lookup.materialization_bytes,
            "cache_transferred_bytes": lookup.transferred_bytes,
            "selector_executed_online": False,
            "selection_executed_online": False,
            "history_prefill_executed_online": False,
            "candidate_group_lookup_online": True,
            "occupied_groups_deduplicated": True,
            "write_tokens_per_layer": 0,
            "read_tokens_per_layer": packed.packed_tokens,
            "query_event_gate": 0,
            "state_layout": "N_occupied,layers,H,Dk,Dv",
            "state_dtype": "float32",
            "return_final_states": return_final_states,
            "global_group_cache_lookup_ms": lookup_ms if timings_valid else None,
            "global_group_candidate_packing_ms": packing_ms if timings_valid else None,
            "global_group_candidate_executor_ms": executor_ms if timings_valid else None,
            "global_group_lookup_gdr_total_ms": total_ms if timings_valid else None,
            **dict(executor_output.auxiliary),
        }
        self.last_grouping = None
        self.last_selection = None
        self.last_packed = None
        self.last_final_states = final_states
        self.last_diagnostics = diagnostics
        return queries, final_states, dict(diagnostics)

    def cached_state_forward(
        self,
        *,
        candidate_x: torch.Tensor,
        candidate_item_ids: torch.Tensor,
        user_ids: torch.Tensor,
        history_versions: torch.Tensor,
        history_lengths: torch.Tensor,
        state_cache: GroupStateLookupCache,
        return_final_states: bool = True,
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        GroupingOutput,
        dict[str, Any],
    ]:
        """Serve candidate-only reads from persisted selected-history states.

        The timed serving core is exact-key lookup/materialization, candidate
        grouping reconstruction, candidate-only packing, and one initialized
        GDR call per layer.  No history feature tensor, selector, selection,
        or history-prefill execution is accepted by this API.
        """

        if not isinstance(state_cache, GroupStateLookupCache):
            raise TypeError("state_cache must be a GroupStateLookupCache")
        if candidate_x.ndim != 3 or not torch.is_floating_point(candidate_x):
            raise ValueError("candidate_x must be floating point [B,K,D]")
        batch, candidates, _ = candidate_x.shape
        if batch < 1 or candidates < 1:
            raise ValueError("candidate-only serving requires nonempty B and K")
        if candidate_item_ids.ndim == 1:
            if candidate_item_ids.numel() != batch * candidates:
                raise ValueError("candidate IDs do not match candidate_x")
            candidate_ids = candidate_item_ids.reshape(batch, candidates)
        elif candidate_item_ids.shape == (batch, candidates):
            candidate_ids = candidate_item_ids
        else:
            raise ValueError("candidate_item_ids must have shape [B*K] or [B,K]")
        if candidate_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("candidate_item_ids must be integer")
        if history_lengths.shape != (batch,) or history_lengths.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("history_lengths must be an integer tensor [B]")
        self._validate_cache_key_fields(
            user_ids=user_ids,
            history_versions=history_versions,
            batch=batch,
        )
        if state_cache.version != self.state_cache_version:
            raise RuntimeError("state cache does not match this frozen serving model")

        profile_synchronized = bool(candidate_x.is_cuda and self.profile_stages)

        def synchronize() -> None:
            if profile_synchronized:
                torch.cuda.synchronize(candidate_x.device)

        synchronize()
        serving_started = time.perf_counter()
        lookup, lookup_ms = self._timed(
            lambda: state_cache.lookup(
                user_ids=user_ids,
                history_versions=history_versions,
                candidate_ids=candidate_ids,
                history_lengths=history_lengths,
                version=self.state_cache_version,
                target_device=candidate_x.device,
                validate_runtime=self.config.validate_runtime,
                non_blocking=True,
            ),
            synchronize=synchronize,
        )
        grouping = lookup.grouping
        packed, packing_ms = self._timed(
            lambda: pack_group_queries(
                candidate_x,
                grouping,
                validate_runtime=self.config.validate_runtime,
            ),
            synchronize=synchronize,
        )
        for layer in self.layers:
            layer.profile_stages = profile_synchronized
        executor_output, executor_ms = self._timed(
            lambda: self.packed_executor.forward_from_states(
                candidate_x=candidate_x,
                packed=packed,
                states=lookup.states,
                return_final_states=return_final_states,
            ),
            synchronize=synchronize,
        )
        synchronize()
        serving_total_ms = (time.perf_counter() - serving_started) * 1000.0
        queries = executor_output.queries
        final_states = executor_output.final_states
        if queries.shape[:2] != (batch, candidates):
            raise RuntimeError("state-cache executor did not restore candidate order")
        if final_states is not None and (
            final_states.shape != lookup.states.shape
            or final_states.dtype != torch.float32
        ):
            raise RuntimeError("state-cache executor returned invalid FP32 states")

        layer_projection_ms: Optional[float] = None
        recurrent_ms: Optional[float] = None
        candidate_output_ms: Optional[float] = None
        if profile_synchronized:
            layer_times = [layer.last_stage_times_ms() for layer in self.layers]
            layer_projection_ms = sum(
                stage.get("projection_ms", 0.0) for stage in layer_times
            )
            recurrent_ms = sum(
                stage.get("prefill_ms", 0.0) for stage in layer_times
            )
            candidate_output_ms = sum(
                stage.get("output_ms", 0.0) for stage in layer_times
            )
        timings_valid = profile_synchronized or not candidate_x.is_cuda
        stage_times_ms: dict[str, Optional[float]] = {
            "state_cache_lookup_materialization_ms": (
                lookup_ms if timings_valid else None
            ),
            "candidate_packing_ms": packing_ms if timings_valid else None,
            "candidate_only_executor_ms": executor_ms if timings_valid else None,
            "candidate_projection_ms": layer_projection_ms,
            "candidate_recurrent_read_ms": recurrent_ms,
            "candidate_output_ms": candidate_output_ms,
            "lookup_candidate_gdr_total_ms": (
                serving_total_ms if timings_valid else None
            ),
        }
        diagnostics: dict[str, Any] = {
            "cache_hit": True,
            "cache_hit_count": lookup.hit_count,
            "cache_miss_count": 0,
            "state_cache_version": self.state_cache_version.fingerprint,
            "model_sha256": self.serving_model_sha256,
            "cache_key": (
                "user_id,history_version,canonical_candidate_set_sha256,"
                "state_cache_contract_version"
            ),
            "cache_entry_count": state_cache.entry_count,
            "cache_payload_bytes": state_cache.payload_bytes,
            "cache_bytes_per_entry": state_cache.bytes_per_entry,
            "state_bytes_per_entry": state_cache.state_bytes_per_entry,
            "cache_storage_device": str(state_cache.storage_device),
            "cache_materialization_bytes": lookup.materialization_bytes,
            "cache_transferred_bytes": lookup.transferred_bytes,
            "selection_metadata_materialization_bytes": (
                lookup.selection_materialization_bytes
            ),
            "state_materialization_bytes": lookup.state_materialization_bytes,
            "table_build_excluded_from_serving_latency": True,
            "selector_executed_online": False,
            "grouping_executed_online": False,
            "pooling_executed_online": False,
            "selection_executed_online": False,
            "history_fetch_executed_online": False,
            "history_projection_executed_online": False,
            "history_prefill_executed_online": False,
            "serving_measurement_scope": (
                "composite_key_lookup+state_materialization+candidate_only_pack+"
                "initialized_candidate_gdr+candidate_query_restore"
            ),
            "batch_size": batch,
            "candidate_count": candidates,
            "candidate_group_count": grouping.group_count,
            "online_history_tokens": 0,
            "packed_tokens_per_layer": packed.packed_tokens,
            "write_tokens_per_layer": 0,
            "read_tokens_per_layer": packed.read_tokens,
            "logical_encodes": packed.sequence_count,
            "logical_states": packed.sequence_count * len(self.layers),
            "physical_gdr_calls": executor_output.physical_gdr_calls,
            "query_event_gate": 0,
            "candidate_order": "original_input_order",
            "state_layout": "B,G,layers,H,Dk,Dv",
            "state_dtype": "float32",
            "return_final_states": return_final_states,
            "component_timings_synchronized": profile_synchronized,
            "stage_times_ms": stage_times_ms,
            "actual_layer_backends": [layer.kernel_backend for layer in self.layers],
            **dict(executor_output.auxiliary),
        }
        diagnostics.update(stage_times_ms)
        self.last_grouping = grouping
        self.last_selection = None
        self.last_packed = packed
        self.last_final_states = final_states
        self.last_diagnostics = diagnostics
        return queries, final_states, grouping, dict(diagnostics)

    def cached_forward(
        self,
        *,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        num_targets: torch.Tensor,
        user_ids: torch.Tensor,
        history_versions: torch.Tensor,
        candidate_item_ids: torch.Tensor,
        selection_cache: GroupSelectionLookupCache,
        return_final_states: bool = True,
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        GroupingOutput,
        GroupSelectionOutput,
        dict[str, Any],
    ]:
        """Serve an exact cache hit as lookup + packing + packed GDR.

        The selector and table construction are deliberately absent.  Misses,
        stale history versions, contract mismatches, and duplicate candidate
        IDs fail closed in ``GroupSelectionLookupCache.lookup``; callers may
        explicitly route those requests to ``forward_group_shared``.
        """

        profile_synchronized = bool(x.is_cuda and self.profile_stages)

        def synchronize() -> None:
            if profile_synchronized:
                torch.cuda.synchronize(x.device)

        validated, input_preparation_ms = self._timed(
            lambda: self._validate_cached_inputs(
                x=x,
                x_lengths=x_lengths,
                x_offsets=x_offsets,
                num_targets=num_targets,
                user_ids=user_ids,
                history_versions=history_versions,
                candidate_item_ids=candidate_item_ids,
            ),
            synchronize=synchronize,
        )
        batch, candidates, history_lengths = validated
        candidate_ids = candidate_item_ids.reshape(batch, candidates)

        synchronize()
        serving_started = time.perf_counter()
        lookup, lookup_ms = self._timed(
            lambda: selection_cache.lookup(
                user_ids=user_ids,
                history_versions=history_versions,
                candidate_ids=candidate_ids,
                history_lengths=history_lengths,
                version=self.selection_cache_version,
                target_device=x.device,
                validate_runtime=self.config.validate_runtime,
                non_blocking=True,
            ),
            synchronize=synchronize,
        )
        grouping = lookup.grouping
        selection = lookup.selection
        packed, packing_ms = self._timed(
            lambda: pack_group_streams(
                x=x,
                x_lengths=x_lengths,
                x_offsets=x_offsets,
                num_targets=num_targets,
                grouping=grouping,
                selection=selection,
                contextual_seq_len=self.contextual_seq_len,
                validate_runtime=self.config.validate_runtime,
            ),
            synchronize=synchronize,
        )
        for layer in self.layers:
            layer.profile_stages = profile_synchronized
        executor_output, executor_ms = self._timed(
            lambda: self.packed_executor(
                source_x=x,
                packed=packed,
                return_final_states=return_final_states,
            ),
            synchronize=synchronize,
        )
        synchronize()
        serving_total_ms = (time.perf_counter() - serving_started) * 1000.0
        if executor_output.physical_gdr_calls != len(self.layers):
            raise RuntimeError("cached group executor did not issue one GDR call per layer")

        queries = executor_output.queries
        final_states = executor_output.final_states
        if queries.shape[:2] != (batch, candidates):
            raise RuntimeError("cached group executor did not restore candidate order")
        if final_states is not None and (
            final_states.shape[:3]
            != (batch, grouping.group_count, len(self.layers))
            or final_states.dtype != torch.float32
        ):
            raise RuntimeError("cached group executor returned an invalid FP32 state layout")

        layer_projection_ms: Optional[float] = None
        recurrent_ms: Optional[float] = None
        candidate_output_ms: Optional[float] = None
        if profile_synchronized:
            layer_times = [layer.last_stage_times_ms() for layer in self.layers]
            layer_projection_ms = sum(
                stage.get("projection_ms", 0.0) for stage in layer_times
            )
            recurrent_ms = sum(stage.get("prefill_ms", 0.0) for stage in layer_times)
            candidate_output_ms = sum(
                stage.get("output_ms", 0.0) for stage in layer_times
            )

        timings_valid = profile_synchronized or not x.is_cuda
        stage_times_ms: dict[str, Optional[float]] = {
            "input_preparation_ms": input_preparation_ms if timings_valid else None,
            "cache_lookup_materialization_ms": lookup_ms if timings_valid else None,
            "packing_ms": packing_ms if timings_valid else None,
            "executor_ms": executor_ms if timings_valid else None,
            "layer_internal_projection_ms": layer_projection_ms,
            "recurrent_ms": recurrent_ms,
            "candidate_output_ms": candidate_output_ms,
            "lookup_packing_gdr_total_ms": (
                serving_total_ms if timings_valid else None
            ),
        }
        diagnostics: dict[str, Any] = {
            "cache_hit": True,
            "cache_hit_count": lookup.hit_count,
            "cache_miss_count": 0,
            "cache_version": self.selection_cache_version.fingerprint,
            "cache_key": (
                "user_id,history_version,canonical_candidate_set_sha256,"
                "cache_contract_version"
            ),
            "cache_entry_count": selection_cache.entry_count,
            "cache_payload_bytes": selection_cache.payload_bytes,
            "cache_bytes_per_entry": selection_cache.bytes_per_entry,
            "cache_storage_device": str(selection_cache.storage_device),
            "cache_materialization_bytes": lookup.materialization_bytes,
            "table_build_excluded_from_serving_latency": True,
            "selector_executed_online": False,
            "grouping_executed_online": False,
            "pooling_executed_online": False,
            "selection_executed_online": False,
            "serving_measurement_scope": (
                "composite_key_lookup+payload_materialization+packing+packed_gdr+"
                "candidate_query_restore"
            ),
            "batch_size": batch,
            "candidate_count": candidates,
            "candidate_group_count": self.config.candidate_group_count,
            "effective_group_count": grouping.group_count,
            "grouping_policy": self.config.grouping_policy,
            "group_pool": self.config.group_pool,
            "retention_ratio": self.config.retention_ratio,
            "recent_floor": self.config.recent_floor,
            "input_history_tokens": int(history_lengths.sum()),
            "selected_group_history_tokens": selection.selected_tokens,
            "packed_tokens_per_layer": packed.packed_tokens,
            "write_tokens_per_layer": packed.write_tokens,
            "read_tokens_per_layer": packed.read_tokens,
            "logical_encodes": packed.sequence_count,
            "logical_states": packed.sequence_count * len(self.layers),
            "physical_gdr_calls": executor_output.physical_gdr_calls,
            "all_groups_single_packed_submission": True,
            "query_event_gate": 0,
            "candidate_order": "original_input_order",
            "state_layout": "B,G,layers,H,Dk,Dv",
            "state_dtype": "float32",
            "return_final_states": return_final_states,
            "component_timings_synchronized": profile_synchronized,
            "stage_times_ms": stage_times_ms,
            "actual_layer_backends": [layer.kernel_backend for layer in self.layers],
            **dict(executor_output.auxiliary),
        }
        diagnostics.update(stage_times_ms)
        self.last_grouping = grouping
        self.last_selection = selection
        self.last_packed = packed
        self.last_final_states = final_states
        self.last_diagnostics = diagnostics
        return queries, final_states, grouping, selection, dict(diagnostics)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise RuntimeError(
            "DLRMv3GroupSharedSTUStack requires forward_group_shared(); the "
            "normal STU forward would silently discard group-aligned state"
        )

__all__ = [
    "DLRMv3GroupSharedSTUStack",
    "GroupSharedDeltaRecConfig",
]

