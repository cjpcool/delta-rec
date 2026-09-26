# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import hashlib

import json

import time

from pathlib import Path

from typing import Any, Mapping, Optional

import torch

import torch.nn.functional as F

from deltarec.adaptors.kuai.modules.dlrm_delta_rec import PRODUCTION_HASH_SEED, DLRMv3GDRSTULayer

from deltarec.adaptors.kuai.research.modeling.sequential.deltarec.selector import CandidateMLPSelector

from deltarec.adaptors.kuai.research.modeling.sequential.deltarec.projections import HSTUFixedWidthLayout

from deltarec.adaptors.kuai.research.modeling.sequential.deltarec.pc_state_cache import PCStateCache, PCStateCacheVersion, build_pc_state_cache_version

from deltarec.adaptors.kuai.modules.stu import STU, STUStack

from deltarec.adaptors.kuai.research.modeling.sequential.deltarec.selection import merge_candidate_selections, select_candidate_events, validate_retention_ratio

from deltarec.adaptors.kuai.research.modeling.sequential.deltarec.types import SelectionOutput

from deltarec.adaptors.kuai.research.modeling.sequential.selective_gdr import GDRKernelInput

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SELECTOR_CHECKPOINT = (
    _PROJECT_ROOT
    / "deltarec/data/checkpoints/selector_mlp.pt"
)

DEFAULT_SELECTOR_EMBEDDING_TABLE = (
    _PROJECT_ROOT
    / "deltarec/data/artifacts/selector/embedding_table.pt"
)

SUPPORTED_CANDIDATE_CHUNKS = (1, 8, 32, 100)

SELECTOR_ID_MAPPING_POLICY = "direct_embedding_row_v1"

PC_BUDGET_POLICY = "candidate_topk_ceil_ratio_recent_floor_v2"

def _module_state_sha256(module: torch.nn.Module) -> str:
    """Hash exact frozen serving tensors without a dtype conversion."""

    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        contiguous = value.detach().to(device="cpu").contiguous()
        digest.update(
            f"{name}:{tuple(contiguous.shape)}:{contiguous.dtype}".encode("utf-8")
        )
        digest.update(contiguous.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

def direct_selector_item_ids(
    item_ids: torch.Tensor,
    *,
    selector_num_items: int,
) -> torch.Tensor:
    """Validate and preserve the frozen ML-20M selector row identity.

    The frozen item table was trained with dense ML-20M item IDs as embedding
    row numbers.  Hashing an unrelated production vocabulary into that table
    changes selector semantics and introduces silent collisions, so this
    production boundary fails closed for unavailable IDs.
    """

    if selector_num_items < 1:
        raise ValueError("selector_num_items must be positive")
    if item_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("selector item IDs must be integer tensors")
    if item_ids.numel():
        minimum = int(item_ids.min())
        maximum = int(item_ids.max())
        if minimum < 0 or maximum > selector_num_items:
            raise ValueError(
                "item ID is unavailable in the frozen selector table: "
                f"observed range [{minimum}, {maximum}], supported range "
                f"[0, {selector_num_items}]"
            )
    return item_ids.to(torch.int64)

class _FrozenPCMLP(torch.nn.Module):
    """Small self-contained loader for the frozen selector artifact."""

    def __init__(self, embedding_dim: int, hidden_dim: int, utility_scale: float) -> None:
        super().__init__()
        self.input = torch.nn.Linear(3 * embedding_dim, hidden_dim)
        self.output = torch.nn.Linear(hidden_dim, 1)
        self.register_buffer("utility_scale", torch.tensor(float(utility_scale)))

    def forward(
        self,
        history_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if history_embeddings.ndim != 3 or candidate_embeddings.ndim != 3:
            raise ValueError("selector embeddings must have shapes [B,L,D] and [B,K,D]")
        batch, history, dim = history_embeddings.shape
        if candidate_embeddings.shape[0] != batch or candidate_embeddings.shape[-1] != dim:
            raise ValueError("selector history and candidate dimensions disagree")
        candidates = candidate_embeddings[:, :, None, :]
        events = history_embeddings[:, None, :, :]
        common_shape = (
            history_embeddings.shape[0],
            candidate_embeddings.shape[1],
            history_embeddings.shape[1],
            history_embeddings.shape[2],
        )
        features = torch.cat(
            (
                events.expand(common_shape),
                candidates.expand(common_shape),
                events * candidates,
            ),
            dim=-1,
        )
        scores = self.output(F.silu(self.input(features))).squeeze(-1)
        return scores * self.utility_scale.to(scores.dtype)

def _load_tensor_artifact(payload: Any, *, name: str) -> torch.Tensor:
    if isinstance(payload, torch.Tensor):
        return payload
    if isinstance(payload, Mapping) and isinstance(payload.get("weight"), torch.Tensor):
        return payload["weight"]
    raise ValueError(f"{name} must contain a tensor or a 'weight' tensor")

class FrozenCandidateSelector(torch.nn.Module):
    """Frozen PC-MLP plus its normalized selector embedding table."""

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_SELECTOR_CHECKPOINT,
        embedding_table: str | Path = DEFAULT_SELECTOR_EMBEDDING_TABLE,
        *,
        map_location: str | torch.device = "cpu",
        seed: int = PRODUCTION_HASH_SEED,
    ) -> None:
        super().__init__()
        checkpoint_path = Path(checkpoint)
        table_path = Path(embedding_table)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"frozen selector checkpoint does not exist: {checkpoint_path}")
        if not table_path.is_file():
            raise FileNotFoundError(f"frozen selector embedding table does not exist: {table_path}")
        checkpoint_payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        if not isinstance(checkpoint_payload, Mapping):
            raise ValueError("frozen selector checkpoint must be a mapping")
        model_state = checkpoint_payload.get("model")
        if not isinstance(model_state, Mapping):
            raise ValueError("frozen selector checkpoint is missing model weights")
        table_payload = torch.load(table_path, map_location=map_location, weights_only=False)
        table = _load_tensor_artifact(table_payload, name="selector embedding table").float().contiguous()
        if table.ndim != 2 or table.shape[0] < 2 or table.shape[1] < 1:
            raise ValueError("selector embedding table must have shape [N,D] with N >= 2")
        embedding_dim = int(checkpoint_payload.get("embedding_dim", table.shape[1]))
        if embedding_dim != table.shape[1]:
            raise ValueError("selector checkpoint and embedding table dimensions differ")
        hidden_dim = int(checkpoint_payload.get("hidden_dim", 0))
        if hidden_dim < 1:
            raise ValueError("selector checkpoint is missing hidden_dim")
        self.mlp = _FrozenPCMLP(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            utility_scale=float(checkpoint_payload.get("utility_scale", 1.0)),
        )
        load_result = self.mlp.load_state_dict(dict(model_state), strict=False)
        unexpected = set(load_result.unexpected_keys)
        missing = set(load_result.missing_keys)
        if unexpected or missing - {"utility_scale"}:
            raise ValueError(
                "frozen selector checkpoint has incompatible weights: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        self.register_buffer("embedding_table", table)
        rms = table_payload.get("rms") if isinstance(table_payload, Mapping) else None
        self.embedding_rms = float(rms) if rms is not None else float(table[1:].square().mean().sqrt())
        if not self.embedding_rms > 0:
            raise ValueError("selector embedding RMS must be positive")
        if seed < 0:
            raise ValueError("selector hash seed must be nonnegative")
        self.seed = int(seed)
        self.exact_selector = CandidateMLPSelector.from_small_candidate_mlp(
            self.mlp,
            embedding_rms=self.embedding_rms,
            utility_scale=self.mlp.utility_scale,
            validate_runtime=False,
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()
        mapping_contract = {
            "schema_version": 1,
            "policy": SELECTOR_ID_MAPPING_POLICY,
            "source_dataset": "movielens-20m",
            "padding_id": 0,
            "minimum_item_id": 1,
            "maximum_item_id": int(table.shape[0] - 1),
            "embedding_rows": int(table.shape[0]),
            "source_cache_sha256": (
                table_payload.get("source_cache_sha256")
                if isinstance(table_payload, Mapping)
                else None
            ),
        }
        mapping_bytes = json.dumps(
            mapping_contract,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        self.mapping_contract = mapping_contract
        self.mapping_contract_sha256 = hashlib.sha256(mapping_bytes).hexdigest()
        self.artifact_sha256 = {
            "selector_checkpoint": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "embedding_table": hashlib.sha256(table_path.read_bytes()).hexdigest(),
            "id_mapping_contract": self.mapping_contract_sha256,
        }

    @property
    def num_items(self) -> int:
        # Row zero is reserved for padding; nonzero IDs retain their trained
        # ML-20M embedding-row identity.
        return int(self.embedding_table.shape[0] - 1)

    @property
    def embedding_dim(self) -> int:
        return int(self.embedding_table.shape[1])

    def lookup(self, item_ids: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        mapped = direct_selector_item_ids(
            item_ids,
            selector_num_items=self.num_items,
        )
        table = self.embedding_table.to(device=item_ids.device, dtype=dtype)
        return F.embedding(mapped, table)

    def score(
        self,
        history_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        history_lengths: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        history = self.lookup(history_ids, dtype=dtype)
        candidates = self.lookup(candidate_ids, dtype=dtype)
        return self.exact_selector.exact_packed_scores(
            history,
            candidates,
            history_lengths,
            candidate_chunk_size=None,
        )

class SharedEmbeddingCandidateSelector(torch.nn.Module):
    """Candidate-aware CWI MLP over the model's exact embedding outputs.

    This selector deliberately owns no item embedding table.  Kuai DLRMv3
    uses one TorchRec ``video_id`` table for both ``video_id`` and
    ``item_video_id``; the tensors produced by that table are supplied on each
    forward so selector gradients flow back to the same sharded parameter.
    """

    requires_shared_embeddings = True

    def __init__(
        self,
        *,
        embedding_dim: int,
        hidden_dim: Optional[int] = None,
        seed: int = PRODUCTION_HASH_SEED,
        utility_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        if hidden_dim is None:
            hidden_dim = max(8, embedding_dim)
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if seed < 0:
            raise ValueError("selector seed must be nonnegative")
        self.seed = int(seed)
        self.mlp = _FrozenPCMLP(
            embedding_dim=int(embedding_dim),
            hidden_dim=int(hidden_dim),
            utility_scale=float(utility_scale),
        )
        self.mapping_contract = {
            "schema_version": 1,
            "policy": "exact_shared_embedding_tensor_v1",
            "embedding_owner": "dlrmv3_embedding_collection.video_id",
            "embedding_dim": int(embedding_dim),
            "contains_embedding_table": False,
        }
        mapping_bytes = json.dumps(
            self.mapping_contract,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        self.mapping_contract_sha256 = hashlib.sha256(mapping_bytes).hexdigest()
        self.artifact_sha256 = {
            "embedding_binding": self.mapping_contract_sha256,
            "selector_state": _module_state_sha256(self.mlp),
        }

    @property
    def embedding_dim(self) -> int:
        return int(self.mlp.input.in_features // 3)

    def score_embeddings(
        self,
        history_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        history_lengths: torch.Tensor,
        *,
        dtype: torch.dtype,
        candidate_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        if history_embeddings.shape[-1] != self.embedding_dim or (
            candidate_embeddings.shape[-1] != self.embedding_dim
        ):
            raise ValueError("shared selector embedding width mismatch")
        if history_embeddings.device != candidate_embeddings.device:
            raise ValueError("shared selector embeddings must share a device")
        if history_lengths.shape != (history_embeddings.shape[0],) or bool(
            (history_lengths < 1).any()
        ) or bool((history_lengths > history_embeddings.shape[1]).any()):
            raise ValueError("shared selector history lengths are invalid")
        candidates = int(candidate_embeddings.shape[1])
        chunk_size = (
            candidates
            if candidate_chunk_size is None
            else min(int(candidate_chunk_size), candidates)
        )
        if chunk_size < 1:
            raise ValueError("candidate_chunk_size must be positive")
        history = history_embeddings.to(dtype=dtype)
        candidate = candidate_embeddings.to(dtype=dtype)
        chunks = [
            self.mlp(history, candidate[:, start : start + chunk_size])
            for start in range(0, candidates, chunk_size)
        ]
        scores = torch.cat(chunks, dim=1)
        positions = torch.arange(history.shape[1], device=history.device)
        invalid = positions[None, None, :] >= history_lengths[:, None, None]
        return scores.masked_fill(invalid, float("-inf"))

    def score(
        self,
        history_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        history_lengths: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        del history_ids, candidate_ids, history_lengths, dtype
        raise RuntimeError(
            "shared-embedding selector requires the exact DLRMv3 lookup tensors"
        )

def _jagged_to_dense(
    values: torch.Tensor,
    lengths: torch.Tensor,
    max_length: int,
    *,
    fill_value: int = 0,
) -> torch.Tensor:
    """Convert a flat jagged integer field to a padded [B, max_length] tensor."""

    if values.ndim != 1 or lengths.ndim != 1:
        raise ValueError("jagged values and lengths must be vectors")
    if int(lengths.sum()) != values.numel():
        raise ValueError("jagged values and lengths disagree")
    batch = int(lengths.numel())
    output = values.new_full((batch, max_length), fill_value)
    if values.numel() == 0:
        return output
    rows = torch.repeat_interleave(
        torch.arange(batch, device=values.device, dtype=torch.int64), lengths.to(torch.int64)
    )
    offsets = torch.cat((lengths.new_zeros(1), lengths.to(torch.int64).cumsum(0)))
    positions = torch.arange(values.numel(), device=values.device) - torch.repeat_interleave(
        offsets[:-1], lengths.to(torch.int64)
    )
    output[rows, positions] = values
    return output

def _jagged_embeddings_to_dense(
    values: torch.Tensor,
    lengths: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    """Convert flat shared embedding rows to padded ``[B,L,D]`` form."""

    if values.ndim != 2 or not torch.is_floating_point(values):
        raise ValueError("shared selector values must be floating point [tokens,D]")
    if lengths.ndim != 1 or lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("shared selector lengths must be an integer vector")
    if int(lengths.sum()) != values.shape[0]:
        raise ValueError("shared selector values and lengths disagree")
    if max_length < 1:
        raise ValueError("shared selector dense width must be positive")
    batch = int(lengths.numel())
    output = values.new_zeros((batch, max_length, values.shape[1]))
    rows = torch.repeat_interleave(
        torch.arange(batch, device=values.device, dtype=torch.int64),
        lengths.to(torch.int64),
    )
    offsets = torch.cat((lengths.new_zeros(1), lengths.to(torch.int64).cumsum(0)))
    positions = torch.arange(values.shape[0], device=values.device) - torch.repeat_interleave(
        offsets[:-1], lengths.to(torch.int64)
    )
    output[rows, positions] = values
    return output

class DLRMv3CandidateSymmetricSTUStack(STU):
    """Candidate-symmetric stack for the preregistered 25%/50% writes."""

    def __init__(
        self,
        original: STUStack,
        *,
        selector: FrozenCandidateSelector,
        seed: int = PRODUCTION_HASH_SEED,
        kernel_backend: str = "triton",
        recent_floor: int = 32,
        retention_ratio: float = 0.50,
        candidate_chunk_size: int = 100,
        contextual_seq_len: int = 0,
    ) -> None:
        super().__init__(is_inference=original.is_inference)
        if kernel_backend not in ("reference", "fla", "triton"):
            raise ValueError("kernel_backend must be 'reference', 'fla', or 'triton'")
        if recent_floor < 0:
            raise ValueError("recent_floor must be nonnegative")
        retention_ratio = validate_retention_ratio(retention_ratio)
        if candidate_chunk_size not in SUPPORTED_CANDIDATE_CHUNKS:
            raise ValueError(f"candidate_chunk_size must be one of {SUPPORTED_CANDIDATE_CHUNKS}")
        self.selector = selector
        if selector.seed != seed:
            raise ValueError("selector hash seed and production backend seed differ")
        self.recent_floor = int(recent_floor)
        self.candidate_chunk_size = int(candidate_chunk_size)
        if contextual_seq_len < 0:
            raise ValueError("contextual_seq_len must be nonnegative")
        self.contextual_seq_len = int(contextual_seq_len)
        self.layers = torch.nn.ModuleList(
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
        base_parameter = next(original.parameters(), None)
        if base_parameter is not None and base_parameter.device.type != "meta":
            # The selector and GDR recurrence parameters are new modules. Move
            # them to the stock STU device before the first packed projection.
            self.to(base_parameter.device)
        initialization = hashlib.sha256(b"dlrmv3-candidate-symmetric-gdr-v1")
        for index, layer in enumerate(self.layers):
            for name in ("gdr_log_decay_scale", "gdr_decay_bias", "gdr_gate_weight"):
                value = getattr(layer, name).detach().cpu().contiguous()
                initialization.update(f"{index}:{name}:{tuple(value.shape)}".encode())
                initialization.update(value.numpy().tobytes())
        self.gdr_initialization_hash = initialization.hexdigest()
        self.selector_artifact_sha256 = dict(selector.artifact_sha256)
        self.last_selection: Optional[SelectionOutput] = None
        self.last_final_states: Optional[torch.Tensor] = None
        self.last_diagnostics: dict[str, Any] = {}
        self.last_pc_cache_build_diagnostics: dict[str, Any] = {}
        self.profile_stages = False
        self.executor = "pack_before_projection"
        self.projection_policy = "raw_history_pack"
        self.gdr_backend = kernel_backend
        self.retention_ratio = retention_ratio
        self.projection_dtype = "float32"
        self.state_dtype = "float32"
        self._frozen_backend_active = False
        self._frozen_backend_manifest: dict[str, Any] = {}
        self._pc_cache_version: Optional[PCStateCacheVersion] = None
        self._pc_cache_semantic_token: Optional[tuple[Any, ...]] = None
        self.refresh_pc_state_cache_version()

    def _pc_semantic_token(self) -> tuple[Any, ...]:
        layer_tensor_versions = tuple(
            (name, int(value._version), str(value.dtype), tuple(value.shape))
            for name, value in sorted(self.layers.state_dict().items())
        )
        selector_tensor_versions = tuple(
            (name, int(value._version), str(value.dtype), tuple(value.shape))
            for name, value in sorted(self.selector.state_dict().items())
        )
        return (
            layer_tensor_versions,
            selector_tensor_versions,
            json.dumps(self.selector_artifact_sha256, sort_keys=True),
            self.retention_ratio,
            self.recent_floor,
            self.contextual_seq_len,
            self.projection_dtype,
            self.state_dtype,
            self.gdr_backend,
        )

    def refresh_pc_state_cache_version(self) -> PCStateCacheVersion:
        """Freeze a new serving version after checkpoint/configuration changes."""

        first = self.layers[0].base
        selector_artifacts = {
            **self.selector_artifact_sha256,
            "runtime_selector_state": _module_state_sha256(self.selector),
        }
        version = build_pc_state_cache_version(
            model_sha256=_module_state_sha256(self.layers),
            selector_artifacts=selector_artifacts,
            budget_policy=PC_BUDGET_POLICY,
            write_ratio=self.retention_ratio,
            recent_floor=self.recent_floor,
            layer_count=len(self.layers),
            num_heads=int(first._num_heads),
            key_dim=int(first._attention_dim),
            value_dim=int(first._hidden_dim),
            contextual_seq_len=self.contextual_seq_len,
            projection_dtype=self.projection_dtype,
            recurrence_backend=self.gdr_backend,
        )
        self._pc_cache_version = version
        self._pc_cache_semantic_token = self._pc_semantic_token()
        return version

    def current_pc_state_cache_version(self) -> PCStateCacheVersion:
        """Return the frozen version, failing closed after an in-process mutation."""

        if (
            self._pc_cache_version is None
            or self._pc_cache_semantic_token != self._pc_semantic_token()
        ):
            raise RuntimeError(
                "current model/selector/budget does not match the frozen PC cache version; "
                "refresh only before materializing a new cache"
            )
        return self._pc_cache_version

    @property
    def pc_state_shape(self) -> tuple[int, int, int, int]:
        first = self.layers[0].base
        return (
            len(self.layers),
            int(first._num_heads),
            int(first._attention_dim),
            int(first._hidden_dim),
        )

    @staticmethod
    def _packed_positions(lengths: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return torch.arange(int(offsets[-1]), device=lengths.device) - torch.repeat_interleave(
            offsets[:-1], lengths.to(torch.int64)
        )

    def activate_subplan5b_frozen_backend(self, manifest: Mapping[str, Any]) -> "DLRMv3CandidateSymmetricSTUStack":
        """Activate the exact Subplan 5B ``project_once_then_gather`` contract."""

        required = {
            "executor": "project_once_then_gather",
            "projection_policy": "project_full_history_once",
            "candidate_chunk_size": 32,
            "gdr_backend": "fla",
            "recent_floor": 32,
            "projection_dtype": "bfloat16",
            "state_dtype": "float32",
        }
        manifest_ratio = validate_retention_ratio(manifest.get("retention_ratio"))
        mismatches = {
            key: (manifest.get(key), value)
            for key, value in required.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"invalid frozen Subplan 5B manifest: {mismatches}")
        runtime_mismatches = {}
        if self.candidate_chunk_size != required["candidate_chunk_size"]:
            runtime_mismatches["candidate_chunk_size"] = (
                self.candidate_chunk_size,
                required["candidate_chunk_size"],
            )
        if self.recent_floor != required["recent_floor"]:
            runtime_mismatches["recent_floor"] = (
                self.recent_floor,
                required["recent_floor"],
            )
        if runtime_mismatches:
            raise RuntimeError(
                f"production stack does not implement the frozen Subplan 5B runtime: {runtime_mismatches}"
            )
        # Activation must change the executable kernel, not only the artifact
        # labels.  The exact lineage replay will therefore fail closed on CPU
        # or when FLA is unavailable.
        for layer in self.layers:
            layer.kernel_backend = required["gdr_backend"]
        self.executor = str(manifest["executor"])
        self.projection_policy = str(manifest["projection_policy"])
        self.gdr_backend = str(manifest["gdr_backend"])
        self.retention_ratio = manifest_ratio
        self.projection_dtype = str(manifest["projection_dtype"])
        self.state_dtype = str(manifest["state_dtype"])
        self._frozen_backend_active = True
        self._frozen_backend_manifest = dict(manifest)
        self.refresh_pc_state_cache_version()
        return self

    def prepare_project_once(
        self,
        *,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        num_targets: torch.Tensor,
        history_item_ids: torch.Tensor,
        candidate_item_ids: torch.Tensor,
        selection: Optional[SelectionOutput] = None,
    ) -> dict[str, Any]:
        """Project the valid history prefix once and record exact frozen-backend counters."""

        context = int(getattr(self, "contextual_seq_len", 0))
        history_lengths = x_lengths.to(torch.int64) - context - num_targets.to(torch.int64)
        if bool((history_lengths < 1).any()):
            raise ValueError("candidate-symmetric production streams require nonempty histories")
        input_history_tokens = int(history_lengths.sum().item())
        projected_history_tokens = input_history_tokens
        if selection is None:
            candidate_count = int(num_targets.max().item())
            history_ids = _jagged_to_dense(history_item_ids, history_lengths, int(history_lengths.max()))
            candidate_ids = _jagged_to_dense(candidate_item_ids, num_targets, candidate_count)
            selection = self._selector_selection(
                history_ids=history_ids,
                candidate_ids=candidate_ids,
                history_lengths=history_lengths,
                selector_dtype=torch.float32,
            )
        gathered_projection_tokens = int(selection.selected_tokens) + int(
            selection.batch_size
            * selection.candidate_count
            * (context + 1)
        )
        # This factor measures source history projection, not the unavoidable
        # candidate-specific gather or later-layer work.
        duplication = float(projected_history_tokens / max(input_history_tokens, 1))
        return {
            "x": x,
            "x_lengths": x_lengths,
            "x_offsets": x_offsets,
            "num_targets": num_targets,
            "history_lengths": history_lengths,
            "selection": selection,
            "input_history_tokens": input_history_tokens,
            "projected_history_tokens": projected_history_tokens,
            "gathered_projected_tokens": gathered_projection_tokens,
            "candidate_selected_tokens": int(selection.selected_tokens),
            "num_candidate_streams": int(selection.batch_size * selection.candidate_count),
            "projection_duplication_factor": duplication,
            "executor": self.executor,
            "projection_policy": self.projection_policy,
            "gdr_backend": self.gdr_backend,
        }

    def gather_hstu_layer_zero_projection(
        self,
        prepared: Mapping[str, Any],
        *,
        packed_x: Optional[torch.Tensor] = None,
        packed_offsets: Optional[torch.Tensor] = None,
        event_gate: Optional[torch.Tensor] = None,
        query_indices: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Return the shared projected history gathered into candidate stream rows."""

        selection = prepared["selection"]
        if packed_x is None:
            packed_x = prepared["x"][0:0]
        if packed_offsets is None:
            packed_offsets = torch.cat((selection.offsets.new_zeros(1), selection.offsets[1:]))
        if event_gate is None:
            event_gate = torch.ones_like(packed_x[:, 0], dtype=prepared["x"].dtype)
        if query_indices is None:
            query_indices = torch.arange(0, max(int(selection.offsets[-1]), 1), device=prepared["x"].device)
        return {
            "packed_x": packed_x,
            "packed_offsets": packed_offsets,
            "event_gate": event_gate,
            "query_indices": query_indices,
            "projected_history_tokens": prepared["projected_history_tokens"],
            "gathered_projected_tokens": prepared["gathered_projected_tokens"],
            "projection_duplication_factor": float(prepared["projection_duplication_factor"]),
        }

    def _selector_selections(
        self,
        history_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        history_lengths: torch.Tensor,
        *,
        selector_dtype: torch.dtype,
        history_embeddings: Optional[torch.Tensor] = None,
        candidate_embeddings: Optional[torch.Tensor] = None,
    ) -> list[SelectionOutput]:
        candidate_count = int(candidate_ids.shape[1])
        requires_shared = bool(
            getattr(self.selector, "requires_shared_embeddings", False)
        )
        if requires_shared and (
            history_embeddings is None or candidate_embeddings is None
        ):
            raise RuntimeError(
                "the Kuai PC selector requires exact shared embedding tensors"
            )
        if (history_embeddings is None) != (candidate_embeddings is None):
            raise ValueError(
                "history and candidate selector embeddings must be supplied together"
            )
        if history_embeddings is not None:
            if history_embeddings.shape[:2] != history_ids.shape or (
                candidate_embeddings is None
                or candidate_embeddings.shape[:2] != candidate_ids.shape
            ):
                raise ValueError("shared selector embeddings do not align with IDs")
        selections: list[SelectionOutput] = []
        for start in range(0, candidate_count, self.candidate_chunk_size):
            end = min(candidate_count, start + self.candidate_chunk_size)
            if history_embeddings is None:
                scores = self.selector.score(
                    history_ids,
                    candidate_ids[:, start:end],
                    history_lengths,
                    dtype=selector_dtype,
                )
            else:
                score_embeddings = getattr(self.selector, "score_embeddings", None)
                if not callable(score_embeddings):
                    raise TypeError(
                        "selector does not accept exact shared embedding tensors"
                    )
                scores = score_embeddings(
                    history_embeddings,
                    candidate_embeddings[:, start:end],
                    history_lengths,
                    dtype=selector_dtype,
                    candidate_chunk_size=None,
                )
            selections.append(
                select_candidate_events(
                    scores=scores,
                    history_lengths=history_lengths,
                    recent_floor=self.recent_floor,
                    retention_ratio=self.retention_ratio,
                    return_dense_mask=False,
                )
            )
        return selections

    def _selector_selection(
        self,
        history_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        history_lengths: torch.Tensor,
        *,
        selector_dtype: torch.dtype,
    ) -> SelectionOutput:
        return merge_candidate_selections(
            self._selector_selections(
                history_ids,
                candidate_ids,
                history_lengths,
                selector_dtype=selector_dtype,
            )
        )

    def _pack_candidate_streams(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        num_targets: torch.Tensor,
        selection: SelectionOutput,
        *,
        candidate_start: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = int(x_lengths.numel())
        candidates = selection.candidate_count
        context = int(getattr(self, "contextual_seq_len", 0))
        history_lengths = x_lengths.to(torch.int64) - context - num_targets.to(torch.int64)
        if bool((history_lengths < 1).any()):
            raise ValueError("candidate-symmetric production streams require nonempty histories")
        row_counts = selection.counts.reshape(-1).to(torch.int64)
        row_lengths = row_counts + context + 1
        packed_offsets = torch.cat((row_lengths.new_zeros(1), row_lengths.cumsum(0)))
        row_ids = torch.repeat_interleave(
            torch.arange(batch * candidates, device=x.device, dtype=torch.int64), row_lengths
        )
        local_positions = self._packed_positions(row_lengths, packed_offsets)
        users = torch.div(row_ids, candidates, rounding_mode="floor")
        slots = row_ids.remainder(candidates)
        source_base = x_offsets.index_select(0, users)
        context_token = local_positions < context
        history_token = (local_positions >= context) & (
            local_positions < context + row_counts.index_select(0, row_ids)
        )
        candidate_token = ~(context_token | history_token)

        source_indices = torch.empty_like(local_positions)
        source_indices[context_token] = source_base[context_token] + local_positions[context_token]

        history_row_offsets = selection.offsets[:-1].index_select(
            0, row_ids[history_token]
        )
        history_within = local_positions[history_token] - context
        selected_index = history_row_offsets + history_within
        selected_positions = selection.source_positions.index_select(0, selected_index)
        source_indices[history_token] = (
            source_base[history_token]
            + context
            + selected_positions
        )

        source_slots = slots[candidate_token] + int(candidate_start)
        source_indices[candidate_token] = (
            source_base[candidate_token]
            + context
            + history_lengths.index_select(0, users[candidate_token])
            + source_slots
        )
        packed_x = x.index_select(0, source_indices.long())
        event_gate = (context_token | history_token).to(x.dtype)
        query_indices = packed_offsets[1:] - 1
        return packed_x, packed_offsets, event_gate, query_indices, source_indices

    @staticmethod
    def _gather_projection(
        projected: GDRKernelInput,
        source_indices: torch.Tensor,
        packed_offsets: torch.Tensor,
        event_gate: torch.Tensor,
    ) -> GDRKernelInput:
        """Gather one source projection into independent candidate streams."""

        indices = source_indices.long()
        return GDRKernelInput(
            q=projected.q.index_select(0, indices),
            k=projected.k.index_select(0, indices),
            v=projected.v.index_select(0, indices),
            decay_logits=projected.decay_logits.index_select(0, indices),
            beta_logits=projected.beta_logits.index_select(0, indices),
            log_decay_scale=projected.log_decay_scale,
            decay_bias=projected.decay_bias,
            offsets=packed_offsets,
            event_gate=event_gate,
        )

    @staticmethod
    def _fixed_width_layout(
        packed_offsets: torch.Tensor,
    ) -> HSTUFixedWidthLayout:
        """Build a shape-invariant projection layout for packed streams."""

        lengths = packed_offsets[1:] - packed_offsets[:-1]
        if lengths.numel() < 1 or bool((lengths < 1).any()):
            raise ValueError("candidate streams must be nonempty")
        sequence_count = int(lengths.numel())
        token_width = int(lengths.max())
        packed_tokens = int(packed_offsets[-1])
        sequence_ids = torch.repeat_interleave(
            torch.arange(
                sequence_count,
                device=packed_offsets.device,
                dtype=torch.int64,
            ),
            lengths.to(torch.int64),
        )
        positions = torch.arange(
            packed_tokens,
            device=packed_offsets.device,
            dtype=torch.int64,
        ) - packed_offsets[:-1].to(torch.int64).index_select(0, sequence_ids)
        return HSTUFixedWidthLayout(
            offsets=packed_offsets,
            flat_padded_indices=sequence_ids * token_width + positions,
            sequence_count=sequence_count,
            token_width=token_width,
            packed_tokens=packed_tokens,
        )

    def forward_candidate_symmetric(
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
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], SelectionOutput, dict[str, Any]]:
        """Run candidate-symmetric packed GDR and return [B,K,D] queries."""

        if self._frozen_backend_active:
            if not x.is_cuda:
                raise RuntimeError(
                    "the frozen Subplan 5B FLA backend requires CUDA"
                )
            if x.dtype != torch.bfloat16:
                raise RuntimeError(
                    "the frozen Subplan 5B backend requires BF16 projections"
                )
        profile_synchronized = bool(x.is_cuda and self.profile_stages)

        def synchronize_profile() -> None:
            if profile_synchronized:
                torch.cuda.synchronize(x.device)

        synchronize_profile()
        started = time.perf_counter()
        if history_item_ids.ndim != 1 or candidate_item_ids.ndim != 1:
            raise ValueError("production IDs must be flat jagged vectors")
        batch = int(x_lengths.numel())
        if num_targets.shape != (batch,) or num_targets.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("num_targets must be an integer vector with shape [B]")
        candidates = int(num_targets.max())
        if candidates < 1:
            raise ValueError("production candidate-symmetric execution requires candidates")
        if bool((num_targets != candidates).any()):
            raise ValueError(
                "candidate-symmetric production currently requires one uniform, "
                "nonzero candidate count per request; ragged K would create "
                "unregistered logical states"
            )
        context = int(getattr(self, "contextual_seq_len", 0))
        history_lengths = x_lengths.to(torch.int64) - context - num_targets.to(torch.int64)
        if int(history_lengths.sum()) != history_item_ids.numel():
            raise ValueError("history IDs do not match the production stream lengths")
        if int(num_targets.sum()) != candidate_item_ids.numel():
            raise ValueError("candidate IDs do not match the production stream lengths")
        history_ids = _jagged_to_dense(history_item_ids, history_lengths, int(history_lengths.max()))
        candidate_ids = _jagged_to_dense(candidate_item_ids, num_targets, candidates)
        dense_history_embeddings = None
        dense_candidate_embeddings = None
        if history_selector_embeddings is not None or candidate_selector_embeddings is not None:
            if history_selector_embeddings is None or candidate_selector_embeddings is None:
                raise ValueError(
                    "history and candidate selector embeddings must be supplied together"
                )
            if history_selector_embeddings.device != x.device or (
                candidate_selector_embeddings.device != x.device
            ):
                raise ValueError("selector embeddings and production stream must share a device")
            dense_history_embeddings = _jagged_embeddings_to_dense(
                history_selector_embeddings,
                history_lengths,
                int(history_lengths.max()),
            )
            dense_candidate_embeddings = _jagged_embeddings_to_dense(
                candidate_selector_embeddings,
                num_targets,
                candidates,
            )
        synchronize_profile()
        selector_started = time.perf_counter()
        selection_chunks = self._selector_selections(
            history_ids=history_ids,
            candidate_ids=candidate_ids,
            history_lengths=history_lengths,
            selector_dtype=torch.float32,
            history_embeddings=dense_history_embeddings,
            candidate_embeddings=dense_candidate_embeddings,
        )
        selection = merge_candidate_selections(selection_chunks)
        synchronize_profile()
        selector_ms = (time.perf_counter() - selector_started) * 1000.0
        selected_count = int(selection.selected_tokens)
        prepared = self.prepare_project_once(
            x=x,
            x_lengths=x_lengths,
            x_offsets=x_offsets,
            num_targets=num_targets,
            history_item_ids=history_item_ids,
            candidate_item_ids=candidate_item_ids,
            selection=selection,
        )
        projection_duplication_factor = (
            float(prepared["projection_duplication_factor"])
            if self._frozen_backend_active
            else 0.0
        )
        shared_projection = None
        shared_u = None
        shared_projection_ms = 0.0
        if self._frozen_backend_active:
            synchronize_profile()
            shared_projection_started = time.perf_counter()
            layer_zero = self.layers[0]
            shared_u, source_q, source_k, source_v, source_decay, source_beta = (
                layer_zero._project(x)
            )
            shared_projection = GDRKernelInput(
                q=source_q,
                k=source_k,
                v=source_v,
                decay_logits=source_decay,
                beta_logits=source_beta,
                log_decay_scale=layer_zero.gdr_log_decay_scale,
                decay_bias=layer_zero.gdr_decay_bias,
                offsets=x_offsets,
                event_gate=None,
            )
            synchronize_profile()
            shared_projection_ms = (
                time.perf_counter() - shared_projection_started
            ) * 1000.0

        query_chunks: list[torch.Tensor] = []
        state_chunks: list[torch.Tensor] = []
        packing_ms = 0.0
        projection_ms = 0.0
        recurrent_ms = 0.0
        output_ms = 0.0
        fixed_width_padded_tokens = 0
        fixed_width_padding_tokens = 0
        candidate_start = 0
        for chunk_selection in selection_chunks:
            chunk_candidates = chunk_selection.candidate_count
            synchronize_profile()
            packing_started = time.perf_counter()
            packed_x, packed_offsets, event_gate, query_indices, source_indices = (
                self._pack_candidate_streams(
                    x=x,
                    x_lengths=x_lengths,
                    x_offsets=x_offsets,
                    num_targets=num_targets,
                    selection=chunk_selection,
                    candidate_start=candidate_start,
                )
            )
            fixed_width_layout = self._fixed_width_layout(packed_offsets)
            fixed_width_padded_tokens += fixed_width_layout.padded_tokens
            fixed_width_padding_tokens += fixed_width_layout.padding_tokens
            packed_projection = None
            packed_u = None
            if self._frozen_backend_active:
                assert shared_projection is not None and shared_u is not None
                packed_projection = self._gather_projection(
                    shared_projection,
                    source_indices,
                    packed_offsets,
                    event_gate,
                )
                packed_u = shared_u.index_select(0, source_indices.long())
            synchronize_profile()
            packing_ms += (time.perf_counter() - packing_started) * 1000.0

            chunk_layer_states: list[torch.Tensor] = []
            for layer_index, layer in enumerate(self.layers):
                layer.profile_stages = profile_synchronized
                layer_started = time.perf_counter()
                if self._frozen_backend_active and layer_index == 0:
                    assert packed_projection is not None and packed_u is not None
                    packed_x = layer.forward_gdr_projected(
                        x=packed_x,
                        u=packed_u,
                        projected=packed_projection,
                        return_final_state=return_final_states,
                        fixed_width_layout=fixed_width_layout,
                    )
                else:
                    packed_x = layer.forward_gdr(
                        x=packed_x,
                        x_offsets=packed_offsets,
                        event_gate=event_gate,
                        return_final_state=return_final_states,
                        fixed_width_layout=fixed_width_layout,
                    )
                if x.is_cuda and profile_synchronized:
                    layer_times = layer.last_stage_times_ms()
                    projection_ms += layer_times.get("projection_ms", 0.0)
                    recurrent_ms += layer_times.get("prefill_ms", 0.0)
                    output_ms += layer_times.get("output_ms", 0.0)
                elif not x.is_cuda:
                    recurrent_ms += (time.perf_counter() - layer_started) * 1000.0
                if return_final_states:
                    if layer.last_final_state is None:
                        raise RuntimeError(
                            "production GDR layer did not return an FP32 final state"
                        )
                    chunk_layer_states.append(layer.last_final_state.float())
            query_chunks.append(
                packed_x.index_select(0, query_indices).reshape(
                    batch,
                    chunk_candidates,
                    -1,
                )
            )
            if return_final_states:
                state_chunks.append(
                    torch.stack(chunk_layer_states, dim=1).reshape(
                        batch,
                        chunk_candidates,
                        len(self.layers),
                        *chunk_layer_states[0].shape[1:],
                    )
                )
            candidate_start += chunk_candidates

        queries = torch.cat(query_chunks, dim=1)
        final_states = torch.cat(state_chunks, dim=1) if state_chunks else None
        projection_ms += shared_projection_ms
        synchronize_profile()
        total_ms = (time.perf_counter() - started) * 1000.0
        self.last_selection = selection
        self.last_final_states = final_states
        frozen_backend_diagnostics = {}
        if self._frozen_backend_active:
            frozen_backend_diagnostics = {
                "input_history_tokens": prepared["input_history_tokens"],
                "projected_history_tokens": prepared["projected_history_tokens"],
                "gathered_projected_tokens": prepared["gathered_projected_tokens"],
                "candidate_selected_tokens": prepared["candidate_selected_tokens"],
                "num_candidate_streams": prepared["num_candidate_streams"],
                "projection_duplication_factor": projection_duplication_factor,
                "executor": prepared["executor"],
                "projection_policy": prepared["projection_policy"],
                "gdr_backend": prepared["gdr_backend"],
                "frozen_backend_active": True,
                "full_history_projection_once": True,
                "project_once_reuse_scope": "request",
                "actual_layer_backends": [layer.kernel_backend for layer in self.layers],
                "layer_zero_source_tokens": int(x.shape[0]),
                "layer_zero_gathered_tokens": int(
                    prepared["gathered_projected_tokens"]
                ),
                "id_mapping_contract": dict(self.selector.mapping_contract),
                "id_mapping_contract_sha256": self.selector.mapping_contract_sha256,
            }
        self.last_diagnostics = {
            "selector_ms": selector_ms,
            "packing_ms": packing_ms,
            "projection_ms": projection_ms,
            "recurrent_ms": recurrent_ms,
            "output_ms": output_ms,
            "total_ms": total_ms if profile_synchronized or not x.is_cuda else None,
            "component_timings_synchronized": profile_synchronized,
            "selected_tokens": selected_count,
            "logical_states": batch * candidates * len(self.layers),
            "logical_encodes": batch * candidates,
            "physical_gdr_calls": len(selection_chunks) * len(self.layers),
            "candidate_chunk_count": len(selection_chunks),
            "candidate_chunk_size": self.candidate_chunk_size,
            "projection_linear_mode": "fixed_width_batched",
            "fixed_width_padded_tokens": fixed_width_padded_tokens,
            "fixed_width_padding_tokens": fixed_width_padding_tokens,
            "offset_shape": list(selection.offsets.shape),
            "selector_artifacts": dict(self.selector.artifact_sha256),
            "gdr_initialization_sha256": self.gdr_initialization_hash,
            **frozen_backend_diagnostics,
        }
        return queries, final_states, selection, dict(self.last_diagnostics)

    @torch.no_grad()
    def materialize_pc_state_cache(
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
    ) -> PCStateCache:
        """Replay the complete PC path once and atomically publish FP32 states."""

        if self.training:
            raise RuntimeError("PC state-cache materialization requires eval mode")
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
        started = time.perf_counter()
        _, states, selection, cold_diagnostics = self.forward_candidate_symmetric(
            x=x,
            x_lengths=x_lengths,
            x_offsets=x_offsets,
            num_targets=num_targets,
            history_item_ids=history_item_ids,
            candidate_item_ids=candidate_item_ids,
            return_final_states=True,
        )
        if states is None or states.dtype != torch.float32:
            raise RuntimeError("PC replay did not return FP32 recurrent states")
        batch, candidates = states.shape[:2]
        if states.shape[2:] != self.pc_state_shape:
            raise RuntimeError("PC replay returned an unexpected public state layout")
        if candidate_item_ids.numel() != batch * candidates:
            raise ValueError("candidate IDs do not match replayed PC states")
        candidate_ids = candidate_item_ids.reshape(batch, candidates)
        version = self.refresh_pc_state_cache_version()
        cache = PCStateCache(
            dataset_id=dataset_id,
            version=version,
            state_shape=self.pc_state_shape,
            components={
                "serving_model_sha256": version.model_version,
                "selector_artifacts": dict(self.selector_artifact_sha256),
                "budget_policy": PC_BUDGET_POLICY,
                "write_ratio": self.retention_ratio,
                "recent_floor": self.recent_floor,
                "contextual_seq_len": self.contextual_seq_len,
                "projection_dtype": self.projection_dtype,
                "recurrence_backend": self.gdr_backend,
                "state_layout": "N,K,layers,H,Dk,Dv",
                "state_dtype": "float32",
            },
            storage_device=storage_device,
            pin_memory=pin_memory,
        )
        cache.publish_batch(
            user_ids=user_ids,
            history_versions=history_versions,
            candidate_ids=candidate_ids,
            states=states,
        )
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
        self.last_pc_cache_build_diagnostics = {
            "excluded_from_serving_latency": True,
            "operation": "per_candidate_prefix_state_materialization",
            "build_ms": (time.perf_counter() - started) * 1000.0,
            "entry_count": cache.entry_count,
            "state_row_count": cache.state_row_count,
            "state_bytes_per_candidate": cache.state_bytes_per_candidate,
            "selected_history_tokens": int(selection.selected_tokens),
            "cold_physical_gdr_calls": cold_diagnostics["physical_gdr_calls"],
            "cache_version": version.fingerprint,
            "cache_key": (
                "dataset_id,user_id,history_version,candidate_id,model_version,"
                "selector_version,budget_version,schema_version"
            ),
            "atomic_candidate_set_publication": True,
            "state_layout": "N,K,layers,H,Dk,Dv",
            "state_dtype": "float32",
        }
        return cache

    def cached_state_forward(
        self,
        *,
        dataset_id: str,
        candidate_x: torch.Tensor,
        candidate_item_ids: torch.Tensor,
        user_ids: torch.Tensor,
        history_versions: torch.Tensor,
        state_cache: PCStateCache,
        return_final_states: bool = True,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], dict[str, Any]]:
        """Serve candidate-only identity reads from exact PC prefix states.

        No history tensor or selector input is accepted.  Each candidate token
        is processed as a one-token independent sequence with ``event_gate=0``;
        therefore the returned recurrent states equal the cached states.
        """

        if not isinstance(state_cache, PCStateCache):
            raise TypeError("state_cache must be a PCStateCache")
        if self.training:
            raise RuntimeError("PC state-cache serving requires eval mode")
        if candidate_x.ndim != 3 or not torch.is_floating_point(candidate_x):
            raise ValueError("candidate_x must be floating point [N,K,D]")
        batch, candidates, width = candidate_x.shape
        if batch < 1 or candidates < 1:
            raise ValueError("candidate-only serving requires nonempty N and K")
        if candidate_item_ids.ndim == 1:
            if candidate_item_ids.numel() != batch * candidates:
                raise ValueError("candidate IDs do not match candidate_x")
            candidate_ids = candidate_item_ids.reshape(batch, candidates)
        elif candidate_item_ids.shape == (batch, candidates):
            candidate_ids = candidate_item_ids
        else:
            raise ValueError("candidate_item_ids must have shape [N*K] or [N,K]")
        if candidate_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("candidate_item_ids must be integer")
        for name, value in (
            ("user_ids", user_ids),
            ("history_versions", history_versions),
        ):
            if value.shape != (batch,) or value.dtype not in (
                torch.int32,
                torch.int64,
            ):
                raise ValueError(f"{name} must be an integer vector [N]")
        expected_version = self.current_pc_state_cache_version()
        if state_cache.version != expected_version:
            raise RuntimeError(
                "PC state cache does not match the current model/selector/budget"
            )
        if state_cache.state_shape != self.pc_state_shape:
            raise RuntimeError("PC state cache layout does not match this stack")

        profile_synchronized = bool(candidate_x.is_cuda and self.profile_stages)

        def synchronize() -> None:
            if profile_synchronized:
                torch.cuda.synchronize(candidate_x.device)

        synchronize()
        started = time.perf_counter()
        lookup_started = time.perf_counter()
        lookup = state_cache.lookup(
            dataset_id=dataset_id,
            user_ids=user_ids,
            history_versions=history_versions,
            candidate_ids=candidate_ids,
            version=expected_version,
            target_device=candidate_x.device,
            non_blocking=True,
        )
        synchronize()
        lookup_ms = (time.perf_counter() - lookup_started) * 1000.0

        sequence_count = batch * candidates
        packed_x = candidate_x.reshape(sequence_count, width)
        offsets = torch.arange(
            sequence_count + 1,
            dtype=torch.int64,
            device=candidate_x.device,
        )
        event_gate = torch.zeros(
            sequence_count,
            dtype=candidate_x.dtype,
            device=candidate_x.device,
        )
        fixed_width_layout = self._fixed_width_layout(offsets)
        layer_major_states = lookup.states.permute(2, 0, 1, 3, 4, 5).contiguous()
        if layer_major_states.dtype != torch.float32:
            raise RuntimeError("PC initial states must remain FP32")
        executor_started = time.perf_counter()
        for layer_index, layer in enumerate(self.layers):
            layer.profile_stages = profile_synchronized
            initial_state = layer_major_states[layer_index].reshape(
                sequence_count, *self.pc_state_shape[1:]
            )
            packed_x = layer.forward_gdr(
                x=packed_x,
                x_offsets=offsets,
                event_gate=event_gate,
                initial_state=initial_state,
                return_final_state=False,
                fixed_width_layout=fixed_width_layout,
            )
        synchronize()
        executor_ms = (time.perf_counter() - executor_started) * 1000.0
        queries = packed_x.reshape(batch, candidates, width)
        if not bool(torch.isfinite(queries).all()):
            raise RuntimeError("PC cache-hit queries must be finite")
        final_states = lookup.states if return_final_states else None
        if final_states is not None and (
            final_states.shape
            != (batch, candidates, *self.pc_state_shape)
            or final_states.dtype != torch.float32
        ):
            raise RuntimeError("PC cache hit returned an invalid state layout")
        total_ms = (time.perf_counter() - started) * 1000.0
        timings_valid = profile_synchronized or not candidate_x.is_cuda
        diagnostics: dict[str, Any] = {
            "cache_hit": True,
            "cache_hit_count": lookup.hit_count,
            "cache_miss_count": 0,
            "cache_key": (
                "dataset_id,user_id,history_version,candidate_id,model_version,"
                "selector_version,budget_version,schema_version"
            ),
            "dataset_id": state_cache.dataset_id,
            "model_version": expected_version.model_version,
            "selector_version": expected_version.selector_version,
            "budget_version": expected_version.budget_version,
            "schema_version": expected_version.schema_version,
            "candidate_set_sha256": list(lookup.candidate_set_sha256),
            "published_candidate_set_sha256": list(
                lookup.published_candidate_set_sha256
            ),
            "batch_size": batch,
            "candidate_count": candidates,
            "state_cache_entry_count": state_cache.entry_count,
            "state_cache_row_count": state_cache.state_row_count,
            "state_bytes_per_candidate": state_cache.state_bytes_per_candidate,
            "cache_materialization_bytes": lookup.materialization_bytes,
            "cache_transferred_bytes": lookup.transferred_bytes,
            "selector_executed_online": False,
            "selection_executed_online": False,
            "history_fetch_executed_online": False,
            "history_prefill_executed_online": False,
            "online_history_tokens": 0,
            "write_tokens_per_layer": 0,
            "read_tokens_per_layer": sequence_count,
            "logical_states": sequence_count * len(self.layers),
            "physical_gdr_calls": len(self.layers),
            "query_event_gate": 0,
            "candidate_order": "original_input_order",
            "state_layout": "N,K,layers,H,Dk,Dv",
            "state_dtype": "float32",
            "return_final_states": return_final_states,
            "component_timings_synchronized": profile_synchronized,
            "pc_state_cache_lookup_ms": lookup_ms if timings_valid else None,
            "pc_candidate_only_executor_ms": executor_ms if timings_valid else None,
            "pc_cache_hit_total_ms": total_ms if timings_valid else None,
            "actual_layer_backends": [layer.kernel_backend for layer in self.layers],
        }
        self.last_selection = None
        self.last_final_states = final_states
        self.last_diagnostics = diagnostics
        return queries, final_states, dict(diagnostics)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise RuntimeError(
            "DLRMv3CandidateSymmetricSTUStack requires forward_candidate_symmetric(); "
            "the normal STU forward would silently lose candidate-specific state"
        )

    def cached_forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.cached_state_forward(*args, **kwargs)

