from __future__ import annotations

from collections.abc import Callable, Iterator

import hashlib

import json

import math

import os

from pathlib import Path

import random

import tempfile

from typing import Any, Mapping, Sequence

import torch

from torch import nn

from deltarec.utils.io import atomic_write_json, protocol_hash, sha256_file

PC_SELECTOR_LOSS = "smooth-l1-asinh-cwi1"

from deltarec.adaptors.hstu_model import MetaBridgeError

CWI_LABEL_SHARD_SCHEMA = "deltarec-headline-cwi-label-shard-v1"

CWI_LABEL_MANIFEST_SCHEMA = "deltarec-headline-cwi-label-manifest-v1"

TRAJECTORY_CWI_MANIFEST_SCHEMA = "deltarec-headline-trajectory-cwi-manifest-v1"

FIT_SCOPE = {
    "split": "train",
    "histories": "training-histories-only-strict-prefix",
    "catalog": "training-catalog-only",
    "validation_data_mounted": False,
    "test_data_mounted": False,
}

SELECTION_SCOPE = {
    "split": "validation",
    "purpose": "checkpoint-and-registered-hyperparameter-selection-only",
    "gradient_visible": False,
    "cwi_labels_generated": False,
    "test_data_mounted": False,
}

_SHARD_TENSORS = (
    "user_ids",
    "history_item_ids",
    "history_lengths",
    "candidate_item_ids",
    "candidate_positions",
    "cwi1",
)

def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")

def _content_hash(value: Mapping[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return protocol_hash(unsigned)

def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256(
        f"{tuple(tensor.shape)}:{tensor.dtype}".encode("ascii")
    )
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

def _file_record(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise MetaBridgeError(f"selector-stage artifact is missing: {source}")
    return {
        "path": str(source),
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }

def _verify_file_record(value: Any, *, name: str) -> Path:
    if not isinstance(value, Mapping):
        raise MetaBridgeError(f"selector-stage {name} binding is missing")
    source = Path(str(value.get("path", ""))).expanduser().resolve()
    if (
        not source.is_file()
        or source.stat().st_size != int(value.get("size_bytes", -1))
        or sha256_file(source) != str(value.get("sha256", ""))
    ):
        raise MetaBridgeError(f"selector-stage {name} checksum changed")
    return source

def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    source = path.expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MetaBridgeError(f"invalid {name} JSON: {source}") from error
    if not isinstance(value, dict):
        raise MetaBridgeError(f"{name} JSON must be an object")
    return value

def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    output = path.expanduser().resolve()
    if output.exists():
        raise MetaBridgeError(f"refusing to overwrite selector artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

def _validate_shard_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != CWI_LABEL_SHARD_SCHEMA:
        raise MetaBridgeError("unexpected CWI label shard schema")
    for name in _SHARD_TENSORS:
        if not isinstance(payload.get(name), torch.Tensor):
            raise MetaBridgeError(f"CWI label shard lacks tensor {name}")
    users = payload["user_ids"]
    histories = payload["history_item_ids"]
    lengths = payload["history_lengths"]
    candidates = payload["candidate_item_ids"]
    candidate_positions = payload["candidate_positions"]
    labels = payload["cwi1"]
    if (
        users.ndim != 1
        or histories.ndim != 2
        or lengths.shape != users.shape
        or candidates.ndim != 2
        or candidates.shape[0] != len(users)
        or candidate_positions.shape != candidates.shape
        or labels.shape != (*candidates.shape, histories.shape[1])
    ):
        raise MetaBridgeError("CWI label shard tensor shapes are inconsistent")
    if any(
        value.dtype not in (torch.int32, torch.int64)
        for value in (users, histories, lengths, candidates, candidate_positions)
    ) or not torch.is_floating_point(labels):
        raise MetaBridgeError("CWI label shard tensor dtypes are inconsistent")
    if bool((lengths < 1).any()) or bool((lengths > histories.shape[1]).any()):
        raise MetaBridgeError("CWI label shard history lengths are invalid")
    expected_positions = torch.arange(candidates.shape[1], dtype=torch.int64)[
        None
    ].expand_as(candidate_positions.cpu().to(torch.int64))
    if not torch.equal(candidate_positions.cpu().to(torch.int64), expected_positions):
        raise MetaBridgeError("CWI candidate exposure positions changed")
    valid = torch.arange(histories.shape[1])[None, None, :] < lengths.cpu()[
        :, None, None
    ]
    if not bool(torch.isfinite(labels.cpu().masked_select(valid)).all()) or bool(
        labels.cpu().masked_select(~valid).ne(0).any()
    ):
        raise MetaBridgeError("CWI labels are nonfinite or nonzero in padding")
    if payload.get("split_role") != "train" or payload.get(
        "test_data_mounted"
    ) is not False:
        raise MetaBridgeError("CWI label shard is not train-only")
    unsigned = dict(payload)
    observed = unsigned.pop("artifact_content_sha256", None)
    tensor_hashes = {
        name: _tensor_sha256(unsigned.pop(name)) for name in _SHARD_TENSORS
    }
    expected = hashlib.sha256(
        _canonical_bytes({"metadata": unsigned, "tensors": tensor_hashes})
    ).hexdigest()
    if observed != expected:
        raise MetaBridgeError("CWI label shard content checksum mismatch")

def write_cwi_label_shard(
    output_path: Path,
    *,
    dataset: str,
    seed: int,
    parent_full_checkpoint_sha256: str,
    candidate_identity: str,
    user_ids: torch.Tensor,
    history_item_ids: torch.Tensor,
    history_lengths: torch.Tensor,
    candidate_item_ids: torch.Tensor,
    cwi1: torch.Tensor,
) -> dict[str, Any]:
    """Write one immutable train-only CWI shard with positioned candidates."""

    candidate_positions = torch.arange(
        candidate_item_ids.shape[1], dtype=torch.int64
    )[None].expand_as(candidate_item_ids.cpu()).clone()
    payload: dict[str, Any] = {
        "schema": CWI_LABEL_SHARD_SCHEMA,
        "dataset": dataset,
        "seed": int(seed),
        "split_role": "train",
        "fit_scope": dict(FIT_SCOPE),
        "parent_full_checkpoint_sha256": parent_full_checkpoint_sha256,
        "candidate_identity": candidate_identity,
        "duplicate_item_ids_allowed": dataset == "kuairand-1k",
        "test_data_mounted": False,
        "user_ids": user_ids.detach().cpu().to(torch.int64).contiguous(),
        "history_item_ids": history_item_ids.detach()
        .cpu()
        .to(torch.int64)
        .contiguous(),
        "history_lengths": history_lengths.detach()
        .cpu()
        .to(torch.int64)
        .contiguous(),
        "candidate_item_ids": candidate_item_ids.detach()
        .cpu()
        .to(torch.int64)
        .contiguous(),
        "candidate_positions": candidate_positions.contiguous(),
        "cwi1": cwi1.detach().cpu().float().contiguous(),
    }
    metadata = {name: value for name, value in payload.items() if name not in _SHARD_TENSORS}
    tensor_hashes = {name: _tensor_sha256(payload[name]) for name in _SHARD_TENSORS}
    payload["artifact_content_sha256"] = hashlib.sha256(
        _canonical_bytes({"metadata": metadata, "tensors": tensor_hashes})
    ).hexdigest()
    _validate_shard_payload(payload)
    _atomic_torch_save(output_path, payload)
    return {
        **_file_record(output_path),
        "artifact_content_sha256": payload["artifact_content_sha256"],
        "rows": int(len(user_ids)),
        "candidate_count": int(candidate_item_ids.shape[1]),
        "valid_label_count": int(
            history_lengths.to(torch.int64).sum().item()
            * candidate_item_ids.shape[1]
        ),
    }

def load_cwi_label_shard(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except Exception as error:
        raise MetaBridgeError(f"cannot load CWI label shard: {source}") from error
    if not isinstance(payload, Mapping):
        raise MetaBridgeError("CWI label shard must be a mapping")
    _validate_shard_payload(payload)
    return dict(payload)

def create_cwi_label_manifest(
    output_path: Path,
    *,
    dataset: str,
    seed: int,
    protocol_sha256: str,
    parent_full_checkpoint_sha256: str,
    parent_full_manifest_content_sha256: str,
    split_manifest: Path,
    training_histories: Path,
    training_catalog: Path,
    item_id_map: Path,
    candidate_identity: str,
    shard_paths: Sequence[Path],
) -> dict[str, Any]:
    """Bind CWI labels to training histories/catalog with no validation/test mount."""

    if not shard_paths:
        raise MetaBridgeError("CWI label manifest requires at least one shard")
    shards: list[dict[str, Any]] = []
    root = output_path.expanduser().resolve().parent
    for path in shard_paths:
        source = path.expanduser().resolve()
        if source.parent != root or source.name != str(source.name):
            raise MetaBridgeError("CWI shards must be local to their manifest")
        payload = load_cwi_label_shard(source)
        expected = {
            "dataset": dataset,
            "seed": seed,
            "split_role": "train",
            "parent_full_checkpoint_sha256": parent_full_checkpoint_sha256,
            "candidate_identity": candidate_identity,
        }
        for field, value in expected.items():
            if payload.get(field) != value:
                raise MetaBridgeError(f"CWI shard mismatch for {field}")
        shards.append(
            {
                "filename": source.name,
                "size_bytes": source.stat().st_size,
                "sha256": sha256_file(source),
                "artifact_content_sha256": payload["artifact_content_sha256"],
                "rows": int(payload["user_ids"].numel()),
                "candidate_count": int(payload["candidate_item_ids"].shape[1]),
                "valid_label_count": int(
                    payload["history_lengths"].sum().item()
                    * payload["candidate_item_ids"].shape[1]
                ),
            }
        )
    manifest: dict[str, Any] = {
        "schema": CWI_LABEL_MANIFEST_SCHEMA,
        "dataset": dataset,
        "seed": int(seed),
        "protocol_sha256": protocol_sha256,
        "split_role": "train",
        "label_fit_scope": dict(FIT_SCOPE),
        "parent_full_checkpoint_sha256": parent_full_checkpoint_sha256,
        "parent_full_manifest_content_sha256": parent_full_manifest_content_sha256,
        "candidate_identity": candidate_identity,
        "duplicate_item_ids_allowed": dataset == "kuairand-1k",
        "teacher": "same-seed-trained-official-full-gdr-cwi1",
        "history_boundary": "strictly-before-supervision-target",
        "split_hashes": {
            "split_manifest_sha256": sha256_file(split_manifest),
            "training_histories_sha256": sha256_file(training_histories),
            "training_catalog_sha256": sha256_file(training_catalog),
            "item_id_map_sha256": sha256_file(item_id_map),
        },
        "source_files": {
            "split_manifest": _file_record(split_manifest),
            "training_histories": _file_record(training_histories),
            "training_catalog": _file_record(training_catalog),
            "item_id_map": _file_record(item_id_map),
        },
        "validation_candidates_mounted": False,
        "test_data_mounted": False,
        "shards": shards,
    }
    manifest["manifest_content_sha256"] = _content_hash(
        manifest, "manifest_content_sha256"
    )
    atomic_write_json(output_path, manifest)
    return manifest

def load_cwi_label_manifest(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    manifest = _read_json(source, name="CWI label manifest")
    if (
        manifest.get("schema") != CWI_LABEL_MANIFEST_SCHEMA
        or manifest.get("manifest_content_sha256")
        != _content_hash(manifest, "manifest_content_sha256")
        or manifest.get("label_fit_scope") != FIT_SCOPE
        or manifest.get("validation_candidates_mounted") is not False
        or manifest.get("test_data_mounted") is not False
    ):
        raise MetaBridgeError("CWI label manifest scope/content changed")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise MetaBridgeError("CWI label manifest has no shards")
    for record in manifest.get("source_files", {}).values():
        _verify_file_record(record, name="CWI training source")
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise MetaBridgeError("CWI shard record is malformed")
        filename = str(shard.get("filename", ""))
        if Path(filename).name != filename:
            raise MetaBridgeError("CWI shard path is not local")
        shard_path = (source.parent / filename).resolve()
        _verify_file_record(
            {
                "path": str(shard_path),
                "size_bytes": shard.get("size_bytes"),
                "sha256": shard.get("sha256"),
            },
            name="CWI label shard",
        )
        payload = load_cwi_label_shard(shard_path)
        if payload.get("artifact_content_sha256") != shard.get(
            "artifact_content_sha256"
        ):
            raise MetaBridgeError("CWI shard content binding changed")
    return manifest

def normalized_trajectory_weights(
    count: int, explicit: Sequence[float] | None = None
) -> list[float]:
    """Return positive late-biased teacher probabilities for an ordered trajectory."""

    if count < 1:
        raise ValueError("a trajectory needs at least one teacher checkpoint")
    raw = list(range(1, count + 1)) if explicit is None else list(explicit)
    if len(raw) != count or any(not math.isfinite(value) or value <= 0 for value in raw):
        raise ValueError("trajectory weights must be finite, positive, and match teachers")
    total = float(sum(raw))
    return [float(value) / total for value in raw]

def create_trajectory_cwi_manifest(
    output_path: Path,
    *,
    teacher_manifests: Sequence[Path],
    teacher_weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Bind ordered checkpoint-specific CWI manifests without averaging labels."""

    if not teacher_manifests:
        raise MetaBridgeError("trajectory CWI requires at least one teacher manifest")
    loaded = [load_cwi_label_manifest(path) for path in teacher_manifests]
    first = loaded[0]
    invariant_fields = (
        "dataset",
        "seed",
        "protocol_sha256",
        "split_hashes",
        "candidate_identity",
        "duplicate_item_ids_allowed",
        "label_fit_scope",
    )
    for index, manifest in enumerate(loaded[1:], 2):
        for field in invariant_fields:
            if manifest.get(field) != first.get(field):
                raise MetaBridgeError(
                    f"trajectory teacher {index} differs for invariant {field}"
                )
    parent_hashes = [str(item["parent_full_checkpoint_sha256"]) for item in loaded]
    if len(set(parent_hashes)) != len(parent_hashes):
        raise MetaBridgeError("trajectory teachers must use distinct ordered checkpoints")
    weights = normalized_trajectory_weights(len(loaded), teacher_weights)
    manifest: dict[str, Any] = {
        "schema": TRAJECTORY_CWI_MANIFEST_SCHEMA,
        "dataset": first["dataset"],
        "seed": int(first["seed"]),
        "protocol_sha256": first["protocol_sha256"],
        "split_role": "train",
        "label_fit_scope": dict(FIT_SCOPE),
        "split_hashes": dict(first["split_hashes"]),
        "candidate_identity": first["candidate_identity"],
        "duplicate_item_ids_allowed": first["duplicate_item_ids_allowed"],
        "teacher_weights": weights,
        "sampling_policy": "seeded-minibatch-teacher-sampling-late-biased",
        "target_policy": "checkpoint-specific-observations-no-label-averaging",
        "final_selector_embedding_checkpoint_sha256": parent_hashes[-1],
        "validation_candidates_mounted": False,
        "test_data_mounted": False,
        "teachers": [
            {
                "trajectory_index": index,
                "parent_full_checkpoint_sha256": parent_hash,
                "cwi_label_manifest": _file_record(path),
                "cwi_label_manifest_content_sha256": labels[
                    "manifest_content_sha256"
                ],
            }
            for index, (path, labels, parent_hash) in enumerate(
                zip(teacher_manifests, loaded, parent_hashes), 1
            )
        ],
    }
    manifest["manifest_content_sha256"] = _content_hash(
        manifest, "manifest_content_sha256"
    )
    atomic_write_json(output_path, manifest)
    return manifest

def load_trajectory_cwi_manifest(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    manifest = _read_json(source, name="trajectory CWI manifest")
    if (
        manifest.get("schema") != TRAJECTORY_CWI_MANIFEST_SCHEMA
        or manifest.get("manifest_content_sha256")
        != _content_hash(manifest, "manifest_content_sha256")
        or manifest.get("label_fit_scope") != FIT_SCOPE
        or manifest.get("target_policy")
        != "checkpoint-specific-observations-no-label-averaging"
        or manifest.get("validation_candidates_mounted") is not False
        or manifest.get("test_data_mounted") is not False
    ):
        raise MetaBridgeError("trajectory CWI manifest scope/content changed")
    teachers = manifest.get("teachers")
    if not isinstance(teachers, list) or not teachers:
        raise MetaBridgeError("trajectory CWI manifest has no teachers")
    weights = normalized_trajectory_weights(
        len(teachers), manifest.get("teacher_weights")
    )
    if weights != manifest.get("teacher_weights"):
        raise MetaBridgeError("trajectory CWI weights are not normalized")
    parent_hashes: list[str] = []
    for expected_index, teacher in enumerate(teachers, 1):
        if not isinstance(teacher, Mapping) or teacher.get(
            "trajectory_index"
        ) != expected_index:
            raise MetaBridgeError("trajectory teacher order changed")
        label_path = _verify_file_record(
            teacher.get("cwi_label_manifest"), name="teacher CWI manifest"
        )
        labels = load_cwi_label_manifest(label_path)
        expected = {
            "dataset": manifest["dataset"],
            "seed": manifest["seed"],
            "protocol_sha256": manifest["protocol_sha256"],
            "split_hashes": manifest["split_hashes"],
            "candidate_identity": manifest["candidate_identity"],
            "parent_full_checkpoint_sha256": teacher.get(
                "parent_full_checkpoint_sha256"
            ),
            "manifest_content_sha256": teacher.get(
                "cwi_label_manifest_content_sha256"
            ),
        }
        for field, value in expected.items():
            if labels.get(field) != value:
                raise MetaBridgeError(
                    f"trajectory teacher {expected_index} binding changed for {field}"
                )
        parent_hashes.append(str(labels["parent_full_checkpoint_sha256"]))
    if len(set(parent_hashes)) != len(parent_hashes) or manifest.get(
        "final_selector_embedding_checkpoint_sha256"
    ) != parent_hashes[-1]:
        raise MetaBridgeError("trajectory checkpoint identity/order changed")
    return manifest

def _training_batches(
    manifest_path: Path, *, microbatch_size: int
) -> Iterator[tuple[torch.Tensor, ...]]:
    manifest = load_cwi_label_manifest(manifest_path)
    for record in manifest["shards"]:
        payload = load_cwi_label_shard(manifest_path.parent / record["filename"])
        rows = len(payload["user_ids"])
        for start in range(0, rows, microbatch_size):
            stop = min(rows, start + microbatch_size)
            yield (
                payload["history_item_ids"][start:stop],
                payload["history_lengths"][start:stop],
                payload["candidate_item_ids"][start:stop],
                payload["cwi1"][start:stop],
            )

def selector_validation_key(
    dataset: str, metrics: Mapping[str, float]
) -> tuple[float, float]:
    """Return the preregistered validation ordering for selector checkpoints."""

    if dataset == "kuairand-1k":
        if set(metrics) < {"macro_gauc", "multitask_loss"}:
            raise MetaBridgeError("Kuai selector validation lacks GAUC/loss")
        primary = float(metrics["macro_gauc"])
        # The scheduler maximizes both tuple elements, so lower official
        # multitask loss must enter with a negative sign.
        tie_break = -float(metrics["multitask_loss"])
    elif dataset in {"ml-20m", "amazon-books"}:
        if "ndcg_at_10" not in metrics:
            raise MetaBridgeError("rating selector validation lacks NDCG@10")
        primary = float(metrics["ndcg_at_10"])
        tie_break = 0.0
    else:
        raise MetaBridgeError(f"unsupported selector dataset: {dataset}")
    if not all(math.isfinite(value) for value in (primary, tie_break)):
        raise MetaBridgeError("selector validation selection metric is nonfinite")
    return primary, tie_break

def train_candidate_cwi_mlp_trajectory(
    selector: nn.Module,
    *,
    trajectory_cwi_manifest: Path,
    teacher_embeddings: Sequence[nn.Embedding],
    validation_score_fn: Callable[[nn.Module, int], Mapping[str, float]],
    seed: int,
    device: torch.device | str,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    microbatch_size: int = 32,
    resume_path: Path | None = None,
) -> dict[str, Any]:
    """Fit one MLP from separately sampled checkpoint-specific CWI observations."""

    if epochs != 5 or learning_rate != 1e-3 or microbatch_size < 1:
        raise ValueError("headline selector uses registered 5-epoch AdamW at 1e-3")
    if not hasattr(selector, "exact_scores_from_embeddings") or not hasattr(
        selector, "official_embedding"
    ):
        raise TypeError("trajectory selector must expose exact embedding scoring")
    manifest = load_trajectory_cwi_manifest(trajectory_cwi_manifest)
    teachers = list(manifest["teachers"])
    if len(teacher_embeddings) != len(teachers):
        raise MetaBridgeError("teacher embedding count differs from trajectory")
    if int(manifest["seed"]) != int(seed):
        raise MetaBridgeError("selector seed differs from trajectory CWI labels")
    target_device = torch.device(device)
    selector.to(target_device)
    final_embedding = selector.official_embedding
    if not isinstance(final_embedding, nn.Embedding) or not torch.equal(
        final_embedding.weight.detach().cpu(),
        teacher_embeddings[-1].weight.detach().cpu(),
    ):
        raise MetaBridgeError("selector feature snapshot is not theta_T")
    frozen_teacher_weights = [
        embedding.weight.detach().cpu().contiguous().clone()
        for embedding in teacher_embeddings
    ]
    for embedding in teacher_embeddings:
        if (
            not isinstance(embedding, nn.Embedding)
            or embedding.num_embeddings != final_embedding.num_embeddings
            or embedding.embedding_dim != final_embedding.embedding_dim
        ):
            raise MetaBridgeError("trajectory teacher embedding shape changed")
        embedding.to(target_device)
        embedding.weight.requires_grad_(False)
    snapshot_prefixes = {
        name
        for name, module in selector.named_modules(remove_duplicate=False)
        if name and module is final_embedding
    }
    snapshot_state_keys = {
        name
        for name in selector.state_dict()
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in snapshot_prefixes
        )
    }
    named_parameters = [
        (name, parameter)
        for name, parameter in selector.named_parameters()
        if parameter.requires_grad
    ]
    parameters = [parameter for _, parameter in named_parameters]
    if not parameters or any(
        parameter is embedding.weight
        for parameter in parameters
        for embedding in teacher_embeddings
    ):
        raise MetaBridgeError("trajectory optimizer must contain MLP parameters only")
    label_paths = [
        Path(teacher["cwi_label_manifest"]["path"]) for teacher in teachers
    ]
    batch_counts: list[int] = []
    for path in label_paths:
        labels = load_cwi_label_manifest(path)
        rows = sum(int(shard["rows"]) for shard in labels["shards"])
        batch_counts.append((rows + microbatch_size - 1) // microbatch_size)
    if any(count < 1 for count in batch_counts):
        raise MetaBridgeError("trajectory teacher contains no CWI batches")
    weights = list(manifest["teacher_weights"])
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    best_key: tuple[float, float, int] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    sampled_totals = [0 for _ in teachers]
    first_epoch = 1
    if resume_path is not None and resume_path.is_file():
        saved = torch.load(resume_path, map_location=target_device, weights_only=False)
        if saved['manifest_sha256'] != sha256_file(trajectory_cwi_manifest):
            raise ValueError('Selector resume CWI source changed')
        selector.load_state_dict(saved['selector'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        first_epoch = saved['epoch'] + 1
        best_key, best_state = saved['best_key'], saved['best_state']
        history, sampled_totals = saved['history'], saved['sampled_totals']
        from deltarec.utils.hstu_resume import restore_rng_state
        restore_rng_state(torch, saved['rng'])
    for epoch in range(first_epoch, epochs + 1):
        selector.train()
        rng = random.Random(int(seed) + 3000 + epoch)
        assignments = rng.choices(
            range(len(teachers)), weights=weights, k=max(batch_counts)
        )
        epoch_counts = [assignments.count(index) for index in range(len(teachers))]
        sampled_totals = [
            total + count for total, count in zip(sampled_totals, epoch_counts)
        ]
        iterators = [
            iter(_training_batches(path, microbatch_size=microbatch_size))
            for path in label_paths
        ]
        loss_sum = 0.0
        label_count = 0
        for teacher_index in assignments:
            try:
                histories, lengths, candidates, cwi1 = next(iterators[teacher_index])
            except StopIteration:
                iterators[teacher_index] = iter(
                    _training_batches(
                        label_paths[teacher_index], microbatch_size=microbatch_size
                    )
                )
                histories, lengths, candidates, cwi1 = next(iterators[teacher_index])
            histories = histories.to(target_device)
            lengths = lengths.to(target_device)
            candidates = candidates.to(target_device)
            # CWI targets are teacher-specific, but the selector feature space
            # is deliberately not.  Every teacher observation is projected
            # through the immutable theta_T snapshot owned by the selector;
            # otherwise the MLP would learn a moving mixture of embedding
            # spaces and sparse execution would no longer match distillation.
            history_features = final_embedding(histories.to(torch.int64))
            candidate_features = final_embedding(candidates.to(torch.int64))
            targets = torch.asinh(cwi1.to(target_device).float())
            optimizer.zero_grad(set_to_none=True)
            scores = selector.exact_scores_from_embeddings(
                history_features, candidate_features
            ).float()
            valid = torch.arange(scores.shape[-1], device=target_device)[
                None, None, :
            ] < lengths[:, None, None]
            residual = scores.masked_select(valid) - targets.masked_select(valid)
            loss = torch.nn.functional.smooth_l1_loss(
                residual, torch.zeros_like(residual), reduction="mean"
            )
            if not bool(torch.isfinite(loss)):
                raise MetaBridgeError("trajectory selector loss is nonfinite")
            loss.backward()
            optimizer.step()
            count = int(valid.sum()) * int(scores.shape[1])
            loss_sum += float(loss.detach()) * count
            label_count += count
        if label_count < 1:
            raise MetaBridgeError("trajectory selector consumed no CWI labels")
        selector.eval()
        with torch.inference_mode():
            validation_metrics = dict(validation_score_fn(selector, epoch))
        primary, tie_break = selector_validation_key(
            str(manifest["dataset"]), validation_metrics
        )
        key = (float(primary), float(tie_break), -epoch)
        record = {
            "epoch": epoch,
            "training_loss": loss_sum / label_count,
            "training_label_count": label_count,
            "sampled_teacher_counts": epoch_counts,
            "validation_primary": float(primary),
            "validation_tie_break": float(tie_break),
            "validation_metrics": validation_metrics,
        }
        history.append(record)
        if best_key is None or key > best_key:
            best_key = key
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in selector.state_dict().items()
                if name not in snapshot_state_keys
            }
        if resume_path is not None:
            from deltarec.utils.checkpoint import atomic_torch_save
            from deltarec.utils.hstu_resume import capture_rng_state
            atomic_torch_save(torch, dict(manifest_sha256=sha256_file(trajectory_cwi_manifest),
                epoch=epoch, selector=selector.state_dict(), optimizer=optimizer.state_dict(),
                best_key=best_key, best_state=best_state, history=history,
                sampled_totals=sampled_totals, rng=capture_rng_state(torch)), resume_path)
    if best_state is None:
        raise MetaBridgeError("trajectory selector did not select a checkpoint")
    loaded = selector.load_state_dict(best_state, strict=False)
    if set(loaded.missing_keys) != snapshot_state_keys or loaded.unexpected_keys:
        raise MetaBridgeError("trajectory best-state reload changed state coverage")
    for embedding, before in zip(teacher_embeddings, frozen_teacher_weights):
        if not torch.equal(embedding.weight.detach().cpu().contiguous(), before):
            raise MetaBridgeError("trajectory training mutated a teacher embedding")
    if not torch.equal(
        final_embedding.weight.detach().cpu(), frozen_teacher_weights[-1]
    ):
        raise MetaBridgeError("trajectory training mutated theta_T selector snapshot")
    return {
        "schema": "deltarec-headline-pc-selector-trajectory-training-v1",
        "label_fit_scope": dict(FIT_SCOPE),
        "selection_scope": dict(SELECTION_SCOPE),
        "optimizer": "adamw",
        "learning_rate": learning_rate,
        "epochs": epochs,
        "loss": PC_SELECTOR_LOSS,
        "selected_epoch": int(-best_key[2]),
        "trajectory_weights": weights,
        "sampled_teacher_counts": sampled_totals,
        "target_policy": "checkpoint-specific-observations-no-label-averaging",
        "final_selector_embedding_checkpoint_sha256": manifest[
            "final_selector_embedding_checkpoint_sha256"
        ],
        "optimizer_parameter_names": [name for name, _ in named_parameters],
        "embedding_parameter_in_optimizer": False,
        "best_epoch_reloaded": True,
        "history": history,
        "test_data_mounted": False,
    }
