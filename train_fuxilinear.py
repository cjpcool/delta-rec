from __future__ import annotations

import itertools

import json

import os

from pathlib import Path

import random

import socket

import time

from contextlib import nullcontext

from typing import Any, Iterable, Mapping, Sequence

import torch

import torch.nn.functional as F

from deltarec.models.fuxi_linear import FuxiLinearDeltaRec

from deltarec.models.hstu_multitask import checkpoint_state_without_legacy_head, hstu_multitask_bce

from deltarec.utils.io import atomic_write_json, sha256_file

from deltarec.utils.trajectory import SAMPLING_POLICY, TARGET_POLICY, TRAJECTORY_LATE_CHECKPOINT_SCHEMA, TRAJECTORY_LATE_SCHEMA, content_sha256, file_record, full_gdr_stage_plan, immutable_json, normalized_trajectory_weights, selector_lineage, tensor_state_sha256, trajectory_contract_fields, verify_selector_lineage_artifacts

from deltarec.adaptors import fuxi_model as bridge

from deltarec.data import training as data

from deltarec.utils.checkpoint import atomic_torch_save

from deltarec.utils.early_stopping import EarlyStoppingState

from deltarec.utils.accumulation import FuxiGradientAccumulation

from deltarec.data.history import iter_numeric_rating_examples

from deltarec.utils.protocols import AdapterOutput, ModelMode, RerankingBatch

from deltarec.metrics.evaluation import AtomicJsonlWriter, evaluate_kuai_stream, evaluate_rating_stream



CHECKPOINT_SCHEMA = TRAJECTORY_LATE_CHECKPOINT_SCHEMA

CONFIG_SCHEMA = TRAJECTORY_LATE_SCHEMA

STOP_REQUESTED = False

def read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value

def _write_or_verify_json(path: Path, value: Mapping[str, Any]) -> None:
    rendered = json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"refusing to overwrite immutable JSON: {path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")

def expected_trajectory_paths(config: Mapping[str, Any]) -> list[Path]:
    reused = [Path(path) for path in config.get("trajectory_checkpoint_paths", ())]
    count = int(config.get("trajectory_count", 0))
    if count < 1 or len(reused) > count:
        raise ValueError("invalid FuXi trajectory count/reused prefix")
    root = Path(config["output"]) / "full_gdr_trajectory"
    return reused + [
        root / f"epoch-{index:03d}.pt"
        for index in range(len(reused) + 1, count + 1)
    ]

def _contract(config: Mapping[str, Any]) -> dict[str, Any]:
    return trajectory_contract_fields(config)

def _state_from_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("model", "model_state_dict", "state_dict"):
        value = payload.get(key)
        if isinstance(value, Mapping) and value:
            return value
    if payload and all(isinstance(key, str) for key in payload):
        values = list(payload.values())
        if values and all(torch.is_tensor(value) for value in values):
            return payload
    raise ValueError("FuXi checkpoint has no model state mapping")

def _strip_uniform_prefixes(state: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(state)
    for prefix in ("module.", "_orig_mod.", "backbone."):
        if value and all(key.startswith(prefix) for key in value):
            value = {key[len(prefix):]: tensor for key, tensor in value.items()}
    return value

def _base_model_state(model: FuxiLinearDeltaRec) -> dict[str, torch.Tensor]:
    if model.task_head is not None:
        return {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
            if key.startswith(("backbone.", "task_head."))
        }
    return model.base_model_state()

def _load_base_model_state(model: FuxiLinearDeltaRec, state: Mapping[str, Any]) -> None:
    if model.task_head is None and any(key.startswith("task_head.") for key in state):
        model.initialize_ranking_head()
    if model.task_head is not None:
        state = _strip_uniform_prefixes(state)
        backbone_keys = set(model.backbone.state_dict())
        if state and not any(key.startswith("backbone.") for key in state):
            state = {
                f"backbone.{key}" if key in backbone_keys else key: value
                for key, value in state.items()
            }
        compatible, fresh_head = checkpoint_state_without_legacy_head(
            model, state
        )
        result = model.load_state_dict(compatible, strict=False)
        missing = [
            key for key in result.missing_keys
            if not key.startswith((
                "selector.", "selector_", "prototypes", "item_to_group"
            )) and key not in fresh_head
        ]
        if result.unexpected_keys or missing:
            raise ValueError(
                "FuXi checkpoint state coverage changed: "
                f"missing={missing}, unexpected={result.unexpected_keys}"
            )
        return
    model.load_base_model_state(_strip_uniform_prefixes(state))

def _read_trajectory_checkpoint(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate/read a trajectory checkpoint without mutating a live model."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    schema = payload.get("schema")
    state = _state_from_payload(payload)
    legacy = schema != CHECKPOINT_SCHEMA
    if schema == CHECKPOINT_SCHEMA:
        if payload.get("contract") != _contract(config):
            raise ValueError(f"FuXi trajectory checkpoint contract changed: {path}")
        expected_hash = payload.get("checkpoint_content_sha256")
        if expected_hash and expected_hash != tensor_state_sha256(state):
            raise ValueError(f"FuXi trajectory checkpoint tensor hash changed: {path}")
    else:
        legacy_config = payload.get("config", {})
        explicit_full_gdr = (
            payload.get("stage") in {"full_gdr", "full_gdr_trajectory"}
            or "full-gdr" in str(legacy_config.get("pipeline", "")).lower()
            or "full_gdr" in str(legacy_config.get("stage", "")).lower()
        )
        if not explicit_full_gdr:
            raise ValueError(
                f"{path} is not an explicitly identified FuXi Full-GDR checkpoint; "
                "ordinary FuXi baseline checkpoints cannot be promoted silently"
            )
        identity_ok = (
            payload.get("run_fingerprint") == config.get("run_fingerprint")
            or payload.get("model_fingerprint") == config.get("model_fingerprint")
            or (
                isinstance(legacy_config, Mapping)
                and legacy_config.get("model_fingerprint")
                == config.get("model_fingerprint")
            )
        )
        if not identity_ok:
            raise ValueError(
                f"legacy FuXi checkpoint lacks an explicit compatible model fingerprint: {path}"
            )
        payload["_trajectory_compatibility"] = "legacy-dense-checkpoint-read-only"
    if legacy:
        payload["_trajectory_compatibility"] = "legacy-dense-checkpoint-read-only"
    return dict(payload)

def load_trajectory_checkpoint(
    path: Path,
    config: Mapping[str, Any],
    model: FuxiLinearDeltaRec,
) -> dict[str, Any]:
    payload = _read_trajectory_checkpoint(path, config)
    _load_base_model_state(model, _state_from_payload(payload))
    return payload

def _write_trajectory_manifest(config: Mapping[str, Any], paths: Sequence[Path]) -> dict[str, Any]:
    weights = normalized_trajectory_weights(
        int(config["trajectory_count"]), config["trajectory_weights"]
    )
    records = []
    previous_epoch = 0
    previous_step = -1
    for index, path in enumerate(paths, 1):
        payload = _read_trajectory_checkpoint(path, config)
        state = _state_from_payload(payload)
        epoch = int(payload.get("epoch", index))
        raw_step = payload.get("global_step")
        step = int(raw_step if raw_step is not None else payload.get("step", 0))
        if epoch <= previous_epoch or step < previous_step:
            raise ValueError("FuXi trajectory checkpoint order is not increasing")
        if payload.get("schema") == CHECKPOINT_SCHEMA:
            if payload.get("trajectory_index") != index:
                raise ValueError("native FuXi trajectory index is not contiguous")
            if index > 1 and payload.get("parent_checkpoint") != str(Path(paths[index - 2]).resolve()):
                raise ValueError("native FuXi trajectory parent is not the previous checkpoint")
        parent_checkpoint = payload.get("parent_checkpoint")
        if parent_checkpoint is None and index > 1:
            # Keep the old checkpoint byte-for-byte intact while making its
            # ordered-prefix lineage explicit in the new manifest.
            parent_checkpoint = str(Path(paths[index - 2]).resolve())
        parent_checkpoint_sha256 = payload.get("parent_checkpoint_sha256")
        if parent_checkpoint_sha256 is None and parent_checkpoint is not None:
            parent_checkpoint_sha256 = sha256_file(parent_checkpoint)
        records.append(
            dict(
                trajectory_index=index,
                path=str(path.resolve()),
                sha256=sha256_file(path),
                epoch=epoch,
                global_step=step,
                parent_checkpoint=parent_checkpoint,
                parent_checkpoint_sha256=parent_checkpoint_sha256,
                checkpoint_content_sha256=payload.get(
                    "checkpoint_content_sha256", tensor_state_sha256(state)
                ),
                teacher_weight=weights[index - 1],
                compatibility=payload.get(
                    "_trajectory_compatibility", "native-v2"
                ),
            )
        )
        previous_epoch = epoch
        previous_step = step
    manifest = dict(
        schema="deltarec-fuxi-linear-trajectory-lineage-v2",
        protocol_schema=CONFIG_SCHEMA,
        dataset=config["dataset"],
        backbone=config["backbone"],
        variant=config["variant"],
        seed=config["seed"],
        contract=_contract(config),
        trajectory_count=int(config["trajectory_count"]),
        checkpoints_present=len(paths),
        trajectory_complete=len(paths) == int(config["trajectory_count"]),
        trajectory_sampling="late",
        normalized_trajectory_weights=weights,
        target_policy=TARGET_POLICY,
        checkpoints=records,
        selector_feature_snapshot="final-theta-T-item-table",
        optimizer_scheduler_rng_restored=False,
    )
    manifest["manifest_content_sha256"] = content_sha256(
        manifest, "manifest_content_sha256"
    )
    manifest_path = Path(config["output"]) / "trajectory_manifest.json"
    if manifest_path.exists():
        previous = read(manifest_path)
        if previous.get("manifest_content_sha256") != content_sha256(
            previous, "manifest_content_sha256"
        ):
            raise ValueError("existing FuXi trajectory lineage checksum changed")
        previous_checkpoints = previous.get("checkpoints", [])
        if (
            previous.get("schema") != manifest["schema"]
            or previous.get("contract") != manifest["contract"]
            or previous.get("trajectory_count") != manifest["trajectory_count"]
            or previous.get("normalized_trajectory_weights")
            != manifest["normalized_trajectory_weights"]
            or previous.get("target_policy") != manifest["target_policy"]
            or not isinstance(previous_checkpoints, list)
            or len(previous_checkpoints) > len(records)
            or previous_checkpoints != records[: len(previous_checkpoints)]
        ):
            raise ValueError("existing FuXi trajectory lineage is not an immutable prefix")
        if len(previous_checkpoints) == len(records):
            immutable_json(manifest_path, manifest)
            return manifest
    # The lineage manifest is atomically replaced as each new epoch becomes
    # durable; checkpoint files themselves remain immutable.
    atomic_write_json(manifest_path, manifest)
    return manifest

def save_trajectory_checkpoint(
    path: Path,
    config: Mapping[str, Any],
    model: FuxiLinearDeltaRec,
    *,
    epoch: int,
    global_step: int,
    trajectory_index: int,
    parent_checkpoint: Path | None,
    continuation: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = torch.load(path, map_location="cpu", weights_only=False)
        if existing.get("schema") == CHECKPOINT_SCHEMA and existing.get("contract") != _contract(config):
            raise ValueError(f"refusing to overwrite incompatible checkpoint: {path}")
        if existing.get("schema") == CHECKPOINT_SCHEMA:
            expected_hash = existing.get("checkpoint_content_sha256")
            if expected_hash and expected_hash != tensor_state_sha256(
                _state_from_payload(existing)
            ):
                raise ValueError(f"existing FuXi trajectory checkpoint checksum changed: {path}")
        else:
            # A legacy checkpoint can be reused only through the explicit
            # compatibility loader; never silently replace it in-place.
            load_trajectory_checkpoint(path, config, model)
        return
    state = _base_model_state(model)
    parent = None if parent_checkpoint is None else parent_checkpoint.resolve()
    weights = normalized_trajectory_weights(
        int(config["trajectory_count"]), config["trajectory_weights"]
    )
    if trajectory_index < 1 or trajectory_index > len(weights):
        raise ValueError(
            f"invalid FuXi trajectory index {trajectory_index} for count {len(weights)}"
        )
    payload = dict(
        schema=CHECKPOINT_SCHEMA,
        contract=_contract(config),
        config=dict(config),
        model=state,
        epoch=int(epoch),
        global_step=int(global_step),
        step=int(global_step),
        stage="full_gdr_trajectory",
        full_gdr_execution=dict(
            dense_dynamic_width=model.dense_dynamic_width,
            length_bucketing=getattr(model, "full_gdr_length_bucketing", False),
        ),
        trajectory_index=int(trajectory_index),
        trajectory_teacher_weight=weights[trajectory_index - 1],
        parent_checkpoint=None if parent is None else str(parent),
        parent_checkpoint_sha256=None if parent is None else sha256_file(parent),
        continuation_mode=(
            "model-only-fresh-optimizer-and-schedule"
            if continuation
            else "scratch-or-true-model-state"
        ),
        optimizer_restored=False,
        scheduler_restored=False,
        rng_restored=False,
        model_fingerprint=config.get("model_fingerprint"),
        run_fingerprint=config.get("run_fingerprint"),
        protocol_config_sha256=config.get("config_content_sha256"),
        checkpoint_content_sha256=tensor_state_sha256(state),
    )
    atomic_torch_save(torch, payload, path, overwrite=False)
    paths = expected_trajectory_paths(config)
    if len(paths) >= trajectory_index and paths[trajectory_index - 1] == path:
        _write_trajectory_manifest(config, paths[:trajectory_index])

def _numeric_rows(config: Mapping[str, Any], files: Mapping[str, Path]) -> Iterable[Mapping[str, Any]]:
    if config.get("dataset") == "kuairand-1k":
        return data._slate_examples(files["train"], role="train")
    return iter_numeric_rating_examples(
        files["train"], max_history_length=1024, include_timestamps=True
    )

def _bounded_shuffle(rows: Iterable[Mapping[str, Any]], seed: int, buffer_size: int = 32768):
    rng = random.Random(seed)
    buffer: list[Mapping[str, Any]] = []
    for row in rows:
        if len(buffer) < buffer_size:
            buffer.append(row)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = row
    rng.shuffle(buffer)
    yield from buffer

def _batches(config: Mapping[str, Any], files: Mapping[str, Path], epoch: int, *, size=None):
    rows = _bounded_shuffle(_numeric_rows(config, files), int(config["seed"]) + epoch)
    size = int(config["microbatch"] if size is None else size)
    while True:
        batch = list(itertools.islice(rows, size))
        if not batch:
            return
        yield batch

def _sample_training_rows(
    config: Mapping[str, Any], files: Mapping[str, Path], limit: int, *, seed: int, min_history: int
) -> list[Mapping[str, Any]]:
    cache = Path(config["output"]) / f"training-samples-s{seed}-n{limit}-l{min_history}.pt"
    binding = dict(
        data_binding_sha256=config["binding_sha256"],
        code=config["code"], seed=seed, limit=limit, min_history=min_history,
    )
    if cache.exists():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        if payload.get("binding") != binding:
            raise ValueError("FuXi training sample cache binding changed")
        return payload["rows"]
    rng = random.Random(seed)
    result: list[Mapping[str, Any]] = []
    seen = 0
    for row in _numeric_rows(config, files):
        if len(row["history"]) < min_history:
            continue
        seen += 1
        if len(result) < limit:
            result.append(row)
        else:
            index = rng.randrange(seen)
            if index < limit:
                result[index] = row
    if not result:
        raise ValueError("no eligible FuXi CWI histories")
    rng.shuffle(result)
    atomic_torch_save(torch, dict(binding=binding, rows=result), cache, overwrite=False)
    return result

def _bucket_window(window):
    """Repack only this optimizer update; preserve its samples and batch sizes."""
    rows = iter(sorted(
        itertools.chain.from_iterable(window), key=lambda row: len(row["history"])
    ))
    return [list(itertools.islice(rows, len(micro))) for micro in window]

def _collate(config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], device: Any) -> dict[str, torch.Tensor]:
    if config.get("dataset") == "kuairand-1k":
        batch = dict(data._collate_slates(torch, rows, device))
        # The canonical frozen Kuai slates do not expose history timestamps to
        # model training.  Match the official FuXi bridge's zero-time fallback.
        batch["timestamps"] = torch.zeros_like(batch["histories"])
        return batch
    width = max(len(row["history"]) for row in rows)
    batch = data._collate_rating(torch, rows, device, history_width=width)
    timestamp_rows = []
    for row in rows:
        values = row.get("history_timestamps")
        if values is None or len(values) != len(row["history"]):
            raise ValueError("FuXi training rows must carry aligned history timestamps")
        timestamp_rows.append(list(values) + [0] * (width - len(values)))
    batch["timestamps"] = torch.tensor(timestamp_rows, dtype=torch.long).to(device)
    return batch

def _candidates(config: Mapping[str, Any], batch: Mapping[str, torch.Tensor], catalog_ids: torch.Tensor) -> torch.Tensor:
    if config.get("dataset") == "kuairand-1k":
        return batch["candidates"]
    negatives = catalog_ids[torch.randint(
        catalog_ids.numel(),
        (batch["targets"].shape[0], int(config["negatives"])),
        device=catalog_ids.device,
    )]
    return torch.cat((batch["targets"][:, None], negatives), dim=1)

def _loss(config: Mapping[str, Any], model: FuxiLinearDeltaRec, history: Any,
          batch: Mapping[str, torch.Tensor], candidates: torch.Tensor) -> torch.Tensor:
    scores = model.read(history, candidates)
    if config.get("dataset") == "kuairand-1k":
        return hstu_multitask_bce(
            scores, batch["labels"], batch.get("label_weights")
        )
    accidental_hits = candidates[:, 1:] == batch["targets"][:, None]
    logits = scores.float()
    if model.task_head is None:
        # Preserve the original dense teacher objective.  Sparse rating runs
        # attach the vector-level ranker and use its candidate logits directly.
        logits = logits / float(config["temperature"])
    logits = torch.cat(
        (logits[:, :1], logits[:, 1:].masked_fill(accidental_hits, -torch.inf)), dim=-1
    )
    return F.cross_entropy(
        logits, torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    )

def _autocast(model: FuxiLinearDeltaRec):
    return torch.autocast("cuda", dtype=torch.bfloat16) if model.item_embedding.weight.is_cuda else nullcontext()

def _emit(output: Path, stage: str, **values: Any) -> None:
    record = dict(stage=stage, time=time.time(), **values)
    atomic_write_json(output / "progress.json", record)
    print(json.dumps(record, sort_keys=True), flush=True)

class _Backend:
    """Adapter of the shared trajectory engine vocabulary for FuXi."""

    batches = staticmethod(_batches)
    collate = staticmethod(_collate)
    candidates_for_loss = staticmethod(_candidates)
    recommendation_loss = staticmethod(_loss)
    autocast = staticmethod(_autocast)
    emit = staticmethod(_emit)
    sample_training_rows = staticmethod(_sample_training_rows)

    @staticmethod
    def base_model_state(model):
        return _base_model_state(model)

    @staticmethod
    def load_base_model_state(model, state):
        _load_base_model_state(model, state)

    @staticmethod
    def make_optimizer(config, model):
        return torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]),
        )

    @staticmethod
    def freeze_selector(model):
        model.freeze_selector()

    @staticmethod
    def cwi_selector_loss(scores, importance, valid):
        if scores.shape != importance.shape or scores.shape != valid.shape:
            raise ValueError("FuXi selector tensors have inconsistent shapes")
        if not bool(valid.any()):
            raise ValueError("FuXi selector batch has no valid event")
        return F.smooth_l1_loss(
            scores.float().masked_select(valid),
            torch.asinh(importance.detach().float()).masked_select(valid),
        )

    @staticmethod
    def early_state(config):
        return EarlyStoppingState(
            min_delta=float(config["min_delta"]),
            patience=int(config["patience"]),
            minimum_epochs=int(config["min_epochs"]),
            maximum_epochs=int(config["max_epochs"]),
            tie_breaker_mode=(
                "min" if config["dataset"] == "kuairand-1k" else None
            ),
        )

    @staticmethod
    def primary_metric(_run, metrics):
        return metrics[
            "macro_gauc" if "macro_gauc" in metrics else "ndcg_at_10"
        ]

    @staticmethod
    def save_local(path, config, model, optimizer, stage, **state):
        semantics = state.pop("resume_semantics", "true-resume")
        model_state = {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
            if key != "selector_embedding"
        }
        atomic_torch_save(
            torch,
            dict(
                schema="deltarec-fuxi-linear-resume-v2",
                config=dict(config), model=model_state,
                optimizer=None if optimizer is None else optimizer.state_dict(),
                stage=stage, resume_semantics=semantics,
                rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                python_rng=random.getstate(), scheduler_contract="none",
                scheduler_state=None,
                sampler_contract="deterministic-bounded-shuffle(seed+epoch)",
                full_gdr_execution=dict(
                    dense_dynamic_width=model.dense_dynamic_width,
                    length_bucketing=getattr(model, "full_gdr_length_bucketing", False),
                ),
                **state,
            ),
            path,
        )

    @staticmethod
    def evaluate(config, model, files, output, *, label, split="validation"):
        output.mkdir(parents=True, exist_ok=True)
        evidence_path = output / f"{label}-rows.jsonl"
        was_training = model.training
        model.eval()
        try:
            with AtomicJsonlWriter(evidence_path) as writer:
                common = dict(
                    torch=torch, adapter=_EvaluationAdapter(model),
                    split_role=split, protocol_lock_hash=None,
                    microbatch_size=int(config.get("eval_microbatch", 32)),
                    evidence=writer,
                )
                if config["dataset"] == "kuairand-1k":
                    metrics, details, digest = evaluate_kuai_stream(
                        **common,
                        slate_file=files["validation"].parent / f"{split}_slates.csv",
                        split_manifest=files["split_manifest"],
                    )
                else:
                    metrics, details, digest = evaluate_rating_stream(
                        **common, candidate_manifest=files["validation_candidates"],
                        dataset=config["dataset"],
                    )
        finally:
            model.train(was_training)
        result = dict(
            dataset=config["dataset"], method="DeltaRec-GC (FuXi-Linear backend)",
            split=split, metrics=metrics, details=details,
            input_sha256=digest, evidence=str(evidence_path),
        )
        atomic_write_json(output / f"{label}.json", result)
        return result

class _EvaluationAdapter:
    def __init__(self, model: FuxiLinearDeltaRec):
        self.model = model

    @torch.inference_mode()
    def run(self, batch: RerankingBatch, mode: ModelMode) -> AdapterOutput:
        del mode
        if self.model.training:
            raise RuntimeError("FuXi validation requires model.eval()")
        device = self.model.item_embedding.weight.device
        histories = batch.history_item_ids.to(device)
        lengths = batch.history_lengths.to(device)
        candidates = batch.candidate_item_ids.to(device)
        with _autocast(self.model):
            timestamps = (
                None if batch.timestamps is None else batch.timestamps.to(device)
            )
            history = self.model.prefill(histories, lengths, timestamps=timestamps)
            scores = self.model.read(history, candidates)
        return AdapterOutput(scores)

def _restore_local(path: Path, config: Mapping[str, Any], model: FuxiLinearDeltaRec,
                   catalog_ids: torch.Tensor, final_path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("schema") != "deltarec-fuxi-linear-resume-v2" or saved.get("config") != dict(config):
        raise ValueError("FuXi node-local resume configuration mismatch")
    if saved.get("stage") == "sparse" and config["dataset"] != "kuairand-1k" and config.get("rating_prediction_head", "pair_mlp") == "pair_mlp":
        model.initialize_ranking_head()
    compatible, fresh_head = checkpoint_state_without_legacy_head(
        model, saved["model"]
    ) if model.task_head is not None else (saved["model"], set())
    result = model.load_state_dict(compatible, strict=False)
    missing = [
        key for key in result.missing_keys
        if key != "selector_embedding" and key not in fresh_head
    ]
    if result.unexpected_keys or missing:
        raise ValueError(f"FuXi local resume state coverage changed: missing={missing}, unexpected={result.unexpected_keys}")
    if fresh_head:
        saved["optimizer"] = None
        saved["epoch"] = 0
        saved["early"] = None
        saved["cursor"] = {}
        saved["prediction_head_reinitialized"] = True
    if saved.get("stage") in {"selector", "sparse"}:
        parent = torch.load(final_path, map_location="cpu", weights_only=False)
        model_state = _state_from_payload(parent)
        table = None
        for key, value in model_state.items():
            if str(key).endswith("_embedding_module._item_emb.weight"):
                table = value
                break
        if table is None:
            raise ValueError("FuXi final trajectory lacks its item embedding table")
        model.bind_selector_space(catalog_ids, feature_table=table)
    torch.set_rng_state(saved["rng"])
    torch.cuda.set_rng_state_all(saved["cuda_rng"])
    random.setstate(saved["python_rng"])
    execution = saved.get("full_gdr_execution", {})
    model.dense_dynamic_width = execution.get("dense_dynamic_width", model.dense_dynamic_width)
    model.full_gdr_length_bucketing = execution.get(
        "length_bucketing", getattr(model, "full_gdr_length_bucketing", False)
    )
    return saved

def _selector_record_loss(model, record, device, *, heldout):
    partition = record["users"].remainder(5).eq(0)
    mask = partition if heldout else ~partition
    if not bool(mask.any()):
        return None
    histories = record["history"][mask].to(device)
    lengths = record["lengths"][mask].to(device)
    importance = record["importance"][mask].to(device)
    active = record["active_groups"][mask].to(device)
    scores = model.selector_scores(histories)
    valid = torch.arange(histories.shape[1], device=device)[None, None] < lengths[:, None, None]
    valid = valid.expand_as(scores)
    return _Backend.cwi_selector_loss(scores[active], importance[active], valid[active]), int(active.sum())

def _generate_cwi(config, model, files, catalog_ids, output, checkpoint_path, index, sampled):
    root = output / "cwi_trajectory"
    root.mkdir(parents=True, exist_ok=True)
    shard = root / f"teacher-{index:03d}.pt"
    binding = dict(
        checkpoint=str(checkpoint_path.resolve()), checkpoint_sha256=sha256_file(checkpoint_path),
        index=index, data_binding_sha256=config["binding_sha256"],
    )
    if shard.exists():
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        if payload.get("binding") != binding:
            raise ValueError(f"FuXi CWI shard binding changed: {shard}")
        manifest_path = shard.with_suffix(".json")
        if not manifest_path.is_file() or read(manifest_path).get("target_policy") != TARGET_POLICY:
            raise ValueError(f"FuXi CWI shard manifest is missing/changed: {shard}")
        return payload["records"]
    records = []
    torch.manual_seed(int(config["seed"]) + 2000 + index)
    for batch_index, start in enumerate(range(0, len(sampled), int(config["cwi_microbatch"]))):
        rows = sampled[start : start + int(config["cwi_microbatch"])]
        batch = _collate(config, rows, model.item_embedding.weight.device)
        candidates = _candidates(config, batch, catalog_ids)
        gates = torch.ones(
            batch["histories"].shape[0], model.group_count, batch["histories"].shape[1],
            device=catalog_ids.device, requires_grad=True,
        )
        with _autocast(model):
            history = model.prefill(
                batch["histories"], batch["lengths"], sparse=False,
                event_gates=gates, timestamps=batch["timestamps"],
            )
            loss = _loss(config, model, history, batch, candidates)
        (gradient,) = torch.autograd.grad(loss, gates)
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("nonfinite FuXi CWI label")
        records.append(
            dict(
                history=batch["histories"].cpu(), lengths=batch["lengths"].cpu(),
                timestamps=batch["timestamps"].cpu(),
                users=batch["users"].cpu(), importance=-gradient.detach().cpu(),
                active_groups=F.one_hot(
                    model.item_to_group[candidates], model.group_count
                ).any(1).cpu(),
            )
        )
        if batch_index % 8 == 0:
            _emit(output, "trajectory-cwi-labels", teacher=index,
                  batches=batch_index + 1, total=config["cwi_batches"])
    atomic_torch_save(
        torch,
        dict(schema="deltarec-fuxi-linear-cwi-trajectory-shard-v2",
             binding=binding, target_policy=TARGET_POLICY, records=records),
        shard,
        overwrite=False,
    )
    manifest = dict(
        schema="deltarec-fuxi-linear-cwi-teacher-manifest-v2",
        dataset=config["dataset"], backbone=config["backbone"], seed=config["seed"],
        trajectory_index=index, parent_checkpoint=file_record(checkpoint_path),
        data_binding_sha256=config["binding_sha256"], target_policy=TARGET_POLICY,
        shard=file_record(shard), rows=len(records), label_space="signed-CWI1-per-group-event",
        validation_candidates_mounted=False, test_data_mounted=False,
    )
    manifest["manifest_content_sha256"] = content_sha256(
        manifest, "manifest_content_sha256"
    )
    immutable_json(shard.with_suffix(".json"), manifest)
    return records

def fit_selector(config, model, files, catalog_ids, output, paths):
    sampled = _sample_training_rows(
        config, files, int(config["cwi_batches"]) * int(config["cwi_microbatch"]),
        seed=1000, min_history=129,
    )
    load_trajectory_checkpoint(paths[-1], config, model)
    final_table = model.item_embedding.weight.detach().cpu().contiguous().clone()
    final_table_sha = tensor_state_sha256({"item_embedding.weight": final_table})
    model.bind_selector_space(catalog_ids, feature_table=final_table)
    datasets = []
    teacher_manifests = []
    for index, path in enumerate(paths, 1):
        load_trajectory_checkpoint(path, config, model)
        model.bind_selector_space(catalog_ids, feature_table=final_table)
        model.eval()
        model.requires_grad_(False)
        datasets.append(_generate_cwi(config, model, files, catalog_ids, output, path, index, sampled))
        teacher_manifests.append(output / "cwi_trajectory" / f"teacher-{index:03d}.json")
    weights = normalized_trajectory_weights(len(datasets), config["trajectory_weights"])
    cwi_manifest = dict(
        schema="deltarec-fuxi-linear-cwi-trajectory-manifest-v2",
        protocol_schema=CONFIG_SCHEMA, dataset=config["dataset"], backbone=config["backbone"],
        seed=config["seed"], teachers=[read(path) for path in teacher_manifests],
        normalized_trajectory_weights=weights, target_policy=TARGET_POLICY,
        sampling_policy=SAMPLING_POLICY,
        final_theta_T_checkpoint=str(paths[-1].resolve()),
        final_theta_T_checkpoint_sha256=sha256_file(paths[-1]),
        frozen_embedding_snapshot_sha256=final_table_sha,
        validation_candidates_mounted=False, test_data_mounted=False,
    )
    cwi_manifest["manifest_content_sha256"] = content_sha256(
        cwi_manifest, "manifest_content_sha256"
    )
    cwi_path = output / "cwi_trajectory_manifest.json"
    immutable_json(cwi_path, cwi_manifest)
    model.selector.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.selector.parameters(), lr=1e-3, weight_decay=0.)
    best = float("inf")
    best_state = None
    history = []
    totals = [0] * len(datasets)
    device = catalog_ids.device
    for epoch in range(int(config["selector_epochs"])):
        rng = random.Random(int(config["seed"]) + 3000 + epoch)
        assignments = rng.choices(range(len(datasets)), weights=weights, k=max(len(rows) for rows in datasets))
        counts = [assignments.count(index) for index in range(len(datasets))]
        totals = [left + right for left, right in zip(totals, counts)]
        train_sum = train_count = 0
        for teacher, records in enumerate(datasets):
            chosen = [position for position, assignment in enumerate(assignments) if assignment == teacher]
            for position in chosen:
                result = _selector_record_loss(model, records[position % len(records)], device, heldout=False)
                if result is None:
                    continue
                loss, count = result
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite FuXi selector loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.selector.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                train_sum += float(loss.detach()) * count
                train_count += count
        heldout_sum = heldout_count = 0.
        with torch.no_grad():
            for teacher, records in enumerate(datasets):
                for record in records:
                    result = _selector_record_loss(model, record, device, heldout=True)
                    if result is None:
                        continue
                    loss, count = result
                    heldout_sum += weights[teacher] * float(loss) * count
                    heldout_count += weights[teacher] * count
        if not train_count or not heldout_count:
            raise ValueError("FuXi CWI observations do not cover both selector partitions")
        row = dict(epoch=epoch, train_loss=train_sum / train_count,
                   heldout_loss=heldout_sum / heldout_count, sampled_teacher_counts=counts)
        history.append(row)
        if row["heldout_loss"] < best:
            best = row["heldout_loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.selector.state_dict().items()}
        _emit(output, "trajectory-selector-fit", **row)
    if best_state is None:
        raise ValueError("FuXi selector did not produce a best state")
    model.selector.load_state_dict(best_state, strict=True)
    load_trajectory_checkpoint(paths[-1], config, model)
    model.bind_selector_space(catalog_ids, feature_table=final_table)
    model.requires_grad_(True)
    model.freeze_selector()
    _write_or_verify_json(output / "selector_fit.json", dict(
        schema="deltarec-trajectory-late-v2", history=history, best_heldout_loss=best,
        trajectory_weights_normalized=weights, sampled_teacher_counts=totals,
        target_policy=TARGET_POLICY, sampling_policy=SAMPLING_POLICY,
        trajectory_cwi_manifest=str(cwi_path), trajectory_cwi_manifest_sha256=sha256_file(cwi_path),
        frozen_embedding_snapshot_sha256=final_table_sha,
        frozen_embedding_snapshot_source="theta_T item embedding table; used for every teacher observation",
        final_selector_embedding_checkpoint=str(paths[-1]),
        final_selector_embedding_checkpoint_sha256=sha256_file(paths[-1]),
    ))

def save_selector(output: Path, config: Mapping[str, Any], model, final_path: Path) -> dict[str, Any]:
    path = output / "selector.pt"
    state = {key: value.detach().cpu() for key, value in model.selector.state_dict().items()}
    payload = dict(
        schema="deltarec-trajectory-late-selector-checkpoint-v2",
        protocol_schema=CONFIG_SCHEMA, config=dict(config), state=state,
        selector_state_sha256=tensor_state_sha256(state),
        embedding_ownership="frozen-final-trajectory-snapshot", legacy_selector_resume=False,
    )
    if path.exists():
        existing = torch.load(path, map_location="cpu", weights_only=False)
        existing_state = existing.get("state", {})
        same_state = isinstance(existing_state, dict) and set(existing_state) == set(state) and all(
            torch.equal(existing_state[key], state[key]) for key in state
        )
        if (existing.get("schema") != payload["schema"] or existing.get("config") != dict(config)
                or existing.get("legacy_selector_resume") is not False
                or existing.get("selector_state_sha256") != payload["selector_state_sha256"]
                or not same_state):
            raise ValueError("refusing to overwrite a different FuXi selector")
    else:
        atomic_torch_save(torch, payload, path, overwrite=False)
    trajectory_manifest = read(output / "trajectory_manifest.json")
    cwi_manifest = read(output / "cwi_trajectory_manifest.json")
    lineage = selector_lineage(
        config=config, trajectory=trajectory_manifest["checkpoints"],
        cwi_manifest=output / "cwi_trajectory_manifest.json",
        final_snapshot_sha256=cwi_manifest["frozen_embedding_snapshot_sha256"],
        selector_checkpoint=path,
    )
    binding = dict(
        **lineage, path=str(path.resolve()), sha256=sha256_file(path),
        selector_state_sha256=payload["selector_state_sha256"], grouping_sha256=config["grouping_sha256"],
        feature_space="frozen-final-trajectory-item-table",
        feature_checkpoint=str(final_path.resolve()), feature_checkpoint_sha256=sha256_file(final_path),
        artifact="trajectory-late-cwi-mlp-no-embedding-rows",
        prototype_reduce="training-catalog-group-means", empty_group="training-catalog-global-mean",
    )
    immutable_json(output / "selector_binding.json", binding)
    return binding

def _verify_selector_artifacts(config, output, final_path):
    path = output / "selector.pt"
    binding_path = output / "selector_binding.json"
    if not path.is_file() or not binding_path.is_file():
        raise FileNotFoundError("FuXi selector artifacts are incomplete")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "deltarec-trajectory-late-selector-checkpoint-v2" or payload.get("config") != dict(config):
        raise ValueError("legacy/incompatible selector cannot be resumed as Trajectory-Late")
    if payload.get("legacy_selector_resume") is not False or payload.get("selector_state_sha256") != tensor_state_sha256(payload["state"]):
        raise ValueError("FuXi selector provenance/checksum changed")
    binding = read(binding_path)
    expected_paths = expected_trajectory_paths(config)
    verify_selector_lineage_artifacts(
        binding,
        trajectory_manifest_path=output / "trajectory_manifest.json",
        final_checkpoint=final_path,
        expected_count=len(expected_paths),
        expected_weights=normalized_trajectory_weights(
            len(expected_paths), config.get("trajectory_weights")
        ),
    )
    if binding.get("path") != str(path.resolve()) or binding.get("sha256") != sha256_file(path):
        raise ValueError("FuXi selector binding does not match selector artifact")
    if binding.get("feature_checkpoint") != str(final_path.resolve()):
        raise ValueError("FuXi selector is not bound to theta_T")
    if (binding.get("selector_state_sha256") != payload["selector_state_sha256"]
            or binding.get("grouping_sha256") != config["grouping_sha256"]):
        raise ValueError("FuXi selector binding state/grouping provenance changed")
    return payload, binding

def load_selector(config, model, output, catalog_ids, final_path):
    payload, binding = _verify_selector_artifacts(config, output, final_path)
    load_trajectory_checkpoint(final_path, config, model)
    final_table = model.item_embedding.weight.detach().cpu().contiguous().clone()
    model.bind_selector_space(catalog_ids, feature_table=final_table)
    model.selector.load_state_dict(payload["state"], strict=True)
    model.requires_grad_(True)
    model.freeze_selector()
    return binding

def _train_full_gdr(config, model, files, catalog_ids, output, latest, saved):
    paths = expected_trajectory_paths(config)
    reused_count = len(config.get("trajectory_checkpoint_paths", ()))
    completed = 0
    for path in paths:
        if not path.is_file():
            break
        # Validate the durable prefix without changing the model restored from
        # the node-local checkpoint.  In particular, a true resume at an epoch
        # boundary must not be overwritten by the last trajectory checkpoint.
        _read_trajectory_checkpoint(path, config)
        completed += 1
    if completed < reused_count:
        raise FileNotFoundError(f"reused FuXi trajectory gap at {paths[completed]}")
    if completed == len(paths):
        _write_trajectory_manifest(config, paths)
        return paths

    plan = full_gdr_stage_plan(
        saved=saved,
        completed=completed,
        reused_count=reused_count,
        continuation_parent=config.get("continuation_parent_checkpoint"),
    )
    saved_epoch = plan["saved_epoch"]
    saved_semantics = plan["saved_semantics"]
    resume_local = plan["resume_local"]

    # A supplied Full-GDR parent is an intentional model-only continuation.
    # It never inherits Adam, scheduler, or RNG state from the parent run.
    parent_arg = config.get("continuation_parent_checkpoint")
    continuation = plan["continuation"]
    if resume_local:
        optimizer = _Backend.make_optimizer(config, model)
        if not saved.get("optimizer"):
            raise ValueError("true Full-GDR resume is missing optimizer state")
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = saved_epoch
        saved_cursor = saved.get("cursor", {})
        global_step = int(saved_cursor.get("global_step", saved_cursor.get("step", 0)))
        resume_semantics = saved_semantics
    elif continuation:
        parent_path = paths[completed - 1] if completed else parent_arg
        if parent_path is None:
            raise ValueError("FuXi continuation has no parent checkpoint")
        parent_path = Path(parent_path).resolve()
        if not parent_path.is_file():
            raise FileNotFoundError(parent_path)
        parent = load_trajectory_checkpoint(parent_path, config, model)
        start_epoch = completed
        global_step = int(parent.get("global_step", parent.get("step", 0)))
        optimizer = _Backend.make_optimizer(config, model)
        resume_semantics = "continuation-model-only-fresh-optimizer-schedule"
        saved_cursor = {}
    else:
        optimizer = _Backend.make_optimizer(config, model)
        if saved and saved.get("optimizer"):
            optimizer.load_state_dict(saved["optimizer"])
        start_epoch = saved_epoch
        saved_cursor = saved.get("cursor", {}) if saved else {}
        global_step = int(saved_cursor.get("global_step", saved_cursor.get("step", 0)))
        resume_semantics = "true-resume-model-optimizer-scheduler-rng"
    controller = FuxiGradientAccumulation(
        int(config["accumulation"]), debug_anomaly=bool(config.get("anomaly_detection", False)),
        health_window=int(config.get("startup_health_window", 10)),
        health_interval=int(config.get("health_check_interval", 100)),
        health_path=output / "full_gdr_step_health.jsonl",
    )
    if resume_local:
        controller.optimizer_steps = int(saved.get(
            "health_optimizer_steps", saved_cursor.get("global_step", saved_cursor.get("step", 0))
        ))
    for epoch in range(start_epoch, int(config["trajectory_count"])):
        model.train(); model.freeze_selector()
        rows = _batches(config, files, epoch + 1)
        cursor = saved_cursor if saved and not continuation and epoch == start_epoch else {}
        epoch_loss = float(cursor.get("epoch_loss", 0.)); examples = int(cursor.get("examples", 0)); step = int(cursor.get("step", 0))
        started = time.time() - float(cursor.get("elapsed", 0.))
        if step:
            rows = itertools.islice(rows, step * int(config["accumulation"]), None)
        while window := list(itertools.islice(rows, int(config["accumulation"]))):
            if getattr(model, "full_gdr_length_bucketing", config.get("full_gdr_length_bucketing", False)):
                window = _bucket_window(window)
            controller.start_window(model, optimizer, steps=len(window))
            for index, micro in enumerate(window):
                controller.start_microbatch(model, index)
                batch = _collate(config, micro, catalog_ids.device)
                candidates = _candidates(config, batch, catalog_ids)
                with _autocast(model):
                    history = model.prefill(
                        batch["histories"], batch["lengths"],
                        sparse=False, single_group=True,
                        timestamps=batch["timestamps"],
                    )
                    loss = _loss(config, model, history, batch, candidates)
                controller.backward(loss, len(micro)); examples += len(micro)
            health = controller.finish_window(model, optimizer, gradient_clip=float(config["gradient_clip"]))
            epoch_loss += health["loss"] * health["denominator"]
            step += 1; global_step += 1
            if step % 25 == 0 or step == 1:
                _emit(output, "full-gdr-warmup", epoch=epoch + 1, step=step, global_step=global_step,
                      examples=examples, loss=epoch_loss / max(examples, 1),
                      detailed_parameter_check=health["detailed_parameter_check"])
            if step % 250 == 0 or STOP_REQUESTED:
                cursor = dict(epoch_loss=epoch_loss, examples=examples, step=step, global_step=global_step,
                              elapsed=time.time() - started)
                _Backend.save_local(latest, config, model, optimizer, "full_gdr", epoch=epoch, cursor=cursor,
                                    resume_semantics=resume_semantics,
                                    health_optimizer_steps=controller.optimizer_steps)
                if STOP_REQUESTED:
                    _emit(output, "paused-resumable", checkpoint=str(latest), resume_stage="full_gdr", epoch=epoch,
                          step=step, resume_semantics=resume_semantics)
                    return None
        parent = paths[epoch - 1] if epoch else (Path(parent_arg) if parent_arg else None)
        save_trajectory_checkpoint(paths[epoch], config, model, epoch=epoch + 1, global_step=global_step,
                                   trajectory_index=epoch + 1, parent_checkpoint=parent,
                                   continuation=continuation)
        _Backend.save_local(latest, config, model, optimizer, "full_gdr", epoch=epoch + 1,
                            cursor=dict(epoch_loss=0., examples=0, step=0, global_step=global_step, elapsed=0.),
                            resume_semantics=resume_semantics,
                            health_optimizer_steps=controller.optimizer_steps)
        saved = None
    return paths

def _train_sparse(config, model, files, catalog_ids, output, latest, saved):
    model.freeze_selector()
    optimizer = _Backend.make_optimizer(config, model)
    if saved and saved.get("optimizer"):
        optimizer.load_state_dict(saved["optimizer"])
    early = _Backend.early_state(config)
    start_epoch = int(saved.get("epoch", 0)) if saved else 0
    if saved and saved.get("early"):
        early = EarlyStoppingState.from_dict(saved["early"])
    microbatch = min(int(config["microbatch"]), 4) if config["dataset"] == "kuairand-1k" else int(config["microbatch"])
    accumulation = int(config["effective_batch"]) // microbatch
    controller = FuxiGradientAccumulation(
        accumulation, debug_anomaly=bool(config.get("anomaly_detection", False)),
        health_window=int(config.get("startup_health_window", 10)),
        health_interval=int(config.get("health_check_interval", 100)),
        health_path=output / "sparse_step_health.jsonl",
    )
    if saved:
        controller.optimizer_steps = int(saved.get(
            "health_optimizer_steps", saved.get("cursor", {}).get("step", 0)
        ))
    for epoch in range(start_epoch, int(config["max_epochs"])):
        model.train(); model.freeze_selector()
        rows = _batches(config, files, epoch + 1, size=microbatch)
        cursor = saved.get("cursor", {}) if saved and epoch == start_epoch else {}
        epoch_loss = float(cursor.get("epoch_loss", 0.)); examples = int(cursor.get("examples", 0)); step = int(cursor.get("step", 0))
        selected = int(cursor.get("selected", 0)); eligible = int(cursor.get("eligible", 0))
        started = time.time() - float(cursor.get("elapsed", 0.))
        if step:
            rows = itertools.islice(rows, step * accumulation, None)
        while window := list(itertools.islice(rows, accumulation)):
            controller.start_window(model, optimizer, steps=len(window))
            selected_window = torch.zeros((), device=catalog_ids.device, dtype=torch.long)
            eligible_window = torch.zeros((), device=catalog_ids.device, dtype=torch.long)
            for index, micro in enumerate(window):
                controller.start_microbatch(model, index)
                batch = _collate(config, micro, catalog_ids.device)
                candidates = _candidates(config, batch, catalog_ids)
                with _autocast(model):
                    history = model.prefill(
                        batch["histories"], batch["lengths"],
                        timestamps=batch["timestamps"],
                    )
                    loss = _loss(config, model, history, batch, candidates)
                controller.backward(loss, len(micro)); examples += len(micro)
                selected_window += history.selected_counts.sum()
                eligible_window += batch["lengths"].sum() * model.group_count
            health = controller.finish_window(model, optimizer, gradient_clip=float(config["gradient_clip"]))
            epoch_loss += health["loss"] * health["denominator"]
            selected += int(selected_window.item()); eligible += int(eligible_window.item())
            step += 1
            if step % 25 == 0 or step == 1:
                loss_fields = (
                    {"train_multitask_bce": epoch_loss / max(examples, 1)}
                    if config["dataset"] == "kuairand-1k"
                    else {"train_candidate_ce": epoch_loss / max(examples, 1)}
                )
                _emit(output, "sparse-training", epoch=epoch, step=step, examples=examples,
                      realized_write_ratio=selected / max(eligible, 1), **loss_fields)
            if step % 250 == 0 or STOP_REQUESTED:
                _Backend.save_local(latest, config, model, optimizer, "sparse", epoch=epoch,
                                    early=early.to_dict(), cursor=dict(epoch_loss=epoch_loss, examples=examples,
                                    step=step, elapsed=time.time() - started, selected=selected, eligible=eligible),
                                    health_optimizer_steps=controller.optimizer_steps)
                if STOP_REQUESTED:
                    _emit(output, "paused-resumable", checkpoint=str(latest), resume_stage="sparse", epoch=epoch, step=step)
                    return None
        result = _Backend.evaluate(config, model, files, output / "validation", label=f"epoch-{epoch:03d}")
        if config["dataset"] == "kuairand-1k":
            _emit(
                output,
                "sparse-epoch",
                epoch=epoch,
                train_multitask_bce=epoch_loss / max(examples, 1),
                validation_multitask_bce=result["metrics"]["multitask_loss"],
                validation_macro_gauc=result["metrics"]["macro_gauc"],
                validation_task_gauc=result["details"]["task_gauc"],
            )
        decision = early.observe(
            primary_metric=_Backend.primary_metric(None, result["metrics"]),
            epoch=epoch,
            tie_breaker=result["metrics"].get("multitask_loss"),
        )
        if decision["improved"]:
            binding = read(output / "selector_binding.json")
            atomic_torch_save(torch, dict(
                schema="deltarec-trajectory-late-sparse-best-v2", config=dict(config),
                model=_base_model_state(model), epoch=epoch, validation=result,
                selector_binding=binding,
                parent_trajectory=str((output / "trajectory_manifest.json").resolve()),
                parent_trajectory_sha256=sha256_file(output / "trajectory_manifest.json"),
                sparse_budget=config["retention_ratio"], backbone=config["backbone"],
            ), output / "best.pt")
        _Backend.save_local(latest, config, model, optimizer, "sparse", epoch=epoch + 1,
                            early=early.to_dict(), health_optimizer_steps=controller.optimizer_steps)
        atomic_write_json(output / "early_stopping.json", early.to_dict())
        Path(result["evidence"]).unlink()
        if decision["should_stop"]:
            break
    return early

def _load_best(config, model, output, catalog_ids, final_path):
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    if best.get("schema") != "deltarec-trajectory-late-sparse-best-v2" or best.get("config") != dict(config):
        raise ValueError("FuXi sparse best checkpoint protocol mismatch")
    binding = load_selector(config, model, output, catalog_ids, final_path)
    if binding != best.get("selector_binding"):
        raise ValueError("FuXi sparse best selector lineage changed")
    _load_base_model_state(model, best["model"])
    model.freeze_selector()
    return best

def construct(config, run: bridge.FuxiRunSpec, files: Mapping[str, Path]) -> FuxiLinearDeltaRec:
    # The selector is initialized in this process even when the backbone is
    # immediately restored from a trajectory checkpoint.  Bind the same seed
    # before construction so an interrupted selector stage can be reproduced
    # from its immutable CWI shards instead of silently starting from a new
    # random feature head.
    torch.manual_seed(int(config["seed"]))
    random.seed(int(config["seed"]))
    grouping = torch.load(config["grouping"], map_location="cpu", weights_only=False)
    upstream = bridge.build_upstream_model(
        run.model,
        max_item_id=data._max_catalog_id(files["full_catalog"]),
        
    )
    model = FuxiLinearDeltaRec(
        upstream,
        grouping["item_to_category_group"],
        int(config["group_count"]),
        retention_ratio=float(config["retention_ratio"]),
        recent_floor=int(config["recent_floor"]),
        multitask=run.model.multitask,
        backbone_input_length=run.model.total_sequence_length,
        dense_dynamic_width=bool(config.get("dense_dynamic_width", False)),
        seed=int(config["seed"]),
    )
    model.runtime_validation = bool(config.get("runtime_validation", False))
    model.full_gdr_length_bucketing = bool(config.get("full_gdr_length_bucketing", False))
    return model.to(config.get("device", "cpu"))


from deltarec.utils.config import load_trajectory_config
from deltarec.utils.blossom_training import catalog as _catalog

def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description='DeltaRec FuXi-Linear trajectory training')
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,default=Path('data'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--stop-stage',choices=('full-gdr','selector','sparse'),default='sparse')
    parser.add_argument('--evaluate',action='store_true')
    parser.add_argument('--checkpoint',type=Path)
    args=parser.parse_args(argv)
    c,run,files=load_trajectory_config(args,'fuxilinear')
    output=Path(c['output']);output.mkdir(parents=True,exist_ok=True)
    model=construct(c,run,files)
    catalog_ids=_catalog(files['train_catalog'],args.device)
    if args.evaluate:
        if args.checkpoint is None:parser.error('--checkpoint is required')
        state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        if c.get('rating_prediction_head', 'pair_mlp') == 'pair_mlp' and c['dataset']!='kuairand-1k':model.initialize_ranking_head()
        model.load_state_dict(state['model'],strict=True)
        print(json.dumps(_Backend.evaluate(c,model,files,output/'evaluation',label='evaluation')))
        return
    latest=output/'resume.pt';paths=expected_trajectory_paths(c)
    saved=_restore_local(latest,c,model,catalog_ids,paths[-1])
    stage=saved.get('stage','full_gdr') if saved else 'full_gdr'
    if stage=='full_gdr':
        paths=_train_full_gdr(c,model,files,catalog_ids,output,latest,saved)
        if paths is None:return
        if args.stop_stage=='full-gdr':return
        load_trajectory_checkpoint(paths[-1],c,model)
        stage='selector';saved=None
    if stage=='selector':
        if (output/'selector.pt').exists():load_selector(c,model,output,catalog_ids,paths[-1])
        else:
            fit_selector(c,model,files,catalog_ids,output,paths)
            save_selector(output,c,model,paths[-1])
        if args.stop_stage=='selector':return
        if c.get('rating_prediction_head', 'pair_mlp') == 'pair_mlp' and c['dataset']!='kuairand-1k':model.initialize_ranking_head()
        _Backend.save_local(latest,c,model,_Backend.make_optimizer(c,model),'sparse',epoch=0,early=None)
        stage='sparse';saved=None
    if _train_sparse(c,model,files,catalog_ids,output,latest,saved) is None:return
    best=_load_best(c,model,output,catalog_ids,paths[-1])
    atomic_torch_save(torch,dict(config=json.loads(args.config.read_text()),model=model.state_dict(),validation=best['validation']['metrics']),output/'model.pt')
    print(json.dumps(dict(checkpoint='model.pt',validation=best['validation']['metrics'])))

if __name__=='__main__':
    main()
