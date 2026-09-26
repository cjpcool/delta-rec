from __future__ import annotations

from dataclasses import asdict, dataclass

import hashlib

import json

import math

import os

from pathlib import Path

import random

import shutil

import struct

from typing import Any, Callable, Mapping, Sequence

import uuid

MetaBridgeError = ValueError

from deltarec.utils.early_stopping import EarlyStoppingError, EarlyStoppingState

from deltarec.utils.validation_lr_scheduler import ValidationLRSchedulerError, optimizer_lr_snapshot

from deltarec.utils.checkpoint import AtomicCheckpointError, atomic_torch_save, require_checkpoint_space

SCHEMA = "deltarec-meta-kuai-resumable-checkpoint-v1"

COMPLETE_SCHEMA = "deltarec-meta-kuai-atomic-checkpoint-complete-v1"

BINDING_SCHEMA = "deltarec-meta-kuai-resume-binding-v1"

STATE_FILE = "resume_state.pt"

COMPLETE_FILE = "COMPLETE.json"

DEFAULT_KEEP_LAST_COMPLETE = 3

PROTECTION_MARKERS = ("FINAL", "BEST", "PINNED")

DEFAULT_DMP_EXPECTED_CHECKPOINT_BYTES = 24 * 1024**3

@dataclass(frozen=True)
class LoopCursor:
    epoch: int
    batch_in_epoch: int
    global_step: int

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise MetaBridgeError(f"resume binding file is missing: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }

def build_binding(
    *,
    variant: str,
    seed: int,
    run_fingerprint: str,
    configuration_files: Mapping[str, Path],
    data_files: Mapping[str, Path],
    code_files: Mapping[str, Path],
) -> dict[str, Any]:
    if variant != "standard":
        raise MetaBridgeError("resumable Kuai production training is fixed to standard")
    value: dict[str, Any] = {
        "schema": BINDING_SCHEMA,
        "variant": variant,
        "seed": int(seed),
        "run_fingerprint": run_fingerprint,
        "configuration": {
            name: file_record(path) for name, path in sorted(configuration_files.items())
        },
        "data": {name: file_record(path) for name, path in sorted(data_files.items())},
        "code": {name: file_record(path) for name, path in sorted(code_files.items())},
    }
    value["binding_sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return value

def verify_binding(saved: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    if dict(saved) != dict(current):
        for group in ("configuration", "data", "code"):
            if saved.get(group) != current.get(group):
                raise MetaBridgeError(
                    f"refusing Kuai resume: {group} path/size/SHA binding changed"
                )
        raise MetaBridgeError("refusing Kuai resume: run binding changed")

def write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise MetaBridgeError(f"refusing to replace resume evidence: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, path)

def capture_rng_state(torch: Any, *, loader_generator: Any | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader_generator": (
            loader_generator.get_state() if loader_generator is not None else None
        ),
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        state["numpy"] = None
    return state

def restore_rng_state(
    state: Mapping[str, Any], torch: Any, *, loader_generator: Any | None = None
) -> None:
    random.setstate(state["python"])
    torch.random.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if loader_generator is not None and state.get("loader_generator") is not None:
        loader_generator.set_state(state["loader_generator"])
    if state.get("numpy") is not None:
        import numpy as np

        np.random.set_state(state["numpy"])

def sampler_order_sha256(sampler: Any, epoch: int) -> str:
    previous = int(getattr(sampler, "epoch", 0))
    sampler.set_epoch(epoch)
    digest = hashlib.sha256()
    for index in sampler:
        digest.update(struct.pack("<q", int(index)))
    sampler.set_epoch(previous)
    return digest.hexdigest()

def dataloader_state(dataloader: Any, cursor: LoopCursor) -> dict[str, Any]:
    sampler = dataloader.sampler
    return {
        "sampler_class": f"{type(sampler).__module__}.{type(sampler).__qualname__}",
        "sampler_seed": int(getattr(sampler, "seed", 0)),
        "sampler_epoch": cursor.epoch,
        "next_batch_in_epoch": cursor.batch_in_epoch,
        "epoch_order_sha256": sampler_order_sha256(sampler, cursor.epoch),
        "dataset_length": len(dataloader.dataset),
        "batch_size": int(dataloader.batch_size),
        "drop_last": bool(dataloader.drop_last),
        "batches_per_epoch": len(dataloader),
    }

def verify_dataloader_state(saved: Mapping[str, Any], dataloader: Any) -> None:
    cursor = LoopCursor(
        epoch=int(saved["sampler_epoch"]),
        batch_in_epoch=int(saved["next_batch_in_epoch"]),
        global_step=0,
    )
    current = dataloader_state(dataloader, cursor)
    if dict(saved) != current:
        raise MetaBridgeError("refusing Kuai resume: DataLoader/sampling order changed")

def _atomic_torch_save(torch: Any, payload: Mapping[str, Any], path: Path) -> None:
    try:
        atomic_torch_save(torch, payload, path, overwrite=False)
    except AtomicCheckpointError as error:
        raise MetaBridgeError(str(error)) from error

def _checkpoint_root(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise MetaBridgeError("Kuai checkpoint root may not be a symlink")
    root = expanded.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise MetaBridgeError("Kuai checkpoint root must be a directory")
    return root

def _numeric_checkpoint(root: Path, step: int) -> Path:
    if step < 0:
        raise MetaBridgeError("Kuai checkpoint step must be nonnegative")
    path = root / str(step)
    if path.is_symlink() or path.parent != root or path.resolve().parent != root:
        raise MetaBridgeError("Kuai checkpoint path is symlinked or escaped its root")
    return path

def _complete_step(path: Path, step: int) -> bool:
    if path.is_symlink() or not path.is_dir():
        return False
    marker = path / COMPLETE_FILE
    state = path / STATE_FILE
    if marker.is_symlink() or state.is_symlink() or not marker.is_file() or not state.is_file():
        return False
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        value.get("schema") == COMPLETE_SCHEMA
        and int(value.get("global_step", -1)) == step
        and value.get("state_file") == STATE_FILE
    )

def _snapshot_keys_and_lrs(
    snapshot: Any, *, label: str
) -> tuple[list[str], list[float]]:
    if not isinstance(snapshot, list) or not snapshot:
        raise MetaBridgeError(f"{label} is missing")
    keys: list[str] = []
    learning_rates: list[float] = []
    for record in snapshot:
        if not isinstance(record, Mapping) or "key" not in record or "lr" not in record:
            raise MetaBridgeError(f"{label} contains an invalid optimizer group")
        key = str(record["key"])
        try:
            learning_rate = float(record["lr"])
        except (TypeError, ValueError) as error:
            raise MetaBridgeError(f"{label} contains a non-scalar LR") from error
        if not key or key in keys:
            raise MetaBridgeError(f"{label} contains duplicate optimizer groups")
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise MetaBridgeError(f"{label} contains a non-finite or non-positive LR")
        keys.append(key)
        learning_rates.append(learning_rate)
    return keys, learning_rates

def _require_scheduler_lr_match(
    scheduler_state: Mapping[str, Any],
    optimizer_snapshot: list[dict[str, Any]],
    *,
    label: str,
) -> None:
    keys, learning_rates = _snapshot_keys_and_lrs(
        optimizer_snapshot, label=f"{label} optimizer LR snapshot"
    )
    scheduler_keys = scheduler_state.get("group_keys")
    scheduler_lrs = scheduler_state.get("last_lrs")
    if not isinstance(scheduler_keys, list) or [str(value) for value in scheduler_keys] != keys:
        raise MetaBridgeError(f"{label} scheduler optimizer group keys disagree")
    if not isinstance(scheduler_lrs, list) or len(scheduler_lrs) != len(learning_rates):
        raise MetaBridgeError(f"{label} scheduler LR values are missing")
    try:
        normalized_scheduler_lrs = [float(value) for value in scheduler_lrs]
    except (TypeError, ValueError) as error:
        raise MetaBridgeError(f"{label} scheduler LR values are not scalar") from error
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in normalized_scheduler_lrs
    ):
        raise MetaBridgeError(f"{label} scheduler LR values are invalid")
    if any(
        abs(actual - expected) > 1e-12
        for actual, expected in zip(learning_rates, normalized_scheduler_lrs)
    ):
        raise MetaBridgeError(
            f"{label} optimizer LR changed outside the validation scheduler"
        )

def _fresh_optimizer_lr_snapshot(
    optimizer: Any, lr_scheduler: Any | None
) -> list[dict[str, Any]]:
    try:
        snapshot = [dict(record) for record in optimizer_lr_snapshot(optimizer)]
    except ValidationLRSchedulerError as error:
        raise MetaBridgeError(f"cannot checkpoint Kuai optimizer LR state: {error}") from error
    if lr_scheduler is not None:
        try:
            scheduler_state = lr_scheduler.state_dict()
        except (AttributeError, TypeError, ValueError) as error:
            raise MetaBridgeError("cannot checkpoint Kuai scheduler LR state") from error
        if not isinstance(scheduler_state, Mapping):
            raise MetaBridgeError("cannot checkpoint Kuai scheduler LR state")
        _require_scheduler_lr_match(
            scheduler_state,
            snapshot,
            label="live Kuai checkpoint",
        )
    return snapshot

def _require_live_optimizer_lr_snapshot(
    optimizer: Any,
    lr_scheduler: Any | None,
    expected_snapshot: list[dict[str, Any]],
) -> None:
    actual_snapshot = _fresh_optimizer_lr_snapshot(optimizer, lr_scheduler)
    actual_keys, actual_lrs = _snapshot_keys_and_lrs(
        actual_snapshot, label="restored optimizer LR snapshot"
    )
    expected_keys, expected_lrs = _snapshot_keys_and_lrs(
        expected_snapshot, label="checkpoint optimizer LR snapshot"
    )
    if actual_keys != expected_keys or any(
        abs(actual - expected) > 1e-12
        for actual, expected in zip(actual_lrs, expected_lrs)
    ):
        raise MetaBridgeError(
            "refusing Kuai resume: restored optimizer LR snapshot disagrees"
        )

def _expected_validation_best_step(
    *,
    checkpoint_root: Path,
    early_stopping_state: Mapping[str, Any],
    validation_history: Sequence[Mapping[str, Any]],
) -> int | None:
    try:
        best_epoch = int(early_stopping_state["best_epoch"])
        last_completed_epoch = int(early_stopping_state["last_completed_epoch"])
    except (KeyError, TypeError, ValueError) as error:
        raise MetaBridgeError("Kuai early-stopping best cursor is invalid") from error
    history = list(validation_history)
    if last_completed_epoch < -1 or len(history) != last_completed_epoch + 1:
        raise MetaBridgeError("Kuai validation history disagrees with early-stopping cursor")
    if best_epoch < -1 or best_epoch > last_completed_epoch:
        raise MetaBridgeError("Kuai early-stopping best epoch is invalid")
    previous_step = -1
    for index, record in enumerate(history):
        if not isinstance(record, Mapping):
            raise MetaBridgeError("Kuai validation history contains an invalid record")
        try:
            record_epoch = int(record["epoch"])
            record_step = int(record["global_step"])
            macro_gauc = float(record["macro_gauc"])
            multitask_loss = float(record["multitask_loss"])
        except (KeyError, TypeError, ValueError) as error:
            raise MetaBridgeError("Kuai validation history record is invalid") from error
        if record_epoch != index + 1 or record_step <= previous_step:
            raise MetaBridgeError("Kuai validation history cursor is not contiguous")
        if not math.isfinite(macro_gauc) or not math.isfinite(multitask_loss):
            raise MetaBridgeError("Kuai validation history contains a non-finite metric")
        previous_step = record_step
    marker = checkpoint_root / "BEST"
    if best_epoch < 0:
        if history or marker.exists() or marker.is_symlink():
            raise MetaBridgeError("Kuai BEST marker exists without a validation best")
        return None
    best_record = history[best_epoch]
    if best_record.get("improved") is not True:
        raise MetaBridgeError("Kuai validation-best history record is not marked improved")
    best_metric = early_stopping_state.get(
        "best_metric", early_stopping_state.get("primary_metric")
    )
    best_tie = early_stopping_state.get(
        "best_tie_breaker", early_stopping_state.get("tie_breaker")
    )
    try:
        metric_matches = float(best_metric) == float(best_record["macro_gauc"])
        tie_matches = float(best_tie) == float(best_record["multitask_loss"])
    except (TypeError, ValueError) as error:
        raise MetaBridgeError("Kuai validation-best metrics are invalid") from error
    if not metric_matches or not tie_matches:
        raise MetaBridgeError("Kuai validation-best metrics disagree with history")
    expected_step = int(best_record["global_step"])
    expected_checkpoint = _numeric_checkpoint(checkpoint_root, expected_step)
    if not _complete_step(expected_checkpoint, expected_step):
        raise MetaBridgeError(
            "expected Kuai validation-best checkpoint is missing or incomplete"
        )
    return expected_step

def _reconcile_best_marker(
    checkpoint_root: Path,
    *,
    early_stopping_state: Mapping[str, Any],
    validation_history: Sequence[Mapping[str, Any]],
) -> int | None:
    root = _checkpoint_root(checkpoint_root)
    expected_step = _expected_validation_best_step(
        checkpoint_root=root,
        early_stopping_state=early_stopping_state,
        validation_history=validation_history,
    )
    if expected_step is None:
        return None
    marker = root / "BEST"
    if marker.is_symlink():
        raise MetaBridgeError("Kuai checkpoint BEST marker may not be a symlink")
    try:
        current = marker.read_text(encoding="ascii").strip() if marker.is_file() else None
    except (OSError, UnicodeError):
        current = None
    if current == str(expected_step):
        return expected_step
    temporary = root / f".BEST.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    temporary.write_text(f"{expected_step}\n", encoding="ascii")
    os.replace(temporary, marker)
    return expected_step

def pin_complete_checkpoint(checkpoint_root: Path, step: int, marker: str) -> None:
    """Protect a completed checkpoint as FINAL, BEST, or PINNED."""

    label = marker.upper()
    if label not in PROTECTION_MARKERS:
        raise MetaBridgeError("unknown Kuai checkpoint protection marker")
    root = _checkpoint_root(checkpoint_root)
    checkpoint = _numeric_checkpoint(root, step)
    if not _complete_step(checkpoint, step):
        raise MetaBridgeError("cannot protect an incomplete Kuai checkpoint")
    temporary = root / f".{label}.{os.getpid()}.tmp"
    temporary.write_text(f"{step}\n", encoding="ascii")
    os.replace(temporary, root / label)

def _protected_steps(root: Path) -> set[int]:
    protected: set[int] = set()
    for label in PROTECTION_MARKERS:
        marker = root / label
        if marker.is_symlink():
            raise MetaBridgeError(f"Kuai checkpoint {label} marker may not be a symlink")
        if not marker.exists():
            continue
        try:
            text = marker.read_text(encoding="ascii").strip()
            step = int(text)
        except (OSError, UnicodeError, ValueError) as error:
            raise MetaBridgeError(f"invalid Kuai checkpoint {label} marker") from error
        if text != str(step) or not _complete_step(_numeric_checkpoint(root, step), step):
            raise MetaBridgeError(f"Kuai checkpoint {label} marker is not complete")
        protected.add(step)
    return protected

def prune_complete_checkpoints(
    checkpoint_root: Path, *, keep_last_complete: int = DEFAULT_KEEP_LAST_COMPLETE
) -> list[int]:
    """Delete only old, verified complete numeric checkpoints.

    Staging and incomplete directories are deliberately ignored.  Protected
    FINAL/BEST/PINNED steps are retained in addition to the newest N complete
    checkpoints.
    """

    if keep_last_complete < 1:
        raise MetaBridgeError("keep_last_complete must be positive")
    root = _checkpoint_root(checkpoint_root)
    latest_path = root / "LATEST"
    if latest_path.is_symlink() or not latest_path.is_file():
        raise MetaBridgeError("Kuai checkpoint LATEST marker is missing or symlinked")
    try:
        latest_text = latest_path.read_text(encoding="ascii").strip()
        latest = int(latest_text)
    except (OSError, UnicodeError, ValueError) as error:
        raise MetaBridgeError("invalid Kuai checkpoint LATEST marker") from error
    if latest_text != str(latest):
        raise MetaBridgeError("invalid Kuai checkpoint LATEST marker")
    complete: list[int] = []
    for entry in root.iterdir():
        if not entry.name.isdigit():
            continue
        if entry.is_symlink():
            raise MetaBridgeError("numeric Kuai checkpoint may not be a symlink")
        step = int(entry.name)
        if entry.name != str(step) or entry.resolve().parent != root:
            raise MetaBridgeError("invalid numeric Kuai checkpoint path")
        if _complete_step(entry, step):
            complete.append(step)
    complete.sort()
    if latest not in complete:
        raise MetaBridgeError("LATEST does not reference a complete Kuai checkpoint")
    keep = set(complete[-keep_last_complete:]) | _protected_steps(root) | {latest}
    removed: list[int] = []
    for step in complete:
        if step in keep:
            continue
        candidate = _numeric_checkpoint(root, step)
        if not _complete_step(candidate, step):
            raise MetaBridgeError("checkpoint changed before retention deletion")
        shutil.rmtree(candidate)
        removed.append(step)
    return removed

def resolve_latest_complete_checkpoint(checkpoint_root: Path) -> Path | None:
    """Resolve LATEST, falling back to the newest completed numeric step."""

    expanded = checkpoint_root.expanduser()
    if not expanded.exists():
        return None
    root = _checkpoint_root(checkpoint_root)
    latest_path = root / "LATEST"
    if latest_path.is_symlink():
        raise MetaBridgeError("Kuai checkpoint LATEST marker may not be a symlink")
    requested: int | None = None
    if latest_path.is_file():
        try:
            text = latest_path.read_text(encoding="ascii").strip()
            requested = int(text)
            if text != str(requested):
                requested = None
        except (OSError, UnicodeError, ValueError):
            requested = None
    complete: list[int] = []
    for entry in root.iterdir():
        if not entry.name.isdigit():
            continue
        if entry.is_symlink():
            raise MetaBridgeError("numeric Kuai checkpoint may not be a symlink")
        step = int(entry.name)
        if entry.name != str(step) or entry.resolve().parent != root:
            raise MetaBridgeError("invalid numeric Kuai checkpoint path")
        if _complete_step(entry, step):
            complete.append(step)
    if not complete:
        return None
    selected = requested if requested in complete else max(complete)
    if requested != selected:
        temporary = root / f".LATEST.{os.getpid()}.fallback.tmp"
        temporary.write_text(f"{selected}\n", encoding="ascii")
        os.replace(temporary, latest_path)
    return _numeric_checkpoint(root, selected)

def save_atomic_dmp_checkpoint(
    *,
    checkpoint_module: Any,
    model: Any,
    optimizer: Any,
    metric_logger: Any,
    rank: int,
    checkpoint_root: Path,
    cursor: LoopCursor,
    binding: Mapping[str, Any],
    dataloader: Any,
    torch: Any,
    loader_generator: Any | None,
    keep_last_complete: int = DEFAULT_KEEP_LAST_COMPLETE,
    early_stopping_state: Mapping[str, Any] | None = None,
    validation_history: Sequence[Mapping[str, Any]] | None = None,
    lr_scheduler: Any | None = None,
    lr_scheduler_history: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    root = _checkpoint_root(checkpoint_root)
    final = _numeric_checkpoint(root, cursor.global_step)
    if final.exists():
        raise MetaBridgeError(f"refusing to overwrite Kuai checkpoint: {final}")
    optimizer_learning_rates = _fresh_optimizer_lr_snapshot(optimizer, lr_scheduler)
    complete_footprints = []
    for candidate in root.iterdir():
        if candidate.is_dir() and candidate.name.isdigit() and _complete_step(
            candidate, int(candidate.name)
        ):
            complete_footprints.append(
                sum(path.stat().st_size for path in candidate.rglob("*") if path.is_file())
            )
    expected_bytes = max(complete_footprints, default=None)
    if expected_bytes is None:
        expected_bytes = int(
            os.environ.get(
                "DELTAREC_DMP_CHECKPOINT_EXPECTED_BYTES",
                DEFAULT_DMP_EXPECTED_CHECKPOINT_BYTES,
            )
        )
    spool = Path(os.environ.get("DELTAREC_CHECKPOINT_SPOOL", "/tmp"))
    spool.mkdir(parents=True, exist_ok=True)
    try:
        require_checkpoint_space(spool, expected_bytes, label="local spool")
        require_checkpoint_space(root, expected_bytes, label="DMP destination")
    except AtomicCheckpointError as error:
        raise MetaBridgeError(str(error)) from error
    staging_root = root / (
        f".staging-{cursor.global_step}-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        checkpoint_module.save_dmp_checkpoint(
            model=model,
            optimizer=optimizer,
            metric_logger=metric_logger,
            rank=rank,
            batch_idx=cursor.global_step,
            path=str(staging_root),
        )
        physical = staging_root / str(cursor.global_step)
        optimizer_learning_rates = _fresh_optimizer_lr_snapshot(optimizer, lr_scheduler)
        scheduler_payload: dict[str, Any]
        if lr_scheduler is None:
            scheduler_payload = {"kind": "none", "state_dict": None}
        else:
            scheduler_state = dict(lr_scheduler.state_dict())
            _require_scheduler_lr_match(
                scheduler_state,
                optimizer_learning_rates,
                label="serialized Kuai checkpoint",
            )
            scheduler_payload = {
                "kind": str(lr_scheduler.kind),
                "state_dict": scheduler_state,
                "history": [
                    dict(record) for record in (lr_scheduler_history or ())
                ],
                "checkpoint_learning_rates": optimizer_learning_rates,
            }
        state = {
            "schema": SCHEMA,
            "cursor": asdict(cursor),
            "scheduler": scheduler_payload,
            "optimizer_learning_rates": optimizer_learning_rates,
            "rng": capture_rng_state(torch, loader_generator=loader_generator),
            "dataloader": dataloader_state(dataloader, cursor),
            "binding": dict(binding),
        }
        if early_stopping_state is not None:
            state["early_stopping"] = dict(early_stopping_state)
            state["validation_history"] = [
                dict(record) for record in (validation_history or ())
            ]
        state_path = physical / STATE_FILE
        _atomic_torch_save(torch, state, state_path)
        complete = {
            "schema": COMPLETE_SCHEMA,
            "global_step": cursor.global_step,
            "state_file": STATE_FILE,
            "state_sha256": sha256_file(state_path),
            "binding_sha256": binding["binding_sha256"],
        }
        write_immutable_json(physical / COMPLETE_FILE, complete)
        os.replace(physical, final)
        staging_root.rmdir()
        latest_tmp = root / f".LATEST.{os.getpid()}.tmp"
        latest_tmp.write_text(f"{cursor.global_step}\n", encoding="ascii")
        os.replace(latest_tmp, root / "LATEST")
        if early_stopping_state is not None:
            _reconcile_best_marker(
                root,
                early_stopping_state=early_stopping_state,
                validation_history=validation_history or (),
            )
        prune_complete_checkpoints(
            root, keep_last_complete=keep_last_complete
        )
        return final
    except BaseException:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise

def load_resume_state(
    checkpoint: Path,
    *,
    current_binding: Mapping[str, Any],
    dataloader: Any,
    torch: Any,
) -> tuple[LoopCursor, Mapping[str, Any]]:
    root = checkpoint.expanduser().resolve()
    complete_path = root / COMPLETE_FILE
    state_path = root / STATE_FILE
    if not complete_path.is_file() or not state_path.is_file():
        raise MetaBridgeError("refusing Kuai resume: checkpoint is incomplete")
    try:
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MetaBridgeError("refusing Kuai resume: invalid completion marker") from error
    if (
        complete.get("schema") != COMPLETE_SCHEMA
        or complete.get("state_sha256") != sha256_file(state_path)
    ):
        raise MetaBridgeError("refusing Kuai resume: checkpoint state SHA mismatch")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state.get("schema") != SCHEMA:
        raise MetaBridgeError("refusing Kuai resume: unsupported checkpoint schema")
    verify_binding(state["binding"], current_binding)
    verify_dataloader_state(state["dataloader"], dataloader)
    scheduler = state.get("scheduler")
    if not isinstance(scheduler, Mapping) or "kind" not in scheduler:
        raise MetaBridgeError("refusing Kuai resume: scheduler state is missing")
    raw_optimizer_learning_rates = state.get("optimizer_learning_rates")
    if not isinstance(raw_optimizer_learning_rates, list) or not all(
        isinstance(record, Mapping) for record in raw_optimizer_learning_rates
    ):
        raise MetaBridgeError(
            "refusing Kuai resume: optimizer LR snapshot is missing"
        )
    optimizer_learning_rates = [
        dict(record) for record in raw_optimizer_learning_rates
    ]
    _snapshot_keys_and_lrs(
        optimizer_learning_rates,
        label="checkpoint optimizer LR snapshot",
    )
    if scheduler.get("kind") != "none":
        scheduler_state = scheduler.get("state_dict")
        if not isinstance(scheduler_state, Mapping):
            raise MetaBridgeError(
                "refusing Kuai resume: scheduler state is missing"
            )
        _require_scheduler_lr_match(
            scheduler_state,
            optimizer_learning_rates,
            label="checkpoint",
        )
        checkpoint_learning_rates = scheduler.get("checkpoint_learning_rates")
        if checkpoint_learning_rates != raw_optimizer_learning_rates:
            raise MetaBridgeError(
                "refusing Kuai resume: scheduler checkpoint LR snapshot disagrees"
            )
    cursor = LoopCursor(**{key: int(value) for key, value in state["cursor"].items()})
    if cursor.global_step != int(complete["global_step"]):
        raise MetaBridgeError("refusing Kuai resume: global step marker mismatch")
    return cursor, state

def resumable_train_loop(
    *,
    rank: int,
    model: Any,
    dataloader: Any,
    optimizer: Any,
    metric_logger: Any,
    device: Any,
    checkpoint_module: Any,
    checkpoint_root: Path,
    binding: Mapping[str, Any],
    torch: Any,
    num_epochs: int,
    checkpoint_frequency: int,
    metric_log_frequency: int,
    output_trace: bool,
    profiler_class: Any,
    resume_checkpoint: Path | None = None,
    keep_last_complete: int = DEFAULT_KEEP_LAST_COMPLETE,
    validation_callback: Callable[[int, LoopCursor], Mapping[str, Any]] | None = None,
    early_stopping_contract: Mapping[str, Any] | None = None,
    early_stopping_state_path: Path | None = None,
    validation_history_path: Path | None = None,
    lr_scheduler: Any | None = None,
    lr_scheduler_history_path: Path | None = None,
) -> LoopCursor:
    if checkpoint_frequency < 1:
        raise ValueError("checkpoint frequency must be positive")
    early_enabled = validation_callback is not None or early_stopping_contract is not None
    if (validation_callback is None) != (early_stopping_contract is None):
        raise MetaBridgeError(
            "Kuai per-epoch validation callback and early-stopping contract must be paired"
        )
    if lr_scheduler is not None and validation_callback is None:
        raise MetaBridgeError(
            "Kuai LR scheduler requires completed-epoch validation"
        )
    early_state: EarlyStoppingState | None = None
    validation_history: list[dict[str, Any]] = []
    lr_scheduler_history: list[dict[str, Any]] = []
    if early_enabled:
        try:
            early_state = EarlyStoppingState.from_contract(early_stopping_contract or {})
        except EarlyStoppingError as error:
            raise MetaBridgeError(str(error)) from error

    def persist_validation_evidence() -> None:
        payloads: list[tuple[Path | None, Any]] = []
        if early_state is not None:
            payloads.extend(
                (
                    (early_stopping_state_path, early_state.to_dict()),
                    (validation_history_path, validation_history),
                )
            )
        if lr_scheduler is not None:
            payloads.append((lr_scheduler_history_path, lr_scheduler_history))
        for path, payload in payloads:
            if path is None:
                continue
            target = path.expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, target)

    loader_generator = getattr(dataloader, "generator", None)
    cursor = LoopCursor(epoch=0, batch_in_epoch=0, global_step=0)
    resume_state: Mapping[str, Any] | None = None
    if resume_checkpoint is not None:
        cursor, resume_state = load_resume_state(
            resume_checkpoint,
            current_binding=binding,
            dataloader=dataloader,
            torch=torch,
        )
        checkpoint_module.load_dmp_checkpoint(
            model=model,
            optimizer=optimizer,
            metric_logger=metric_logger,
            device=device,
            path=str(resume_checkpoint.expanduser().resolve()),
        )
        scheduler_payload = resume_state.get("scheduler")
        if not isinstance(scheduler_payload, Mapping):
            raise MetaBridgeError("refusing Kuai resume: scheduler state is missing")
        if lr_scheduler is None:
            if dict(scheduler_payload) != {"kind": "none", "state_dict": None}:
                raise MetaBridgeError(
                    "refusing Kuai resume: scheduler contract changed"
                )
        else:
            if (
                scheduler_payload.get("kind") != str(lr_scheduler.kind)
                or not isinstance(scheduler_payload.get("state_dict"), Mapping)
            ):
                raise MetaBridgeError(
                    "refusing Kuai resume: plateau scheduler state is missing"
                )
            raw_scheduler_history = scheduler_payload.get("history")
            if not isinstance(raw_scheduler_history, list) or not all(
                isinstance(record, Mapping) for record in raw_scheduler_history
            ):
                raise MetaBridgeError(
                    "refusing Kuai resume: scheduler history is invalid"
                )
            lr_scheduler_history = [
                dict(record) for record in raw_scheduler_history
            ]
            try:
                lr_scheduler.load_state_dict(scheduler_payload["state_dict"])
            except (TypeError, ValueError) as error:
                raise MetaBridgeError(
                    f"refusing Kuai resume: invalid LR scheduler state: {error}"
                ) from error
            if (
                int(lr_scheduler.last_completed_epoch) != cursor.epoch - 1
                or len(lr_scheduler_history) != cursor.epoch
            ):
                raise MetaBridgeError(
                    "refusing Kuai resume: scheduler cursor disagrees with "
                    "training cursor"
                )
        restored_lr_snapshot = resume_state.get("optimizer_learning_rates")
        if not isinstance(restored_lr_snapshot, list) or not all(
            isinstance(record, Mapping) for record in restored_lr_snapshot
        ):
            raise MetaBridgeError(
                "refusing Kuai resume: optimizer LR snapshot is invalid"
            )
        _require_live_optimizer_lr_snapshot(
            optimizer,
            lr_scheduler,
            [dict(record) for record in restored_lr_snapshot],
        )
        if int(metric_logger.global_step["train"]) != cursor.global_step:
            raise MetaBridgeError("refusing Kuai resume: metric/global step changed")
        if early_state is not None:
            if "early_stopping" not in resume_state or "validation_history" not in resume_state:
                raise MetaBridgeError(
                    "refusing Kuai resume: per-epoch validation state is missing"
                )
            try:
                early_state = EarlyStoppingState.from_dict(
                    resume_state["early_stopping"],
                    contract=early_stopping_contract,
                )
            except (EarlyStoppingError, TypeError) as error:
                raise MetaBridgeError(
                    f"refusing Kuai resume: invalid early-stopping state: {error}"
                ) from error
            raw_history = resume_state["validation_history"]
            if not isinstance(raw_history, list) or not all(
                isinstance(record, Mapping) for record in raw_history
            ):
                raise MetaBridgeError(
                    "refusing Kuai resume: validation history is invalid"
                )
            validation_history = [dict(record) for record in raw_history]
            if (
                early_state.last_completed_epoch != cursor.epoch - 1
                or len(validation_history) != cursor.epoch
            ):
                raise MetaBridgeError(
                    "refusing Kuai resume: validation cursor disagrees with training cursor"
                )
            _reconcile_best_marker(
                checkpoint_root,
                early_stopping_state=early_state.to_dict(),
                validation_history=validation_history,
            )
            persist_validation_evidence()
            if early_state.stopped:
                return cursor
    model.train()
    profiler = profiler_class(rank, active=10) if output_trace else None
    restored_resume_rng = False
    for epoch in range(cursor.epoch, num_epochs):
        dataloader.sampler.set_epoch(epoch)
        start_batch = cursor.batch_in_epoch if epoch == cursor.epoch else 0
        iterator = iter(dataloader)
        for batch_index in range(start_batch):
            try:
                next(iterator)
            except StopIteration as error:
                raise MetaBridgeError("resume batch offset exceeds DataLoader") from error
        if resume_state is not None and not restored_resume_rng:
            restore_rng_state(
                resume_state["rng"], torch, loader_generator=loader_generator
            )
            restored_resume_rng = True
        for batch_index, sample in enumerate(iterator, start=start_batch):
            optimizer.zero_grad()
            sample.to(device)
            (_, _, aux_losses, predictions, labels, weights) = model.forward(
                sample.uih_features_kjt, sample.candidates_features_kjt
            )
            nonfinite_losses = [
                name
                for name, value in aux_losses.items()
                if not bool(torch.isfinite(value.detach()).all().item())
            ]
            if nonfinite_losses:
                raise MetaBridgeError(
                    "non-finite training loss before backward at "
                    f"epoch={epoch + 1}, batch={batch_index}, "
                    f"global_step={cursor.global_step + 1}, "
                    f"tasks={nonfinite_losses}"
                )
            sum(aux_losses.values()).backward()
            optimizer.step()
            metric_logger.update(
                mode="train",
                predictions=predictions,
                labels=labels,
                weights=weights,
                num_candidates=sample.candidates_features_kjt.lengths().view(
                    len(sample.candidates_features_kjt.keys()), -1
                )[0],
            )
            global_step = cursor.global_step + 1
            if global_step % metric_log_frequency == 0:
                metric_logger.compute_and_log(
                    mode="train", additional_logs={"losses": aux_losses}
                )
            next_epoch = epoch
            next_batch = batch_index + 1
            if next_batch == len(dataloader):
                next_epoch, next_batch = epoch + 1, 0
            cursor = LoopCursor(next_epoch, next_batch, global_step)
            epoch_complete = next_batch == 0
            if global_step % checkpoint_frequency == 0 and not (
                early_state is not None and epoch_complete
            ):
                save_atomic_dmp_checkpoint(
                    checkpoint_module=checkpoint_module,
                    model=model,
                    optimizer=optimizer,
                    metric_logger=metric_logger,
                    rank=rank,
                    checkpoint_root=checkpoint_root,
                    cursor=cursor,
                    binding=binding,
                    dataloader=dataloader,
                    torch=torch,
                    loader_generator=loader_generator,
                    keep_last_complete=keep_last_complete,
                    early_stopping_state=(
                        None if early_state is None else early_state.to_dict()
                    ),
                    validation_history=validation_history,
                    lr_scheduler=lr_scheduler,
                    lr_scheduler_history=lr_scheduler_history,
                )
            if profiler is not None:
                profiler.step()
        if cursor.batch_in_epoch != 0:
            raise MetaBridgeError("Kuai epoch ended with inconsistent resume cursor")
        if early_state is not None:
            assert validation_callback is not None
            validation = dict(validation_callback(epoch, cursor))
            if "macro_gauc" not in validation or "multitask_loss" not in validation:
                raise MetaBridgeError(
                    "Kuai epoch validation must expose macro_gauc and multitask_loss"
                )
            try:
                macro_gauc = float(validation["macro_gauc"])
                multitask_loss = float(validation["multitask_loss"])
            except (TypeError, ValueError) as error:
                raise MetaBridgeError("Kuai epoch validation metrics are not scalar") from error
            if not math.isfinite(macro_gauc) or not math.isfinite(multitask_loss):
                raise MetaBridgeError(
                    "Kuai epoch validation macro_gauc and multitask_loss must "
                    "both be finite before scheduler, early-stopping, history, "
                    "or checkpoint mutation"
                )
            scheduler_event: Mapping[str, Any] | None = None
            if lr_scheduler is not None:
                try:
                    scheduler_event = lr_scheduler.step(
                        macro_gauc,
                        epoch=epoch,
                        global_step=cursor.global_step,
                    )
                except (TypeError, ValueError) as error:
                    raise MetaBridgeError(
                        f"Kuai validation LR scheduler rejected metric: {error}"
                    ) from error
                lr_scheduler_history.append(dict(scheduler_event))
            decision = early_state.observe(
                primary_metric=macro_gauc,
                tie_breaker=multitask_loss,
                epoch=epoch,
            )
            validation_record = {
                "epoch": epoch + 1,
                "global_step": cursor.global_step,
                "macro_gauc": macro_gauc,
                "multitask_loss": multitask_loss,
                "improved": bool(decision["improved"]),
                "bad_evaluation_count": int(decision["bad_evaluation_count"]),
                "should_stop": bool(decision["should_stop"]),
            }
            if scheduler_event is not None:
                validation_record["lr_scheduler"] = dict(scheduler_event)
            validation_history.append(validation_record)
            model.train()
            save_atomic_dmp_checkpoint(
                checkpoint_module=checkpoint_module,
                model=model,
                optimizer=optimizer,
                metric_logger=metric_logger,
                rank=rank,
                checkpoint_root=checkpoint_root,
                cursor=cursor,
                binding=binding,
                dataloader=dataloader,
                torch=torch,
                loader_generator=loader_generator,
                keep_last_complete=keep_last_complete,
                early_stopping_state=early_state.to_dict(),
                validation_history=validation_history,
                lr_scheduler=lr_scheduler,
                lr_scheduler_history=lr_scheduler_history,
            )
            if decision["improved"]:
                pin_complete_checkpoint(checkpoint_root, cursor.global_step, "BEST")
            persist_validation_evidence()
            if decision["should_stop"]:
                pin_complete_checkpoint(checkpoint_root, cursor.global_step, "FINAL")
                break
    return cursor
