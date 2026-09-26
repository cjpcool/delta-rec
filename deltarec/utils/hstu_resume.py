from __future__ import annotations

from dataclasses import dataclass

import hashlib

import json

import os

from pathlib import Path

import random

import tempfile

from typing import Any, Mapping

from .checkpoint import AtomicCheckpointError, atomic_torch_save

from .early_stopping import EarlyStoppingError, EarlyStoppingState

from .validation_lr_scheduler import ValidationLRSchedulerError, ValidationReduceLROnPlateau, audit_optimizer_peak_lrs, optimizer_lr_snapshot

CHECKPOINT_SCHEMA = "deltarec-meta-resumable-checkpoint-v1"

COMPLETION_SCHEMA = "deltarec-meta-resumable-completion-v1"

class MetaResumeError(RuntimeError):
    pass

def _identity_sha256(identity: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), default=repr
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    rendered = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

def _atomic_text(path: Path, value: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

def capture_rng_state(torch: Any) -> dict[str, Any]:
    import numpy as np

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
    }

def restore_rng_state(torch: Any, state: Mapping[str, Any]) -> None:
    import numpy as np

    required = {"python", "numpy", "torch_cpu", "torch_cuda_all"}
    if set(state) != required:
        raise MetaResumeError("checkpoint RNG state is incomplete")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        cuda_states = list(state["torch_cuda_all"])
        if len(cuda_states) != torch.cuda.device_count():
            raise MetaResumeError("checkpoint CUDA RNG device count changed")
        torch.cuda.set_rng_state_all(cuda_states)

class ResumableWarmupScheduler:
    """Compose legacy warmup with optional validation-only LR reduction.

    Plateau-controlled Task2 runs require zero warmup. In that mode the
    per-optimizer-step hook validates the cursor but deliberately leaves the
    optimizer LR untouched, so a validation reduction cannot be overwritten by
    the legacy warmup path on the next training batch.
    """

    def __init__(
        self,
        optimizer: Any,
        *,
        learning_rate: float,
        warmup_steps: int,
        validation_scheduler_contract: Mapping[str, Any] | None = None,
        validation_history_path: Path | None = None,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if validation_scheduler_contract is not None and warmup_steps != 0:
            raise ValueError("validation LR scheduling requires zero warmup steps")
        if validation_history_path is not None and validation_scheduler_contract is None:
            raise ValueError("validation history path requires a scheduler contract")
        self.optimizer = optimizer
        self.learning_rate = float(learning_rate)
        self.warmup_steps = int(warmup_steps)
        self.completed_steps = 0
        try:
            self.startup_lr_audit = audit_optimizer_peak_lrs(
                optimizer,
                peak_lr=self.learning_rate,
                gate_lr_multiplier=1.0,
                require_dense_sparse=False,
            )
            self.validation_scheduler = (
                None
                if validation_scheduler_contract is None
                else ValidationReduceLROnPlateau(
                    optimizer, validation_scheduler_contract
                )
            )
        except ValidationLRSchedulerError as error:
            raise ValueError(str(error)) from error
        self.validation_history: list[dict[str, Any]] = []
        self.validation_history_path = (
            None
            if validation_history_path is None
            else validation_history_path.expanduser().resolve()
        )

    def apply_for_step(self, global_step: int) -> float:
        if int(global_step) != self.completed_steps:
            raise MetaResumeError("scheduler/global-step state diverged")
        if self.validation_scheduler is not None:
            snapshot = optimizer_lr_snapshot(self.optimizer)
            learning_rates = [float(group["lr"]) for group in snapshot]
            expected = [
                float(value)
                for value in self.validation_scheduler.state_dict()["last_lrs"]
            ]
            if len(learning_rates) != len(expected) or any(
                abs(actual - declared) > 1e-12
                for actual, declared in zip(learning_rates, expected)
            ):
                raise MetaResumeError(
                    "optimizer LR changed outside the validation scheduler"
                )
            return learning_rates[0]
        if self.warmup_steps and global_step < self.warmup_steps:
            scalar = min(1.0, float(global_step + 1) / self.warmup_steps)
            lr = scalar * self.learning_rate
        else:
            lr = self.learning_rate
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def mark_step_completed(self, global_step: int) -> None:
        expected = self.completed_steps + 1
        if int(global_step) != expected:
            raise MetaResumeError("scheduler step completion is non-contiguous")
        self.completed_steps = expected

    @property
    def has_validation_scheduler(self) -> bool:
        return self.validation_scheduler is not None

    def learning_rate_snapshot(self) -> list[dict[str, Any]]:
        return optimizer_lr_snapshot(self.optimizer)

    def observe_validation(
        self, *, metric: float, epoch: int, global_step: int
    ) -> dict[str, Any] | None:
        """Advance plateau state after one completed full validation only."""

        if self.validation_scheduler is None:
            return None
        if int(global_step) != self.completed_steps:
            raise MetaResumeError("validation scheduler/global-step state diverged")
        try:
            event = self.validation_scheduler.step(
                metric, epoch=epoch, global_step=global_step
            )
        except ValidationLRSchedulerError as error:
            raise MetaResumeError(str(error)) from error
        self.validation_history.append(dict(event))
        self.persist_validation_history()
        return event

    def persist_validation_history(self) -> None:
        if self.validation_history_path is None:
            return
        self.validation_history_path.parent.mkdir(parents=True, exist_ok=True)
        assert self.validation_scheduler is not None
        _atomic_json(
            self.validation_history_path,
            {
                "schema": "deltarec-rating-lr-scheduler-history-v1",
                "scheduler": {
                    "kind": self.validation_scheduler.kind,
                    "contract": dict(self.validation_scheduler.contract),
                },
                "startup_lr_audit": self.startup_lr_audit,
                "events": list(self.validation_history),
            },
        )

    def state_dict(self) -> dict[str, Any]:
        state = {
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "completed_steps": self.completed_steps,
        }
        if self.validation_scheduler is not None:
            state["validation_scheduler_state_dict"] = (
                self.validation_scheduler.state_dict()
            )
            state["validation_scheduler_history"] = list(self.validation_history)
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if float(state.get("learning_rate", -1)) != self.learning_rate:
            raise MetaResumeError("checkpoint scheduler learning rate changed")
        if int(state.get("warmup_steps", -1)) != self.warmup_steps:
            raise MetaResumeError("checkpoint scheduler warmup changed")
        completed = int(state.get("completed_steps", -1))
        if completed < 0:
            raise MetaResumeError("checkpoint scheduler state is invalid")
        self.completed_steps = completed
        validation_state = state.get("validation_scheduler_state_dict")
        history = state.get("validation_scheduler_history")
        if self.validation_scheduler is None:
            if validation_state is not None or history is not None:
                raise MetaResumeError(
                    "checkpoint enables an unexpected validation LR scheduler"
                )
            return
        if not isinstance(validation_state, Mapping) or not isinstance(history, list):
            raise MetaResumeError("checkpoint validation LR scheduler state is missing")
        try:
            self.validation_scheduler.load_state_dict(validation_state)
        except ValidationLRSchedulerError as error:
            raise MetaResumeError(str(error)) from error
        if len(history) != self.validation_scheduler.completed_validations:
            raise MetaResumeError("checkpoint validation LR history is incomplete")
        if history and history[-1] != self.validation_scheduler.last_event:
            raise MetaResumeError("checkpoint validation LR history cursor changed")
        self.validation_history = [dict(event) for event in history]
        self.persist_validation_history()

@dataclass(frozen=True)
class ResumeState:
    epoch: int = 0
    next_microbatch: int = 0
    global_step: int = 0
    resumed_from: Path | None = None
    early_stopping: Mapping[str, Any] | None = None

class ResumableCheckpointManager:
    def __init__(
        self,
        root: Path,
        *,
        identity: Mapping[str, Any],
        step_interval: int = 0,
        keep_last: int = 3,
        early_stopping_contract: Mapping[str, Any] | None = None,
        early_stopping_initial_state: Mapping[str, Any] | None = None,
    ) -> None:
        if step_interval < 0:
            raise ValueError("checkpoint step interval must be non-negative")
        if keep_last < 3:
            raise ValueError("checkpoint keep_last must be at least three")
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = dict(identity)
        self.identity_sha256 = _identity_sha256(self.identity)
        self.step_interval = int(step_interval)
        self.keep_last = int(keep_last)
        self._resume_rng_state: Mapping[str, Any] | None = None
        self.early_stopping_contract = (
            None
            if early_stopping_contract is None
            else dict(early_stopping_contract)
        )
        self.early_stopping_state = (
            None
            if self.early_stopping_contract is None
            else EarlyStoppingState.from_contract(self.early_stopping_contract)
        )
        if early_stopping_initial_state is not None:
            if self.early_stopping_contract is None:
                raise ValueError("initial early-stopping state requires a contract")
            try:
                self.early_stopping_state = EarlyStoppingState.from_dict(
                    early_stopping_initial_state,
                    contract=self.early_stopping_contract,
                )
            except EarlyStoppingError as error:
                raise ValueError(str(error)) from error

    def observe_validation(
        self,
        *,
        primary_metric: float,
        epoch: int,
        tie_breaker: float | None = None,
    ) -> Mapping[str, Any]:
        """Update the contract-bound state after a completed validation epoch."""

        if self.early_stopping_state is None:
            raise MetaResumeError("early-stopping contract is not configured")
        return self.early_stopping_state.observe(
            primary_metric=primary_metric,
            tie_breaker=tie_breaker,
            epoch=epoch,
        )

    def pin_best(self, checkpoint: Path) -> None:
        """Atomically point BEST at a complete resumable checkpoint."""

        self._pin_complete_checkpoint("BEST", checkpoint)

    def pin_final(self, checkpoint: Path) -> None:
        self._pin_complete_checkpoint("FINAL", checkpoint)

    def _completion_record_for(self, checkpoint: Path) -> Mapping[str, Any]:
        checkpoint = checkpoint.expanduser().resolve()
        try:
            checkpoint.relative_to(self.root)
        except ValueError as error:
            raise MetaResumeError(
                "checkpoint marker target is outside the resumable root"
            ) from error
        if not checkpoint.is_file():
            raise MetaResumeError(f"checkpoint marker target is missing: {checkpoint}")
        manifest = checkpoint.with_suffix(".complete.json")
        try:
            record = json.loads(manifest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise MetaResumeError(
                f"checkpoint marker target is not complete: {checkpoint}"
            ) from error
        if (
            not isinstance(record, Mapping)
            or record.get("schema") != COMPLETION_SCHEMA
            or record.get("identity_sha256") != self.identity_sha256
            or record.get("checkpoint") != checkpoint.name
        ):
            raise MetaResumeError(
                f"checkpoint marker target completion changed: {checkpoint}"
            )
        try:
            progress = tuple(
                int(record.get(field, -1))
                for field in ("epoch", "next_microbatch", "global_step")
            )
        except (TypeError, ValueError) as error:
            raise MetaResumeError(
                f"checkpoint marker target progress is invalid: {checkpoint}"
            ) from error
        if min(progress) < 0:
            raise MetaResumeError(
                f"checkpoint marker target progress is invalid: {checkpoint}"
            )
        return record

    def _pin_complete_checkpoint(self, label: str, checkpoint: Path) -> None:
        if label not in {"BEST", "FINAL", "PINNED"}:
            raise ValueError("unknown checkpoint protection marker")
        checkpoint = checkpoint.expanduser().resolve()
        self._completion_record_for(checkpoint)
        marker = self.root / label
        if (
            marker.is_file()
            and not marker.is_symlink()
            and marker.read_text(encoding="utf-8").strip() == checkpoint.name
        ):
            return
        _atomic_text(marker, checkpoint.name + "\n")

    def _record_early_stopping_state(
        self, record: Mapping[str, Any], *, checkpoint: Path
    ) -> EarlyStoppingState:
        payload = record.get("early_stopping_state")
        if not isinstance(payload, Mapping):
            raise MetaResumeError(
                f"completed checkpoint lacks persisted early-stopping state: {checkpoint}"
            )
        try:
            return EarlyStoppingState.from_dict(
                payload, contract=self.early_stopping_contract
            )
        except EarlyStoppingError as error:
            raise MetaResumeError(
                f"completed checkpoint early-stopping state is invalid: {checkpoint}"
            ) from error

    def _reconcile_best_marker(
        self,
        records: list[tuple[tuple[int, int, int], Path, Mapping[str, Any]]],
    ) -> None:
        state = self.early_stopping_state
        if state is None or state.best_epoch < 0:
            return
        expected_epoch = state.best_epoch + 1
        matches: list[Path] = []
        for progress, checkpoint, record in records:
            if (
                progress[0] != expected_epoch
                or progress[1] != 0
                or not checkpoint.name.startswith("epoch-")
            ):
                continue
            candidate_state = self._record_early_stopping_state(
                record, checkpoint=checkpoint
            )
            if (
                candidate_state.best_epoch == state.best_epoch
                and candidate_state.last_completed_epoch == state.best_epoch
                and candidate_state.primary_metric == state.primary_metric
                and candidate_state.tie_breaker == state.tie_breaker
            ):
                matches.append(checkpoint)
        if len(matches) != 1:
            raise MetaResumeError(
                "true validation-best complete checkpoint is missing or ambiguous"
            )
        self.pin_best(matches[0])

    def reconcile_terminal_checkpoint(self, checkpoint: Path | None) -> None:
        """Validate and atomically pin a restored terminal epoch checkpoint."""

        state = self.early_stopping_state
        if state is None or not state.stopped:
            raise MetaResumeError(
                "terminal checkpoint requires persisted stopped early-stopping state"
            )
        if state.terminal_reason not in {"early_stopped", "max_epochs_reached"}:
            raise MetaResumeError("terminal checkpoint reason is invalid")
        if checkpoint is None:
            raise MetaResumeError("restored terminal checkpoint is missing")
        checkpoint = checkpoint.expanduser().resolve()
        record = self._completion_record_for(checkpoint)
        terminal_state = self._record_early_stopping_state(
            record, checkpoint=checkpoint
        )
        if (
            not checkpoint.name.startswith("epoch-")
            or int(record["epoch"]) != state.last_completed_epoch + 1
            or int(record["next_microbatch"]) != 0
            or terminal_state.to_dict() != state.to_dict()
        ):
            raise MetaResumeError(
                "restored checkpoint is not the persisted terminal epoch checkpoint"
            )
        self.pin_final(checkpoint)

    @staticmethod
    def _sampler_state(
        sampler: Any, *, epoch: int, next_microbatch: int, num_microbatches: int
    ) -> dict[str, Any]:
        state = {
            "epoch": int(epoch),
            "next_microbatch": int(next_microbatch),
            "num_microbatches": int(num_microbatches),
        }
        for name in ("seed", "num_replicas", "rank", "shuffle", "drop_last"):
            if sampler is not None and hasattr(sampler, name):
                state[name] = getattr(sampler, name)
        return state

    def _validate_sampler(self, sampler: Any, state: Mapping[str, Any]) -> None:
        for name in ("seed", "num_replicas", "rank", "shuffle", "drop_last"):
            if name in state:
                if sampler is None or getattr(sampler, name, None) != state[name]:
                    raise MetaResumeError(f"checkpoint sampler {name} changed")

    def _checkpoint_payload(
        self,
        torch: Any,
        *,
        model: Any,
        optimizer: Any,
        scheduler: ResumableWarmupScheduler,
        sampler: Any,
        epoch: int,
        next_microbatch: int,
        num_microbatches: int,
        global_step: int,
        provenance: Mapping[str, Any],
        early_stopping: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if scheduler.completed_steps != int(global_step):
            raise MetaResumeError("refusing checkpoint with divergent scheduler step")
        state = early_stopping
        if state is None and self.early_stopping_state is not None:
            state = self.early_stopping_state.to_dict()
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "identity_sha256": self.identity_sha256,
            "epoch": int(epoch),
            "next_microbatch": int(next_microbatch),
            "global_step": int(global_step),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": scheduler.state_dict(),
            "checkpoint_learning_rates": scheduler.learning_rate_snapshot(),
            "rng_state": capture_rng_state(torch),
            "sampler_state": self._sampler_state(
                sampler,
                epoch=epoch,
                next_microbatch=next_microbatch,
                num_microbatches=num_microbatches,
            ),
            "deltarec_provenance": dict(provenance),
        }
        if state is not None:
            payload["early_stopping_state"] = dict(state)
        return payload

    def _atomic_torch_save(self, torch: Any, path: Path, payload: Any) -> None:
        try:
            atomic_torch_save(torch, payload, path, overwrite=False)
        except AtomicCheckpointError as error:
            raise MetaResumeError(str(error)) from error

    def save(
        self,
        torch: Any,
        *,
        kind: str,
        model: Any,
        optimizer: Any,
        scheduler: ResumableWarmupScheduler,
        sampler: Any,
        epoch: int,
        next_microbatch: int,
        num_microbatches: int,
        global_step: int,
        provenance: Mapping[str, Any],
        early_stopping: Mapping[str, Any] | None = None,
    ) -> Path:
        if kind not in {"step", "epoch"}:
            raise ValueError("checkpoint kind must be step or epoch")
        name = (
            f"{kind}-e{epoch:04d}-m{next_microbatch:08d}-"
            f"s{global_step:012d}.pt"
        )
        checkpoint = self.root / name
        payload = self._checkpoint_payload(
            torch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            epoch=epoch,
            next_microbatch=next_microbatch,
            num_microbatches=num_microbatches,
            global_step=global_step,
            provenance=provenance,
            early_stopping=early_stopping,
        )
        self._atomic_torch_save(torch, checkpoint, payload)
        completion = {
            "schema": COMPLETION_SCHEMA,
            "checkpoint": checkpoint.name,
            "identity_sha256": self.identity_sha256,
            "epoch": int(epoch),
            "next_microbatch": int(next_microbatch),
            "global_step": int(global_step),
        }
        if "early_stopping_state" in payload:
            completion["early_stopping_state"] = payload["early_stopping_state"]
        completion["checkpoint_learning_rates"] = payload[
            "checkpoint_learning_rates"
        ]
        manifest = checkpoint.with_suffix(".complete.json")
        _atomic_json(manifest, completion)
        _atomic_json(self.root / "latest.json", completion)
        self._prune(checkpoint)
        return checkpoint

    def maybe_save_step(self, torch: Any, *, global_step: int, **kwargs: Any) -> Path | None:
        if self.step_interval <= 0 or global_step % self.step_interval:
            return None
        return self.save(torch, kind="step", global_step=global_step, **kwargs)

    def _prune(self, newest: Path) -> None:
        protected = {newest}
        for label in ("BEST", "FINAL", "PINNED"):
            marker = self.root / label
            if not marker.is_file() or marker.is_symlink():
                continue
            candidate = self.root / marker.read_text(encoding="utf-8").strip()
            if candidate.parent == self.root and candidate.is_file():
                protected.add(candidate)
        manifests = sorted(
            self.root.glob("*.complete.json"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for manifest in manifests[self.keep_last :]:
            try:
                record = json.loads(manifest.read_text(encoding="utf-8"))
                checkpoint = self.root / str(record["checkpoint"])
            except (json.JSONDecodeError, KeyError, OSError):
                continue
            if checkpoint not in protected:
                checkpoint.unlink(missing_ok=True)
                manifest.unlink(missing_ok=True)

    def _complete_records(self) -> list[tuple[tuple[int, int, int], Path, Mapping[str, Any]]]:
        records = []
        for manifest in sorted(self.root.glob("*.complete.json")):
            try:
                record = json.loads(manifest.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as error:
                raise MetaResumeError(f"invalid checkpoint completion file: {manifest}") from error
            if record.get("schema") != COMPLETION_SCHEMA:
                raise MetaResumeError(f"unknown checkpoint completion schema: {manifest}")
            if record.get("identity_sha256") != self.identity_sha256:
                raise MetaResumeError("checkpoint config/data identity mismatch")
            checkpoint = self.root / str(record.get("checkpoint", ""))
            if not checkpoint.is_file():
                raise MetaResumeError(f"completed checkpoint file is missing: {checkpoint}")
            progress = (
                int(record.get("epoch", -1)),
                int(record.get("next_microbatch", -1)),
                int(record.get("global_step", -1)),
            )
            if min(progress) < 0:
                raise MetaResumeError(f"checkpoint progress is invalid: {manifest}")
            records.append((progress, checkpoint, record))
        return records

    def restore(
        self,
        torch: Any,
        *,
        model: Any,
        optimizer: Any,
        scheduler: ResumableWarmupScheduler,
        sampler: Any,
        reconcile_markers: bool = True,
    ) -> ResumeState:
        records = self._complete_records()
        if not records:
            return ResumeState()
        payload = None
        checkpoint = None
        checkpoint_record = None
        failures = []
        for _, candidate, record in sorted(
            records, key=lambda value: value[0], reverse=True
        ):
            try:
                payload = torch.load(candidate, map_location="cpu", weights_only=False)
                checkpoint = candidate
                checkpoint_record = record
                break
            except (EOFError, OSError, RuntimeError, ValueError) as error:
                failures.append(f"{candidate.name}: {error}")
        if payload is None or checkpoint is None or checkpoint_record is None:
            raise MetaResumeError(
                "no loadable completed checkpoint; " + "; ".join(failures)
            )
        if not isinstance(payload, Mapping) or payload.get("schema") != CHECKPOINT_SCHEMA:
            raise MetaResumeError("checkpoint payload schema is invalid")
        if payload.get("identity_sha256") != self.identity_sha256:
            raise MetaResumeError("checkpoint payload config/data identity mismatch")
        required = {
            "model_state_dict",
            "optimizer_state_dict",
            "lr_scheduler_state_dict",
            "epoch",
            "next_microbatch",
            "global_step",
            "rng_state",
            "sampler_state",
        }
        if not required.issubset(payload):
            raise MetaResumeError("checkpoint payload is incomplete")
        try:
            payload_progress = tuple(
                int(payload[field])
                for field in ("epoch", "next_microbatch", "global_step")
            )
            manifest_progress = tuple(
                int(checkpoint_record[field])
                for field in ("epoch", "next_microbatch", "global_step")
            )
        except (KeyError, TypeError, ValueError) as error:
            raise MetaResumeError(
                "checkpoint payload/completion progress is invalid"
            ) from error
        if payload_progress != manifest_progress:
            raise MetaResumeError("checkpoint payload/completion progress changed")
        early_payload = payload.get("early_stopping_state")
        manifest_early_payload = checkpoint_record.get("early_stopping_state")
        if isinstance(early_payload, Mapping) != isinstance(
            manifest_early_payload, Mapping
        ):
            raise MetaResumeError(
                "checkpoint payload/completion early-stopping state changed"
            )
        if (
            isinstance(early_payload, Mapping)
            and dict(early_payload) != dict(manifest_early_payload)
        ):
            raise MetaResumeError(
                "checkpoint payload/completion early-stopping state changed"
            )
        payload_lrs = payload.get("checkpoint_learning_rates")
        manifest_lrs = checkpoint_record.get("checkpoint_learning_rates")
        if payload_lrs != manifest_lrs:
            raise MetaResumeError(
                "checkpoint payload/completion learning-rate evidence changed"
            )
        if self.early_stopping_contract is not None:
            if not isinstance(early_payload, Mapping):
                # A migration sidecar may reconstruct state for a legacy
                # checkpoint.  New checkpoints always persist the full state.
                if self.early_stopping_state is None or (
                    self.early_stopping_state.last_completed_epoch < 0
                ):
                    raise MetaResumeError("checkpoint early-stopping state is missing")
                early_payload = self.early_stopping_state.to_dict()
            try:
                self.early_stopping_state = EarlyStoppingState.from_dict(
                    early_payload, contract=self.early_stopping_contract
                )
            except EarlyStoppingError as error:
                raise MetaResumeError(str(error)) from error
        elif isinstance(early_payload, Mapping):
            try:
                self.early_stopping_state = EarlyStoppingState.from_dict(early_payload)
            except EarlyStoppingError as error:
                raise MetaResumeError(str(error)) from error
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["lr_scheduler_state_dict"])
        global_step = int(payload["global_step"])
        if scheduler.completed_steps != global_step:
            raise MetaResumeError("restored scheduler/global-step state diverged")
        self._validate_sampler(sampler, payload["sampler_state"])
        self._resume_rng_state = payload["rng_state"]
        restore_rng_state(torch, self._resume_rng_state)
        if reconcile_markers:
            self._reconcile_best_marker(records)
            if (
                self.early_stopping_state is not None
                and self.early_stopping_state.stopped
            ):
                self.reconcile_terminal_checkpoint(checkpoint)
        return ResumeState(
            epoch=int(payload["epoch"]),
            next_microbatch=int(payload["next_microbatch"]),
            global_step=global_step,
            resumed_from=checkpoint,
            early_stopping=(
                None
                if self.early_stopping_state is None
                else self.early_stopping_state.to_dict()
            ),
        )

    def restore_rng_after_loader_position(self, torch: Any) -> None:
        if self._resume_rng_state is None:
            raise MetaResumeError("no resumed RNG state is available")
        restore_rng_state(torch, self._resume_rng_state)

    def save_final_export(
        self, torch: Any, path: Path, *, use_best: bool = False, **kwargs: Any
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        if use_best:
            marker = self.root / "BEST"
            if not marker.is_file():
                raise MetaResumeError("best checkpoint marker is missing")
            candidate = self.root / marker.read_text(encoding="utf-8").strip()
            try:
                payload = torch.load(candidate, map_location="cpu", weights_only=False)
                if (
                    not isinstance(payload, Mapping)
                    or payload.get("schema") != CHECKPOINT_SCHEMA
                    or payload.get("identity_sha256") != self.identity_sha256
                ):
                    raise MetaResumeError("best checkpoint identity changed")
            except (KeyError, OSError, RuntimeError, ValueError) as error:
                raise MetaResumeError("best checkpoint cannot be exported") from error
            # Export the validation-best state as one coherent snapshot. Loading
            # an older best scheduler into the live objects and pairing it with
            # the terminal global step would create a divergent checkpoint.
            self._atomic_torch_save(torch, path, dict(payload))
            return path
        payload = self._checkpoint_payload(torch, **kwargs)
        self._atomic_torch_save(torch, path, payload)
        return path
