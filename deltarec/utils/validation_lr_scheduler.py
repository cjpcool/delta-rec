"""Validation-only, multi-optimizer ReduceLROnPlateau state.

Task2 uses one logical plateau decision for every optimizer family.  TorchRec's
``CombinedOptimizer`` is not a ``torch.optim.Optimizer``, so PyTorch's stock
``ReduceLROnPlateau`` cannot own it directly.  This controller implements the
same max/min and absolute-threshold decision while applying one reduction to
all of the CombinedOptimizer's named parameter groups.

The public ``patience`` is measured in *completed bad validations*: a value of
two reduces on the second consecutive plateau validation.  This is deliberate
and is recorded in the serialized contract.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence


SCHEMA = "deltarec-validation-reduce-lr-on-plateau-v1"
KIND = "reduce_lr_on_plateau"


class ValidationLRSchedulerError(ValueError):
    """Raised when the scheduler contract or persisted state is unsafe."""


def _is_gate_group(name: str, group: Mapping[str, Any]) -> bool:
    return (
        name.lower() == "gate"
        or bool(group.get("gdr_gate_group", False))
        or bool(group.get("kda_channel_gate_group", False))
    )


def _optimizer_groups(optimizer: Any) -> list[dict[str, Any]]:
    """Return every actual optimizer group once, with a stable logical label."""

    named = getattr(optimizer, "optimizers", None)
    if named is None:
        named_items: list[tuple[str, Any]] = [("dense", optimizer)]
    else:
        if not isinstance(named, (list, tuple)) or not named:
            raise ValidationLRSchedulerError(
                "optimizer must expose a non-empty named optimizer sequence"
            )
        named_items = []
        child_names: set[str] = set()
        for item in named:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValidationLRSchedulerError(
                    "named optimizer entries must be (name, optimizer) pairs"
                )
            name, child = item
            if not isinstance(name, str) or not name:
                raise ValidationLRSchedulerError("optimizer name must be non-empty")
            if name in child_names:
                raise ValidationLRSchedulerError(
                    f"duplicate named optimizer child {name!r}"
                )
            child_names.add(name)
            named_items.append((name, child))

    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    role_counts: dict[str, int] = {}

    def append(name: str, group: Any) -> None:
        if not isinstance(group, dict):
            raise ValidationLRSchedulerError("optimizer parameter group must be a dict")
        identity = id(group)
        if identity in seen:
            raise ValidationLRSchedulerError(
                "optimizer parameter group is referenced by more than one alias"
            )
        if "lr" not in group:
            raise ValidationLRSchedulerError("optimizer parameter group lacks lr")
        role = "gate" if _is_gate_group(name, group) else name
        index = role_counts.get(role, 0)
        role_counts[role] = index + 1
        seen.add(identity)
        records.append(
            {
                "optimizer": role,
                "source_optimizer": name,
                "group": index,
                "key": f"{role}:{index}",
                "param_group": group,
            }
        )

    for name, child in named_items:
        groups = getattr(child, "param_groups", None)
        if not isinstance(groups, list) or not groups:
            raise ValidationLRSchedulerError(
                f"optimizer {name!r} does not expose parameter groups"
            )
        for group in groups:
            append(name, group)

    # Some wrappers append a separately configured gate group to the combined
    # optimizer after construction.  Account for it, but reject any other
    # unnamed group so the startup audit remains fail closed.
    combined_groups = getattr(optimizer, "param_groups", None)
    if named is not None and isinstance(combined_groups, list):
        named_group_ids = set(seen)
        mirrored: set[int] = set()
        for group in combined_groups:
            identity = id(group)
            if identity in named_group_ids:
                # TorchRec exposes one aggregate mirror of each named child
                # group. A second reference is an ambiguous alias.
                if identity in mirrored:
                    raise ValidationLRSchedulerError(
                        "combined optimizer repeats a parameter-group alias"
                    )
                mirrored.add(identity)
                continue
            if identity in seen:
                raise ValidationLRSchedulerError(
                    "combined optimizer repeats a parameter-group alias"
                )
            if isinstance(group, dict) and _is_gate_group("", group):
                append("gate", group)
                continue
            if named is not None:
                raise ValidationLRSchedulerError(
                    "combined optimizer exposes an unnamed non-gate parameter group"
                )
            append("dense", group)
    if not records:
        raise ValidationLRSchedulerError("optimizer exposes no parameter groups")
    return records


def optimizer_lr_snapshot(optimizer: Any) -> list[dict[str, Any]]:
    """Capture finite positive LR values for every actual parameter group."""

    snapshot: list[dict[str, Any]] = []
    for record in _optimizer_groups(optimizer):
        try:
            learning_rate = float(record["param_group"]["lr"])
        except (TypeError, ValueError) as error:
            raise ValidationLRSchedulerError(
                f"optimizer group {record['key']} LR is not scalar"
            ) from error
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValidationLRSchedulerError(
                f"optimizer group {record['key']} LR must be finite and positive"
            )
        snapshot.append(
            {
                "optimizer": record["optimizer"],
                "source_optimizer": record["source_optimizer"],
                "group": record["group"],
                "key": record["key"],
                "lr": learning_rate,
            }
        )
    return snapshot


def audit_optimizer_peak_lrs(
    optimizer: Any,
    *,
    peak_lr: float,
    gate_lr_multiplier: float = 1.0,
    require_dense_sparse: bool = True,
) -> dict[str, Any]:
    """Fail closed unless every group has its explicitly declared peak LR."""

    peak = float(peak_lr)
    gate_multiplier = float(gate_lr_multiplier)
    if (
        not math.isfinite(peak)
        or peak <= 0.0
        or not math.isfinite(gate_multiplier)
        or gate_multiplier <= 0.0
    ):
        raise ValidationLRSchedulerError("peak and gate LR multiplier must be positive")
    groups = optimizer_lr_snapshot(optimizer)
    names = {str(group["optimizer"]) for group in groups}
    unexpected = names - {"dense", "sparse_fused", "gate"}
    if unexpected:
        raise ValidationLRSchedulerError(
            "optimizer exposes unexpected Task2 roles: "
            + ", ".join(sorted(unexpected))
        )
    if require_dense_sparse and not {"dense", "sparse_fused"}.issubset(names):
        raise ValidationLRSchedulerError(
            "optimizer must expose dense and sparse_fused groups"
        )
    for group in groups:
        expected = peak * (gate_multiplier if group["optimizer"] == "gate" else 1.0)
        if abs(float(group["lr"]) - expected) > 1e-12:
            raise ValidationLRSchedulerError(
                f"optimizer group {group['key']} LR {group['lr']} != {expected}"
            )
        group["expected_peak_lr"] = expected
    return {
        "peak_lr": peak,
        "gate_lr_multiplier": gate_multiplier,
        "group_count": len(groups),
        "groups": groups,
    }


class ValidationReduceLROnPlateau:
    """One serializable plateau controller for a named optimizer tree."""

    kind = KIND

    def __init__(self, optimizer: Any, contract: Mapping[str, Any]) -> None:
        self.optimizer = optimizer
        self.contract = self._normalize_contract(contract)
        initial = optimizer_lr_snapshot(optimizer)
        if any(float(group["lr"]) < self.contract["min_lr"] for group in initial):
            raise ValidationLRSchedulerError(
                "optimizer LR may not start below the scheduler minimum"
            )
        self._group_keys = [str(group["key"]) for group in initial]
        self._last_lrs = [float(group["lr"]) for group in initial]
        self.best: float | None = None
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.last_completed_epoch = -1
        self.completed_validations = 0
        self.last_event: dict[str, Any] | None = None

    @staticmethod
    def _normalize_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
        mode = str(contract.get("mode", "max"))
        threshold_mode = str(contract.get("threshold_mode", "abs"))
        factor = float(contract.get("factor", 0.5))
        patience = int(contract.get("patience", 2))
        threshold = float(contract.get("threshold", 1e-4))
        cooldown = int(contract.get("cooldown", 0))
        minimum = float(contract.get("min_lr", 3e-6))
        metric_name = str(contract.get("metric_name", ""))
        if mode not in {"max", "min"}:
            raise ValidationLRSchedulerError("scheduler mode must be max or min")
        if threshold_mode != "abs":
            raise ValidationLRSchedulerError("Task2 scheduler threshold_mode must be abs")
        if not (0.0 < factor < 1.0):
            raise ValidationLRSchedulerError("scheduler factor must be between zero and one")
        if patience < 1:
            raise ValidationLRSchedulerError("scheduler patience must be positive")
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValidationLRSchedulerError("scheduler threshold must be finite and nonnegative")
        if cooldown < 0:
            raise ValidationLRSchedulerError("scheduler cooldown must be nonnegative")
        if not math.isfinite(minimum) or minimum <= 0.0:
            raise ValidationLRSchedulerError("scheduler minimum LR must be positive")
        if not metric_name:
            raise ValidationLRSchedulerError("scheduler metric_name is required")
        return {
            "mode": mode,
            "factor": factor,
            "patience": patience,
            "patience_unit": "completed_bad_validations",
            "threshold": threshold,
            "threshold_mode": threshold_mode,
            "cooldown": cooldown,
            "min_lr": minimum,
            "metric_name": metric_name,
        }

    def _snapshot(self) -> list[dict[str, Any]]:
        snapshot = optimizer_lr_snapshot(self.optimizer)
        keys = [str(group["key"]) for group in snapshot]
        if keys != self._group_keys:
            raise ValidationLRSchedulerError(
                "optimizer parameter groups changed after scheduler construction"
            )
        return snapshot

    def _improved(self, metric: float) -> bool:
        if self.best is None:
            return True
        threshold = float(self.contract["threshold"])
        if self.contract["mode"] == "max":
            return metric > self.best + threshold
        return metric < self.best - threshold

    def step(self, metric: float, *, epoch: int, global_step: int) -> dict[str, Any]:
        """Consume exactly one completed epoch validation and return its LR log."""

        value = float(metric)
        if not math.isfinite(value):
            raise ValidationLRSchedulerError(
                "non-finite validation metric cannot advance the LR scheduler"
            )
        completed_epoch = int(epoch)
        if completed_epoch != self.last_completed_epoch + 1:
            raise ValidationLRSchedulerError(
                "scheduler validations must be contiguous completed epochs"
            )
        step_value = int(global_step)
        if step_value < 0:
            raise ValidationLRSchedulerError("scheduler global_step must be nonnegative")

        before = self._snapshot()
        before_lrs = [float(group["lr"]) for group in before]
        if any(abs(left - right) > 1e-12 for left, right in zip(before_lrs, self._last_lrs)):
            raise ValidationLRSchedulerError(
                "optimizer LR changed outside the validation scheduler"
            )

        bad_before = self.num_bad_epochs
        improved = self._improved(value)
        if improved:
            self.best = value
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0

        reduced = False
        if self.num_bad_epochs >= int(self.contract["patience"]):
            factor = float(self.contract["factor"])
            minimum = float(self.contract["min_lr"])
            for record in _optimizer_groups(self.optimizer):
                old_lr = float(record["param_group"]["lr"])
                new_lr = max(old_lr * factor, minimum)
                record["param_group"]["lr"] = new_lr
                reduced = reduced or new_lr < old_lr
            self.num_bad_epochs = 0
            self.cooldown_counter = int(self.contract["cooldown"])

        after = self._snapshot()
        self._last_lrs = [float(group["lr"]) for group in after]
        self.last_completed_epoch = completed_epoch
        self.completed_validations += 1
        event = {
            "epoch": completed_epoch + 1,
            "global_step": step_value,
            "metric_name": self.contract["metric_name"],
            "validation_metric": value,
            "learning_rates_before": before,
            "learning_rates_after": after,
            "improved": improved,
            "reduced": reduced,
            "bad_epoch_counter_before": bad_before,
            "bad_epoch_counter": self.num_bad_epochs,
            "cooldown_counter": self.cooldown_counter,
        }
        self.last_event = event
        return event

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "kind": KIND,
            "contract": dict(self.contract),
            "group_keys": list(self._group_keys),
            "last_lrs": list(self._last_lrs),
            "best": self.best,
            "num_bad_epochs": self.num_bad_epochs,
            "cooldown_counter": self.cooldown_counter,
            "last_completed_epoch": self.last_completed_epoch,
            "completed_validations": self.completed_validations,
            "last_event": copy.deepcopy(self.last_event),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Validate fully, then restore optimizer LRs and scheduler state atomically."""

        if not isinstance(state, Mapping):
            raise ValidationLRSchedulerError(
                "checkpoint LR scheduler state is not a mapping"
            )
        required_fields = {
            "schema",
            "kind",
            "contract",
            "group_keys",
            "last_lrs",
            "best",
            "num_bad_epochs",
            "cooldown_counter",
            "last_completed_epoch",
            "completed_validations",
            "last_event",
        }
        if set(state) != required_fields:
            raise ValidationLRSchedulerError(
                "checkpoint LR scheduler payload fields changed"
            )
        if state.get("schema") != SCHEMA or state.get("kind") != KIND:
            raise ValidationLRSchedulerError("checkpoint LR scheduler schema changed")
        if state.get("contract") != self.contract:
            raise ValidationLRSchedulerError("checkpoint LR scheduler contract changed")

        keys = state.get("group_keys")
        if (
            not isinstance(keys, list)
            or any(not isinstance(value, str) for value in keys)
            or keys != self._group_keys
            or len(set(keys)) != len(keys)
        ):
            raise ValidationLRSchedulerError("checkpoint optimizer LR groups changed")

        last_lrs = state.get("last_lrs")
        if not isinstance(last_lrs, list) or len(last_lrs) != len(keys):
            raise ValidationLRSchedulerError(
                "checkpoint LR scheduler values are missing"
            )
        try:
            restored_lrs = [float(value) for value in last_lrs]
        except (TypeError, ValueError, OverflowError) as error:
            raise ValidationLRSchedulerError(
                "checkpoint LR scheduler value is invalid"
            ) from error
        if any(
            isinstance(value, bool)
            or not math.isfinite(restored)
            or restored < self.contract["min_lr"]
            for value, restored in zip(last_lrs, restored_lrs)
        ):
            raise ValidationLRSchedulerError(
                "checkpoint LR scheduler value is invalid"
            )

        raw_best = state.get("best")
        try:
            restored_best = None if raw_best is None else float(raw_best)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValidationLRSchedulerError(
                "checkpoint scheduler best metric is invalid"
            ) from error
        if (
            isinstance(raw_best, bool)
            or (
                restored_best is not None
                and not math.isfinite(restored_best)
            )
        ):
            raise ValidationLRSchedulerError(
                "checkpoint scheduler best metric is invalid"
            )

        def state_int(field: str) -> int:
            value = state.get(field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValidationLRSchedulerError(
                    f"checkpoint scheduler {field} is not an integer"
                )
            return value

        bad = state_int("num_bad_epochs")
        cooldown = state_int("cooldown_counter")
        last_epoch = state_int("last_completed_epoch")
        completed = state_int("completed_validations")
        if bad < 0 or bad >= int(self.contract["patience"]):
            raise ValidationLRSchedulerError(
                "checkpoint scheduler bad-epoch state is invalid"
            )
        if cooldown < 0 or cooldown > int(self.contract["cooldown"]):
            raise ValidationLRSchedulerError(
                "checkpoint scheduler cooldown state is invalid"
            )
        if completed < 0 or last_epoch != completed - 1:
            raise ValidationLRSchedulerError(
                "checkpoint validation cursor is invalid"
            )

        last_event = state.get("last_event")
        if completed == 0:
            if (
                last_event is not None
                or restored_best is not None
                or bad != 0
                or cooldown != 0
            ):
                raise ValidationLRSchedulerError(
                    "empty checkpoint scheduler history is inconsistent"
                )
            restored_event = None
        else:
            if restored_best is None or not isinstance(last_event, Mapping):
                raise ValidationLRSchedulerError(
                    "checkpoint scheduler last event is inconsistent"
                )
            event_fields = {
                "epoch",
                "global_step",
                "metric_name",
                "validation_metric",
                "learning_rates_before",
                "learning_rates_after",
                "improved",
                "reduced",
                "bad_epoch_counter_before",
                "bad_epoch_counter",
                "cooldown_counter",
            }
            if set(last_event) != event_fields:
                raise ValidationLRSchedulerError(
                    "checkpoint scheduler last-event fields changed"
                )

            def event_int(field: str) -> int:
                value = last_event.get(field)
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValidationLRSchedulerError(
                        f"checkpoint scheduler event {field} is not an integer"
                    )
                return value

            event_epoch = event_int("epoch")
            event_step = event_int("global_step")
            event_bad_before = event_int("bad_epoch_counter_before")
            event_bad = event_int("bad_epoch_counter")
            event_cooldown = event_int("cooldown_counter")
            if (
                event_epoch != completed
                or event_step < 0
                or event_bad_before < 0
                or event_bad_before >= int(self.contract["patience"])
                or event_bad != bad
                or event_cooldown != cooldown
            ):
                raise ValidationLRSchedulerError(
                    "checkpoint scheduler last-event counters are inconsistent"
                )
            if last_event.get("metric_name") != self.contract["metric_name"]:
                raise ValidationLRSchedulerError(
                    "checkpoint scheduler last-event metric changed"
                )
            raw_metric = last_event.get("validation_metric")
            try:
                event_metric = float(raw_metric)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValidationLRSchedulerError(
                    "checkpoint scheduler last-event metric is invalid"
                ) from error
            if isinstance(raw_metric, bool) or not math.isfinite(event_metric):
                raise ValidationLRSchedulerError(
                    "checkpoint scheduler last-event metric is invalid"
                )
            improved = last_event.get("improved")
            reduced = last_event.get("reduced")
            if not isinstance(improved, bool) or not isinstance(reduced, bool):
                raise ValidationLRSchedulerError(
                    "checkpoint scheduler last-event decisions are invalid"
                )
            if improved and (
                reduced
                or bad != 0
                or event_metric != restored_best
            ):
                raise ValidationLRSchedulerError(
                    "checkpoint scheduler best metric/history diverged"
                )
            threshold = float(self.contract["threshold"])
            would_improve = (
                event_metric > restored_best + threshold
                if self.contract["mode"] == "max"
                else event_metric < restored_best - threshold
            )
            if not improved and would_improve:
                raise ValidationLRSchedulerError(
                    "checkpoint scheduler best metric/history diverged"
                )

            def event_lrs(field: str) -> list[float]:
                records = last_event.get(field)
                if not isinstance(records, list) or len(records) != len(keys):
                    raise ValidationLRSchedulerError(
                        f"checkpoint scheduler event {field} is invalid"
                    )
                result: list[float] = []
                record_fields = {
                    "optimizer",
                    "source_optimizer",
                    "group",
                    "key",
                    "lr",
                }
                for record, key in zip(records, keys):
                    if (
                        not isinstance(record, Mapping)
                        or set(record) != record_fields
                    ):
                        raise ValidationLRSchedulerError(
                            f"checkpoint scheduler event {field} is invalid"
                        )
                    group_index = record.get("group")
                    if (
                        record.get("key") != key
                        or not isinstance(record.get("optimizer"), str)
                        or not record.get("optimizer")
                        or not isinstance(record.get("source_optimizer"), str)
                        or not record.get("source_optimizer")
                        or isinstance(group_index, bool)
                        or not isinstance(group_index, int)
                        or group_index < 0
                    ):
                        raise ValidationLRSchedulerError(
                            f"checkpoint scheduler event {field} topology changed"
                        )
                    raw_lr = record.get("lr")
                    try:
                        learning_rate = float(raw_lr)
                    except (TypeError, ValueError, OverflowError) as error:
                        raise ValidationLRSchedulerError(
                            f"checkpoint scheduler event {field} LR is invalid"
                        ) from error
                    if (
                        isinstance(raw_lr, bool)
                        or not math.isfinite(learning_rate)
                        or learning_rate < self.contract["min_lr"]
                    ):
                        raise ValidationLRSchedulerError(
                            f"checkpoint scheduler event {field} LR is invalid"
                        )
                    result.append(learning_rate)
                return result

            before_lrs = event_lrs("learning_rates_before")
            after_lrs = event_lrs("learning_rates_after")
            if any(
                abs(actual - expected) > 1e-12
                for actual, expected in zip(after_lrs, restored_lrs)
            ):
                raise ValidationLRSchedulerError(
                    "checkpoint scheduler last-event LR/history diverged"
                )
            if any(
                after > before + 1e-12
                for before, after in zip(before_lrs, after_lrs)
            ):
                raise ValidationLRSchedulerError(
                    "checkpoint scheduler last-event LR increased"
                )
            event_reduced = any(
                after < before - 1e-12
                for before, after in zip(before_lrs, after_lrs)
            )
            if reduced != event_reduced:
                raise ValidationLRSchedulerError(
                    "checkpoint scheduler last-event reduction is inconsistent"
                )
            restored_event = copy.deepcopy(dict(last_event))

        # TorchRec KeyedOptimizer checkpoints may intentionally omit parameter
        # groups. The scheduler group_keys/last_lrs pair is the canonical LR
        # source, so write it to every actual group only after full validation.
        groups = _optimizer_groups(self.optimizer)
        actual_keys = [str(group["key"]) for group in groups]
        if actual_keys != keys:
            raise ValidationLRSchedulerError(
                "checkpoint optimizer LR groups changed"
            )
        try:
            original_lrs = [
                float(group["param_group"]["lr"]) for group in groups
            ]
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValidationLRSchedulerError(
                "optimizer LR groups cannot be snapshotted for restore"
            ) from error
        if any(
            not math.isfinite(value) or value <= 0.0 for value in original_lrs
        ):
            raise ValidationLRSchedulerError(
                "optimizer LR groups cannot be snapshotted for restore"
            )

        targets = dict(zip(keys, restored_lrs))
        try:
            for group in groups:
                group["param_group"]["lr"] = targets[str(group["key"])]
            post_groups = _optimizer_groups(self.optimizer)
            if (
                [str(group["key"]) for group in post_groups] != keys
                or [id(group["param_group"]) for group in post_groups]
                != [id(group["param_group"]) for group in groups]
            ):
                raise ValidationLRSchedulerError(
                    "optimizer LR group topology changed during restore"
                )
            actual_lrs = [
                float(group["param_group"]["lr"]) for group in post_groups
            ]
            if any(
                not math.isfinite(actual)
                or abs(actual - expected) > 1e-12
                for actual, expected in zip(actual_lrs, restored_lrs)
            ):
                raise ValidationLRSchedulerError(
                    "failed to restore optimizer LR groups"
                )
        except Exception as error:
            rollback_errors: list[Exception] = []
            for group, learning_rate in zip(groups, original_lrs):
                try:
                    group["param_group"]["lr"] = learning_rate
                except Exception as rollback_error:  # pragma: no cover
                    rollback_errors.append(rollback_error)
            try:
                rolled_back_lrs = [
                    float(group["param_group"]["lr"]) for group in groups
                ]
            except Exception as rollback_error:  # pragma: no cover
                rollback_errors.append(rollback_error)
                rolled_back_lrs = []
            rollback_mismatch = (
                len(rolled_back_lrs) != len(original_lrs)
                or any(
                    abs(actual - expected) > 1e-12
                    for actual, expected in zip(
                        rolled_back_lrs, original_lrs
                    )
                )
            )
            if rollback_errors or rollback_mismatch:
                raise ValidationLRSchedulerError(
                    "optimizer LR restore failed and rollback was incomplete"
                ) from error
            raise ValidationLRSchedulerError(
                "optimizer LR restore failed; original LRs restored"
            ) from error

        # Commit scheduler counters/history only after every LR assignment and
        # its postcondition have succeeded.
        self._last_lrs = list(restored_lrs)
        self.best = restored_best
        self.num_bad_epochs = bad
        self.cooldown_counter = cooldown
        self.last_completed_epoch = last_epoch
        self.completed_validations = completed
        self.last_event = restored_event


__all__: Sequence[str] = (
    "KIND",
    "SCHEMA",
    "ValidationLRSchedulerError",
    "ValidationReduceLROnPlateau",
    "audit_optimizer_peak_lrs",
    "optimizer_lr_snapshot",
)
