"""Small, serializable validation early-stopping state used by baseline seams.

The class deliberately contains no model or optimizer code.  It only makes the
validation decision deterministic across an uninterrupted run and a restart;
the caller remains responsible for saving the checkpoint at the epoch boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


class EarlyStoppingError(ValueError):
    """Raised when persisted early-stopping state is incompatible."""


@dataclass
class EarlyStoppingState:
    """Validation-selection cursor with deterministic patience semantics."""

    primary_metric: float = -math.inf
    tie_breaker: float = math.inf
    best_epoch: int = -1
    bad_evaluation_count: int = 0
    last_completed_epoch: int = -1
    min_delta: float = 1e-4
    patience: int = 10
    minimum_epochs: int = 5
    maximum_epochs: int | None = None
    primary_mode: str = "max"
    tie_breaker_mode: str | None = None
    stopped: bool = False
    terminal_reason: str | None = None

    @classmethod
    def from_contract(cls, contract: Mapping[str, Any]) -> "EarlyStoppingState":
        mode = str(contract.get("primary_mode", "max"))
        if mode not in {"max", "min"}:
            raise EarlyStoppingError("primary_mode must be max or min")
        tie_mode = contract.get("tie_breaker_mode")
        if tie_mode is not None and str(tie_mode) not in {"max", "min"}:
            raise EarlyStoppingError("tie_breaker_mode must be max, min, or null")
        minimum = int(contract.get("minimum_epochs", 5))
        patience = int(contract.get("patience", 10))
        delta = float(contract.get("min_delta", 1e-4))
        maximum = contract.get("maximum_epochs")
        if minimum < 0 or patience <= 0 or not math.isfinite(delta) or delta < 0:
            raise EarlyStoppingError("invalid early-stopping contract")
        if maximum is not None and int(maximum) < minimum:
            raise EarlyStoppingError("maximum_epochs must cover minimum_epochs")
        initial = -math.inf if mode == "max" else math.inf
        tie_initial = math.inf if tie_mode == "min" else -math.inf
        return cls(
            primary_metric=initial,
            tie_breaker=tie_initial,
            min_delta=delta,
            patience=patience,
            minimum_epochs=minimum,
            maximum_epochs=None if maximum is None else int(maximum),
            primary_mode=mode,
            tie_breaker_mode=None if tie_mode is None else str(tie_mode),
        )

    def contract(self) -> dict[str, Any]:
        return {
            "min_delta": float(self.min_delta),
            "patience": int(self.patience),
            "minimum_epochs": int(self.minimum_epochs),
            "maximum_epochs": self.maximum_epochs,
            "primary_mode": self.primary_mode,
            "tie_breaker_mode": self.tie_breaker_mode,
        }

    def to_dict(self) -> dict[str, Any]:
        best_metric = (
            None if not math.isfinite(self.primary_metric) else float(self.primary_metric)
        )
        best_tie_breaker = (
            None if not math.isfinite(self.tie_breaker) else float(self.tie_breaker)
        )
        return {
            # Keep the descriptive names for the generic manager and the
            # explicit best_* names required by the headline handoff.
            "primary_metric": best_metric,
            "tie_breaker": best_tie_breaker,
            "best_metric": best_metric,
            "best_tie_breaker": best_tie_breaker,
            "best_epoch": int(self.best_epoch),
            "bad_evaluation_count": int(self.bad_evaluation_count),
            "last_completed_epoch": int(self.last_completed_epoch),
            "min_delta": float(self.min_delta),
            "patience": int(self.patience),
            "minimum_epochs": int(self.minimum_epochs),
            "maximum_epochs": self.maximum_epochs,
            "primary_mode": self.primary_mode,
            "tie_breaker_mode": self.tie_breaker_mode,
            "stopped": bool(self.stopped),
            "terminal_reason": self.terminal_reason,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], *, contract: Mapping[str, Any] | None = None
    ) -> "EarlyStoppingState":
        required = {
            "best_epoch",
            "bad_evaluation_count",
            "last_completed_epoch",
            "min_delta",
            "patience",
            "minimum_epochs",
            "maximum_epochs",
            "primary_mode",
            "tie_breaker_mode",
            "stopped",
            "terminal_reason",
        }
        if not required.issubset(payload):
            raise EarlyStoppingError("early-stopping state is incomplete")
        if "primary_metric" not in payload and "best_metric" not in payload:
            raise EarlyStoppingError("early-stopping best metric is missing")
        if "tie_breaker" not in payload and "best_tie_breaker" not in payload:
            raise EarlyStoppingError("early-stopping tie-breaker is missing")
        primary_mode = str(payload["primary_mode"])
        tie_mode = (
            None
            if payload["tie_breaker_mode"] is None
            else str(payload["tie_breaker_mode"])
        )
        primary_payload = payload.get("primary_metric", payload.get("best_metric"))
        if (
            "primary_metric" in payload
            and "best_metric" in payload
            and payload["primary_metric"] != payload["best_metric"]
        ):
            raise EarlyStoppingError("early-stopping best metric aliases disagree")
        tie_payload = payload.get("tie_breaker", payload.get("best_tie_breaker"))
        if (
            "tie_breaker" in payload
            and "best_tie_breaker" in payload
            and payload["tie_breaker"] != payload["best_tie_breaker"]
        ):
            raise EarlyStoppingError("early-stopping tie-breaker aliases disagree")
        primary_value = (
            (-math.inf if primary_mode == "max" else math.inf)
            if primary_payload is None
            else float(primary_payload)
        )
        tie_value = (
            (math.inf if tie_mode == "min" else -math.inf)
            if tie_payload is None
            else float(tie_payload)
        )
        state = cls(
            primary_metric=primary_value,
            tie_breaker=tie_value,
            best_epoch=int(payload["best_epoch"]),
            bad_evaluation_count=int(payload["bad_evaluation_count"]),
            last_completed_epoch=int(payload["last_completed_epoch"]),
            min_delta=float(payload["min_delta"]),
            patience=int(payload["patience"]),
            minimum_epochs=int(payload["minimum_epochs"]),
            maximum_epochs=(
                None
                if payload["maximum_epochs"] is None
                else int(payload["maximum_epochs"])
            ),
            primary_mode=primary_mode,
            tie_breaker_mode=tie_mode,
            stopped=bool(payload["stopped"]),
            terminal_reason=(
                None
                if payload["terminal_reason"] is None
                else str(payload["terminal_reason"])
            ),
        )
        # Validate finite values except the intentional +/- infinity sentinels.
        if math.isnan(state.primary_metric) or math.isnan(state.tie_breaker):
            raise EarlyStoppingError("early-stopping metric is NaN")
        if state.primary_mode not in {"max", "min"}:
            raise EarlyStoppingError("invalid persisted primary mode")
        if state.tie_breaker_mode not in {None, "max", "min"}:
            raise EarlyStoppingError("invalid persisted tie-breaker mode")
        if state.best_epoch < -1 or state.bad_evaluation_count < 0:
            raise EarlyStoppingError("invalid persisted early-stopping cursor")
        if (
            state.last_completed_epoch < -1
            or state.best_epoch > state.last_completed_epoch
            or (state.best_epoch >= 0 and not math.isfinite(state.primary_metric))
        ):
            raise EarlyStoppingError("invalid persisted early-stopping metric cursor")
        if contract is not None:
            expected = cls.from_contract(contract)
            for field in (
                "min_delta",
                "patience",
                "minimum_epochs",
                "maximum_epochs",
                "primary_mode",
                "tie_breaker_mode",
            ):
                if getattr(state, field) != getattr(expected, field):
                    raise EarlyStoppingError(
                        f"early-stopping contract changed: {field}"
                    )
        return state

    def observe(
        self,
        *,
        primary_metric: float,
        epoch: int,
        tie_breaker: float | None = None,
    ) -> dict[str, Any]:
        """Consume one completed validation evaluation and return its decision."""

        value = float(primary_metric)
        tie = None if tie_breaker is None else float(tie_breaker)
        valid = math.isfinite(value)
        tie_valid = tie is not None and math.isfinite(tie)
        if self.stopped:
            return {
                "improved": False,
                "primary_improved": False,
                "tie_breaker_improved": False,
                "valid": valid,
                "should_stop": True,
                "bad_evaluation_count": self.bad_evaluation_count,
                "best_epoch": self.best_epoch,
            }
        primary_improves = False
        if valid:
            if self.best_epoch < 0:
                primary_improves = True
            elif self.primary_mode == "max":
                primary_improves = value > self.primary_metric + self.min_delta
            else:
                primary_improves = value < self.primary_metric - self.min_delta
        tie_improves = False
        if (
            valid
            and not primary_improves
            and tie_valid
            and self.tie_breaker_mode is not None
            and self.best_epoch >= 0
        ):
            # A tie-breaker is allowed to choose a deterministic checkpoint only
            # inside the primary metric's min_delta band.
            within_band = (
                abs(value - self.primary_metric) <= self.min_delta
                if self.primary_mode == "max"
                else abs(value - self.primary_metric) <= self.min_delta
            )
            if within_band:
                tie_improves = (
                    tie < self.tie_breaker
                    if self.tie_breaker_mode == "min"
                    else tie > self.tie_breaker
                )
        improved = primary_improves or tie_improves
        if improved:
            self.primary_metric = value
            if tie_valid:
                self.tie_breaker = tie
            self.best_epoch = int(epoch)
            self.bad_evaluation_count = 0
        else:
            self.bad_evaluation_count += 1
        self.last_completed_epoch = int(epoch)
        minimum_reached = int(epoch) + 1 >= self.minimum_epochs
        should_stop = minimum_reached and self.bad_evaluation_count >= self.patience
        if self.maximum_epochs is not None and int(epoch) + 1 >= self.maximum_epochs:
            should_stop = True
        if should_stop:
            self.stopped = True
            self.terminal_reason = (
                "max_epochs_reached"
                if self.maximum_epochs is not None
                and int(epoch) + 1 >= self.maximum_epochs
                else "early_stopped"
            )
        return {
            "improved": improved,
            "primary_improved": primary_improves,
            "tie_breaker_improved": tie_improves,
            "valid": valid,
            "should_stop": should_stop,
            "bad_evaluation_count": self.bad_evaluation_count,
            "best_epoch": self.best_epoch,
        }


__all__ = ["EarlyStoppingError", "EarlyStoppingState"]
