"""Low-synchronization accumulation for the FuXi DeltaRec ports.

The published FuXi trainer predates the HSTU numerical patch and enables
autograd anomaly detection for every microbatch.  This controller keeps the
same weighted-mean update while checking the complete gradient set once per
optimizer update.  Detailed parameter checks cover the first ten updates and
then run periodically, matching the HSTU startup-health policy.
"""

from __future__ import annotations

from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping


class FuxiAccumulationError(RuntimeError):
    pass


def _finite_gradient_names(model: Any) -> list[str]:
    """Use the same per-device aggregate finite check as the HSTU path."""

    from .numerics import nonfinite_gradient_names

    return nonfinite_gradient_names(model.named_parameters())


class FuxiGradientAccumulation:
    """Accumulate exact numerator gradients with one normal check per update."""

    def __init__(
        self,
        steps: int,
        *,
        debug_anomaly: bool = False,
        health_window: int = 10,
        health_interval: int = 100,
        health_path: str | Path | None = None,
    ) -> None:
        if isinstance(steps, bool) or steps < 1:
            raise ValueError("FuXi accumulation steps must be positive")
        if health_window < 1 or health_interval < 1:
            raise ValueError("FuXi health windows must be positive")
        self.steps = int(steps)
        self.debug_anomaly = bool(debug_anomaly)
        self.health_window = int(health_window)
        self.health_interval = int(health_interval)
        self.health_path = None if health_path is None else Path(health_path)
        self.optimizer_steps = 0
        self._denominator = 0.0
        self._loss_numerator: Any | None = None
        self._losses: list[Any] = []
        self._microbatches = 0
        self._window_steps = self.steps
        self._window_open = False

    @property
    def in_startup_health_window(self) -> bool:
        return self.optimizer_steps < self.health_window

    def start_window(
        self, model: Any, optimizer: Any, *, steps: int | None = None
    ) -> None:
        if self._window_open:
            raise FuxiAccumulationError("cannot start a nonempty accumulation window")
        window_steps = self.steps if steps is None else int(steps)
        if window_steps < 1 or window_steps > self.steps:
            raise FuxiAccumulationError(
                f"window steps must be in [1, {self.steps}], got {window_steps}"
            )
        optimizer.zero_grad(set_to_none=True)
        self._denominator = 0.0
        self._loss_numerator = None
        self._losses = []
        self._microbatches = 0
        self._window_steps = window_steps
        self._window_open = True

    def start_microbatch(self, model: Any, index: int) -> None:
        if not self._window_open:
            raise FuxiAccumulationError("start_window must precede every microbatch")
        if index != self._microbatches or index >= self._window_steps:
            raise FuxiAccumulationError("FuXi microbatch indices are not contiguous")
        if hasattr(model, "require_backward_grad_sync"):
            model.require_backward_grad_sync = index == self._window_steps - 1

    def backward(self, loss: Any, denominator: int | float) -> None:
        """Backpropagate an unnormalized numerator; normalize at the step."""

        if not self._window_open:
            raise FuxiAccumulationError("backward called outside an accumulation window")
        value = float(denominator)
        if not math.isfinite(value) or value <= 0:
            raise FuxiAccumulationError("FuXi loss denominator must be finite and positive")
        context = (
            torch_detect_anomaly()
            if self.debug_anomaly
            else nullcontext()
        )
        with context:
            (loss * value).backward()
        detached_loss = loss.detach()
        weighted_loss = detached_loss.float() * value
        self._loss_numerator = (
            weighted_loss if self._loss_numerator is None
            else self._loss_numerator + weighted_loss
        )
        self._losses.append(detached_loss)
        self._denominator += value
        self._microbatches += 1

    def finish_window(
        self,
        model: Any,
        optimizer: Any,
        *,
        gradient_clip: float | None = None,
    ) -> Mapping[str, Any]:
        if not self._window_open or self._microbatches != self._window_steps:
            raise FuxiAccumulationError(
                f"expected {self._window_steps} microbatches, got {self._microbatches}"
            )
        import torch

        # Both checks are intentionally batched.  They synchronize once before
        # the optimizer mutation, so a bad update cannot alter model/optimizer
        # state and normal execution does not synchronize per parameter.
        loss_finite = bool(torch.stack([value.float() for value in self._losses]).isfinite().all())
        bad_gradients = _finite_gradient_names(model)
        if not loss_finite or bad_gradients:
            raise FloatingPointError(
                f"FuXi nonfinite update: loss_finite={loss_finite}, gradients={bad_gradients}"
            )
        detailed = self.optimizer_steps < self.health_window or (
            self.optimizer_steps + 1
        ) % self.health_interval == 0
        gradient_norms = []
        gradient_abs_maxima = []
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            values = parameter.grad.coalesce().values() if parameter.grad.is_sparse else parameter.grad
            if detailed:
                gradient_norms.append(values.detach().float().norm().square())
                gradient_abs_maxima.append(values.detach().abs().max().float())
        if detailed:
            gradient_l2 = float(torch.stack(gradient_norms).sum().sqrt()) if gradient_norms else 0.0
            gradient_abs_max = float(torch.stack(gradient_abs_maxima).max()) if gradient_abs_maxima else 0.0
            if not gradient_l2 > 0.0:
                raise FloatingPointError("FuXi startup/update health found no nonzero gradient")
        else:
            gradient_l2 = None
            gradient_abs_max = None
        for parameter in model.parameters():
            if parameter.grad is not None:
                if parameter.grad.is_sparse:
                    parameter.grad = parameter.grad.coalesce()
                    parameter.grad._values().div_(self._denominator)
                else:
                    parameter.grad.div_(self._denominator)
        if gradient_clip is not None:
            if not math.isfinite(float(gradient_clip)) or float(gradient_clip) <= 0:
                raise ValueError("FuXi gradient_clip must be finite and positive")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(gradient_clip), error_if_nonfinite=True
            )
        optimizer.step()
        self.optimizer_steps += 1
        parameter_finite = True
        if detailed:
            parameter_flags = []
            for parameter in model.parameters():
                parameter_flags.append(torch.isfinite(parameter.detach()).all())
            parameter_finite = bool(torch.stack(parameter_flags).all()) if parameter_flags else True
            if not parameter_finite:
                raise FloatingPointError("FuXi optimizer produced a nonfinite parameter")
        if self._loss_numerator is None:
            raise FuxiAccumulationError("FuXi loss numerator was not recorded")
        loss_value = float(self._loss_numerator / self._denominator)
        record = {
            "schema": "deltarec-fuxi-step-health-v1",
            "step": self.optimizer_steps,
            "loss": loss_value,
            "denominator": self._denominator,
            "microbatches": self._microbatches,
            "detailed_parameter_check": detailed,
            "gradient_l2": gradient_l2,
            "gradient_abs_max": gradient_abs_max,
            "parameters_finite": parameter_finite,
            "anomaly_detection": self.debug_anomaly,
            "status": "ok",

        }
        self._write_record(record)
        if self.optimizer_steps == self.health_window:
            marker = (
                (self.health_path.parent if self.health_path is not None else Path.cwd())
                / "STARTUP_HEALTH_PASSED.json"
            )
            marker.write_text(
                json.dumps(
                    {
                        "checked_optimizer_steps": self.health_window,
                        "status": "passed",
                        "scope": "first startup health window; not a convergence guarantee",

                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        self._window_open = False
        self._loss_numerator = None
        self._losses = []
        self._denominator = 0.0
        self._microbatches = 0
        return record

    def _write_record(self, record: Mapping[str, Any]) -> None:
        if self.health_path is None:
            return
        self.health_path.parent.mkdir(parents=True, exist_ok=True)
        with self.health_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(record), allow_nan=False) + "\n")


def torch_detect_anomaly() -> Any:
    import torch

    return torch.autograd.detect_anomaly()


__all__ = ["FuxiAccumulationError", "FuxiGradientAccumulation"]
