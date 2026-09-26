from __future__ import annotations

import math

from typing import Any

class GradientAccumulationController:
    """Controls zero/sync/step boundaries for one epoch-streamed optimizer."""

    def __init__(self, accumulation_steps: int) -> None:
        if accumulation_steps <= 0:
            raise ValueError("accumulation_steps must be positive")
        self.accumulation_steps = accumulation_steps
        self.microbatches_in_window = 0
        self.current_window_target = 0
        self.window_token_count = 0
        self.window_loss_numerator = 0.0
        self.last_window_token_count = 0
        self.last_window_mean_loss = 0.0
        self.optimizer_steps = 0
        self._should_step = False

    def start_microbatch(
        self,
        *,
        model: Any,
        optimizer: Any,
        microbatch_index: int,
        microbatches_in_epoch: int,
    ) -> bool:
        if not 0 <= microbatch_index < microbatches_in_epoch:
            raise ValueError("microbatch index is outside the epoch")
        if self.microbatches_in_window == 0:
            self.current_window_target = min(
                self.accumulation_steps,
                microbatches_in_epoch - microbatch_index,
            )
            self.window_token_count = 0
            self.window_loss_numerator = 0.0
            optimizer.zero_grad()
        self._should_step = (
            self.microbatches_in_window + 1 == self.current_window_target
        )
        # This is the flag toggled by torch.nn.parallel.DistributedDataParallel
        # no_sync().  It must be set before the forward call.
        if hasattr(model, "require_backward_grad_sync"):
            model.require_backward_grad_sync = self._should_step
        return self._should_step

    def numerator_loss(self, loss: Any, token_count: Any) -> Any:
        """Convert an upstream token mean back to its differentiable numerator."""

        if self.current_window_target <= 0:
            raise RuntimeError("start_microbatch must precede numerator_loss")
        count = (
            int(token_count.detach().item())
            if hasattr(token_count, "detach")
            else int(token_count)
        )
        if count <= 0:
            raise ValueError("each microbatch must contain a valid supervision token")
        self.window_token_count += count
        value = float(loss.detach().item())
        if not math.isfinite(value):
            raise FloatingPointError('nonfinite training loss before backward')
        self.window_loss_numerator += value * count
        return loss * count

    @staticmethod
    def _divide_gradients(model: Any, denominator: int) -> None:
        if denominator <= 0:
            raise RuntimeError("cannot normalize gradients by an empty token window")
        for parameter in model.parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            if getattr(gradient, "is_sparse", False):
                gradient._values().div_(denominator)
            else:
                gradient.div_(denominator)

    @property
    def current_window_mean_loss(self) -> float:
        if self.window_token_count <= 0:
            raise RuntimeError("numerator_loss must precede loss reporting")
        return self.window_loss_numerator / self.window_token_count

    def finish_microbatch(self, model: Any, optimizer: Any) -> bool:
        if self.current_window_target <= 0:
            raise RuntimeError("start_microbatch must precede finish_microbatch")
        self.microbatches_in_window += 1
        if not self._should_step:
            return False
        if self.microbatches_in_window != self.current_window_target:
            raise RuntimeError("gradient accumulation window ended inconsistently")
        self._divide_gradients(model, self.window_token_count)
        from .numerics import nonfinite_gradient_names
        bad = nonfinite_gradient_names(model.named_parameters())
        if bad:
            raise FloatingPointError(f'nonfinite gradients before optimizer update: {bad}')
        optimizer.step()
        self.optimizer_steps += 1
        self.last_window_token_count = self.window_token_count
        self.last_window_mean_loss = self.current_window_mean_loss
        self.microbatches_in_window = 0
        self.current_window_target = 0
        self.window_token_count = 0
        self.window_loss_numerator = 0.0
        self._should_step = False
        return True
