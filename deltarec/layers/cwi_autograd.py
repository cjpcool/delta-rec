from __future__ import annotations

from collections.abc import Callable

import torch

def candidate_aware_cwi1(
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    history_lengths: torch.Tensor,
    candidate_count: int,
    history_width: int,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-candidate losses and positioned CWI1 labels.

    ``loss_fn`` receives a full-write gate ``[B,K,L]`` and must return one
    scalar loss per candidate position ``[B,K]``.  Summing those independent
    outputs before differentiation preserves duplicate Kuai item IDs because
    the exposure-position axis, not item identity, names the stream.
    """

    if (
        history_lengths.ndim != 1
        or candidate_count < 1
        or history_width < 1
        or bool((history_lengths < 1).any())
        or bool((history_lengths > history_width).any())
        or not dtype.is_floating_point
    ):
        raise ValueError("invalid candidate-aware CWI gate dimensions")
    batch = int(history_lengths.numel())
    positions = torch.arange(history_width, device=history_lengths.device)
    valid = positions[None, None, :] < history_lengths[:, None, None]
    valid = valid.expand(batch, candidate_count, history_width)
    gate = torch.where(
        valid,
        torch.ones((), device=history_lengths.device, dtype=dtype),
        torch.zeros((), device=history_lengths.device, dtype=dtype),
    ).requires_grad_(True)
    losses = loss_fn(gate)
    if losses.shape != (batch, candidate_count) or not bool(
        torch.isfinite(losses).all()
    ):
        raise ValueError("candidate-aware teacher loss must be finite [B,K]")
    gradient = torch.autograd.grad(losses.sum(), gate, create_graph=False)[0]
    cwi1 = torch.where(valid, -gradient, torch.zeros_like(gradient))
    if not bool(torch.isfinite(cwi1).all()):
        raise RuntimeError("candidate-aware CWI1 contains non-finite labels")
    return losses.detach(), cwi1.detach()
