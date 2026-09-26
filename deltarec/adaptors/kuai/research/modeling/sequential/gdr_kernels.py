# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""Reference and optimized kernels for packed Gated Delta Rule attention."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
import inspect
import linecache
import re
import sys
import textwrap
import threading
from typing import Literal, Optional

import torch
import torch.nn.functional as F

from deltarec.adaptors.kuai.research.modeling.sequential.selective_gdr import (
    GDRKernel,
    GDRKernelInput,
    GDRKernelOutput,
    apply_gdr_event_gate,
)

GDRKernelBackend = Literal["reference", "fla"]
_FLA_IMPORT_LOCK = threading.Lock()


@dataclass(frozen=True)
class GDRTransitionFactorTrace:
    """Differentiable transition cut used only by offline CWI teachers.

    Each tensor has shape ``[tokens, heads]``. Canonical GDR event gating
    transforms only decay and beta and leaves values unchanged.
    """

    base_decay: torch.Tensor
    base_beta: torch.Tensor
    effective_decay: torch.Tensor
    effective_beta: torch.Tensor


_GDR_TRANSITION_CAPTURE: ContextVar[Optional[list[GDRTransitionFactorTrace]]] = (
    ContextVar("gdr_transition_capture", default=None)
)


@contextmanager
def capture_gdr_transition_factors():
    """Capture per-layer GDR transition factors without changing model APIs."""

    traces: list[GDRTransitionFactorTrace] = []
    token = _GDR_TRANSITION_CAPTURE.set(traces)
    try:
        yield traces
    finally:
        _GDR_TRANSITION_CAPTURE.reset(token)


@lru_cache(maxsize=1)
def _import_fla_chunk_gated_delta_rule():
    """Import FLA with its Python 3.10 stacked-decorator workaround."""

    if sys.version_info >= (3, 11):
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        return chunk_gated_delta_rule

    original_getsourcelines = inspect.getsourcelines

    def compatible_getsourcelines(obj):
        lines, start = original_getsourcelines(obj)
        source = textwrap.dedent("".join(lines))
        if inspect.isfunction(obj) and re.search(
            rf"^def\s+{re.escape(obj.__name__)}\s*\(",
            source,
            re.MULTILINE,
        ) is None:
            filename = inspect.getsourcefile(obj)
            if filename is not None:
                file_lines = linecache.getlines(filename)
                first_line = max(obj.__code__.co_firstlineno - 1, 0)
                for index in range(first_line, len(file_lines)):
                    if re.match(
                        rf"^\s*def\s+{re.escape(obj.__name__)}\s*\(",
                        file_lines[index],
                    ):
                        return inspect.getblock(file_lines[index:]), index + 1
        return lines, start

    with _FLA_IMPORT_LOCK:
        inspect.getsourcelines = compatible_getsourcelines
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        finally:
            inspect.getsourcelines = original_getsourcelines
    return chunk_gated_delta_rule


def build_gdr_kernel(
    backend: GDRKernelBackend,
    *,
    chunk_size: int = 64,
    eps: float = 1e-6,
) -> GDRKernel:
    """Build a GDR computation backend without changing its tensor contract."""

    if backend == "reference":
        return ReferenceGDRKernel(eps=eps)
    if backend == "fla":
        return FLAGDRKernel(chunk_size=chunk_size, eps=eps)
    raise ValueError(f"Unsupported GDR kernel backend {backend}")


def _l2_normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x / torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + eps)


def validate_gdr_kernel_input(
    projected: GDRKernelInput,
    initial_state: Optional[torch.Tensor] = None,
    *,
    validate_values: bool = True,
) -> tuple[int, int, int, int]:
    """Validate the architecture-facing packed GDR contract."""

    q, k, v = projected.q, projected.k, projected.v
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must have shape [tokens, heads, dim]")
    if q.shape != k.shape:
        raise ValueError("q and k must have identical shapes")
    if q.shape[:2] != v.shape[:2]:
        raise ValueError("q, k, and v must share token and head dimensions")
    tokens, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    for name, gate in (
        ("decay_logits", projected.decay_logits),
        ("beta_logits", projected.beta_logits),
    ):
        if gate.shape != (tokens, heads):
            raise ValueError(f"{name} must have shape [tokens, heads]")
    for name, parameter in (
        ("log_decay_scale", projected.log_decay_scale),
        ("decay_bias", projected.decay_bias),
    ):
        if parameter.shape != (heads,):
            raise ValueError(f"{name} must have shape [heads]")
    offsets = projected.offsets
    if offsets.ndim != 1 or offsets.numel() < 1:
        raise ValueError("offsets must have shape [batch + 1]")
    if offsets.dtype not in (torch.int32, torch.int64):
        raise ValueError("offsets must be an integer tensor")
    if validate_values:
        if int(offsets[0]) != 0 or int(offsets[-1]) != tokens:
            raise ValueError("offsets must span every packed token")
        if bool((offsets[1:] < offsets[:-1]).any()):
            raise ValueError("offsets must be nondecreasing")
    if projected.offsets_cpu is not None:
        offsets_cpu = projected.offsets_cpu
        if (
            offsets_cpu.device.type != "cpu"
            or offsets_cpu.dtype != torch.int64
            or offsets_cpu.shape != offsets.shape
        ):
            raise ValueError("offsets_cpu must be an int64 CPU copy of offsets")
        if validate_values:
            if int(offsets_cpu[0]) != 0 or int(offsets_cpu[-1]) != tokens:
                raise ValueError("offsets_cpu must span every packed token")
            if bool((offsets_cpu[1:] < offsets_cpu[:-1]).any()):
                raise ValueError("offsets_cpu must be nondecreasing")
    if projected.event_gate is not None:
        if projected.event_gate.shape != (tokens,):
            raise ValueError("event_gate must have shape [tokens]")
        if not torch.is_floating_point(projected.event_gate):
            raise ValueError("event_gate must be floating point")
        if validate_values:
            if not bool(torch.isfinite(projected.event_gate).all()):
                raise ValueError("event_gate must contain finite values")
            if bool((projected.event_gate < 0).any()) or bool(
                (projected.event_gate > 1).any()
            ):
                raise ValueError("event_gate values must lie in [0, 1]")
    batch = offsets.numel() - 1
    if initial_state is not None and initial_state.shape != (
        batch,
        heads,
        key_dim,
        value_dim,
    ):
        raise ValueError(
            "initial_state must have shape [batch, heads, key_dim, value_dim]"
        )
    return batch, heads, key_dim, value_dim


def _materialize_base_gdr_gates(
    projected: GDRKernelInput,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform raw GDR parameters before applying an optional event gate."""

    log_decay = -torch.exp(projected.log_decay_scale.to(dtype))[None, :] * F.softplus(
        projected.decay_logits.to(dtype) + projected.decay_bias.to(dtype)[None, :]
    )
    decay = torch.exp(log_decay)
    beta = torch.sigmoid(projected.beta_logits.to(dtype))
    return decay, beta


def materialize_gdr_gates(
    projected: GDRKernelInput,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform raw GDR parameters into multiplicative decay and write gates."""

    decay, beta = _materialize_base_gdr_gates(projected, dtype=dtype)
    if projected.event_gate is not None:
        decay, beta = apply_gdr_event_gate(decay, beta, projected.event_gate)
    return decay, beta


def materialize_gdr_transition(
    projected: GDRKernelInput,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return canonical value, decay, and beta transition parameters.

    Event gating changes only decay and beta. The projected value is returned
    unchanged (apart from the requested compute dtype).
    """

    decay, beta = _materialize_base_gdr_gates(projected, dtype=dtype)
    value = projected.v.to(dtype)
    if projected.event_gate is None:
        return value, decay, beta

    effective_decay, effective_beta = apply_gdr_event_gate(
        decay,
        beta,
        projected.event_gate,
    )
    traces = _GDR_TRANSITION_CAPTURE.get()
    if traces is not None:
        traces.append(
            GDRTransitionFactorTrace(
                base_decay=decay,
                base_beta=beta,
                effective_decay=effective_decay,
                effective_beta=effective_beta,
            )
        )
    return value, effective_decay, effective_beta


class ReferenceGDRKernel(GDRKernel):
    """Transparent packed recurrence with an FP32 state."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = eps

    def forward(
        self,
        projected: GDRKernelInput,
        initial_state: Optional[torch.Tensor] = None,
        return_final_state: bool = False,
    ) -> GDRKernelOutput:
        batch, heads, key_dim, value_dim = validate_gdr_kernel_input(
            projected, initial_state
        )
        compute_dtype = (
            torch.float64 if projected.q.dtype == torch.float64 else torch.float32
        )
        q = _l2_normalize(projected.q.to(compute_dtype), self.eps)
        k = _l2_normalize(projected.k.to(compute_dtype), self.eps)
        v, decay, beta = materialize_gdr_transition(projected, compute_dtype)
        scale = key_dim**-0.5

        contexts: list[torch.Tensor] = []
        final_states: list[torch.Tensor] = []
        for sequence_index in range(batch):
            start = int(projected.offsets[sequence_index])
            end = int(projected.offsets[sequence_index + 1])
            state = (
                q.new_zeros(heads, key_dim, value_dim)
                if initial_state is None
                else initial_state[sequence_index].to(compute_dtype)
            )
            for token_index in range(start, end):
                full_state = decay[token_index, :, None, None] * state
                prediction = torch.einsum(
                    "hkd,hk->hd", full_state, k[token_index]
                )
                residual = v[token_index] - prediction
                full_state = full_state + beta[
                    token_index, :, None, None
                ] * torch.einsum("hk,hd->hkd", k[token_index], residual)
                state = full_state
                contexts.append(
                    scale
                    * torch.einsum("hk,hkd->hd", q[token_index], state)
                )
            final_states.append(state)

        context = (
            torch.stack(contexts).to(projected.v.dtype)
            if contexts
            else projected.v.new_empty((0, heads, value_dim))
        )
        final_state = torch.stack(final_states) if return_final_state else None
        return GDRKernelOutput(context=context, final_state=final_state)


class FLAGDRKernel(GDRKernel):
    """Adapter for FLA's packed chunkwise Gated Delta Rule kernel."""

    def __init__(
        self,
        chunk_size: int = 64,
        eps: float = 1e-6,
        validate_inputs: bool = True,
        assume_binary_event_gate: bool = False,
    ) -> None:
        super().__init__()
        if chunk_size != 64:
            raise ValueError("fla-core==0.5.1 GDR requires chunk_size=64")
        if eps != 1e-6:
            raise ValueError("fla-core==0.5.1 GDR requires eps=1e-6")
        self.chunk_size = chunk_size
        self.eps = eps
        self.validate_inputs = validate_inputs
        if not isinstance(assume_binary_event_gate, bool):
            raise TypeError("assume_binary_event_gate must be boolean")
        self.assume_binary_event_gate = assume_binary_event_gate

    def forward(
        self,
        projected: GDRKernelInput,
        initial_state: Optional[torch.Tensor] = None,
        return_final_state: bool = False,
    ) -> GDRKernelOutput:
        _, _, key_dim, _ = validate_gdr_kernel_input(
            projected,
            initial_state,
            validate_values=self.validate_inputs,
        )
        if (
            self.assume_binary_event_gate
            and self.validate_inputs
            and projected.event_gate is not None
            and not projected.event_gate.requires_grad
            and _GDR_TRANSITION_CAPTURE.get() is None
            and bool(
                (
                    (projected.event_gate != 0)
                    & (projected.event_gate != 1)
                ).any()
            )
        ):
            raise ValueError(
                "assume_binary_event_gate requires event_gate values in {0, 1}"
            )
        if not projected.q.is_cuda:
            raise RuntimeError(
                "FLAGDRKernel requires CUDA; use ReferenceGDRKernel on CPU"
            )
        try:
            chunk_gated_delta_rule = _import_fla_chunk_gated_delta_rule()
        except ImportError as error:
            raise ImportError(
                "FLAGDRKernel requires optional dependency fla-core==0.5.1"
            ) from error

        q = projected.q.unsqueeze(0).contiguous()
        k = projected.k.unsqueeze(0).contiguous()
        offsets = projected.offsets.to(device=q.device, dtype=torch.int64).contiguous()
        common = {
            "q": q,
            "k": k,
            "initial_state": initial_state,
            "output_final_state": return_final_state,
            "use_qk_l2norm_in_kernel": True,
            "allow_neg_eigval": False,
            "state_v_first": False,
            "cu_seqlens": offsets,
            "cu_seqlens_cpu": projected.offsets_cpu,
            "scale": key_dim**-0.5,
            "chunk_size": self.chunk_size,
        }
        if projected.event_gate is None:
            v = projected.v.unsqueeze(0).contiguous()
            context, final_state = chunk_gated_delta_rule(
                v=v,
                g=projected.decay_logits.unsqueeze(0).contiguous(),
                beta=projected.beta_logits.unsqueeze(0).contiguous(),
                use_gate_in_kernel=True,
                A_log=projected.log_decay_scale,
                dt_bias=projected.decay_bias,
                use_beta_sigmoid_in_kernel=True,
                **common,
            )
        else:
            use_binary_event_gate = bool(
                self.assume_binary_event_gate
                and not projected.event_gate.requires_grad
                and _GDR_TRANSITION_CAPTURE.get() is None
            )
            if use_binary_event_gate:
                # For z in {0,1}, the canonical event gate leaves V unchanged,
                # sets (decay, beta) to (1, 0) for z=0, and keeps the original
                # transition for z=1. The boolean where also avoids
                # cancellation in 1 + z * (decay - 1) for very small decay.
                decay, beta = _materialize_base_gdr_gates(projected)
                writes = projected.event_gate.to(torch.bool).unsqueeze(-1)
                decay = torch.where(writes, decay, torch.ones_like(decay))
                beta = torch.where(writes, beta, torch.zeros_like(beta))
                v = projected.v.unsqueeze(0).contiguous()
            else:
                value, decay, beta = materialize_gdr_transition(projected)
                v = value.to(projected.v.dtype).unsqueeze(0).contiguous()
            context, final_state = chunk_gated_delta_rule(
                v=v,
                g=torch.log(decay).unsqueeze(0).to(q.dtype).contiguous(),
                beta=beta.unsqueeze(0).to(q.dtype).contiguous(),
                use_gate_in_kernel=False,
                use_beta_sigmoid_in_kernel=False,
                **common,
            )
        return GDRKernelOutput(
            context=context.squeeze(0),
            final_state=final_state,
        )

