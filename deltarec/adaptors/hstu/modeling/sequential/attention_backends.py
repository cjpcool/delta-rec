# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""Reference attention backends for matched sequential-recommender studies."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


LinearAttentionImplementation = Literal["recurrent", "cumsum", "chunked"]


def apply_rotary_embedding(x: torch.Tensor) -> torch.Tensor:
    """Apply standard RoPE to ``[batch, heads, length, dim]`` tensors."""
    dim = x.shape[-1]
    if dim % 2 != 0:
        raise ValueError("Rotary head dimension must be even")
    positions = torch.arange(x.shape[2], device=x.device, dtype=torch.float32)
    inverse_frequency = 1.0 / (
        10000
        ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    )
    angles = torch.outer(positions, inverse_frequency)
    cos = angles.cos()[None, None]
    sin = angles.sin()[None, None]
    even = x.float()[..., 0::2]
    odd = x.float()[..., 1::2]
    rotated = torch.stack(
        (even * cos - odd * sin, even * sin + odd * cos), dim=-1
    ).flatten(-2)
    return rotated.to(x.dtype)


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [batch, heads, length, dim]")
    if q.shape[:3] != k.shape[:3] or q.shape[:3] != v.shape[:3]:
        raise ValueError("q, k, and v must share batch, head, and length dimensions")
    if valid_mask.shape != (q.shape[0], q.shape[2]):
        raise ValueError("valid_mask must have shape [batch, length]")
    if valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean")


def _feature_map(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x.float()) + 1.0


def normalized_linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    implementation: LinearAttentionImplementation = "cumsum",
    chunk_size: int = 64,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Causal normalized Linear Attention with FP32 recurrent state."""
    _validate_inputs(q, k, v, valid_mask)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")

    qf = _feature_map(q)
    kf = _feature_map(k)
    vf = v.float()
    mask = valid_mask[:, None, :, None].float()
    kf = kf * mask
    vf = vf * mask

    if implementation == "recurrent":
        output = _linear_attention_recurrent(qf, kf, vf, eps)
    elif implementation == "cumsum":
        output = _linear_attention_cumsum(qf, kf, vf, eps)
    elif implementation == "chunked":
        output = _linear_attention_chunked(qf, kf, vf, chunk_size, eps)
    else:
        raise ValueError(f"Unknown Linear Attention implementation {implementation}")

    output = output * mask
    return output.to(v.dtype)


def _linear_attention_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    batch, heads, length, key_dim = q.shape
    value_dim = v.shape[-1]
    state = q.new_zeros(batch, heads, key_dim, value_dim)
    normalizer = q.new_zeros(batch, heads, key_dim)
    outputs = []
    for index in range(length):
        kt = k[:, :, index]
        vt = v[:, :, index]
        state = state + torch.einsum("bhd,bhe->bhde", kt, vt)
        normalizer = normalizer + kt
        qt = q[:, :, index]
        numerator = torch.einsum("bhd,bhde->bhe", qt, state)
        denominator = torch.einsum("bhd,bhd->bh", qt, normalizer).clamp_min(eps)
        outputs.append(numerator / denominator[..., None])
    return torch.stack(outputs, dim=2)


def _linear_attention_cumsum(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    state = torch.einsum("bhnd,bhne->bhnde", k, v).cumsum(dim=2)
    normalizer = k.cumsum(dim=2)
    numerator = torch.einsum("bhnd,bhnde->bhne", q, state)
    denominator = torch.einsum("bhnd,bhnd->bhn", q, normalizer).clamp_min(eps)
    return numerator / denominator[..., None]


def _linear_attention_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int,
    eps: float,
) -> torch.Tensor:
    batch, heads, length, key_dim = q.shape
    value_dim = v.shape[-1]
    state = q.new_zeros(batch, heads, key_dim, value_dim)
    normalizer = q.new_zeros(batch, heads, key_dim)
    outputs = []
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        q_chunk = q[:, :, start:end]
        k_chunk = k[:, :, start:end]
        v_chunk = v[:, :, start:end]
        updates = torch.einsum("bhnd,bhne->bhnde", k_chunk, v_chunk).cumsum(dim=2)
        chunk_state = updates + state[:, :, None]
        chunk_normalizer = k_chunk.cumsum(dim=2) + normalizer[:, :, None]
        numerator = torch.einsum("bhnd,bhnde->bhne", q_chunk, chunk_state)
        denominator = torch.einsum(
            "bhnd,bhnd->bhn", q_chunk, chunk_normalizer
        ).clamp_min(eps)
        outputs.append(numerator / denominator[..., None])
        state = chunk_state[:, :, -1]
        normalizer = chunk_normalizer[:, :, -1]
    return torch.cat(outputs, dim=2)


class NormalizedLinearAttention(torch.nn.Module):
    """Multi-head projection wrapper matching ``nn.MultiheadAttention`` roles."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        implementation: LinearAttentionImplementation = "chunked",
        chunk_size: int = 64,
        eps: float = 1e-6,
        use_rotary: bool = False,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.implementation = implementation
        self.chunk_size = chunk_size
        self.eps = eps
        self.use_rotary = use_rotary
        self.q_proj = torch.nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = torch.nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = torch.nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = torch.nn.Linear(embedding_dim, embedding_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))
        if self.use_rotary:
            q = apply_rotary_embedding(q)
            k = apply_rotary_embedding(k)
        output = normalized_linear_attention(
            q,
            k,
            v,
            valid_mask,
            implementation=self.implementation,
            chunk_size=self.chunk_size,
            eps=self.eps,
        )
        output = output.transpose(1, 2).reshape(query.shape[0], query.shape[1], -1)
        return self.out_proj(self.dropout(output))
