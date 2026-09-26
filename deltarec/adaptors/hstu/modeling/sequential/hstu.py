# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

"""
Implements HSTU (Hierarchical Sequential Transduction Unit) in
Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations
(https://arxiv.org/abs/2402.17152, ICML'24).
"""

import abc
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from deltarec.layers import jagged
import torch.nn.functional as F
from deltarec.adaptors.hstu.modeling.sequential.attention_backends import (
    normalized_linear_attention,
)
from deltarec.adaptors.hstu.modeling.sequential.embedding_modules import (
    EmbeddingModule,
)
from deltarec.layers.gdr_kernels import (
    build_gdr_kernel,
)
from deltarec.adaptors.hstu.modeling.sequential.input_features_preprocessors import (
    InputFeaturesPreprocessorModule,
)
from deltarec.adaptors.hstu.modeling.sequential.output_postprocessors import (
    OutputPostprocessorModule,
)
from deltarec.layers.selective_gdr import (
    GDRKernel,
    GDRKernelInput,
)
from deltarec.adaptors.hstu.modeling.sequential.utils import (
    get_current_embeddings,
)
from deltarec.adaptors.hstu.modeling.similarity_module import (
    SequentialEncoderWithLearnedSimilarityModule,
)
from deltarec.adaptors.hstu.rails.similarities.module import SimilarityModule


TIMESTAMPS_KEY = "timestamps"


class RelativeAttentionBiasModule(torch.nn.Module):
    @abc.abstractmethod
    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            all_timestamps: [B, N] x int64
        Returns:
            torch.float tensor broadcastable to [B, N, N]
        """
        pass


class RelativePositionalBias(RelativeAttentionBiasModule):
    def __init__(self, max_seq_len: int) -> None:
        super().__init__()

        self._max_seq_len: int = max_seq_len
        self._w = torch.nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02),
        )

    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        del all_timestamps
        n: int = self._max_seq_len
        t = F.pad(self._w[: 2 * n - 1], [0, n]).repeat(n)
        t = t[..., :-n].reshape(1, n, 3 * n - 2)
        r = (2 * n - 1) // 2
        return t[..., r:-r]


class RelativeBucketedTimeAndPositionBasedBias(RelativeAttentionBiasModule):
    """
    Bucketizes timespans based on ts(next-item) - ts(current-item).
    """

    def __init__(
        self,
        max_seq_len: int,
        num_buckets: int,
        bucketization_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()

        self._max_seq_len: int = max_seq_len
        self._ts_w = torch.nn.Parameter(
            torch.empty(num_buckets + 1).normal_(mean=0, std=0.02),
        )
        self._pos_w = torch.nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02),
        )
        self._num_buckets: int = num_buckets
        self._bucketization_fn: Callable[[torch.Tensor], torch.Tensor] = (
            bucketization_fn
        )

    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            all_timestamps: (B, N).
        Returns:
            (B, N, N).
        """
        B = all_timestamps.size(0)
        N = self._max_seq_len
        t = F.pad(self._pos_w[: 2 * N - 1], [0, N]).repeat(N)
        t = t[..., :-N].reshape(1, N, 3 * N - 2)
        r = (2 * N - 1) // 2

        # [B, N + 1] to simplify tensor manipulations.
        ext_timestamps = torch.cat(
            [all_timestamps, all_timestamps[:, N - 1 : N]], dim=1
        )
        # causal masking. Otherwise [:, :-1] - [:, 1:] works
        bucketed_timestamps = torch.clamp(
            self._bucketization_fn(
                ext_timestamps[:, 1:].unsqueeze(2) - ext_timestamps[:, :-1].unsqueeze(1)
            ),
            min=0,
            max=self._num_buckets,
        ).detach()
        rel_pos_bias = t[:, :, r:-r]
        rel_ts_bias = torch.index_select(
            self._ts_w, dim=0, index=bucketed_timestamps.view(-1)
        ).view(B, N, N)
        return rel_pos_bias + rel_ts_bias


HSTUCacheState = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _hstu_attention_maybe_from_cache(
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cached_q: Optional[torch.Tensor],
    cached_k: Optional[torch.Tensor],
    delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]],
    x_offsets: torch.Tensor,
    all_timestamps: Optional[torch.Tensor],
    invalid_attn_mask: torch.Tensor,
    rel_attn_bias: RelativeAttentionBiasModule,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B: int = x_offsets.size(0) - 1
    n: int = invalid_attn_mask.size(-1)
    if delta_x_offsets is not None:
        padded_q, padded_k = cached_q, cached_k
        flattened_offsets = delta_x_offsets[1] + torch.arange(
            start=0,
            end=B * n,
            step=n,
            device=delta_x_offsets[1].device,
            dtype=delta_x_offsets[1].dtype,
        )
        assert isinstance(padded_q, torch.Tensor)
        assert isinstance(padded_k, torch.Tensor)
        padded_q = (
            padded_q.view(B * n, -1)
            .index_copy_(
                dim=0,
                index=flattened_offsets,
                source=q,
            )
            .view(B, n, -1)
        )
        padded_k = (
            padded_k.view(B * n, -1)
            .index_copy_(
                dim=0,
                index=flattened_offsets,
                source=k,
            )
            .view(B, n, -1)
        )
    else:
        padded_q = jagged.jagged_to_padded_dense(
            values=q, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
        )
        padded_k = jagged.jagged_to_padded_dense(
            values=k, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
        )

    qk_attn = torch.einsum(
        "bnhd,bmhd->bhnm",
        padded_q.view(B, n, num_heads, attention_dim),
        padded_k.view(B, n, num_heads, attention_dim),
    )
    if all_timestamps is not None:
        qk_attn = qk_attn + rel_attn_bias(all_timestamps).unsqueeze(1)
    qk_attn = F.silu(qk_attn) / n
    qk_attn = qk_attn * invalid_attn_mask.unsqueeze(0).unsqueeze(0)
    attn_output = jagged.dense_to_jagged(
        torch.einsum(
            "bhnm,bmhd->bnhd",
            qk_attn,
            jagged.jagged_to_padded_dense(v, [x_offsets], [n]).reshape(
                B, n, num_heads, linear_dim
            ),
        ).reshape(B, n, num_heads * linear_dim),
        [x_offsets],
    )[0]
    return attn_output, padded_q, padded_k


class SequentialTransductionUnitJagged(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        linear_hidden_dim: int,
        attention_dim: int,
        dropout_ratio: float,
        attn_dropout_ratio: float,
        num_heads: int,
        linear_activation: str,
        relative_attention_bias_module: Optional[RelativeAttentionBiasModule] = None,
        normalization: str = "rel_bias",
        linear_config: str = "uvqk",
        concat_ua: bool = False,
        epsilon: float = 1e-6,
        max_length: Optional[int] = None,
        attention_backend: str = "original",
        gdr_kernel_backend: str = "fla",
        gdr_kernel: Optional[GDRKernel] = None,
    ) -> None:
        super().__init__()
        self._embedding_dim: int = embedding_dim
        self._linear_dim: int = linear_hidden_dim
        self._attention_dim: int = attention_dim
        self._dropout_ratio: float = dropout_ratio
        self._attn_dropout_ratio: float = attn_dropout_ratio
        self._num_heads: int = num_heads
        self._normalization: str = normalization
        self._attention_backend: str = attention_backend
        if attention_backend not in (
            "original",
            "softmax_no_bias",
            "linear_elu",
            "gdr",
        ):
            raise ValueError(f"Unsupported HSTU attention backend {attention_backend}")
        if gdr_kernel is not None and attention_backend != "gdr":
            raise ValueError("gdr_kernel may only be set for the gdr attention backend")
        self._rel_attn_bias: Optional[RelativeAttentionBiasModule] = (
            relative_attention_bias_module if attention_backend == "original" else None
        )
        self._linear_config: str = linear_config
        if self._linear_config == "uvqk":
            self._uvqk: torch.nn.Parameter = torch.nn.Parameter(
                torch.empty(
                    (
                        embedding_dim,
                        linear_hidden_dim * 2 * num_heads
                        + attention_dim * num_heads * 2,
                    )
                ).normal_(mean=0, std=0.02),
            )
        else:
            raise ValueError(f"Unknown linear_config {self._linear_config}")
        self._linear_activation: str = linear_activation
        self._concat_ua: bool = concat_ua
        self._gdr_kernel_backend = gdr_kernel_backend
        if attention_backend == "gdr":
            self._gdr_gate = torch.nn.Linear(
                embedding_dim,
                2 * num_heads,
                bias=False,
            )
            torch.nn.init.zeros_(self._gdr_gate.weight)
            decay_scale = torch.empty(num_heads, dtype=torch.float32).uniform_(0, 16)
            self._gdr_log_decay_scale = torch.nn.Parameter(
                torch.log(decay_scale.clamp_min(1e-4))
            )
            self._gdr_log_decay_scale._no_weight_decay = True
            dt = torch.exp(
                torch.rand(num_heads)
                * (math.log(0.1) - math.log(0.001))
                + math.log(0.001)
            ).clamp_min(1e-4)
            self._gdr_decay_bias = torch.nn.Parameter(
                dt + torch.log(-torch.expm1(-dt))
            )
            self._gdr_decay_bias._no_weight_decay = True
            self._gdr_kernel = gdr_kernel or build_gdr_kernel(
                gdr_kernel_backend,
                eps=epsilon,
            )
        else:
            self._gdr_gate = None
            self._gdr_log_decay_scale = None
            self._gdr_decay_bias = None
            self._gdr_kernel = None
        self._o = torch.nn.Linear(
            in_features=linear_hidden_dim * num_heads * (3 if concat_ua else 1),
            out_features=embedding_dim,
        )
        torch.nn.init.xavier_uniform_(self._o.weight)
        self._eps: float = epsilon

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, normalized_shape=[self._embedding_dim], eps=self._eps)

    def _norm_attn_output(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, normalized_shape=[self._linear_dim * self._num_heads], eps=self._eps
        )

    def forward(  # pyre-ignore [3]
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[HSTUCacheState] = None,
        return_cache_states: bool = False,
        gdr_event_gate: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: (\sum_i N_i, D) x float.
            x_offsets: (B + 1) x int32.
            all_timestamps: optional (B, N) x int64.
            invalid_attn_mask: (B, N, N) x float, each element in {0, 1}.
            delta_x_offsets: optional 2-tuple ((B,) x int32, (B,) x int32).
                For the 1st element in the tuple, each element is in [0, x_offsets[-1]). For the
                2nd element in the tuple, each element is in [0, N).
            cache: Optional 4-tuple of (v, padded_q, padded_k, output) from prior runs,
                where all except padded_q, padded_k are jagged.
        Returns:
            x' = f(x), (\sum_i N_i, D) x float.
        """
        n: int = invalid_attn_mask.size(-1)
        cached_q = None
        cached_k = None
        padded_q = None
        padded_k = None
        if delta_x_offsets is not None:
            # In this case, for all the following code, x, u, v, q, k become restricted to
            # [delta_x_offsets[0], :].
            assert cache is not None
            x = x[delta_x_offsets[0], :]
            cached_v, cached_q, cached_k, cached_outputs = cache

        normed_x = self._norm_input(x)

        if self._linear_config == "uvqk":
            batched_mm_output = torch.mm(normed_x, self._uvqk)
            if self._linear_activation == "silu":
                batched_mm_output = F.silu(batched_mm_output)
            elif self._linear_activation == "none":
                batched_mm_output = batched_mm_output
            u, v, q, k = torch.split(
                batched_mm_output,
                [
                    self._linear_dim * self._num_heads,
                    self._linear_dim * self._num_heads,
                    self._attention_dim * self._num_heads,
                    self._attention_dim * self._num_heads,
                ],
                dim=1,
            )
        else:
            raise ValueError(f"Unknown self._linear_config {self._linear_config}")

        decay_logits = None
        beta_logits = None
        if self._attention_backend == "gdr":
            if return_cache_states:
                raise NotImplementedError(
                    "GDR attention does not support HSTU cache materialization"
                )
            assert self._gdr_gate is not None
            decay_logits, beta_logits = self._gdr_gate(normed_x).chunk(2, dim=-1)

        if delta_x_offsets is not None:
            v = cached_v.index_copy_(dim=0, index=delta_x_offsets[0], source=v)

        B: int = x_offsets.size(0) - 1
        if self._attention_backend == "gdr":
            if delta_x_offsets is not None:
                raise NotImplementedError(
                    "GDR attention does not support incremental HSTU cache updates"
                )
            assert decay_logits is not None
            assert beta_logits is not None
            assert self._gdr_log_decay_scale is not None
            assert self._gdr_decay_bias is not None
            assert self._gdr_kernel is not None
            kernel_output = self._gdr_kernel(
                GDRKernelInput(
                    q=q.view(-1, self._num_heads, self._attention_dim),
                    k=k.view(-1, self._num_heads, self._attention_dim),
                    v=v.view(-1, self._num_heads, self._linear_dim),
                    decay_logits=decay_logits,
                    beta_logits=beta_logits,
                    log_decay_scale=self._gdr_log_decay_scale,
                    decay_bias=self._gdr_decay_bias,
                    offsets=x_offsets,
                    event_gate=gdr_event_gate,
                ),
            )
            attn_output = kernel_output.context.reshape(
                -1, self._num_heads * self._linear_dim
            )
        elif self._attention_backend == "linear_elu":
            if delta_x_offsets is not None:
                raise NotImplementedError(
                    "Linear Attention reference does not support incremental cache"
                )
            padded_q = jagged.jagged_to_padded_dense(
                values=q, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
            )
            padded_k = jagged.jagged_to_padded_dense(
                values=k, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
            )
            padded_v = jagged.jagged_to_padded_dense(
                values=v, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
            )
            lengths = x_offsets[1:] - x_offsets[:-1]
            valid_mask = torch.arange(n, device=x_offsets.device)[None, :] < lengths[:, None]
            dense_output = normalized_linear_attention(
                padded_q.view(B, n, self._num_heads, self._attention_dim).transpose(1, 2),
                padded_k.view(B, n, self._num_heads, self._attention_dim).transpose(1, 2),
                padded_v.view(B, n, self._num_heads, self._linear_dim).transpose(1, 2),
                valid_mask,
                implementation="chunked",
            )
            dense_output = dense_output.transpose(1, 2).reshape(
                B, n, self._num_heads * self._linear_dim
            )
            attn_output = jagged.dense_to_jagged(
                dense_output, [x_offsets]
            )[0]
        elif self._attention_backend == "softmax_no_bias":
            if delta_x_offsets is not None:
                raise NotImplementedError(
                    "Softmax diagnostic does not support incremental cache"
                )
            padded_q = jagged.jagged_to_padded_dense(
                values=q, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
            )
            padded_k = jagged.jagged_to_padded_dense(
                values=k, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
            )
            padded_v = jagged.jagged_to_padded_dense(
                values=v, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
            )
            dense_q = padded_q.view(B, n, self._num_heads, self._attention_dim)
            dense_k = padded_k.view(B, n, self._num_heads, self._attention_dim)
            dense_v = padded_v.view(B, n, self._num_heads, self._linear_dim)
            logits = torch.einsum("bnhd,bmhd->bhnm", dense_q, dense_k)
            allowed = invalid_attn_mask.bool()
            if allowed.ndim == 2:
                allowed = allowed[None, None]
            elif allowed.ndim == 3:
                allowed = allowed[:, None]
            logits = logits.float() / math.sqrt(self._attention_dim)
            logits = logits.masked_fill(~allowed, float("-inf"))
            weights = torch.softmax(logits, dim=-1).to(dense_v.dtype)
            dense_output = torch.einsum("bhnm,bmhd->bnhd", weights, dense_v).reshape(
                B, n, self._num_heads * self._linear_dim
            )
            attn_output = jagged.dense_to_jagged(
                dense_output, [x_offsets]
            )[0]
        elif self._normalization == "rel_bias" or self._normalization == "hstu_rel_bias":
            assert self._rel_attn_bias is not None
            attn_output, padded_q, padded_k = _hstu_attention_maybe_from_cache(
                num_heads=self._num_heads,
                attention_dim=self._attention_dim,
                linear_dim=self._linear_dim,
                q=q,
                k=k,
                v=v,
                cached_q=cached_q,
                cached_k=cached_k,
                delta_x_offsets=delta_x_offsets,
                x_offsets=x_offsets,
                all_timestamps=all_timestamps,
                invalid_attn_mask=invalid_attn_mask,
                rel_attn_bias=self._rel_attn_bias,
            )
        elif self._normalization == "softmax_rel_bias":
            if delta_x_offsets is not None:
                B = x_offsets.size(0) - 1
                padded_q, padded_k = cached_q, cached_k
                flattened_offsets = delta_x_offsets[1] + torch.arange(
                    start=0,
                    end=B * n,
                    step=n,
                    device=delta_x_offsets[1].device,
                    dtype=delta_x_offsets[1].dtype,
                )
                assert padded_q is not None
                assert padded_k is not None
                padded_q = (
                    padded_q.view(B * n, -1)
                    .index_copy_(
                        dim=0,
                        index=flattened_offsets,
                        source=q,
                    )
                    .view(B, n, -1)
                )
                padded_k = (
                    padded_k.view(B * n, -1)
                    .index_copy_(
                        dim=0,
                        index=flattened_offsets,
                        source=k,
                    )
                    .view(B, n, -1)
                )
            else:
                padded_q = jagged.jagged_to_padded_dense(
                    values=q, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
                )
                padded_k = jagged.jagged_to_padded_dense(
                    values=k, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
                )

            qk_attn = torch.einsum("bnd,bmd->bnm", padded_q, padded_k)
            if self._rel_attn_bias is not None:
                qk_attn = qk_attn + self._rel_attn_bias(all_timestamps)
            qk_attn = F.softmax(qk_attn / math.sqrt(self._attention_dim), dim=-1)
            qk_attn = qk_attn * invalid_attn_mask
            attn_output = jagged.dense_to_jagged(
                torch.bmm(
                    qk_attn,
                    jagged.jagged_to_padded_dense(v, [x_offsets], [n]),
                ),
                [x_offsets],
            )[0]
        else:
            raise ValueError(f"Unknown normalization method {self._normalization}")

        attn_output = (
            attn_output
            if delta_x_offsets is None
            else attn_output[delta_x_offsets[0], :]
        )
        if self._concat_ua:
            a = self._norm_attn_output(attn_output)
            o_input = torch.cat([u, a, u * a], dim=-1)
        else:
            o_input = u * self._norm_attn_output(attn_output)

        new_outputs = (
            self._o(
                F.dropout(
                    o_input,
                    p=self._dropout_ratio,
                    training=self.training,
                )
            )
            + x
        )

        if delta_x_offsets is not None:
            new_outputs = cached_outputs.index_copy_(
                dim=0, index=delta_x_offsets[0], source=new_outputs
            )

        if return_cache_states and delta_x_offsets is None:
            v = v.contiguous()

        return new_outputs, (v, padded_q, padded_k, new_outputs)


class HSTUJagged(torch.nn.Module):
    def __init__(
        self,
        modules: List[SequentialTransductionUnitJagged],
        autocast_dtype: Optional[torch.dtype],
    ) -> None:
        super().__init__()

        self._attention_layers: torch.nn.ModuleList = torch.nn.ModuleList(
            modules=modules
        )
        self._autocast_dtype: Optional[torch.dtype] = autocast_dtype
        self._use_pytorch_jagged = all(
            layer._attention_backend == "gdr" for layer in self._attention_layers
        )

    def jagged_forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
        gdr_event_gate: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[HSTUCacheState]]:
        """
        Args:
            x: (\sum_i N_i, D) x float
            x_offsets: (B + 1) x int32
            all_timestamps: (B, 1 + N) x int64
            invalid_attn_mask: (B, N, N) x float, each element in {0, 1}
            return_cache_states: bool. True if we should return cache states.

        Returns:
            x' = f(x), (\sum_i N_i, D) x float
        """
        cache_states: List[HSTUCacheState] = []

        with torch.autocast(
            "cuda",
            enabled=self._autocast_dtype is not None,
            dtype=self._autocast_dtype or torch.float16,
        ):
            for i, layer in enumerate(self._attention_layers):
                x, cache_states_i = layer(
                    x=x,
                    x_offsets=x_offsets,
                    all_timestamps=all_timestamps,
                    invalid_attn_mask=invalid_attn_mask,
                    delta_x_offsets=delta_x_offsets,
                    cache=cache[i] if cache is not None else None,
                    return_cache_states=return_cache_states,
                    gdr_event_gate=gdr_event_gate,
                )
                if return_cache_states:
                    cache_states.append(cache_states_i)

        return x, cache_states

    def forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
        gdr_event_gate: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[HSTUCacheState]]:
        """
        Args:
            x: (B, N, D) x float.
            x_offsets: (B + 1) x int32.
            all_timestamps: (B, 1 + N) x int64
            invalid_attn_mask: (B, N, N) x float, each element in {0, 1}.
        Returns:
            x' = f(x), (B, N, D) x float
        """
        flat_valid_indices = None
        if len(x.size()) == 3:
            if self._use_pytorch_jagged:
                batch, sequence_length, embedding_dim = x.shape
                lengths = x_offsets[1:] - x_offsets[:-1]
                valid = (
                    torch.arange(sequence_length, device=x.device)[None, :]
                    < lengths[:, None]
                )
                flat_valid_indices = torch.nonzero(
                    valid.reshape(-1),
                    as_tuple=False,
                ).squeeze(-1)
                x = x.reshape(batch * sequence_length, embedding_dim).index_select(
                    0,
                    flat_valid_indices,
                )
            else:
                x = jagged.dense_to_jagged(x, [x_offsets])[0]

        jagged_x, cache_states = self.jagged_forward(
            x=x,
            x_offsets=x_offsets,
            all_timestamps=all_timestamps,
            invalid_attn_mask=invalid_attn_mask,
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
            gdr_event_gate=gdr_event_gate,
        )
        if self._use_pytorch_jagged:
            batch = x_offsets.numel() - 1
            sequence_length = invalid_attn_mask.size(1)
            if flat_valid_indices is None:
                lengths = x_offsets[1:] - x_offsets[:-1]
                valid = (
                    torch.arange(sequence_length, device=jagged_x.device)[None, :]
                    < lengths[:, None]
                )
                flat_valid_indices = torch.nonzero(
                    valid.reshape(-1),
                    as_tuple=False,
                ).squeeze(-1)
            y = jagged_x.new_zeros(
                batch * sequence_length,
                jagged_x.shape[-1],
            ).index_copy(0, flat_valid_indices, jagged_x)
            y = y.view(batch, sequence_length, jagged_x.shape[-1])
        else:
            y = jagged.jagged_to_padded_dense(
                values=jagged_x,
                offsets=[x_offsets],
                max_lengths=[invalid_attn_mask.size(1)],
                padding_value=0.0,
            )
        return y, cache_states


class HSTU(SequentialEncoderWithLearnedSimilarityModule):
    """
    Implements HSTU (Hierarchical Sequential Transduction Unit) in
    Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations,
    https://arxiv.org/abs/2402.17152.

    Note that this implementation is intended for reproducing experiments in
    the traditional sequential recommender setting (Section 4.1.1), and does
    not yet use optimized kernels discussed in the paper.
    """

    def __init__(
        self,
        max_sequence_len: int,
        max_output_len: int,
        embedding_dim: int,
        num_blocks: int,
        num_heads: int,
        linear_dim: int,
        attention_dim: int,
        normalization: str,
        linear_config: str,
        linear_activation: str,
        linear_dropout_rate: float,
        attn_dropout_rate: float,
        embedding_module: EmbeddingModule,
        similarity_module: SimilarityModule,
        input_features_preproc_module: InputFeaturesPreprocessorModule,
        output_postproc_module: OutputPostprocessorModule,
        enable_relative_attention_bias: bool = True,
        concat_ua: bool = False,
        verbose: bool = True,
        attention_backend: str = "original",
        gdr_kernel_backend: str = "fla",
    ) -> None:
        super().__init__(ndp_module=similarity_module)

        self._embedding_dim: int = embedding_dim
        self._item_embedding_dim: int = embedding_module.item_embedding_dim
        self._max_sequence_length: int = max_sequence_len
        self._embedding_module: EmbeddingModule = embedding_module
        self._input_features_preproc: InputFeaturesPreprocessorModule = (
            input_features_preproc_module
        )
        self._output_postproc: OutputPostprocessorModule = output_postproc_module
        self._num_blocks: int = num_blocks
        self._num_heads: int = num_heads
        self._dqk: int = attention_dim
        self._dv: int = linear_dim
        self._linear_activation: str = linear_activation
        self._linear_dropout_rate: float = linear_dropout_rate
        self._attn_dropout_rate: float = attn_dropout_rate
        self._enable_relative_attention_bias: bool = enable_relative_attention_bias
        self._attention_backend: str = attention_backend
        self._gdr_kernel_backend: str = gdr_kernel_backend
        self._hstu = HSTUJagged(
            modules=[
                SequentialTransductionUnitJagged(
                    embedding_dim=self._embedding_dim,
                    linear_hidden_dim=linear_dim,
                    attention_dim=attention_dim,
                    normalization=normalization,
                    linear_config=linear_config,
                    linear_activation=linear_activation,
                    num_heads=num_heads,
                    # TODO: change to lambda x.
                    relative_attention_bias_module=(
                        RelativeBucketedTimeAndPositionBasedBias(
                            max_seq_len=max_sequence_len
                            + max_output_len,  # accounts for next item.
                            num_buckets=128,
                            bucketization_fn=lambda x: (
                                torch.log(torch.abs(x).clamp(min=1)) / 0.301
                            ).long(),
                        )
                        if enable_relative_attention_bias
                        else None
                    ),
                    dropout_ratio=linear_dropout_rate,
                    attn_dropout_ratio=attn_dropout_rate,
                    concat_ua=concat_ua,
                    attention_backend=attention_backend,
                    gdr_kernel_backend=gdr_kernel_backend,
                )
                for _ in range(num_blocks)
            ],
            autocast_dtype=None,
        )
        # causal forward, w/ +1 for padding.
        self.register_buffer(
            "_attn_mask",
            torch.triu(
                torch.ones(
                    (
                        self._max_sequence_length + max_output_len,
                        self._max_sequence_length + max_output_len,
                    ),
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
        )
        self._verbose: bool = verbose
        self.reset_params()

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if ("_hstu" in name) or ("_embedding_module" in name):
                if self._verbose:
                    print(f"Skipping init for {name}")
                continue
            try:
                torch.nn.init.xavier_normal_(params.data)
                if self._verbose:
                    print(
                        f"Initialize {name} as xavier normal: {params.data.size()} params"
                    )
            except:
                if self._verbose:
                    print(f"Failed to initialize {name}: {params.data.size()} params")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._embedding_module.get_item_embeddings(item_ids)

    def debug_str(self) -> str:
        debug_str = (
            f"HSTU-b{self._num_blocks}-h{self._num_heads}-dqk{self._dqk}-dv{self._dv}"
            + f"-l{self._linear_activation}d{self._linear_dropout_rate}"
            + f"-ad{self._attn_dropout_rate}"
        )
        if not self._enable_relative_attention_bias:
            debug_str += "-norab"
        if self._attention_backend != "original":
            debug_str += f"-attn{self._attention_backend}"
        if self._attention_backend == "gdr":
            debug_str += f"-kernel{self._gdr_kernel_backend}"
        return debug_str

    def generate_user_embeddings(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
        gdr_event_gate: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[HSTUCacheState]]:
        """
        [B, N] -> [B, N, D].
        """
        device = past_lengths.device
        float_dtype = past_embeddings.dtype
        B, N, _ = past_embeddings.size()

        past_lengths, user_embeddings, _ = self._input_features_preproc(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )

        float_dtype = user_embeddings.dtype
        x_offsets = jagged.asynchronous_complete_cumsum(past_lengths)
        packed_gdr_event_gate = None
        if gdr_event_gate is not None:
            if self._attention_backend != "gdr":
                raise ValueError("gdr_event_gate requires the gdr attention backend")
            if gdr_event_gate.shape != past_ids.shape:
                raise ValueError("gdr_event_gate must match past_ids shape")
            if not torch.is_floating_point(gdr_event_gate):
                gdr_event_gate = gdr_event_gate.to(user_embeddings.dtype)
            if not bool(torch.isfinite(gdr_event_gate).all()):
                raise ValueError("gdr_event_gate must contain finite values")
            if bool((gdr_event_gate < 0).any()) or bool((gdr_event_gate > 1).any()):
                raise ValueError("gdr_event_gate values must lie in [0, 1]")
            positions = torch.arange(N, device=device)[None, :]
            packed_gdr_event_gate = gdr_event_gate[
                positions < past_lengths[:, None]
            ]
        user_embeddings, cached_states = self._hstu(
            x=user_embeddings,
            x_offsets=x_offsets,
            all_timestamps=(
                past_payloads[TIMESTAMPS_KEY]
                if TIMESTAMPS_KEY in past_payloads
                else None
            ),
            invalid_attn_mask=1.0 - self._attn_mask.to(float_dtype),
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
            gdr_event_gate=packed_gdr_event_gate,
        )
        return self._output_postproc(user_embeddings), cached_states

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        batch_id: Optional[int] = None,
        gdr_event_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Runs the main encoder.

        Args:
            past_lengths: (B,) x int64
            past_ids: (B, N,) x int64 where the latest engaged ids come first. In
                particular, past_ids[i, past_lengths[i] - 1] should correspond to
                the latest engaged values.
            past_embeddings: (B, N, D) x float or (\sum_b N_b, D) x float.
            past_payloads: implementation-specific keyed tensors of shape (B, N, ...).

        Returns:
            encoded_embeddings of [B, N, D].
        """
        encoded_embeddings, _ = self.generate_user_embeddings(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
            gdr_event_gate=gdr_event_gate,
        )
        return encoded_embeddings

    def _encode(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]],
        cache: Optional[List[HSTUCacheState]],
        return_cache_states: bool,
        gdr_event_gate: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[HSTUCacheState]]]:
        """
        Args:
            past_lengths: (B,) x int64.
            past_ids: (B, N,) x int64.
            past_embeddings: (B, N, D,) x float.
            past_payloads: implementation-specific keyed tensors of shape (B, N, ...).
            return_cache_states: bool.

        Returns:
            (B, D) x float, representing embeddings for the current state.
        """
        encoded_seq_embeddings, cache_states = self.generate_user_embeddings(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
            gdr_event_gate=gdr_event_gate,
        )  # [B, N, D]
        current_embeddings = get_current_embeddings(
            lengths=past_lengths, encoded_embeddings=encoded_seq_embeddings
        )
        if return_cache_states:
            return current_embeddings, cache_states
        else:
            return current_embeddings

    def encode(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
        gdr_event_gate: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[HSTUCacheState]]]:
        """
        Runs encoder to obtain the current hidden states.

        Args:
            past_lengths: (B,) x int.
            past_ids: (B, N,) x int.
            past_embeddings: (B, N, D) x float.
            past_payloads: implementation-specific keyed tensors of shape (B, N, ...).

        Returns:
            (B, D,) x float, representing encoded states at the most recent time step.
        """
        return self._encode(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
            gdr_event_gate=gdr_event_gate,
        )
