# Copyright 2025-2026 USTC-StarTeam
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import math
from typing import Callable, Dict, List, Optional, Tuple, Union, Literal

import torch
from deltarec.layers import jagged
import torch.nn as nn
import torch.nn.functional as F
from deltarec.adaptors.fuxi.modeling.ndp_module import NDPModule

import einops
import logging

from deltarec.adaptors.fuxi.modeling.sequential.embedding_modules import (
    EmbeddingModule,
)
from deltarec.adaptors.fuxi.modeling.sequential.input_features_preprocessors import (
    InputFeaturesPreprocessorModule,
)
from deltarec.adaptors.fuxi.modeling.sequential.output_postprocessors import (
    OutputPostprocessorModule,
)
from deltarec.adaptors.fuxi.modeling.sequential.utils import get_current_embeddings
from deltarec.adaptors.fuxi.modeling.similarity_module import (
    GeneralizedInteractionModule,
)

from deltarec.adaptors.fuxi.modeling.sequential.hstu import RelativeAttentionBiasModule

from deltarec.adaptors.fuxi.modeling.sequential.hstu import RelativeBucketedTimeAndPositionBasedBias
from deltarec.adaptors.fuxi.modeling.sequential.fuxi_modules import (
    Retention,
    MultistageFeedforwardNeuralNetwork,
)
from deltarec.adaptors.fuxi.modeling.sequential.fuxi_modules.attn import (
    LinearPositionalChannel,
    LinearTemporalChannel,
)

TIMESTAMPS_KEY = "timestamps"

class FuXiLinearBlockJagged(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        linear_hidden_dim: int,
        attention_dim: int,
        dropout_ratio: float,
        attn_dropout_ratio: float,
        num_heads: int,
        linear_activation: str,
        channel_t_config: Optional[Dict],
        channel_p_config: Optional[Dict],
        id_layer: int,
        use_rope: bool = False,
        normalization: str = "rel_bias",
        epsilon: float = 1e-6,
        ffn_multiply: float = 1,
        chunk_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._embedding_dim: int = embedding_dim
        self._linear_dim: int = linear_hidden_dim
        self._attention_dim: int = attention_dim
        self._dropout_ratio: float = dropout_ratio
        self._attn_dropout_ratio: float = attn_dropout_ratio
        self._num_heads: int = num_heads
        self._normalization: str = normalization
        
        query_dim = attention_dim * num_heads
        key_dim = query_dim
        value_dim = linear_hidden_dim * num_heads
        
        self._attn_dim = value_dim * (1 + (channel_p_config != None) + (channel_t_config != None))
        
        self._uvqk: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(
                (
                    embedding_dim,
                    query_dim + key_dim + value_dim + self._attn_dim,
                )
            ).normal_(mean=0, std=0.02),
        )
        
        self._linear_activation: str = linear_activation
        
        self._retention = Retention(
            head_dim = attention_dim,
            num_heads = num_heads,
            use_rope = use_rope,
            chunk_size = chunk_size,
        )
        
        self._channel_t_config = channel_t_config
        # type == 'linear'
        self._channel_t = LinearTemporalChannel(
            linear_dim = value_dim,
            num_heads = channel_t_config.get('num_heads', num_heads),
            base = channel_t_config.get('base', 2),
            start_index = channel_t_config.get('start_index', 0),
            base_stride = channel_t_config.get('base_stride', 1),
            chunk_size = chunk_size,
            use_proj = channel_t_config.get('use_proj', True),
            learnable_gamma = channel_t_config.get('learnable_gamma', False) and id_layer == 0,
            no_temporal_qk = channel_t_config.get('no_temporal_qk', False),
            use_augment_connection = channel_t_config.get('aug_current', False),
        )
        
        self._channel_p_config = channel_p_config
        # type == 'linear'
        self._channel_p = LinearPositionalChannel(
            max_seq_len = channel_p_config.get('max_sequence_length', None),
            embedding_dim = channel_p_config.get('dim', 32),
            aug_current = channel_p_config.get('aug_current', True),
            use_proj = True,
            value_dim = value_dim,
            chunk_size = chunk_size,
        )
        
        logging.info(f'Temporal channel config: {channel_t_config}')
        logging.info(f'Positional channel config: {channel_p_config}')
        
        self._mffn = MultistageFeedforwardNeuralNetwork(
            ams_output_size = self._attn_dim,
            input_size = embedding_dim,
            hidden_size = int(embedding_dim * ffn_multiply),
            output_size = embedding_dim,
            dropout_ratio = dropout_ratio,
            single_stage = False,
            epsilon = epsilon
        )
        self._mffn.init()
        
        self._eps: float = epsilon

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, normalized_shape=[self._embedding_dim], eps=self._eps)

    def _norm_attn_output(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            x, normalized_shape=[self._attn_dim], eps=self._eps
        )

    def forward(  # pyre-ignore [3]
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        past_lengths: torch.Tensor,
        target_timestamps: torch.Tensor,
        current_length: int,
        cache = None,
        early_lengths: Optional[torch.Tensor] = None,
        early_timestamps: Optional[torch.Tensor] = None,
        return_cache_states: bool = False,
        channel_t_buffer = None,
    ):
        B: int = past_lengths.size(0)
        n: int = current_length
        
        normed_x = self._norm_input(x)
        
        if cache is not None :
            cache_s, cache_t, cache_p = cache
        else :
            cache_s, cache_t, cache_p = None, None, None
        
        batched_mm_output = torch.mm(normed_x, self._uvqk)
        batched_mm_output = F.silu(batched_mm_output)
        
        u, q, k, v = torch.split(
            batched_mm_output,
            [
                self._attn_dim,
                self._attention_dim * self._num_heads,
                self._attention_dim * self._num_heads,
                self._linear_dim * self._num_heads,
            ],
            dim=1,
        )

        B: int = x_offsets.size(0) - 1
        
        padded_v = jagged.jagged_to_padded_dense(v, [x_offsets], [n])
        
        outputs = []
       
        output_latent, returned_cache = self._retention(
            q=q,
            k=k,
            v=padded_v,
            x_offsets=x_offsets,
            all_timestamps=all_timestamps,
            invalid_attn_mask=invalid_attn_mask,
            is_v_padded=True,
            current_length=current_length,
            past_lengths=past_lengths,
            cache=cache_s,
            early_lengths=early_lengths,
            early_timestamps=early_timestamps,
            return_cache_states=return_cache_states,
        )

        outputs.append(self._norm_input(output_latent))
        
        if self._channel_t != None :
            log_decay_t = torch.ones(normed_x.shape[0], self._channel_t_config.get('num_heads') * 2, dtype=torch.float32, device=normed_x.device)
            
            output_latent_t, channel_t_buffer, cache_t = self._channel_t(
                q=None,
                k=None,
                v=normed_x,
                x_offsets=x_offsets,
                all_timestamps=all_timestamps,
                invalid_attn_mask=invalid_attn_mask,
                past_lengths=past_lengths,
                is_v_padded=False,
                cache=cache_t,
                buffered_attn_map=channel_t_buffer,
                target_timestamps=target_timestamps,
                return_cache_states=return_cache_states,
                log_decay=log_decay_t,
            )
            outputs.append(self._norm_input(output_latent_t))

        if self._channel_p != None :
            output_latent_p, channel_p_buffer, cache_p = self._channel_p(
                q=q,
                k=k,
                v=normed_x,
                # w=w,
                x_offsets=x_offsets,
                all_timestamps=all_timestamps,
                invalid_attn_mask=invalid_attn_mask,
                is_v_padded=False,
                cache=cache_p,
                past_lengths=past_lengths,
                buffered_attn_map=None,
                return_cache_states=return_cache_states,
            )
            outputs.append(self._norm_input(output_latent_p))
        
        combined_output = torch.concat(
            outputs,
            dim=-1
        ).reshape(B, n, self._attn_dim)

        attn_output = jagged.dense_to_jagged(
            combined_output, [x_offsets],
        )[0]
        
        attn_output = u * attn_output

        new_outputs = self._mffn(attn_output, x, past_lengths)
        
        return new_outputs, (returned_cache, cache_t, cache_p), channel_t_buffer


class FuXiLinearJagged(torch.nn.Module):

    def __init__(
        self,
        modules: List[FuXiLinearBlockJagged],
        autocast_dtype: Optional[torch.dtype],
    ) -> None:
        super().__init__()

        self._attention_layers: torch.nn.ModuleList = torch.nn.ModuleList(
            modules=modules
        )
        self._autocast_dtype: Optional[torch.dtype] = autocast_dtype

    def jagged_forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        past_lengths: torch.Tensor,
        target_timestamps: torch.Tensor,
        current_length: int,
        cache = None,
        early_lengths: Optional[torch.Tensor] = None,
        early_timestamps: Optional[torch.Tensor] = None,
        return_cache_states: bool = False,
    ) -> torch.Tensor :

        cache_states: List[torch.Tensor] = []
        
        with torch.autocast(
            "cuda",
            enabled=self._autocast_dtype is not None,
            dtype=self._autocast_dtype or torch.float16,
        ):
            buffer_states = None
            for i, layer in enumerate(self._attention_layers):
                x, cache_states_i, buffer_states = layer(
                    x=x,
                    x_offsets=x_offsets,
                    all_timestamps=all_timestamps,
                    invalid_attn_mask=invalid_attn_mask,
                    past_lengths=past_lengths,
                    target_timestamps=target_timestamps,
                    cache=cache[i] if cache is not None else None,
                    return_cache_states=return_cache_states,
                    current_length=current_length,
                    early_lengths=early_lengths,
                    early_timestamps=early_timestamps,
                    channel_t_buffer=buffer_states,
                )
                if return_cache_states :
                    cache_states.append(cache_states_i)

        return x, cache_states

    def forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        current_length: int,
        past_lengths: torch.Tensor,
        target_timestamps: torch.Tensor,
        cache = None,
        early_lengths: Optional[torch.Tensor] = None,
        early_timestamps: Optional[torch.Tensor] = None,
        return_cache_states: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) x float.
            x_offsets: (B + 1) x int32.
            all_timestamps: (B, 1 + N) x int64
            invalid_attn_mask: (B, N, N) x float, each element in {0, 1}.
        Returns:
            x' = f(x), (B, N, D) x float
        """
        if len(x.size()) == 3:
            x = jagged.dense_to_jagged(x, [x_offsets])[0]

        jagged_x, cache_states = self.jagged_forward(
            x=x,
            x_offsets=x_offsets,
            all_timestamps=all_timestamps,
            invalid_attn_mask=invalid_attn_mask,
            current_length=current_length,
            past_lengths=past_lengths,
            target_timestamps=target_timestamps,
            cache=cache,
            early_timestamps=early_timestamps,
            early_lengths=early_lengths,
            return_cache_states=return_cache_states,
        )
        y = jagged.jagged_to_padded_dense(
            values=jagged_x,
            offsets=[x_offsets],
            max_lengths=[current_length],
            padding_value=0.0,
        )
        return y, cache_states


class FuXiLinear(GeneralizedInteractionModule):
    """
    Implements FuXi Block
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
        linear_activation: str,
        linear_dropout_rate: float,
        attn_dropout_rate: float,
        ffn_multiply: int,
        embedding_module: EmbeddingModule,
        similarity_module: NDPModule,
        input_features_preproc_module: InputFeaturesPreprocessorModule,
        output_postproc_module: OutputPostprocessorModule,
        channel_t_config: Optional[Dict],
        channel_p_config: Optional[Dict],
        use_rope: bool = False, 
        enable_relative_attention_bias: bool = True,
        chunk_size: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        super().__init__(ndp_module=similarity_module)
        
        logging.info(f'chunk_size = {chunk_size}')

        self._embedding_dim: int = embedding_dim
        self._item_embedding_dim: int = embedding_module.item_embedding_dim
        self._max_sequence_length: int = max_sequence_len
        self._embedding_module: EmbeddingModule = embedding_module
        self._input_features_preproc: InputFeaturesPreprocessorModule = (
            input_features_preproc_module
        )
        logging.info(type(input_features_preproc_module))
        self._output_postproc: OutputPostprocessorModule = output_postproc_module
        self._num_blocks: int = num_blocks
        self._num_heads: int = num_heads
        self._dqk: int = attention_dim
        self._dv: int = linear_dim
        self._linear_activation: str = linear_activation
        self._linear_dropout_rate: float = linear_dropout_rate
        self._attn_dropout_rate: float = attn_dropout_rate
        self._enable_relative_attention_bias: bool = enable_relative_attention_bias
        
        if channel_p_config is not None : 
            channel_p_config['max_sequence_length'] = max_sequence_len + max_output_len
        
        channel_t_config['chunk_size'] = chunk_size
        
        self.chunk_size = chunk_size
        
        self._fuxi = FuXiLinearJagged(
            modules=[
                FuXiLinearBlockJagged(
                    embedding_dim=self._embedding_dim,
                    linear_hidden_dim=linear_dim,
                    attention_dim=attention_dim,
                    normalization=normalization,
                    linear_activation=linear_activation,
                    num_heads=num_heads,
                    channel_t_config=channel_t_config,
                    channel_p_config=channel_p_config,
                    use_rope=use_rope, 
                    chunk_size=chunk_size,
                    dropout_ratio=linear_dropout_rate,
                    attn_dropout_ratio=attn_dropout_rate,
                    ffn_multiply=ffn_multiply,
                    id_layer=_,
                )
                for _ in range(num_blocks)
            ],
            autocast_dtype=None,
        )
        
        N = self._max_sequence_length + max_output_len
        if self.chunk_size is not None :
            N = (N + chunk_size - 1) // chunk_size * chunk_size    
        
        # causal forward, w/ +1 for padding.
        self.register_buffer(
            "_attn_mask",
            torch.triu(
                torch.ones(
                    (
                        N,
                        N,
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
            if ("_fuxi" in name) or ("_embedding_module" in name):
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
            f"FuXi-Linear-b{self._num_blocks}-h{self._num_heads}-dqk{self._dqk}-dv{self._dv}"
            + f"-l{self._linear_activation}d{self._linear_dropout_rate}"
            + f"-ad{self._attn_dropout_rate}"
        )
        if not self._enable_relative_attention_bias:
            debug_str += "-norab"
        return debug_str

    def generate_user_embeddings(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        cache: Optional[List[torch.Tensor]] = None,
        target_timestamps: torch.Tensor = None,
        early_lengths: Optional[torch.Tensor] = None,
        early_payloads: Optional[Dict[str, torch.Tensor]] = None,
        return_cache_states: bool = False,
    ) -> torch.Tensor :
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
            position_offsets=early_lengths,
        )
        
        if cache is None :
            invalid_attn_mask = 1.0 - self._attn_mask.to(float_dtype)
        else :
            invalid_attn_mask = None
        
        all_timestamps=(
                past_payloads[TIMESTAMPS_KEY]
                if TIMESTAMPS_KEY in past_payloads
                else None
            )
        
        if self.chunk_size is not None and cache is None:
            old_length = N
            current_length = ((N + self.chunk_size - 1) // self.chunk_size) * self.chunk_size  
            user_embeddings = F.pad(user_embeddings, (0, 0, 0, current_length - N))
            all_timestamps = F.pad(all_timestamps, (0, current_length - N))
            N = current_length
        
        early_timestamps = None if early_payloads is None or TIMESTAMPS_KEY not in early_payloads else early_payloads[TIMESTAMPS_KEY]
        
        if cache is not None :
            assert target_timestamps is not None
        
        float_dtype = user_embeddings.dtype
        user_embeddings, cached_states = self._fuxi(
            x=user_embeddings,
            x_offsets=jagged.asynchronous_complete_cumsum(past_lengths),
            all_timestamps=all_timestamps,
            invalid_attn_mask=invalid_attn_mask,
            cache=cache,
            past_lengths=past_lengths,
            target_timestamps=target_timestamps,
            return_cache_states=return_cache_states,
            current_length=N,
            early_lengths=early_lengths,
            early_timestamps=early_timestamps,
        )
        
        if self.chunk_size is not None and cache is None:
            user_embeddings = user_embeddings[:, :old_length]
        return self._output_postproc(user_embeddings), cached_states

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        batch_id: Optional[int] = None,
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
        )
        return encoded_embeddings

    def _encode(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        cache,
        target_timestamps: Optional[torch.Tensor],
        early_lengths: Optional[torch.Tensor],
        early_payloads: Optional[Dict[str, torch.Tensor]],
        return_cache_states: bool,
    ) -> torch.Tensor:
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
            cache=cache,
            target_timestamps=target_timestamps,
            early_lengths=early_lengths,
            early_payloads=early_payloads,
            return_cache_states=return_cache_states,
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
        cache = None,
        target_timestamps: Optional[torch.Tensor] = None,
        early_lengths: Optional[torch.Tensor] = None,
        early_payloads: Optional[Dict[str, torch.Tensor]] = None,
        return_cache_states: bool = False,
    ) -> torch.Tensor:
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
        if cache is not None :
            assert target_timestamps is not None
        return self._encode(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
            target_timestamps=target_timestamps,
            cache=cache,
            early_lengths=early_lengths,
            early_payloads=early_payloads,
            return_cache_states=return_cache_states,
        )
