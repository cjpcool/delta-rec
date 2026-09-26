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
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union, Literal

import numpy as np

import torch
from deltarec.layers import jagged
import torch.nn as nn
import torch.nn.functional as F

import einops
import logging

from einops import rearrange

from . import linear_attn
from . import linear_attn_fn

def apply_multiplication_with_lambda(
    inputs: torch.Tensor,       # (B, n, h, d)
    sin_bases: torch.Tensor,    # (n, h, d) / (B, n, h, d) 
    cos_bases: torch.Tensor,    # (n, h, d) / (B, n, h, d)
) -> torch.Tensor :
    # logging.info(f'inputs {inputs.shape}')
    # logging.info(f'sin_bases {sin_bases.shape}')
    dim = inputs.shape[-1]
    assert dim % 2 == 0
    
    half_dim = dim // 2
    
    chk0, chk1 = inputs.split([half_dim, half_dim], dim=-1)
    
    if sin_bases.dim() == 3 :
        sin_bases = sin_bases.unsqueeze(0)
        cos_bases = cos_bases.unsqueeze(0)    
    
    pos_real = chk0 * cos_bases - chk1 * sin_bases
    pos_img = chk0 * sin_bases + chk1 * cos_bases
    
    pos_value = torch.concat([pos_real, pos_img], dim=-1)
    
    return pos_value

class Retention (nn.Module) :
    def __init__(
        self, 
        head_dim: int,
        num_heads: int,
        use_rope: bool = False,
        chunk_size: Optional[int] = None,
        report_precision: bool = False,
        dtype = torch.float32,
    ):
        super().__init__()
        attn_dim = head_dim * num_heads
        self._attn_dim = attn_dim
        self._num_heads = num_heads
        logging.info(f'Retention: num heads = {num_heads}, chunk_size = {chunk_size}')
        self._head_dim = head_dim
        self._use_rope: bool = use_rope
        
        self._chunk_size = chunk_size
        self._report_precision = report_precision
        
        
        if self._use_rope: 
            requires_grad_flg = False
            logging.info(f'retnet: use RoPE, requires_grad={requires_grad_flg}')
            head_dim = attn_dim // num_heads
            half_head_dim = head_dim // 2
            theta = torch.exp(-math.log(10000) * torch.arange(half_head_dim) / half_head_dim)
            self._lambda = nn.Parameter(theta, requires_grad=requires_grad_flg)
            
        self._gamma = nn.Parameter(torch.empty(num_heads, dtype=dtype).normal_(mean=0, std=0.02))
        
        if chunk_size is not None :
            self.register_buffer(
                '_chunkwise_casual_mask', 
                torch.tril(torch.ones(chunk_size, chunk_size))
            )
            
    def _get_gamma(self) :
        gamma = F.softplus(self._gamma)
        gamma = torch.cumsum(gamma, dim=0)
        return torch.exp(-gamma)
    
    def _input_preprocess(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        current_length: int,
        is_v_padded: bool = False,
        max_length = None,
    ) :
        h, dim = self._num_heads, self._head_dim
        B, n = all_timestamps.shape[0], current_length
        
        if max_length is None :
            max_length = n
        
        padded_q = jagged.jagged_to_padded_dense(
            values=q, offsets=[x_offsets], max_lengths=[max_length], padding_value=0.0
        ).view(B, max_length, h, -1)
        padded_k = jagged.jagged_to_padded_dense(
            values=k, offsets=[x_offsets], max_lengths=[max_length], padding_value=0.0
        ).view(B, max_length, h, -1)
        if is_v_padded :
            # padded_v = v.reshape(B, n, self._attn_dim)
            padded_v = v
            if n < max_length :
                padded_v = torch.concat([
                        padded_v.reshape(B, n, h, -1),
                        torch.zeros(B, max_length - n, h, dim, dtype=padded_v.dtype, device=padded_v.device)
                    ],
                    dim=1,
                )
            else :
                padded_v = padded_v.reshape(B, n, h, -1)
            v = None
        else :
            padded_v = jagged.jagged_to_padded_dense(
                values=v, offsets=[x_offsets], max_lengths=[max_length], padding_value=0.0
            ).reshape(B, max_length, h, -1)
        
        if self._use_rope :
            position = torch.arange(max_length, dtype=padded_v.dtype, device=padded_v.device)
            bases = torch.einsum('n,d->nd', position, self._lambda)
            bases = bases.unsqueeze(dim=-2)             # head axis
            sin_bases = torch.sin(bases)
            cos_bases = torch.cos(bases)
            
            tilde_q = apply_multiplication_with_lambda(padded_q, sin_bases, cos_bases)
            tilde_k = apply_multiplication_with_lambda(padded_k, sin_bases, cos_bases)
        else :
            tilde_q = padded_q
            tilde_k = padded_k
        
        return tilde_q, tilde_k, padded_v
    
    def _get_positional_attn_map(self, N, step=1, gamma=None, shift=1) :
        ''' returned tensor shape [H, N, N] '''
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        positions = torch.arange(0, N, step, dtype=dtype, device=device)
        diff = torch.clamp(positions.unsqueeze(1) - positions.unsqueeze(0), min=0) + shift
        if gamma is None :
            gamma = self._get_gamma()
        exp_x = torch.exp(torch.einsum('...,h->h...', diff, -gamma))
        return exp_x

    def chunkwise_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) :
        chunk_size = self._chunk_size
        n_chunks = v.size(1) // chunk_size
        
        gamma = self._get_gamma()
        chunkwise_decay = torch.exp(-gamma * chunk_size).unsqueeze(0)
        decay = torch.exp(-gamma)
        inchunk_decay = (self._get_positional_attn_map(chunk_size, shift=0) * self._chunkwise_casual_mask).unsqueeze(0)
        position_frwd = torch.arange(1, chunk_size + 1, dtype=q.dtype, device=q.device)
        inchunk_decay_frwd = torch.exp(-gamma[None, :] * position_frwd[:, None]).unsqueeze(0)
        inchunk_decay_bkwd = torch.exp(-gamma[None, :] * (chunk_size - position_frwd)[:, None]).unsqueeze(0)
        
        chunkwise_forward_fn = linear_attn_fn.chunkwise_parallel_forward
        
        return chunkwise_forward_fn(
            q=q*decay[None, None, :, None],
            k=k,
            v=v,
            chunk_size=chunk_size,
            chunkwise_decay=chunkwise_decay.unsqueeze(0).repeat(n_chunks, 1, 1),
            inchunk_decay=inchunk_decay.unsqueeze(0).repeat(n_chunks, 1, 1, 1, 1),
            inchunk_decay_frwd=inchunk_decay_frwd.unsqueeze(0).repeat(n_chunks, 1, 1, 1),
            inchunk_decay_bkwd=inchunk_decay_bkwd.unsqueeze(0).repeat(n_chunks, 1, 1, 1),
            # eval_chunkwise_decay=lambda x: chunkwise_decay,
            # eval_inchunk_decay=lambda x: inchunk_decay,
            # eval_inchunk_decay_frwd=lambda x: inchunk_decay_frwd,
            # eval_inchunk_decay_bkwd=lambda x: inchunk_decay_bkwd,
        )
           
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        current_length: int,
        past_lengths: torch.Tensor,
        is_v_padded: bool = False,
        cache = None,
        early_lengths: Optional[torch.Tensor] = None,
        early_timestamps: Optional[torch.Tensor] = None,
        return_cache_states: bool = False,
    ) :
        B, n = all_timestamps.shape[0], current_length
        h, d = self._num_heads, self._head_dim
        
        tilde_q, tilde_k, padded_v = self._input_preprocess(
            q=q, 
            k=k, 
            v=v, 
            current_length=current_length,
            x_offsets=x_offsets, 
            all_timestamps=all_timestamps, 
            invalid_attn_mask=invalid_attn_mask, 
            is_v_padded=is_v_padded
        )
         
        if self._chunk_size is None :
            qk_attn = torch.einsum(
                "bnhd,bmhd->bhnm",
                tilde_q,
                tilde_k,
            )
            ts_attn = self._get_positional_attn_map(n)
                
            ts_attn = ts_attn * invalid_attn_mask.unsqueeze(0).unsqueeze(0)
            qk_attn = qk_attn * ts_attn
            
            raw_output = torch.einsum(
                "bhnm,bmhd->bnhd",
                qk_attn,
                padded_v
            ).reshape(B, n, -1)
        else :
            raw_output = self.chunkwise_forward(q=tilde_q, k=tilde_k, v=padded_v)
        
        return raw_output, None

def get_decay_attn_map(log_decay, swap_head_axis=False) :
    cumsum_log_decay = torch.cumsum(log_decay, dim=1)
    # decay_{i, j} = \prod_{k=j+1}^i decay_j
    if not swap_head_axis :
        return torch.clamp(cumsum_log_decay.unsqueeze(2) - cumsum_log_decay.unsqueeze(1), min=0)
    cumsum_log_decay = rearrange(cumsum_log_decay, 'b n h -> b h n')
    return torch.clamp(cumsum_log_decay.unsqueeze(3) - cumsum_log_decay.unsqueeze(2), min=0)
        

def get_ext_decay_attn_map(log_decay, swap_head_axis=True) :
    # decay_{i, j} = \sum_{i <= k <= j} d_k
    ext_log_decay = torch.concat([torch.zeros_like(log_decay[:, 0:1]), log_decay], dim=1)
    cumsum_log_decay = torch.cumsum(ext_log_decay, dim=1)
    if swap_head_axis :
        cumsum_log_decay = rearrange(cumsum_log_decay, 'b n h -> b h n')
        attn_map = torch.clamp(cumsum_log_decay[:, :, 1:, None] - cumsum_log_decay[:, :, None, :-1], min=0)
    else :
        raise NotImplementedError()
    return attn_map

class LinearTemporalChannel (nn.Module) :
    def __init__(
        self, 
        linear_dim, 
        num_heads,
        base,
        start_index=0,
        base_stride=1,
        chunk_size=None,
        use_proj=True,
        learnable_gamma=False,
        use_augment_connection=False,
        no_temporal_qk=False,
    ):
        super().__init__()
        logging.info(f'multi-head krab, chunk_size={chunk_size}')
        self._num_heads = num_heads
        self._use_proj = use_proj
        self._linear_dim = linear_dim
        self._pi2 = torch.pi * 2
        self.register_buffer('_intervals', torch.pow(base, torch.arange(start_index, start_index+num_heads*base_stride, base_stride, dtype=torch.long)))
        logging.info(f'channel_t {self._intervals}')
        self.register_buffer('_scale_factor', self._pi2 * torch.pow(1/base, torch.arange(start_index, start_index+num_heads*base_stride, base_stride, dtype=torch.float32)))
        if not learnable_gamma :
            self.register_buffer('_gamma', torch.empty(1, dtype=torch.float32).fill_(0))
        else :
            logging.info('channel_t: learnable gamma!')
            self._gamma = torch.nn.Parameter(torch.empty(num_heads, dtype=torch.float32).normal_(std=0.02), requires_grad=True)
        
        self._no_temporal_qk = no_temporal_qk
        if no_temporal_qk :
            logging.info('no_temporal_qk!')
        
        self.use_augment_connection = use_augment_connection
        if use_augment_connection :
            logging.info('channel t: use aug conn.')
            self.alpha = torch.nn.Parameter(torch.empty(num_heads * 2).normal_(mean=0, std=0.02), requires_grad=True)
            self.beta = torch.nn.Parameter(torch.empty(num_heads * 2).fill_(1.), requires_grad=True)
        
        if self._use_proj :
            self.proj_v = nn.Linear(linear_dim, linear_dim, bias=False)
        
        self._chunk_size = chunk_size
        
        if chunk_size is not None :
            self.register_buffer(
                '_chunkwise_casual_mask', 
                torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool))
            )
    
    def _get_decay(self, diff, gamma=None) :
        if gamma is None :
            gamma = F.sigmoid(self._gamma)
        hdiff = torch.einsum('...,h->...h', diff, self._scale_factor)
        return torch.pow(gamma, hdiff)
    
    def _get_query_key(
        self,
        all_timestamps: torch.Tensor,
        buffering: Optional[Dict[str, torch.Tensor]],
        gamma: Optional[torch.Tensor] = None,
        use_decay_q: bool = True,
    ) :
        if buffering is not None and 'query' in buffering :
            return buffering['query'], buffering['key']
        if self._no_temporal_qk :
            B, n = all_timestamps.shape
            h = self._intervals.shape[-1]
            q = torch.ones(B, n, h * 2, 2, dtype=torch.float32, device=all_timestamps.device) / math.sqrt(2)
            return q, q
        if gamma is None :
            gamma = F.sigmoid(self._gamma)
        theta_t = all_timestamps[:, :, None] % self._intervals[None, None, :] * self._scale_factor[None, None, :]
        cos_t = torch.cos(theta_t)
        sin_t = torch.sin(theta_t)
        # (B, n, h, 2)
        k = torch.stack([cos_t, sin_t], dim=3).repeat(1, 1, 2, 1)
            
        q_sin = torch.stack([sin_t[:, 1:], -cos_t[:, 1:]], dim=3)
        q_cos = torch.stack([cos_t[:, 1:], sin_t[:, 1:]], dim=3)
        q = torch.concat([q_sin, q_cos], dim=2)
        q = torch.concat([q, q[:, -2:-1]], dim=1)
        if use_decay_q :
            ext_timestamps = torch.concat([all_timestamps, all_timestamps[:, -2:-1]], dim=-1)
            decay_q = self._get_decay(torch.clamp(ext_timestamps[:, 1:] - ext_timestamps[:, :-1], min=0), gamma=gamma).repeat(1, 1, 2)
            q = q * decay_q.unsqueeze(-1)
        return q, k
    
    def _get_query_key_inference(
        self,
        last_timestamps: torch.Tensor,
        current_timestamps: torch.Tensor,
        target_timestamps: torch.Tensor,
        buffering: Optional[Dict[str, torch.Tensor]],
        gamma: Optional[torch.Tensor] = None,
    ) :
        '''
          last_timestamps, current_timestamps, target_timestamps: (batch_size, )
        '''
        if buffering is not None and 'query_inference' in buffering :
            return buffering['query_inference'], buffering['key_inference'], buffering['decay_inference']
        if gamma is None :
            gamma = F.sigmoid(self._gamma)
        theta_q = target_timestamps[:, None] % self._intervals[None, :] * self._scale_factor[None, :]
        theta_k = current_timestamps[:, None] % self._intervals[None, :] * self._scale_factor[None, :]
        k = torch.stack([torch.cos(theta_k), torch.sin(theta_k)], dim=2).repeat(1, 2, 1)
        
        cos_theta_q, sin_theta_q = torch.cos(theta_q), torch.sin(theta_q)
        q_sin = torch.stack([sin_theta_q, -cos_theta_q], dim=2)
        q_cos = torch.stack([cos_theta_q, sin_theta_q], dim=2)
        q = torch.concat([q_sin, q_cos], dim=1)
        
        decay_q = self._get_decay(target_timestamps - current_timestamps, gamma=gamma)
        q = q * decay_q.repeat(1, 2).unsqueeze(dim=-1)
        
        decay_inference = self._get_decay(current_timestamps - last_timestamps)
        return q, k, decay_inference.repeat(1, 2)
        
    def chunkwise_forward(
        self,
        v: torch.Tensor,
        all_timestamps: torch.Tensor,
        x_offsets: torch.Tensor,
        log_decay: torch.Tensor,
        buffering: Optional[Dict[str, torch.Tensor]] = None,
    ) :
        if buffering is None :
            B, n = all_timestamps.shape
            interval = torch.clamp(all_timestamps[:, 1:] - all_timestamps[:, :-1], min=0)
            interval = F.pad(interval, (0, 1))
            hinterval = torch.einsum('h,bn->bnh', self._scale_factor, interval)

            gamma = F.sigmoid(self._gamma)
            log_gamma = torch.log(gamma)
            q, k = self._get_query_key(all_timestamps, None, gamma, False)
            padded_log_decay = jagged.jagged_to_padded_dense(log_decay, [x_offsets], [n])
            log_decay_pos = ((hinterval * -log_gamma[None, None, :]).repeat(1, 1, 2) * padded_log_decay)
        
            buffering = {
                'key': k,
                'query': q,
                'log_decay_pos': log_decay_pos,
            }
        else :
            k = buffering['key']
            q = buffering['query']
            log_decay_pos = buffering['log_decay_pos']
            
        return (
            linear_attn.chunkwise_forward(
                q=q,
                k=k,
                v=v,
                log_decay_step=log_decay_pos,
                chunk_size=self._chunk_size,
            ),
            buffering,
        )
        
    def forward(
        self,
        q: None,
        k: None,
        v: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        past_lengths: torch.Tensor,
        is_v_padded: bool = False,
        buffered_attn_map: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache = None,
        target_timestamps: Optional[torch.Tensor] = None,
        return_cache_states: bool = False,
        record_attn_map: bool = False,
        log_decay = None,
    ) :
        dim = self._linear_dim
        B, n = all_timestamps.shape[0], all_timestamps.shape[1]
        
        if self._use_proj :
            v = self.proj_v(v)
        
        if is_v_padded :
            padded_v = v.reshape(B, n, dim)
            v = None
        else :
            padded_v = jagged.jagged_to_padded_dense(
                values=v, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
            )
        
        batch_id = torch.arange(B, device=padded_v.device)
        x = padded_v.reshape(B, n, 2 * self._num_heads, -1)
       
        ext_timestamps = torch.concat([all_timestamps, all_timestamps[:, -1: ]], dim=-1)
        gamma = F.sigmoid(self._gamma)
        
        buffering = {}
        if self._chunk_size is None :
            if buffered_attn_map is not None :
                attn_maps = buffered_attn_map['attn_maps']
                decay = buffered_attn_map['decay']
            else :
                interval = torch.clamp(all_timestamps[:, 1:] - all_timestamps[:, :-1], min=0)
                interval = F.pad(interval, (0, 1))
                hinterval = torch.einsum('h,bn->bnh', self._scale_factor, interval)
                
                q, k = self._get_query_key(all_timestamps, None, gamma, False)
                log_gamma = torch.log(gamma)
                padded_log_decay = jagged.jagged_to_padded_dense(log_decay, [x_offsets], [n])
                log_decay_pos = ((hinterval * -log_gamma[None, None, :]).repeat(1, 1, 2) * padded_log_decay)
                log_decay_map = get_ext_decay_attn_map(log_decay_pos, swap_head_axis=True)
                decay_map = torch.exp(-log_decay_map) * invalid_attn_mask.unsqueeze(0).unsqueeze(0)
                
                attn_maps_sinusoid_parallel = torch.einsum('bnhd,bmhd->bhnm', q, k)
                attn_maps = attn_maps_sinusoid_parallel * decay_map
                buffering.update({
                    'attn_maps': attn_maps,
                    'decay': decay_map,
                })
            
            if self._no_multihead :
                fused_attn_map = torch.einsum('bhnm,h->bnm', attn_maps, self._ws_t)
                y = torch.einsum('bnm,bmhd->bnhd', fused_attn_map, x).reshape(B, n, -1)
            else :
                y = torch.einsum('bhnm,bmhd->bnhd', attn_maps, x)
                if self.use_augment_connection :
                    y_aug = torch.einsum('bnhd,h->bnhd', y, self.alpha) + torch.einsum('bnhd,h->bnhd', x, self.beta)
                    y = y_aug.reshape(B, n, -1)
                else :
                    y = y.reshape(B, n, -1)
                
        if self._chunk_size is not None :
            y, buffering_chunk = self.chunkwise_forward(
                x,
                all_timestamps,
                x_offsets,
                log_decay,
                buffered_attn_map,
            )
            buffering.update(buffering_chunk)
            
            if self.use_augment_connection :
                yh = einops.rearrange(y, 'b n (h d) -> b n h d', h=2*self._num_heads)
                y_aug = torch.einsum('bnhd,h->bnhd', yh, self.alpha) + torch.einsum('bnhd,h->bnhd', x, self.beta)
                y = y_aug.reshape(B, n, -1)
            
        if buffered_attn_map is not None :
            buffering = buffered_attn_map
        
        return y, buffering, None

class LinearPositionalChannel(nn.Module) :
    def __init__(
        self,
        max_seq_len,
        embedding_dim = 32,
        aug_current: bool = True,
        use_proj: bool = True,
        value_dim: Optional[int] = None,
        chunk_size = None,
        sinusoidal_base = 10000,
        is_identical = False,
        num_heads = 1,
    ) :
        super().__init__()
        
        n = max_seq_len
        if chunk_size is not None :
            n = (n + chunk_size - 1) // chunk_size * chunk_size
        half_dim = embedding_dim // 2
        self._is_identical = is_identical
        
        self._use_proj = use_proj
        if use_proj :
            assert value_dim is not None
            self._proj_p = torch.nn.Linear(value_dim, value_dim, bias=False)
        
        theta = torch.exp(-math.log(sinusoidal_base) * torch.arange(half_dim) / half_dim)
        bases = torch.arange(n)[:, None] * theta[None, :]
        emb_weight = torch.concat([torch.sin(bases), torch.cos(bases)], dim=1)
        self._emb = torch.nn.Parameter(emb_weight, requires_grad=True)
        self._chunk_size = chunk_size
        
        self._aug_current = aug_current
        if aug_current :
            self._alpha = torch.nn.Parameter(torch.empty(1, dtype=torch.float32).normal_(std=0.02))
            self._beta = torch.nn.Parameter(torch.empty(1, dtype=torch.float32).fill_(1))
        
        if chunk_size is not None :
            self.register_buffer(
                '_chunkwise_casual_mask', 
                torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool))
            )
    
    def chunkwise_forward(
        self,
        v,
    ) :
        q = self._emb[None, :, None, :] 
        k = q / (self._emb.shape[-1] // 2)
        
        chunkwise_forward_fn = linear_attn_fn.chunkwise_parallel_forward
        
        chunkwise_decay = torch.ones(1, 1, dtype=q.dtype, device=q.device)
        inchunk_decay = self._chunkwise_casual_mask[None, None, :]
        inchunk_decay_frwd = torch.ones(1, 1, 1, dtype=q.dtype, device=q.device)
        inchunk_decay_bkwd = torch.ones(1, 1, 1, dtype=q.dtype, device=q.device)
        
        n_chunks = v.size(1) // self._chunk_size
        
        return chunkwise_forward_fn(
            q=q,
            k=k,
            v=v.unsqueeze(2),
            chunk_size=self._chunk_size,
            chunkwise_decay=chunkwise_decay.unsqueeze(0).repeat(n_chunks, 1, 1),
            inchunk_decay=inchunk_decay.unsqueeze(0).repeat(n_chunks, 1, 1, 1, 1),
            inchunk_decay_frwd=inchunk_decay_frwd.unsqueeze(0).repeat(n_chunks, 1, 1, 1),
            inchunk_decay_bkwd=inchunk_decay_bkwd.unsqueeze(0).repeat(n_chunks, 1, 1, 1),
        )
        
    
    def forward(
        self,
        q: None,
        k: None,
        v: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        past_lengths: torch.Tensor,
        is_v_padded: bool = False,
        buffered_attn_map: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache = None,
        target_timestamps: Optional[torch.Tensor] = None,
        # early_lengths: Optional[torch.Tensor] = None,
        # early_timestamps: Optional[torch.Tensor] = None,
        return_cache_states: bool = False,
        record_attn_map: bool = False,
        log_decay: Optional[torch.Tensor] = None,   # (B, 1)
    ) :
        if self._use_proj :
            n = all_timestamps.shape[1]
            assert not is_v_padded
            v = self._proj_p(v)
            v = jagged.jagged_to_padded_dense(v, [x_offsets], [n])
        else :
            assert is_v_padded
            
        if self._chunk_size is None :
            B, n = v.shape[0: 2] 
            attn_weights = torch.einsum('nd,md->nm', self._emb, self._emb) / (self._emb.shape[-1] // 2)  * invalid_attn_mask
            
            if log_decay is not None :
                padded_log_decay = jagged.jagged_to_padded_dense(log_decay, [x_offsets], [n])
                decay_map = torch.exp(-get_decay_attn_map(padded_log_decay))
                attn_weights = torch.einsum('nm,bnm->bnm', attn_weights, decay_map.squeeze(-1))
                y = torch.einsum('bnm,bmd->bnd', attn_weights, v)
                
            else :
                y = torch.einsum('nm,bmd->bnd', attn_weights, v)
        else :
            y = self.chunkwise_forward(v)
            
        if self._aug_current :
            y = y * self._alpha + v * self._beta
        return y, None, None
