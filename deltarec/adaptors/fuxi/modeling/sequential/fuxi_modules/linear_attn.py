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


import torch
import logging
from torch import nn

import torch.nn.functional as F

from typing import Callable, Tuple

from einops import rearrange

from . import linear_attn_fn

def chunkwise_forward(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  log_decay_step: torch.Tensor,
  chunk_size: int,
) :
  '''
    decay_step: (batch_size, num_heads, seq_len)
      decay_(i->j) = \prod_{k=i}^j decay_step[k]
  '''
  B, n, h, d_v = v.shape
  d_attn = q.size(-1)
  num_chunks = n // chunk_size
  assert n % chunk_size == 0, "n must be divisible by chunk_size"
  q = rearrange(q, 'b (s c) h d -> s b c h d', c=chunk_size)
  k = rearrange(k, 'b (s c) h d -> s b c h d', c=chunk_size)
  v = rearrange(v, 'b (s c) h d -> s b c h d', c=chunk_size)
  
  log_decay_step = rearrange(log_decay_step, 'b (s c) h -> s b h c', c=chunk_size)
  log_cw_decay = log_decay_step.sum(dim=-1)
  log_ic_decay_frwd = log_decay_step.cumsum(dim=-1)
  
  foo = F.pad(log_ic_decay_frwd, (1, 0))
  log_ic_decay_bkwd = foo[:, :, :, -1:] - foo[:, :, :, :-1]
    
  ic_decay_bkwd = torch.exp(-rearrange(log_ic_decay_bkwd, 's b h c -> s b c h'))
  tilde_k = k * ic_decay_bkwd.unsqueeze(-1)
  ic_kv = torch.einsum('sbchk,sbchv->sbhkv', tilde_k, v)
  
  cw_decay = torch.exp(-log_cw_decay)
  if False :
    state_list = []
    current_hidden_state = torch.zeros(B, h, d_attn, d_v, dtype=q.dtype, device=q.device)
    for i, (kv, cs_decay) in enumerate(zip(ic_kv, cw_decay)) :
      state_list.append(current_hidden_state)
      if i != num_chunks - 1 :
        current_hidden_state = current_hidden_state * cs_decay[:, :, None, None] + kv
    hidden_states = torch.stack(state_list, dim=0)
  else :
    hidden_states = linear_attn_fn.eval_chunkwise_hidden_state_fn(ic_kv, cw_decay)
  
  ic_decay_frwd = torch.exp(-rearrange(log_ic_decay_frwd, 's b h c -> s b c h'))
  tilde_q = q * ic_decay_frwd.unsqueeze(-1)
  y_cw = torch.einsum('sbchk,sbhkv->sbchv', tilde_q, hidden_states)
  
  log_ic_decay = torch.clamp(foo[:, :, :, 1:, None] - foo[:, :, :, None, :-1], min=0)
  ic_decay = torch.exp(-log_ic_decay) * torch.tril(torch.ones(chunk_size, chunk_size, device=q.device))[None, None, None, :]
  ic_attnmap = torch.einsum('sbnhd,sbmhd->sbhnm', q, k) * ic_decay
  y_ic = torch.einsum('sbhnm,sbmhd->sbnhd', ic_attnmap, v)
  
  y = rearrange(y_cw + y_ic, 's b c h v -> b (s c) (h v)')
  return y
