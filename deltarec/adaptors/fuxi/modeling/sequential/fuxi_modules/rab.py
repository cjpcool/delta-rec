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
import torch.nn as nn
import torch.nn.functional as F

import einops
import logging

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
        # del all_timestamps
        n: int = self._max_seq_len
        t = F.pad(self._w[: 2 * n - 1], [0, n]).repeat(n)
        t = t[..., :-n].reshape(1, n, 3 * n - 2)
        r = (2 * n - 1) // 2
        return t[..., r:-r]
      
class FunctionalTemporalRelativeAttentionBias(RelativeAttentionBiasModule) :
    AcceptedFunctions: List[str] = ['linear', 'log', 'exp', 'sin', 'pow', 'mixed', 'nn', 'zero', 'spline']
    
    def __init__(
        self,
        # max_seq_len: int,
        func_type: Literal['linear', 'log', 'exp', 'sin', 'pow', 'mixed', 'nn', 'zero', 'fabric'],
    ) -> None:
        super().__init__()
        
        self._func_type = func_type
        
        if func_type == 'linear' :
            # a * x + b
            self._lin_a = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-0.01, 0.01))
            self._lin_b = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-0.2, 0.2))
        elif func_type == 'log' :
            # a * log(1 + b * x) + c
            self._log_a = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-0.01, 0.01))
            self._log_b = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(0.5, 1))
            self._log_c = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-0.05, 0.05))
        elif func_type == 'exp' :
            # a * exp(-b * x)
            self._exp_a = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-1, 1))
            self._exp_b = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-2, 0))
        elif func_type == 'sin' :
            # c * sin(a * x + b) + d
            self._sin_a = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-0.02, 0.02))
            self._sin_b = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-torch.pi, torch.pi))
            self._sin_c = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-2, 2))
            self._sin_d = nn.Parameter(torch.empty(1, dtype=torch.float32).zero_())
        elif func_type == 'pow' :
            # a * pow(x, b)
            self._pow_a = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-0.2, 0.2))
            self._pow_b = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(0.4, 0.8))
        elif func_type == 'mixed' :
            self._lin_a = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-0.01, 0.01))
            self._lin_b = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-0.2, 0.2))
            self._log_a = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-0.01, 0.01))
            self._log_b = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(0.5, 1))
            self._log_c = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-0.05, 0.05))
            self._exp_a = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-0.2, 0.2))
            self._exp_b = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-2, 0))
            self._sin_a = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-0.02, 0.02))
            self._sin_b = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-torch.pi, torch.pi))
            self._sin_c = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-2, 2))
            self._sin_d = nn.Parameter(torch.empty(1, dtype=torch.float32).zero_())
            self._pow_a = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-0.2, 0.2))
            self._pow_b = nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(0.4, 0.8))
        elif func_type == 'nn' :
            self._nn_a = nn.Linear(1, 5)
            self._nn_b = nn.Linear(5, 5)
            self._nn_c = nn.Linear(5, 1)
        elif func_type == 'zero' :
            pass
        elif func_type == 'spline':
            # Spline parameters
            self._spline_knots = nn.Parameter(torch.linspace(0, 1, 5))  # 5 control points
            self._spline_coeffs = nn.Parameter(torch.randn(5, dtype=torch.float32) * 0.1)
        else :
            raise Exception(f'Unknown function type {func_type}')
        
        self._func_map = {
            'linear': self.f_lin,
            'log': self.f_log,
            'exp': self.f_exp,
            'sin': self.f_sin,
            'pow': self.f_pow,
            'mixed': self.f_mix,
            'nn': self.f_nn,
            'zero': self.f_zero,
            'spline': self.f_spline,
        }
        
    def f_lin(self, x) :
        return self._lin_a * x + self._lin_b
    
    def f_log(self, x) :
        x = torch.relu(x)
        assert self._log_b > 0
        return self._log_a * torch.log(1 + self._log_b * x) + self._log_c
    
    def f_exp(self, x) :
        x = torch.relu(x)
        exp_b = torch.exp(self._exp_b)
        assert exp_b > 0
        return self._exp_a * torch.exp(-exp_b * x)
    
    def f_sin(self, x) :
        return self._sin_c * torch.sin(self._sin_a * x + self._sin_b) + self._sin_d
    
    def f_pow(self, x) :
        x = torch.relu(x) + 1
        return self._pow_a * torch.pow(x, -self._pow_b)
    
    def f_mix(self, x) :
        return (self.f_lin(x) + self.f_log(x) + self.f_exp(x) + self.f_sin(x) + self.f_pow(x)) / 5    
    
    def f_nn(self, x) :
        x = self._nn_a(x.to(torch.float32).unsqueeze(-1))
        x = self._nn_b(torch.sin(x))
        x = F.silu(x)
        return self._nn_c(x).squeeze(-1)
    
    def f_zero(self, x) :
        return torch.zeros_like(x, device=x.device)
        
    def f_spline(self, x):
        # 对输入值进行对数变换，使其分布更加均匀def f_spline(self, x):
        # 对输入值进行对数变换，使其分布更加均匀
        x = torch.log(torch.relu(x) + 1)
        
        # 归一化到 [0, 1] 范围
        x_min = x.min()
        x_max = x.max()
        x_norm = (x - x_min) / (x_max - x_min + 1e-8)
        
        # 计算样条插值
        t = x_norm.unsqueeze(-1)  # Shape: [..., 1]
        knots = self._spline_knots  # Shape: [5]
        # knots = knots.view(1, 1, 1, -1)  # Shape: [1, 1, 1, 5]
        coeffs = self._spline_coeffs  # Shape: [5]
        
        # 找到每个 x 所属的区间
        indices = torch.searchsorted(knots, t, right=True) - 1
        indices = indices.clamp(min=0, max=knots.size(-1) - 2)
        
        # 获取对应的控制点和系数
        t0 = torch.gather(knots.expand(t.size(0), t.size(1), t.size(2), -1), 3, indices)
        t1 = torch.gather(knots.expand(t.size(0), t.size(1), t.size(2), -1), 3, indices + 1)
        c0 = torch.gather(coeffs.view(1, 1, 1, -1).expand(t.size(0), t.size(1), t.size(2), -1), 3, indices)
        c1 = torch.gather(coeffs.view(1, 1, 1, -1).expand(t.size(0), t.size(1), t.size(2), -1), 3, indices + 1)
        
        # 计算插值权重
        alpha = (t - t0) / (t1 - t0 + 1e-8)
        
        # 插值计算
        return (c0 + alpha * (c1 - c0)).squeeze(-1)
        
    def f(self, x) :
        func = self._func_map[self._func_type]
        return func(x)
    
    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor :
        """
        Args:
            all_timestamps: (B, N).
        Returns:
            (B, N, N).
        """
        B = all_timestamps.size(0)
        N = all_timestamps.shape[-1]
        # N = self._max_seq_len
        # t = F.pad(self._pos_w[: 2 * N - 1], [0, N]).repeat(N)
        # t = t[..., :-N].reshape(1, N, 3 * N - 2)
        # r = (2 * N - 1) // 2

        # [B, N + 1] to simplify tensor manipulations.
        ext_timestamps = torch.cat(
            [all_timestamps, all_timestamps[:, N - 1 : N]], dim=1
        )
        # causal masking. Otherwise [:, :-1] - [:, 1:] works
        ext_timestamps = ext_timestamps[:, 1:].unsqueeze(2) - ext_timestamps[:, :-1].unsqueeze(1)
        
        rel_ts_bias = self.f(ext_timestamps)
        
        # rel_pos_bias = t[:, :, r:-r]
        return rel_ts_bias

class RelativeBucketedTimeBasedBias(RelativeAttentionBiasModule):
    """
    Bucketizes timespans based on ts(next-item) - ts(current-item).
    """

    def __init__(
        self,
        num_buckets: int,
        bucketization_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()

        self._ts_w = torch.nn.Parameter(
            torch.empty(num_buckets + 1).normal_(mean=0, std=0.02),
        )
        # self._pos_w = torch.nn.Parameter(
        #     torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02),
        # )
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
        N = all_timestamps.shape[-1]
        # t = F.pad(self._pos_w[: 2 * N - 1], [0, N]).repeat(N)
        # t = t[..., :-N].reshape(1, N, 3 * N - 2)
        # r = (2 * N - 1) // 2

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
        # rel_pos_bias = t[:, :, r:-r]
        rel_ts_bias = torch.index_select(
            self._ts_w, dim=0, index=bucketed_timestamps.view(-1)
        ).view(B, N, N)
        return rel_ts_bias
