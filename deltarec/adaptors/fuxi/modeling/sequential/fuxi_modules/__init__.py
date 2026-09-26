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

from .rab import *

from .attn import (
    Retention,
    LinearPositionalChannel,
    LinearTemporalChannel,
)
    
# FFNs    
    
class MultistageFeedforwardNeuralNetwork(torch.nn.Module) :
    def __init__(
        self, 
        ams_output_size, 
        input_size, 
        hidden_size, 
        output_size, 
        dropout_ratio: float,
        bias: bool = False, 
        single_stage: bool = False,
        epsilon: float = 1e-6,
        dtype = torch.float32
    ) :
        super(MultistageFeedforwardNeuralNetwork, self).__init__()
        self.lin0 = torch.nn.Linear(ams_output_size, input_size, dtype=dtype, bias=bias)
        self.is_single_stage = single_stage
        self.dropout_ratio = dropout_ratio
        self.input_size = input_size
        self.eps = epsilon
        self.dtype = dtype
        if not single_stage :
            self.lin1 = torch.nn.Linear(input_size, hidden_size, bias=bias, dtype=dtype)
            self.lin2 = torch.nn.Linear(hidden_size, output_size, bias=bias, dtype=dtype)
            self.lin3 = torch.nn.Linear(input_size, hidden_size, bias=bias, dtype=dtype)
    
    def forward(self, X, X0, past_length=None) :
        X = (
            self.lin0(
                F.dropout(
                    X.to(self.dtype),
                    p = self.dropout_ratio,
                    training = self.training
                )
            ) + X0
        )
        if not self.is_single_stage :
            normed_X = F.rms_norm(X, normalized_shape=[self.input_size], eps=self.eps)
            normed_X = F.dropout(
                normed_X,
                p = self.dropout_ratio,
                training = self.training
            )
            X1 = F.silu(self.lin1(normed_X)) * self.lin3(normed_X)
            X = self.lin2(X1) + X
        return X.to(torch.float32)
    
    def init(self) :
        with torch.no_grad() :
            self.lin0.weight.normal_(mean=0, std=0.02)
            if not self.is_single_stage :
                self.lin1.weight.normal_(mean=0, std=0.02)
                self.lin2.weight.normal_(mean=0, std=0.02)
                self.lin3.weight.normal_(mean=0, std=0.02)
