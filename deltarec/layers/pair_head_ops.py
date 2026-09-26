# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.
"""The PyTorch SwishLayerNorm and binary-task loss used by the pair head."""
import torch
from torch import nn
from torch.nn import functional as F

class SwishLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self._eps = eps
    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        return (x * torch.sigmoid(F.layer_norm(x, [x.shape[-1]], self.weight.float(), self.bias.float(), self._eps))).to(dtype)

def init_mlp_weights_optional_bias(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.)

def _compute_loss(task_offsets, causal_multitask_weights, mt_logits, mt_labels, mt_weights, has_multiple_task_types):
    if task_offsets != [0, 8, 8] or has_multiple_task_types:
        raise ValueError('Only the eight binary recommendation tasks are supported')
    losses = F.binary_cross_entropy_with_logits(mt_logits, mt_labels, reduction='none') * mt_weights
    return losses.sum(-1) / mt_weights.sum(-1).clamp(min=1.) * causal_multitask_weights
