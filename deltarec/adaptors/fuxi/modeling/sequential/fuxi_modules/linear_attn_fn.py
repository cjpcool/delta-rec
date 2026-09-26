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
import torch.nn as nn
from torch.autograd import Function
from typing import Tuple, Callable
from einops import rearrange

class ChunkwiseHiddenStateFunction(Function):
    @staticmethod
    def forward(ctx, inchunk_kv, chunkwise_decay):
        """
        前向传播：计算分块隐藏状态
        
        参数:
            inchunk_kv: [num_chunks, B, h, d_attn, d_v] 块内KV乘积
            chunkwise_decay: [num_chunks, B, h] 块间衰减系数
            
        返回:
            hidden_states: [num_chunks, B, h, d_attn, d_v] 隐藏状态序列
        """
        num_chunks, B, h, d_attn, d_v = inchunk_kv.shape
        
        # 初始化隐藏状态
        current_hidden_state = torch.zeros(
            B, h, d_attn, d_v, 
            dtype=inchunk_kv.dtype, 
            device=inchunk_kv.device
        )
        
        hidden_states = torch.zeros(
            num_chunks, B, h, d_attn, d_v,
            dtype=inchunk_kv.dtype, 
            device=inchunk_kv.device
        )
        
        decay_expanded = chunkwise_decay.unsqueeze(-1).unsqueeze(-1)
        
        # 前向传播循环
        for i in range(num_chunks):
            hidden_states[i] = current_hidden_state
            
            if i != num_chunks - 1:
                # 更新隐藏状态: h_t = h_{t-1} * decay_t + kv_t
                current_hidden_state = torch.addcmul(
                    inchunk_kv[i], 
                    current_hidden_state, 
                    decay_expanded[i]
                )
        
        # 保存中间状态用于反向传播
        ctx.num_chunks = num_chunks
        # 保存中间结果用于反向传播
        ctx.save_for_backward(inchunk_kv, chunkwise_decay, hidden_states)
        
        return hidden_states

    @staticmethod
    def backward(ctx, grad_hidden_states):
        """
        反向传播：计算梯度
        
        参数:
            grad_hidden_states: [num_chunks, B, h, d_attn, d_v] 隐藏状态的梯度
            
        返回:
            grad_inchunk_kv: inchunk_kv的梯度
            grad_chunkwise_decay: chunkwise_decay的梯度
        """
        inchunk_kv, chunkwise_decay, state_list = ctx.saved_tensors
        num_chunks = ctx.num_chunks
        B, h, d_attn, d_v = inchunk_kv.shape[1:]
        
        # 初始化梯度
        grad_inchunk_kv = torch.zeros_like(inchunk_kv)
        grad_chunkwise_decay = torch.zeros_like(chunkwise_decay)
        
        decay_expanded = chunkwise_decay.unsqueeze(-1).unsqueeze(-1)
        
        # 初始化隐藏状态的梯度
        grad_current_hidden = torch.zeros(
            B, h, d_attn, d_v, 
            dtype=grad_hidden_states.dtype,
            device=grad_hidden_states.device
        )
        
        grad_decay_shape = [B, h]
        broadcast_dims = [
            i for i in range(2)
                if grad_chunkwise_decay.shape[i+1] != grad_decay_shape[i]
        ]
        
        # 反向传播循环（从最后一个时间步到第一个）
        for i in range(num_chunks-1, -1, -1):
            # 当前时间步的梯度来自两部分：
            # 1. 直接来自输出的梯度 (grad_hidden_states[i])
            # 2. 来自下一个时间步的梯度 (grad_current_hidden)
            grad_h = grad_hidden_states[i] + grad_current_hidden
            
            if i != num_chunks - 1:
                # 计算chunkwise_decay的梯度
                # ∂L/∂decay_i = ∂L/∂h_i * ∂h_i/∂decay_i = grad_current_hidden * h_{i-1}
                prev_hidden = state_list[i]  # h_{i-1}
                grad_decay = torch.einsum('bhkv,bhkv->bh', grad_current_hidden, prev_hidden)
                # grad_decay = (grad_current_hidden * prev_hidden).sum(dim=[-1, -2])  # 求和到[B, h]
                if len(broadcast_dims) > 0 :
                    grad_decay = grad_decay.sum(dim=broadcast_dims)
                # print(grad_chunkwise_decay.shape)
                grad_chunkwise_decay[i] = grad_decay
                
                # 计算inchunk_kv的梯度
                # ∂L/∂kv_i = ∂L/∂h_i * ∂h_i/∂kv_i = grad_current_hidden * 1
                grad_inchunk_kv[i] = grad_current_hidden
                
                # 计算前一个隐藏状态的梯度
                # ∂L/∂h_{i-1} = ∂L/∂h_i * ∂h_i/∂h_{i-1} = grad_current_hidden * decay_i
                # grad_current_hidden = grad_current_hidden * decay_expanded[i] + grad_hidden_states[i]
         
                grad_current_hidden = torch.addcmul(
                    grad_hidden_states[i], 
                    grad_current_hidden, 
                    decay_expanded[i]
                )

            else:
                # 最后一个时间步没有后续状态，只来自直接梯度
                grad_current_hidden = grad_h
            
            # 更新当前梯度为前一个时间步的梯度
            grad_current_hidden = grad_h if i == num_chunks - 1 else grad_current_hidden
        
        return grad_inchunk_kv, grad_chunkwise_decay

# 使用自定义算子的封装函数
def eval_chunkwise_hidden_state_fn(inchunk_kv, chunkwise_decay):
    return ChunkwiseHiddenStateFunction.apply(inchunk_kv, chunkwise_decay)

def chunkwise_parallel_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunkwise_decay: torch.Tensor,
    inchunk_decay: torch.Tensor,
    inchunk_decay_frwd: torch.Tensor,
    inchunk_decay_bkwd: torch.Tensor,
    chunk_size: int,
) :
    B, n, h, d_v = v.shape
    d_attn = q.size(-1)
    assert n % chunk_size == 0, "n must be divisible by chunk_size"
    num_chunks = n // chunk_size
    q = rearrange(q, 'b (s c) h d -> s b c h d', c=chunk_size)
    k = rearrange(k, 'b (s c) h d -> s b c h d', c=chunk_size)
    v = rearrange(v, 'b (s c) h d -> s b c h d', c=chunk_size)
    
    tilde_k = k * inchunk_decay_bkwd.unsqueeze(-1)
    inchunk_kv = torch.einsum('sbchk,sbchv->sbhkv', tilde_k, v)
    
    if q.requires_grad :
        hidden_states = eval_chunkwise_hidden_state_fn(inchunk_kv, chunkwise_decay)
    else :
        state_list = []
        current_hidden_state = torch.zeros(B, h, d_attn, d_v, dtype=q.dtype, device=q.device)
        for i, (kv, cs_decay) in enumerate(zip(inchunk_kv, chunkwise_decay)) :
            state_list.append(current_hidden_state)
            if i != num_chunks - 1 :
                current_hidden_state = current_hidden_state * cs_decay[:, :, None, None] + kv
        hidden_states = torch.stack(state_list, dim=0)
    
    inchunk_attnmap = torch.einsum('sbnhd,sbmhd->sbhnm', q, k) * inchunk_decay
    y2 = torch.einsum('sbhnm,sbmhd->sbnhd', inchunk_attnmap, v)
    
    tilde_q = q * inchunk_decay_frwd.unsqueeze(-1)
    y1 = torch.einsum('sbchk,sbhkv->sbchv', tilde_q, hidden_states)
    
    y = rearrange(y1 + y2, 's b c h d -> b (s c) (h d)')
    return y
    
if __name__ == "__main__":
    # ==================== Consistency Check ====================
    def _test_forward_consistency():
        from .linear_attn import chunkwise_forward
        
        torch.manual_seed(0)
        B, n, h, d_attn, d_v = 2, 64, 3, 4, 5
        chunk_size = 4
        
        q = torch.randn(B, n, h, d_attn)
        k = torch.randn(B, n, h, d_attn)
        v = torch.randn(B, n, h, d_v)
        
        def make_eval(tensor_list):
            return lambda i: tensor_list[i]
        
        # Precompute decay parameters
        num_chunks = n // chunk_size
        cw_list = [torch.randn(B, h) for _ in range(num_chunks)]
        ic_list = [torch.randn(B, h, chunk_size, chunk_size) for _ in range(num_chunks)]
        ic_frwd_list = [torch.randn(B, chunk_size, h) for _ in range(num_chunks)]
        ic_bkwd_list = [torch.randn(B, chunk_size, h) for _ in range(num_chunks)]
        
        # Original
        out1 = chunkwise_forward(
            q, k, v,
            make_eval(cw_list),
            make_eval(ic_list),
            make_eval(ic_frwd_list),
            make_eval(ic_bkwd_list),
            chunk_size
        )
        
        # Custom function
        out2 = chunkwise_parallel_forward(
            q, k, v,
            torch.stack(cw_list, dim=0),
            torch.stack(ic_list, dim=0),
            torch.stack(ic_frwd_list, dim=0),
            torch.stack(ic_bkwd_list, dim=0),
            chunk_size
        )
        
        print("Forward consistency check:")
        print(f"Max difference: {torch.max(torch.abs(out1 - out2)).item():.2e}")
        assert torch.allclose(out1, out2, atol=1e-6), "Forward outputs do not match!"
        print("✅ Forward consistency PASSED!")

    # ==================== Gradcheck ====================
    def test_gradcheck():
        torch.manual_seed(42)
        B, n, h, d_attn, d_v = 5, 12, 2, 3, 4
        chunk_size = 4
        num_chunks = n // chunk_size
        
        # q = torch.randn(B, n, h, d_attn, dtype=torch.double, requires_grad=True)
        # k = torch.randn(B, n, h, d_attn, dtype=torch.double, requires_grad=True)
        # v = torch.randn(B, n, h, d_v, dtype=torch.double, requires_grad=True)
        
        # cw = torch.randn(B, num_chunks, h, dtype=torch.double, requires_grad=True)
        # ic = torch.randn(B, h, num_chunks, chunk_size, chunk_size, dtype=torch.double, requires_grad=True)
        # ic_frwd = torch.randn(B, num_chunks, chunk_size, h, dtype=torch.double, requires_grad=True)
        # ic_bkwd = torch.randn(B, num_chunks, chunk_size, h, dtype=torch.double, requires_grad=True)
        
        # inputs = (q, k, v, cw, ic, ic_frwd, ic_bkwd, chunk_size)
        
        inchunk_kv = torch.randn(num_chunks, B, h, d_attn, d_v, dtype=torch.double, requires_grad=True)
        chunkwise_decay = torch.randn(num_chunks, 1, 1, dtype=torch.double, requires_grad=True)
        
        print("Running gradcheck...")
        try:
            test_passed = torch.autograd.gradcheck(
                ChunkwiseHiddenStateFunction.apply,
                (inchunk_kv, chunkwise_decay),
                eps=1e-6,
                atol=1e-5,
                rtol=1e-3,
                nondet_tol=0.0
            )
            if test_passed:
                print("✅ Gradcheck PASSED!")
            else:
                print("❌ Gradcheck FAILED!")
        except Exception as e:
            print(f"❌ Gradcheck error: {e}")
            raise

    
    _test_forward_consistency()
    print("\n" + "="*50 + "\n")
    test_gradcheck()
