"""Paper CWI uses interpolation of the COMPLETE transition, not of two gates.

This opt-in offline teacher kernel preserves the production binary endpoints
but fixes the fractional intervention derivative. It is deliberately a slow,
transparent reference; serving continues to use the existing binary kernels.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import math
from typing import Callable

import torch
from torch import nn

TEACHER_INTERVENTION = 'complete-transition-interpolation-v1'


def complete_transition(state, key, value, decay, beta, gate):
    """S + m(F(S)-S), with leading dimensions shared by state/key/value.

    state [...,Dk,Dv], key [...,Dk], value [...,Dv], decay/beta [...].
    gate may be a scalar or have broadcast-compatible leading dimensions.
    """
    decayed = decay[..., None, None] * state
    residual = value - (decayed * key[..., :, None]).sum(-2)
    updated = decayed + beta[..., None, None] * key[..., :, None] * residual[..., None, :]
    if gate is None:
        return updated
    gate = torch.as_tensor(gate, device=state.device, dtype=state.dtype)
    return state + gate[..., None, None] * (updated - state)


class CompleteTransitionReferenceKernel(nn.Module):
    """Drop-in packed GDR kernel for offline differentiation at m=1."""

    def __init__(self, eps=1e-6):
        super().__init__()
        if eps <= 0:
            raise ValueError('eps must be positive')
        self.eps = eps

    def forward(self, projected, initial_state=None, return_final_state=False):
        from deltarec.layers.gdr_kernels import (
            _l2_normalize, materialize_gdr_transition, validate_gdr_kernel_input,
        )
        from deltarec.layers.selective_gdr import GDRKernelOutput
        batch, heads, key_dim, value_dim = validate_gdr_kernel_input(projected, initial_state)
        dtype = torch.float64 if projected.q.dtype == torch.float64 else torch.float32
        q = _l2_normalize(projected.q.to(dtype), self.eps)
        k = _l2_normalize(projected.k.to(dtype), self.eps)
        # The original gate must NOT enter beta/decay materialization. It acts
        # exactly once on F(S)-S below; otherwise the residual creates an m^2 term.
        value, decay, beta = materialize_gdr_transition(replace(projected, event_gate=None), dtype)
        context, final = [], []
        offsets = projected.offsets.detach().cpu().tolist()
        for sequence in range(batch):
            state = (q.new_zeros(heads, key_dim, value_dim) if initial_state is None
                     else initial_state[sequence].to(dtype))
            for i in range(offsets[sequence], offsets[sequence + 1]):
                gate = None if projected.event_gate is None else projected.event_gate[i]
                state = complete_transition(state, k[i], value[i], decay[i], beta[i], gate)
                context.append(key_dim**-0.5 * torch.einsum('hk,hkv->hv', q[i], state))
            final.append(state)
        output = (torch.stack(context).to(projected.v.dtype) if context
                  else projected.v.new_empty((0, heads, value_dim)))
        states = torch.stack(final) if return_final_state else None
        return GDRKernelOutput(context=output, final_state=states)


@contextmanager
def complete_transition_teacher(model: nn.Module):
    """Temporarily select paper kernels in an official ResearchHSTUGDR host.

    The caller still supplies the per-event gates to the host/loss callback.
    The host is frozen/eval during teacher differentiation and restored on exit.
    This mutates only the supplied in-process model, never files or old jobs.
    """
    layers = [module for module in model.modules()
              if hasattr(module, 'reference_kernel') and hasattr(module, 'fla_kernel')
              and hasattr(module, 'kernel_backend')]
    if not layers:
        raise ValueError('model has no compatible official GDR layers')
    modes = [(module, module.training) for module in model.modules()]
    gradients = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    originals = [(layer, layer.reference_kernel, layer.kernel_backend) for layer in layers]
    absent = object()
    final_states = [(layer, getattr(layer, 'last_final_state', absent)) for layer in layers]
    try:
        model.eval().requires_grad_(False)
        for layer, old, _ in originals:
            layer.reference_kernel = CompleteTransitionReferenceKernel(getattr(old, 'eps', 1e-6))
            layer.kernel_backend = 'reference'
        yield model
    finally:
        for layer, old, backend in originals:
            layer.reference_kernel = old
            layer.kernel_backend = backend
        for layer, state in final_states:
            if state is absent:
                if hasattr(layer, 'last_final_state'): delattr(layer, 'last_final_state')
            else:
                layer.last_final_state = state
        for parameter, requires_grad in gradients:
            parameter.requires_grad_(requires_grad)
        for module, training in modes:
            module.training = training


def shared_history_focal_cwi(query_from_gates: Callable, candidate_embeddings: torch.Tensor,
                             history_lengths: torch.Tensor, history_width: int,
                             temperature: float = 1.0):
    """Exact candidate-focal softmax derivative through a shared history readout.

    query_from_gates accepts [N,L] gates and returns [N,D] history queries,
    with the paper complete-transition intervention inside each recurrent layer.
    Candidates [N,K,D] are detached frozen teacher embeddings. Every focal
    candidate is differentiated against the SAME full candidate softmax. Return
    losses [N,K] and signed CWI [N,K,L]; padding labels are zero. Sampling must
    include both observed positives and negative focal candidates in new data.
    This generic differentiable host seam does not invent a Kuai BCE adapter.
    """
    if candidate_embeddings.ndim != 3 or history_lengths.shape != (candidate_embeddings.shape[0],):
        raise ValueError('candidate embeddings must be [N,K,D], lengths [N]')
    if (candidate_embeddings.shape[1] < 1 or not math.isfinite(temperature) or temperature <= 0 or history_width < 1
            or history_lengths.dtype not in (torch.int32, torch.int64)
            or bool((history_lengths < 1).any()) or bool((history_lengths > history_width).any())):
        raise ValueError('invalid candidate count, temperature or history lengths')
    dtype = torch.float64 if candidate_embeddings.dtype == torch.float64 else torch.float32
    candidates = candidate_embeddings.detach().to(dtype)
    valid = torch.arange(history_width, device=candidates.device)[None] < history_lengths.to(candidates.device)[:, None]
    gate = valid.to(dtype).requires_grad_(True)
    queries = query_from_gates(gate)
    if queries.shape != (candidates.shape[0], candidates.shape[-1]):
        raise ValueError('teacher must return one common history query [N,D]')
    logits = torch.einsum('nd,nkd->nk', queries.to(dtype), candidates) / temperature
    losses = -logits.log_softmax(-1)
    if not bool(torch.isfinite(losses).all()):
        raise ValueError('teacher losses must be finite')
    labels = []
    for focal in range(candidates.shape[1]):
        gradient, = torch.autograd.grad(losses[:, focal].sum(), gate,
                                       retain_graph=focal + 1 < candidates.shape[1])
        labels.append((-gradient).masked_fill(~valid, 0))
    cwi = torch.stack(labels, 1)
    if not bool(torch.isfinite(cwi).all()):
        raise ValueError('teacher CWI must be finite')
    return losses.detach(), cwi.detach()


def official_rating_cwi(scorer, history_ids: torch.Tensor, history_lengths: torch.Tensor,
                        candidate_ids: torch.Tensor, *, temperature: float = 1.0,
                        official_loss_from_queries: Callable | None = None):
    """Run paper CWI through the actual frozen ID/position official rating host.

    This explicit teacher path bypasses cached serving kernels and binary-only
    forward_gdr_streams wrappers. Every layer receives the differentiable gate
    and directly calls CompleteTransitionReferenceKernel. It retains the full
    history's readout position while taking the derivative; compacted-history
    inference may instead read the last retained event. That host distinction
    is explicit in the implementation appendix, not claimed as deletion parity.
    """
    from deltarec.layers.hstu_gdr import _research_layers
    from deltarec.models.hstu_packed import finish_layer
    if history_ids.ndim != 2 or candidate_ids.ndim != 2 or candidate_ids.shape[0] != history_ids.shape[0]:
        raise ValueError('histories and candidates must be [N,L] and [N,K]')
    model = scorer.model
    device = next(model.parameters()).device
    histories, lengths = history_ids.to(device), history_lengths.to(device)
    candidates = candidate_ids.to(device)
    if lengths.shape != (histories.shape[0],) or bool((lengths < 1).any()) or bool((lengths > histories.shape[1]).any()):
        raise ValueError('invalid history lengths')
    positions = torch.arange(histories.shape[1], device=device)[None].expand_as(histories)
    valid = positions < lengths[:, None]
    if bool((histories[valid] <= 0).any()) or bool((candidates <= 0).any()):
        raise ValueError('valid events and candidates require positive IDs')
    original_positions = positions[valid]
    offsets_cpu = torch.cat((torch.zeros(1,dtype=torch.long), lengths.detach().cpu().long().cumsum(0)))
    offsets = offsets_cpu.to(device)
    layers = _research_layers(model)
    calls = 0
    modes = [(module, module.training) for module in model.modules()]
    gradients = [(p, p.requires_grad) for p in model.parameters()]
    try:
        model.eval().requires_grad_(False)
        preproc = model._input_features_preproc
        frozen_x = (model.get_item_embeddings(histories[valid]) * preproc._embedding_dim**.5
                    + preproc._pos_emb(original_positions)).detach()
        embeddings = model.get_item_embeddings(candidates).detach()
        config = scorer.model_config
        if config.item_l2_norm:
            embeddings = embeddings / embeddings.norm(dim=-1,keepdim=True).clamp_min(config.l2_norm_eps)

        def query_from_gates(gates):
            nonlocal calls
            x = frozen_x
            for layer in layers:
                u,q,k,v,decay,beta = layer._project(x)
                projected = layer._kernel_input_type(q=q,k=k,v=v,decay_logits=decay,
                    beta_logits=beta,log_decay_scale=layer.gdr_log_decay_scale,
                    decay_bias=layer.gdr_decay_bias,offsets=offsets,offsets_cpu=offsets_cpu,
                    event_gate=gates[valid])
                output = CompleteTransitionReferenceKernel()(projected)
                calls += 1
                x = finish_layer(layer,x,u,output.context)
            return model._output_postproc(x[offsets[1:]-1])

        if official_loss_from_queries is None:
            losses, labels = shared_history_focal_cwi(query_from_gates, embeddings, lengths,
                                                     histories.shape[1],temperature)
        else:
            # Adapter seam: preserve the official HSTU loss and its internal
            # negative sampler while the CWI primitive owns the focal gate.
            from deltarec.layers.cwi_autograd import candidate_aware_cwi1
            if candidates.shape[1] != 1:
                raise ValueError('official-loss HSTU CWI requires one positive focal candidate')
            losses, labels = candidate_aware_cwi1(
                lambda gates: official_loss_from_queries(
                    query_from_gates(gates[:, 0, :]), candidates),
                history_lengths=lengths, candidate_count=1,
                history_width=histories.shape[1], dtype=torch.float32,
            )
        if calls != len(layers):
            raise RuntimeError('paper teacher did not execute each recurrent layer exactly once')
        return losses, labels, {'teacher_intervention':TEACHER_INTERVENTION,
            'recurrent_layer_calls':calls,'candidate_focal_count':candidate_ids.shape[1],
            'teacher_objective':('official-local-sampled-softmax' if official_loss_from_queries
                                 is not None else 'shared-history-full-candidate-softmax'),
            'readout':'fixed-full-history-final-event-host-readout',
            'new_labels_required':True}
    finally:
        for p, required in gradients:
            p.requires_grad_(required)
        for module, training in modes:
            module.training = training
