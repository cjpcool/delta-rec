"""Candidate-position CWI teacher over the loaded Kuai Full-GDR layers.

The parent ranker is frozen. Each exposure gets an independent history gate,
shared across its layers. Context always writes and candidates never write.
Only the gate is differentiated; labels never enter the selector inputs.
"""
from __future__ import annotations

import torch
from torch import nn


def pack_teacher_streams(x, lengths, offsets, targets, context, gate):
    batch, candidates, width = gate.shape
    history = lengths.long() - targets.long() - context
    if not torch.equal(targets, targets.new_full((batch,), candidates)):
        raise ValueError("CWI requires uniform positioned candidate counts")
    if bool((history < 1).any()) or int(history.max()) > width:
        raise ValueError("CWI gate does not cover the full history")
    stream_lengths = (history + context + 1).repeat_interleave(candidates)
    packed_offsets = torch.cat((stream_lengths.new_zeros(1), stream_lengths.cumsum(0)))
    rows = torch.repeat_interleave(torch.arange(batch * candidates, device=x.device), stream_lengths)
    positions = torch.arange(rows.numel(), device=x.device) - packed_offsets[:-1][rows]
    users = rows // candidates
    slots = rows % candidates
    query = positions == context + history[users]
    is_history = (positions >= context) & ~query
    sources = offsets[:-1][users] + positions + torch.where(query, slots, 0)
    event_gate = torch.where(positions < context, 1.0, 0.0).float()
    event_gate = event_gate.masked_scatter(is_history, gate[users[is_history], slots[is_history], positions[is_history] - context])
    return x[sources], packed_offsets, event_gate, packed_offsets[1:] - 1, sources


class PackedCWIStack(nn.Module):
    def __init__(self, full_stack):
        super().__init__()
        self.layers = full_stack.layers
        self.contextual_seq_len = full_stack.contextual_seq_len
        self.gate_override = None
        self.last_gate = None
        self.last_history_lengths = None
        self.last_history_embeddings = None
        self.last_candidate_embeddings = None

    def forward_candidate_symmetric(self, *, x, x_lengths, x_offsets,
            num_targets, history_item_ids, candidate_item_ids,
            history_selector_embeddings=None, candidate_selector_embeddings=None,
            return_final_states=False):
        if return_final_states:
            raise ValueError("CWI does not export serving cache states")
        lengths = x_lengths.long() - num_targets.long() - self.contextual_seq_len
        batch, candidates, width = len(lengths), int(num_targets.max()), int(lengths.max())
        gate = self.gate_override
        if gate is None:
            valid = torch.arange(width, device=x.device)[None, None, :] < lengths[:, None, None]
            gate = valid.expand(batch, candidates, width).float().clone()
            gate.requires_grad_(torch.is_grad_enabled())
        if gate.shape != (batch, candidates, width):
            raise ValueError("CWI override gate shape changed")
        packed, offsets, event_gate, queries, _ = pack_teacher_streams(
            x, x_lengths, x_offsets, num_targets, self.contextual_seq_len, gate)
        for layer in self.layers:
            packed = layer.forward_gdr(x=packed, x_offsets=offsets,
                                      event_gate=event_gate, return_final_state=False)
        self.last_gate = gate
        self.last_history_lengths = lengths
        self.last_history_embeddings = history_selector_embeddings.detach() if history_selector_embeddings is not None else None
        self.last_candidate_embeddings = candidate_selector_embeddings.detach() if candidate_selector_embeddings is not None else None
        return packed[queries].reshape(batch, candidates, -1), None, None, {
            "teacher": "same-seed-full-gdr", "candidate_write_gate": 0,
            "physical_gdr_calls": len(self.layers), "candidate_count": candidates}


def candidate_cwi_losses(raw_logits, labels, weights, *, batch_size):
    """Official weighted eight-task BCE, retaining the exposure axis."""
    logits = raw_logits.float()
    labels, weights = labels.T.float(), weights.T.float()
    if logits.shape != labels.shape or labels.shape != weights.shape or logits.shape != (batch_size * 32, 8):
        raise ValueError("invalid Kuai CWI logits/labels/weights shape")
    if not all(bool(torch.isfinite(v).all()) for v in (logits, labels, weights)):
        raise ValueError("nonfinite Kuai CWI inputs")
    if bool(((labels != 0) & (labels != 1)).any()) or bool((weights < 0).any()):
        raise ValueError("invalid Kuai CWI labels/weights")
    losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    # Sum of exposure contributions reproduces the official batch task loss.
    denominator = weights.sum(0).clamp_min(1)
    return (0.2 * losses * weights / denominator).sum(-1).reshape(batch_size, 32)
