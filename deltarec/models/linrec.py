"""Grouped LinRec: masked batched native attention or packed GDR.

Upstream layers are supplied by the external RecBole bridge, never copied here.
Native rating uses the original dot-product scorer; native Kuai uses a direct
2D -> 8 head. GDR uses the project's existing vector-level prediction heads.
"""
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from deltarec.layers.paper_cwi import CompleteTransitionReferenceKernel
from deltarec.layers.recent_selection import recent_topk_indices
from deltarec.models.hstu_multitask import HSTUMultitaskHead, HSTURankingHead
from deltarec.layers.gdr_kernels import build_gdr_kernel
from deltarec.layers.selective_gdr import GDRKernelInput


def budget(length):
    return max(math.ceil(float(torch.tensor(length, dtype=torch.float32) * .25)), min(32, length))


def pack_histories(histories, device):
    lengths = [len(row) for row in histories]
    if not lengths or min(lengths) < 1 or max(lengths) > 1024:
        raise ValueError('histories must contain 1..1024 real events')
    if any(item <= 0 for row in histories for item in row):
        raise ValueError('packed histories cannot contain padding/nonpositive IDs')
    ids = torch.tensor([item for row in histories for item in row], device=device, dtype=torch.long)
    positions = torch.cat([torch.arange(n) for n in lengths]).to(device)
    return ids, positions, lengths


@contextmanager
def value_gates(layers, gates):
    """An event's scalar intervenes on KV contributions in EVERY native layer."""
    handles = []
    try:
        for layer in layers:
            handles.append(layer.multi_head_attention.value.register_forward_hook(
                lambda _module, _args, output: output * gates[..., None].to(output.dtype)))
        yield
    finally:
        for handle in handles:
            handle.remove()


class LinRecGDRLayer(nn.Module):
    def __init__(self, original, width, heads, backend, seed):
        super().__init__()
        self.attention = original.multi_head_attention
        self.feed_forward = original.feed_forward
        self.heads, self.head_dim = heads, width // heads
        self.gates = nn.Linear(width, 2 * heads, bias=False)
        nn.init.zeros_(self.gates.weight)
        generator = torch.Generator().manual_seed(seed)
        scale = torch.empty(heads).uniform_(0., 16., generator=generator).clamp_min(1e-4)
        dt = torch.exp(torch.rand(heads, generator=generator) * math.log(100.) + math.log(.001))
        self.log_decay_scale = nn.Parameter(scale.log())
        self.decay_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.kernel = build_gdr_kernel(backend)
        self.teacher_kernel = CompleteTransitionReferenceKernel()

    def forward(self, x, offsets, gates=None):
        attention = self.attention
        shape = (-1, self.heads, self.head_dim)
        q, k = F.elu(attention.query(x).reshape(shape)), F.elu(attention.key(x).reshape(shape))
        v = attention.value(x).reshape(shape)
        decay, beta = self.gates(x).chunk(2, -1)
        projected = GDRKernelInput(q=q, k=k, v=v, decay_logits=decay, beta_logits=beta,
            log_decay_scale=self.log_decay_scale, decay_bias=self.decay_bias,
            offsets=offsets.to(x.device), offsets_cpu=offsets, event_gate=gates)
        kernel = self.teacher_kernel if gates is not None else self.kernel
        context = kernel(projected, return_final_state=False).context.flatten(-2)
        x = attention.LayerNorm(x + attention.out_dropout(attention.dense(context.to(x.dtype))))
        return self.feed_forward(x)


class LinRecDeltaRec(nn.Module):
    def __init__(self, upstream, *, core, multitask=False, backend='reference', seed=0):
        super().__init__()
        if core not in ('native', 'gdr'):
            raise ValueError('core must be native or gdr')
        self.core, self.multitask = core, bool(multitask)
        self.item_embedding = upstream.item_embedding
        self.position_embedding = upstream.position_embedding
        self.input_norm, self.input_dropout = upstream.LayerNorm, upstream.dropout
        self.width = self.item_embedding.embedding_dim
        self.layers = nn.ModuleList(list(upstream.trm_encoder.layer) if core == 'native' else [
            LinRecGDRLayer(layer, self.width, upstream.n_heads, backend, seed + i)
            for i, layer in enumerate(upstream.trm_encoder.layer)])
        if core == 'native':
            self.task_head = nn.Linear(2 * self.width, 8) if multitask else None
        else:
            self.task_head = (HSTUMultitaskHead if multitask else HSTURankingHead)(self.width)
        self.executed_tokens = 0
        self._catalog_group_ids = None

    @property
    def device(self):
        return self.item_embedding.weight.device

    def encode_packed(self, ids, positions, lengths, gates=None):
        if ids.ndim != 1 or positions.shape != ids.shape or sum(lengths) != len(ids) or min(lengths) < 1:
            raise ValueError('invalid nonempty packed streams')
        self.executed_tokens += len(ids)
        x = self.input_dropout(self.input_norm(self.item_embedding(ids) + self.position_embedding(positions)))
        offsets = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.long)
        if self.core == 'gdr':
            for layer in self.layers:
                x = layer(x, offsets, gates)
            return x[(offsets[1:] - 1).to(x.device)]
        # Upstream normalizes K over head dimensions, not time, and ignores its
        # attention_mask. Zeroing padded V in EVERY layer therefore exactly
        # removes padding from K^T V while preserving its original equations.
        # Repeat the last real input at padded positions to avoid zero-norm Q/K;
        # these positions have no value contribution and are never read out.
        lengths_device = torch.tensor(lengths, device=x.device)
        local = torch.arange(max(lengths), device=x.device)[None]
        valid = local < lengths_device[:, None]
        index = offsets[:-1].to(x.device)[:, None] + torch.minimum(local, lengths_device[:, None] - 1)
        hidden = x[index]
        mask = valid.to(x.dtype) if gates is None else gates[index] * valid
        with value_gates(self.layers, mask):
            for layer in self.layers:
                hidden = layer(hidden, None)
        return hidden[torch.arange(len(lengths), device=x.device), lengths_device - 1]

    def encode_full(self, histories, gates=None):
        return self.encode_packed(*pack_histories(histories, self.device), gates=gates)

    def pair_scores(self, queries, candidates):
        embedding = self.item_embedding(candidates)
        if queries.ndim == 2:
            queries = queries[:, None].expand_as(embedding)
        if self.task_head is None:
            return (queries * embedding).sum(-1).float()
        if isinstance(self.task_head, nn.Linear):
            return self.task_head(torch.cat((queries, embedding), -1)).float()
        return self.task_head(queries, embedding).float()

    def catalog_logits(self, queries, mapping=None):
        """Exact original full-vocabulary CE logits, including output row zero."""
        if self.core != 'native' or self.multitask:
            raise ValueError('full-vocabulary CE is native rating only')
        if mapping is None:
            return queries @ self.item_embedding.weight.T
        # Each catalog row is scored once with its own group query; do not
        # project every candidate against every group or replace CE by sampling.
        if self._catalog_group_ids is None:
            self.bind_catalog_groups(mapping, queries.shape[1])
        pieces = [queries[:, group] @ self.item_embedding.weight[ids].T
                  for group, ids in enumerate(self._catalog_group_ids)]
        return torch.cat(pieces, 1)[:, self._catalog_inverse]

    def bind_catalog_groups(self, mapping, groups=4):
        self._catalog_group_ids = [(mapping == group).nonzero(as_tuple=True)[0] for group in range(groups)]
        self._catalog_inverse = torch.cat(self._catalog_group_ids).argsort()


class FrozenUtilitySelector(nn.Module):
    def __init__(self, table, catalog):
        super().__init__()
        self.register_buffer('embedding', table.detach().clone(), persistent=False)
        self.register_buffer('rms', table.detach()[catalog].float().square().mean().sqrt().clamp_min(1e-6))
        width = table.shape[1]
        self.input, self.output = nn.Linear(3 * width, width), nn.Linear(width, 1)
        self.to(table.device)

    def forward(self, events, candidates):
        # Outer GDR autocast must never quantize selector ordering or pooling.
        with torch.autocast(device_type=self.embedding.device.type, enabled=False):
            event = self.embedding[events].float() / self.rms
            candidate = self.embedding[candidates].float() / self.rms
            event, candidate = torch.broadcast_tensors(event, candidate)
            return self.output(F.silu(self.input(torch.cat((event, candidate, event * candidate), -1)))).squeeze(-1)


@dataclass
class SelectedStreams:
    ids: torch.Tensor
    positions: torch.Tensor
    lengths: list
    lookup: torch.Tensor
    original_tokens: int


@torch.no_grad()
def select_streams(histories, candidates, selector, mapping, *, core, groups=4, native_table=None, mandatory_recent_suffix=True):
    """Select without labels; all pooling precedes Top-B and encoding."""
    device = mapping.device
    ids, positions, lengths = pack_histories(histories, device)
    owners = torch.repeat_interleave(torch.arange(len(lengths), device=device),
                                    torch.tensor(lengths, device=device))
    if core == 'native':
        if native_table is None:
            raise ValueError('native grouped execution requires frozen representative scores')
        utilities = native_table[ids]
        active = ([(row, group) for row in range(len(lengths)) for group in range(groups)]
                  if candidates is None else sorted(set((row, int(g)) for row, values in enumerate(
                      mapping[candidates].cpu().tolist()) for g in values)))
    else:
        if candidates is None:
            raise ValueError('GDR selection requires the complete candidate slate')
        member_groups = mapping[candidates]
        counts = torch.zeros(len(lengths), groups, device=device)
        counts.scatter_add_(1, member_groups, torch.ones_like(member_groups, dtype=torch.float32))
        utilities = torch.zeros(len(ids), groups, device=device)
        for start in range(0, len(ids), 256):
            stop = min(start + 256, len(ids))
            rows = owners[start:stop]
            for c in range(0, candidates.shape[1], 32):
                values = selector(ids[start:stop, None], candidates[rows, c:c + 32]).sinh()
                utilities[start:stop].scatter_add_(1, member_groups[rows, c:c + 32], values)
        utilities /= counts[owners].clamp_min(1)
        active = [tuple(pair) for pair in (counts > 0).nonzero().cpu().tolist()]
    if not bool(torch.isfinite(utilities).all()):
        raise FloatingPointError('nonfinite frozen utility')
    offsets = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.long)
    buckets = defaultdict(list)
    for stream, (row, group) in enumerate(active):
        buckets[lengths[row]].append((stream, row, group))
    kept, stream_order, kept_lengths = [], [], []
    for length, rows in sorted(buckets.items()):
        streams, users, group_ids = zip(*rows)
        index = (offsets[list(users), None] + torch.arange(length)[None]).to(device)
        scores = utilities[index, torch.tensor(group_ids, device=device)[:, None]]
        selected = (recent_topk_indices(scores, budget(length)) if mandatory_recent_suffix else
                    scores.argsort(dim=-1, descending=True, stable=True)[..., :budget(length)].sort(-1).values)
        kept.append(index.gather(1, selected).flatten())
        stream_order.extend(streams)
        kept_lengths.extend([budget(length)] * len(rows))
    indices = torch.cat(kept)
    lookup = torch.full((len(lengths), groups), -1, dtype=torch.long, device=device)
    for new_stream, original_stream in enumerate(stream_order):
        row, group = active[original_stream]
        lookup[row, group] = new_stream
    return SelectedStreams(ids[indices], positions[indices], kept_lengths, lookup,
                           sum(lengths[row] for row, _ in active))


def encode_selection(model, selected, *, max_tokens=16384):
    """Bound each encoder call; the trainer separately caps each backward graph."""
    outputs, start, offset = [], 0, 0
    while start < len(selected.lengths):
        end, total = start, 0
        while end < len(selected.lengths) and total + selected.lengths[end] <= max_tokens:
            total += selected.lengths[end]
            end += 1
        if end == start:
            raise ValueError('token cap cannot contain one stream')
        outputs.append(model.encode_packed(selected.ids[offset:offset + total],
            selected.positions[offset:offset + total], selected.lengths[start:end]))
        start, offset = end, offset + total
    return torch.cat(outputs)


def grouped_scores(model, histories, candidates, selector, mapping, native_table=None):
    selected = select_streams(histories, candidates, selector, mapping,
                              core=model.core, groups=getattr(model, "selection_groups", 4), native_table=native_table, mandatory_recent_suffix=getattr(model, "mandatory_recent_suffix", True))
    queries = encode_selection(model, selected)
    if candidates is None:
        return model.catalog_logits(queries[selected.lookup], mapping), selected
    streams = selected.lookup.gather(1, mapping[candidates])
    outputs = [model.pair_scores(queries[streams[:, start:start + 32]], candidates[:, start:start + 32])
               for start in range(0, candidates.shape[1], 32)]
    return torch.cat(outputs, 1), selected
