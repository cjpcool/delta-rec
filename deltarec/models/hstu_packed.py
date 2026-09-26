from dataclasses import dataclass

import hashlib

import numpy as np

import torch

from torch.nn import functional as F



from deltarec.layers.hstu_gdr import _research_layers

@dataclass
class WorkCounts:
    supervised_positions: int = 0
    history_streams: int = 0
    history_model_calls: int = 0
    candidate_read_calls: int = 0
    candidate_reads: int = 0
    scoring_calls: int = 0
    backward_calls: int = 0
    first_layer_projection_tokens: int = 0

def budgets(lengths, ratio):
    return torch.maximum(torch.ceil(lengths.float() * np.float32(ratio)).long(),
                         lengths.clamp(max=32).long())

def stable_mask(scores, lengths, ratio):
    """Same FP32 ceil/count floor/stable ties as exact_budget_write_mask."""
    valid = torch.arange(scores.shape[-1], device=scores.device)[None] < lengths[:, None]
    order = scores.float().masked_fill(~valid[:, None], -torch.inf).argsort(
        dim=-1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(-1, order, torch.arange(order.shape[-1], device=order.device).expand_as(order))
    return (ranks < budgets(lengths, ratio)[:, None, None]) & valid[:, None]

def finish_layer(layer, x, u, context):
    base = layer.base
    attention = context.reshape(-1, base._num_heads * base._linear_dim)
    normalized = base._norm_attn_output(attention)
    value = torch.cat((u, attention, u * normalized), -1) if base._concat_ua else u * normalized
    return base._o(F.dropout(value, p=float(base._dropout_ratio), training=layer.training)) + x

def identity_read(layer, x, state):
    """GDR with beta=0 and decay=1: normalized query reads unchanged state.

    This preserves projection, gates/output/residual and differentiates through
    both state and query. No detach or candidate-state replication is needed.
    """
    u, q, _, _, _, _ = layer._project(x)
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q = q.to(dtype)
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    context = torch.einsum('nhd,nhdv->nhv', q, state.to(dtype)) * q.shape[-1]**-.5
    return finish_layer(layer, x, u, context.to(x.dtype))

class BatchedSparseScoring:
    def __init__(self, scorer, *, legacy_shared_read=False, max_stream_tokens=32768,
                 reuse_first_projection=True):
        if scorer.method not in {'full-gdr', 'random', 'recent-only', 'rank-only', 'target-similarity'}:
            raise ValueError('unsupported rating sparse method')
        self.scorer, self.model = scorer, scorer.model
        self.layers = _research_layers(self.model)
        self.counts = WorkCounts()
        self.legacy_shared_read = legacy_shared_read  # benchmark/parity only, never default training
        self.max_stream_tokens = int(max_stream_tokens)
        self.reuse_first_projection = bool(reuse_first_projection)
        if self.max_stream_tokens < 1024:
            raise ValueError('stream-token cap must fit at least one full history')
        # Input values are checked once per physical batch. Keep shape checks in
        # the kernel; avoid repeating CUDA scalar checks at every layer/chunk.
        self.kernels = [type(layer.fla_kernel)(validate_inputs=False, assume_binary_event_gate=True) if layer.kernel_backend == 'fla'
                        else layer.reference_kernel for layer in self.layers]

    def masks(self, histories, lengths, candidates, *, cpu_histories=None, rank_mask=None):
        method = self.scorer.method
        if method == 'rank-only':
            if rank_mask is None:
                raise ValueError('fixed rank mask must be resolved before batching prefixes')
            return rank_mask[:, None].to(device=histories.device, dtype=torch.bool)
        if method == 'full-gdr':
            return (torch.arange(histories.shape[1], device=histories.device)[None] < lengths[:, None])[:, None]
        if method == 'recent-only':
            pos = torch.arange(histories.shape[1], device=histories.device)[None]
            return ((pos < lengths[:, None]) & (pos >= (lengths - budgets(lengths, self.scorer.retention_ratio))[:, None]))[:, None]
        if method == 'random':
            if cpu_histories is None:
                cpu_histories = histories.detach().cpu()
            scores = torch.empty_like(histories, dtype=torch.float32)
            ns = lengths.detach().cpu().tolist()
            for i, n in enumerate(ns):
                raw = cpu_histories[i, :n].contiguous().long().numpy().tobytes()
                digest = hashlib.sha256(b'rating-random-s0-v1' + raw).digest()
                generator = torch.Generator(device=histories.device).manual_seed(
                    int.from_bytes(digest[:8], 'little') % (2**63 - 1))
                # v7 trains unpadded individual prefixes. Generate exactly n
                # draws: CUDA Philox launch geometry can depend on vector size.
                scores[i].zero_()
                scores[i, :n] = torch.rand(n, device=histories.device, generator=generator)
            return stable_mask(scores[:, None], lengths, self.scorer.retention_ratio)
        # The selector is nondifferentiable top-k. It references the live official
        # item table; no stale/frozen embedding copy or privileged positive slot.
        with torch.no_grad():
            history = F.normalize(self.model.get_item_embeddings(histories).float(), dim=-1, eps=1e-6)
            candidate = F.normalize(self.model.get_item_embeddings(candidates).float(), dim=-1, eps=1e-6)
            scores = torch.einsum('bld,bkd->bkl', history, candidate)
            return stable_mask(scores, lengths, self.scorer.retention_ratio)

    def _history(self, histories, lengths, mask, counts_cpu, *, need_states):
        # Embed/preprocess once per prefix, then broadcast selected references
        # over candidate streams. Positions stay ORIGINAL history positions.
        events = self.scorer._preprocess_history(histories, lengths)
        batch, streams, width = mask.shape
        stream, position = mask.reshape(batch * streams, width).nonzero(as_tuple=True)
        user = torch.div(stream, streams, rounding_mode='floor')
        x = events[user, position]
        offsets_cpu = torch.cat((torch.zeros(1, dtype=torch.long), counts_cpu.cumsum(0)))
        offsets = offsets_cpu.to(x.device)
        self.counts.history_streams += batch * streams
        self.counts.history_model_calls += 1
        states = []
        for index, (layer, kernel) in enumerate(zip(self.layers, self.kernels)):
            if index == 0 and streams > 1 and self.reuse_first_projection:
                # Only layer 1 sees candidate-independent event inputs. Project
                # once before candidate expansion, then gather differentiably.
                projected_events = layer._project(events.flatten(0, 1))
                gather = user * width + position
                u, q, k, v, decay, beta = [value[gather] for value in projected_events]
                self.counts.first_layer_projection_tokens += batch * width
            else:
                u, q, k, v, decay, beta = layer._project(x)
                if index == 0:
                    self.counts.first_layer_projection_tokens += len(x)
            projected = layer._kernel_input_type(q=q, k=k, v=v, decay_logits=decay,
                beta_logits=beta, log_decay_scale=layer.gdr_log_decay_scale,
                decay_bias=layer.gdr_decay_bias, offsets=offsets, offsets_cpu=offsets_cpu,
                event_gate=torch.ones(len(x), device=x.device, dtype=x.dtype))
            # Preserve v7's numerically stabilized binary-gate path, including
            # finite extreme-decay gradients. Do not bypass it with fused gates.
            result = kernel(projected, initial_state=None, return_final_state=need_states)
            x = finish_layer(layer, x, u, result.context)
            if need_states:
                states.append(result.final_state)
        return x[offsets[1:] - 1], states

    def _score_head(self, query, candidates):
        item = self.model.get_item_embeddings(candidates)
        config = self.scorer.model_config
        if config.item_l2_norm:
            item = item / item.norm(dim=-1, keepdim=True).clamp_min(config.l2_norm_eps)
        self.counts.scoring_calls += 1
        return self.model.similarity_fn(query_embeddings=query, item_ids=candidates,
                                        item_embeddings=item)[0]

    def _read(self, candidates, lengths, states, *, shared):
        batch, count = candidates.shape
        x = self.scorer._candidate_input(candidates, lengths).reshape(batch * count, -1)
        self.counts.candidate_read_calls += 1
        self.counts.candidate_reads += batch * count
        for layer, state in zip(self.layers, states):
            if shared:
                state = state[:, None].expand(-1, count, *state.shape[1:]).reshape(batch * count, *state.shape[1:])
            x = identity_read(layer, x, state)
        query = self.model._output_postproc(x)
        return self._score_head(query, candidates.reshape(-1, 1)).reshape(batch, count)

    def scores(self, histories, lengths, candidates, *, cpu_histories=None, rank_mask=None):
        n = lengths.detach().cpu()
        b = budgets(n, self.scorer.retention_ratio)
        if self.scorer.method == 'full-gdr':
            b = n.long()
        if self.scorer.method != 'target-similarity':
            mask = self.masks(histories, lengths, candidates, cpu_histories=cpu_histories, rank_mask=rank_mask)
            last, states = self._history(histories, lengths, mask, b, need_states=self.legacy_shared_read)
            if self.legacy_shared_read:
                return self._read(candidates, lengths, states, shared=True)
            return self._score_head(self.model._output_postproc(last), candidates)
        # Every candidate remains a distinct chronological recurrent stream.
        # Pack N_supervised x candidate-chunk, limited by total selected tokens.
        chunk = max(1, min(candidates.shape[1], self.max_stream_tokens // max(1, int(b.sum()))))
        outputs = []
        for ids in candidates.split(chunk, dim=1):
            mask = self.masks(histories, lengths, ids)
            counts = b[:, None].expand(-1, ids.shape[1]).reshape(-1)
            _, states = self._history(histories, lengths, mask, counts, need_states=True)
            outputs.append(self._read(ids, lengths, states, shared=False))
        return torch.cat(outputs, dim=1)

class PrefixBatch:
    """CPU planning, including user-specific rank artifacts; no GPU scalar loop."""
    def __init__(self, full_ids, lengths, scorer):
        self.cpu_ids = full_ids.detach().cpu()
        self.cpu_lengths = lengths.detach().cpu().long()
        ns = self.cpu_lengths.tolist()
        if full_ids.ndim != 2 or len(ns) != len(full_ids) or any(n < 1 or n >= full_ids.shape[1] for n in ns):
            raise ValueError('invalid supervised histories')
        if any(bool((row[:n + 1] <= 0).any()) for row, n in zip(self.cpu_ids, ns)):
            raise ValueError('padding in training history/target')
        self.rows = torch.repeat_interleave(torch.arange(len(ns)), self.cpu_lengths)
        self.ends = torch.cat([torch.arange(1, n + 1) for n in ns])
        self.rank_masks = None

    def blocks(self, positions, token_cap, ratio):
        # Sort only within this physical batch. Restore target/negative mapping
        # through the original flattened position index.
        order = torch.argsort(self.ends, stable=True)
        start = 0
        while start < len(order):
            stop = min(start + positions, len(order))
            while stop > start + 1 and int(budgets(self.ends[order[start:stop]], ratio).sum()) > token_cap:
                stop = start + max(1, (stop - start) // 2)
            yield order[start:stop]
            start = stop
