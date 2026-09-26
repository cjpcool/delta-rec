"""Fixed item->utility-group lookup and physically compressed HSTU training.

The partition is fitted once on frozen-selector utility profiles over common
train-only anchor events. Per-prefix grouping is a lookup, never clustering.
Active-group arithmetic mean utilities select chronological history indices.
Only selected IDs/positions enter the host embedding/projection/recurrence.
"""
from types import SimpleNamespace
import logging

import torch
from deltarec.layers.recent_selection import recent_topk_mask

from deltarec.models.hstu_packed import BatchedSparseScoring, PrefixBatch, budgets
from deltarec.models.hstu_shared import SharedSparseScoring
from deltarec.models.hstu_loss import sample_negative_ids, sampled_softmax


class GroupEncoder(SharedSparseScoring):
    def _score_head(self, query, candidates):
        return query


@torch.no_grad()
def grouped_indices(selector, mapping, group_count, histories, lengths, candidates,
                    *, ratio=.25, candidate_chunk=32):
    """Return active stream rows, lengths, packed indices and candidate->stream.

    Pool ALL candidates before top-k, independent of execution chunking. Empty
    groups have no history stream. Positive labels never enter this function.
    """
    batch, width = histories.shape
    groups = mapping[candidates]
    sums = torch.zeros(batch, group_count, width, device=histories.device)
    counts = torch.zeros(batch, group_count, device=histories.device)
    for start in range(0, candidates.shape[1], candidate_chunk):
        ids = candidates[:, start:start + candidate_chunk]
        member_groups = groups[:, start:start + candidate_chunk]
        utilities = selector.score_ids(histories, ids, lengths).float().sinh()
        sums.scatter_add_(1, member_groups[:, :, None].expand_as(utilities), utilities)
        counts.scatter_add_(1, member_groups, torch.ones_like(member_groups, dtype=sums.dtype))
    if not bool(torch.isfinite(sums).all()):
        raise ValueError('non-finite raw selector utility')
    rows, active_groups = (counts > 0).nonzero(as_tuple=True)
    pooled = sums[rows, active_groups] / counts[rows, active_groups, None]
    stream_lengths = lengths[rows]
    kept_counts = budgets(stream_lengths, ratio)
    selected = recent_topk_mask(pooled[:, None], stream_lengths, kept_counts)[:, 0]
    positions = selected.nonzero(as_tuple=True)[1]
    streams = torch.arange(len(rows), device=histories.device).repeat_interleave(kept_counts)
    offsets_cpu = torch.cat((torch.zeros(1, dtype=torch.long), kept_counts.cpu().cumsum(0)))
    lookup = torch.full((batch, group_count), -1, dtype=torch.long, device=histories.device)
    lookup[rows, active_groups] = torch.arange(len(rows), device=histories.device)
    candidate_streams = lookup.gather(1, groups)
    return rows, stream_lengths, streams, positions, offsets_cpu, candidate_streams


class GlobalUtilityScoring:
    def __init__(self, scorer, selector, mapping, group_count, *, max_stream_tokens=16384):
        self.scorer, self.model, self.selector = scorer, scorer.model, selector
        self.mapping = mapping.to(next(self.model.parameters()).device)
        self.group_count = group_count
        if (mapping.ndim != 1 or mapping.dtype != torch.long or group_count < 1
                or bool((mapping < 0).any()) or bool((mapping >= group_count).any())):
            raise ValueError('invalid global item-to-group mapping')
        self.max_stream_tokens = max_stream_tokens
        # The shared encoder consumes explicit packed indices; its policy is
        # never used for selection. It preserves original positional features.
        proxy = SimpleNamespace(model=self.model, model_config=scorer.model_config,
                                method='random', retention_ratio=.25)
        self.encoder = GroupEncoder(proxy, max_stream_tokens=max_stream_tokens)
        self.counts = self.encoder.counts
        self.last_execution = {}

    def scores(self, histories, lengths, candidates, **unused):
        rows, ends, streams, positions, offsets, candidate_streams = grouped_indices(
            self.selector, self.mapping, self.group_count, histories, lengths, candidates)
        before = self.counts.first_layer_projection_tokens
        queries = self.encoder.packed_scores(histories, rows, ends, streams, positions, offsets, None)
        projected = self.counts.first_layer_projection_tokens - before
        if projected != int(offsets[-1]) or projected != len(positions):
            raise RuntimeError('GDR traversed a different length than the filtered history')
        self.last_execution = dict(active_group_streams=len(rows), retained_tokens=projected,
            original_stream_tokens=int(ends.sum()), candidate_reads=0,
            offset_lengths=offsets.diff().tolist())
        query = queries[candidate_streams.reshape(-1)]
        return BatchedSparseScoring._score_head(self.encoder, query,
            candidates.reshape(-1, 1)).reshape_as(candidates)

    def backward_batch(self, full_ids, lengths, catalog, num_negatives, temperature,
                       *, supervision_block=16):
        plan = PrefixBatch(full_ids, lengths, self.scorer)
        device = next(self.model.parameters()).device
        positives = plan.cpu_ids[plan.rows, plan.ends].to(device)
        negatives = sample_negative_ids(positives, catalog, num_negatives)
        candidates = torch.cat((positives[:, None], negatives), dim=1)
        total = torch.zeros((), device=device, dtype=torch.float64)
        # All active group graphs fit this bound and are released per block.
        for order in plan.blocks(supervision_block, self.max_stream_tokens // self.group_count, .25):
            rows, ends = plan.rows[order], plan.ends[order]
            width = int(ends.max())
            histories = plan.cpu_ids[rows, :width].clone()
            histories.masked_fill_(torch.arange(width)[None] >= ends[:, None], 0)
            ids = candidates[order.to(device)]
            scores = self.scores(histories.to(device), ends.to(device), ids)
            loss = sampled_softmax(scores, ids, temperature).sum()
            if not bool(torch.isfinite(loss.detach())):
                raise ValueError('non-finite grouped training loss')
            loss.backward()
            total += loss.detach().double()
            self.counts.backward_calls += 1
            self.counts.supervised_positions += len(order)
            if self.counts.backward_calls == 1 or self.counts.backward_calls % 32 == 0:
                logging.info('Grouped backward blocks=%s supervised_positions=%s filtered_execution=%s',
                    self.counts.backward_calls, self.counts.supervised_positions, self.last_execution)
            del scores, loss
        return float(total), len(positives)
