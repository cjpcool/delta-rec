"""Shared recent-32 selection: mandatory suffix is included in the total budget."""
import torch


def retention_budget(lengths, ratio=.25, recent=32):
    proportional = torch.ceil(lengths.float() * float(ratio)).long()
    return torch.minimum(lengths.long(), torch.maximum(proportional, lengths.long().clamp(max=recent)))


def recent_topk_mask(scores, lengths, counts, recent=32):
    """Batched [B,K,L] membership; callers validate shapes and finite valid scores."""
    positions = torch.arange(scores.shape[-1], device=scores.device)
    valid = positions[None] < lengths[:, None]
    suffix = valid & (positions[None] >= (lengths - lengths.clamp(max=recent))[:, None])
    # +inf reserves all mandatory slots; the remaining finite scores retain
    # stable utility order. Invalid padding is always -inf, even when poisoned.
    priority = scores.masked_fill(suffix[:, None], torch.inf).masked_fill(~valid[:, None], -torch.inf)
    order = priority.argsort(dim=-1, descending=True, stable=True)
    take = (positions[None, None] < counts[:, None, None]).expand_as(scores)
    return torch.zeros_like(scores, dtype=torch.bool).scatter(-1, order, take)


def recent_topk_indices(scores, count, recent=32):
    """Chronological Top-B for unpadded equal-length rows; short rows avoid sorting."""
    length = scores.shape[-1]
    keep = min(recent, length)
    if not keep <= count <= length:
        raise ValueError('budget must include the complete recent suffix')
    suffix = torch.arange(length-keep, length, device=scores.device).expand(*scores.shape[:-1], keep)
    extra = count-keep
    if not extra:
        return suffix
    earlier = scores[..., :length-keep].argsort(dim=-1, descending=True, stable=True)[..., :extra]
    return torch.cat((earlier.sort(dim=-1).values, suffix), dim=-1)
