import torch

from torch.nn import functional as F

def sample_negative_ids(positive_ids, catalog, num_negatives):
    """Exact LocalNegativesSampler ID/RNG operation, without unused embeddings."""
    if num_negatives < 1 or catalog.ndim != 1 or not len(catalog):
        raise ValueError('nonempty train catalog and positive negative count required')
    offsets = torch.randint(0, len(catalog), positive_ids.size() + (num_negatives,),
                            dtype=positive_ids.dtype, device=positive_ids.device)
    return catalog[offsets.reshape(-1)].reshape(offsets.shape)

def sampled_softmax(scores, candidates, temperature):
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    positive = scores[:, :1] / temperature
    negative = torch.where(candidates[:, 1:].eq(candidates[:, :1]),
                           scores[:, 1:].new_full((), -5e4),
                           scores[:, 1:] / temperature)
    return -F.log_softmax(torch.cat((positive, negative), dim=1), dim=1)[:, 0]
