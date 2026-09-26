import logging

import os

import torch

from deltarec.layers.utility_grouping import UtilityGroupingConfig, prepare_utility_groups

@torch.no_grad()
def global_partition(path, selector, training, catalog, *, identity, groups=4, anchors=64):
    """Cluster training-catalog utility profiles once; unseen IDs use group zero."""
    if path.is_file():
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('global utility grouping inputs changed')
        return saved
    generator = torch.Generator().manual_seed(0)
    rows = torch.randperm(len(training), generator=generator)[:anchors].tolist()
    anchor_ids = []
    for index in rows:
        example = training[index]
        n = int(example['history_lengths'])
        offset = int(torch.randint(n, (1,), generator=generator))
        anchor_ids.append(int(example['historical_ids'][offset]))
    device = next(selector.parameters()).device
    histories = torch.tensor([anchor_ids], device=device)
    lengths = torch.tensor([len(anchor_ids)], device=device)
    profiles = []
    for start in range(0, len(catalog), 256):
        ids = torch.tensor([catalog[start:start + 256]], device=device)
        values = selector.score_ids(histories, ids, lengths)[0].float().sinh()
        if not bool(torch.isfinite(values).all()):
            raise ValueError('non-finite global utility features')
        profiles.append(values.cpu())
        if start % 16384 == 0:
            logging.info('Global utility features %s/%s items', start, len(catalog))
    profiles = torch.cat(profiles)
    plan = prepare_utility_groups(profiles, UtilityGroupingConfig(groups=groups))
    mapping = torch.zeros(selector.num_items + 1, dtype=torch.long)
    mapping[torch.tensor(catalog)] = plan.candidate_group_ids
    saved = dict(schema='hstu-global-utility-partition-v1', identity=identity,
        item_to_group=mapping, group_count=groups, anchor_item_ids=anchor_ids,
        anchor_training_rows=rows, centers=plan.group_utilities.float(),
        group_sizes=torch.bincount(plan.candidate_group_ids, minlength=groups).tolist(),
        unknown_item_policy='non-training-catalog-items-map-to-group-zero')
    temporary = path.with_suffix('.tmp')
    torch.save(saved, temporary)
    os.replace(temporary, path)
    return saved
