from __future__ import annotations

import csv

from pathlib import Path

import random

import torch

from deltarec.data import training as data

from deltarec.utils.checkpoint import atomic_torch_save

def sample_training_rows(c, files, limit, *, seed, min_history=1):
    """Bounded reservoir across the whole training split, independent of CSV order."""
    cache = None
    binding = None
    if 'output' in c:
        cache = Path(c['output']) / f'training-samples-s{seed}-n{limit}-l{min_history}.pt'
        binding = dict(data_binding_sha256=c['binding_sha256'], code=c['code'],
                       seed=seed, limit=limit, min_history=min_history)
        if cache.exists():
            payload = torch.load(cache, map_location='cpu', weights_only=False)
            if payload.get('binding') != binding:
                raise ValueError('training sample cache has a different source/data binding')
            return payload['rows']
    rng = random.Random(seed)
    def rating_rows():
        source = data.validate_sequence_input(files['train'])
        with source.open(newline='') as handle:
            for row in csv.DictReader(handle):
                items = data._parse_ints(row['sequence_item_ids'])
                # len-1 is validation and is never sampled as history or target.
                if len(items) - 2 < min_history:
                    continue
                cutoff = rng.randrange(min_history, len(items) - 1)
                yield dict(user_id=int(row['user_id']),
                    history=items[max(0, cutoff-1024):cutoff], target=items[cutoff])
    source = data._slate_examples(files['train'], role='train') if c['dataset'] == 'kuairand-1k' else rating_rows()
    result = []
    eligible = 0
    for row in source:
        if len(row['history']) < min_history:
            continue
        eligible += 1
        if len(result) < limit:
            result.append(row)
        else:
            index = rng.randrange(eligible)
            if index < limit:
                result[index] = row
    if not result:
        raise ValueError('no eligible training histories for the requested sampling stage')
    rng.shuffle(result)
    if cache is not None:
        atomic_torch_save(torch, dict(binding=binding, rows=result), cache)
    return result
