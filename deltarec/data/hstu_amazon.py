import csv

import random

from pathlib import Path

import numpy as np

from deltarec.utils.io import sha256_file, atomic_write_json

STRATA = ((33, 64), (65, 128), (129, 256), (257, 1024))

def prepare(source, output, count=4096, seed=0):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    reservoirs = [[] for _ in STRATA]
    populations = [0] * len(STRATA)
    quota = count // len(STRATA)
    if count % len(STRATA): raise ValueError('count must divide equally into four strata')
    items, offsets, users = [], [0], []
    max_validation = max_train = 0
    total = zero = 0
    with Path(source).open() as f:
        for row in csv.DictReader(f):
            ids = list(map(int, row['sequence_item_ids'].split(',')))
            times = list(map(int, row['sequence_timestamps'].split(',')))
            if len(ids) != len(times) or any(b < a for a,b in zip(times,times[1:])):
                raise ValueError('unaligned or nonchronological source')
            max_validation = max(max_validation, sum(t < times[-1] for t in times[:-1]))
            ids, times = ids[:-1], times[:-1]  # Remove validation before all sampling.
            max_train = max(max_train, len(ids)); total += 1
            if len(ids) < 2:
                zero += 1; continue
            user = int(row['user_id'])
            items.extend(ids); offsets.append(len(items)); users.append(user)
            # Reservoir uniformly over eligible (user,prefix) pairs within each
            # stratum. Exclude equal-time target/history boundaries for labels.
            for cut in range(33, len(ids)):
                if times[cut - 1] >= times[cut]: continue
                s = next(i for i,(lo,hi) in enumerate(STRATA) if lo <= cut <= hi)
                populations[s] += 1
                slot = populations[s]-1 if populations[s] <= quota else rng.randrange(populations[s])
                if slot < quota:
                    value = dict(user_id=user, history=ids[:cut], target=ids[cut],
                        target_index=cut, stratum=list(STRATA[s]))
                    if slot == len(reservoirs[s]): reservoirs[s].append(value)
                    else: reservoirs[s][slot] = value
    if any(len(r) != quota for r in reservoirs): raise ValueError(f'insufficient eligible prefixes: {populations}')
    for name,value in [('items',items),('offsets',offsets),('users',users)]:
        np.save(output/f'{name}.npy', np.asarray(value, dtype=np.int64))
    record = dict(schema='amazon-hstu-gdr-data-v1', source=Path(source).name,
        source_sha256=sha256_file(source), max_history_length=max(max_train,max_validation),
        max_training_sequence=max_train, max_strict_validation_history=max_validation,
        training_users=len(users), source_users=total, zero_supervision_users=zero,
        training_pairs=len(items)-len(users), test_access=False,
        padding='per-microbatch-maximum-with-length-bucketing',
        cwi_sampling='equal-length-strata-uniform-eligible-prefix-reservoir',
        cwi_count=count, cwi_strata=[dict(bounds=b, population=n, sampled=quota) for b,n in zip(STRATA,populations)],
        validation_removed_before_sampling=True, cwi_strict_timestamp_boundary=True)
    atomic_write_json(output/'data.json', record)
    atomic_write_json(output/'training_prefixes.json', dict(metadata=record, rows=sum(reservoirs, [])))
    return record

def epoch_windows(offsets, seed, epoch, size=128):
    # Randomize all users first, then sort ONLY inside each fixed optimizer
    # window. Dynamic microbatch bucketing does not change window membership.
    order = np.random.default_rng(seed+epoch).permutation(len(offsets)-1)
    for start in range(0,len(order),size):
        window = order[start:start+size]
        lengths = offsets[window+1]-offsets[window]
        yield window[np.argsort(lengths,kind='stable')]
