from __future__ import annotations

import random

from collections.abc import Iterator

from typing import Any

def iter_hstu_cwi_contexts(train_dataset: Any) -> Iterator[dict[str, Any]]:
    """Yield each valid position of ``supervision_ids[:, 1:]`` in train order."""

    for row_index in range(len(train_dataset)):
        row = train_dataset[row_index]
        length = int(row["history_lengths"])
        history = row["historical_ids"][:length].tolist()
        timestamps = row["historical_timestamps"][:length].tolist()
        if len(history) != length or len(timestamps) != length:
            raise ValueError("official HSTU training history is malformed")
        if length == 0:
            # No history event can form the CWI primitive's next-item context.
            continue
        # Official train.py appends target_ids after the historical window,
        # then supervises nonzero entries of past_ids[:, 1:] for `length`
        # positions.  The first historical event is context, never a target.
        targets = history[1:] + [int(row["target_ids"])]
        for prefix_length, target in enumerate(targets, 1):
            if target == 0:  # The official AR supervision mask excludes padding.
                continue
            yield {
                "user_id": int(row["user_id"]),
                "history": history[:prefix_length],
                "history_timestamps": timestamps[:prefix_length],
                "target": target,
                "source_row": row_index,
                "supervision_position": prefix_length,
            }

def sample_hstu_cwi_contexts(
    train_dataset: Any, *, cwi_samples: int, seed: int = 1000,
) -> list[dict[str, Any]]:
    """Reservoir-select a configurable labeled subset, once for both teachers."""

    if cwi_samples < 1:
        raise ValueError("cwi_samples must be positive")
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    seen = 0
    for context in iter_hstu_cwi_contexts(train_dataset):
        seen += 1
        if len(selected) < cwi_samples:
            selected.append(context)
        else:
            replacement = rng.randrange(seen)
            if replacement < cwi_samples:
                selected[replacement] = context
    if not selected:
        raise ValueError("official HSTU training stream has no valid CWI contexts")
    rng.shuffle(selected)
    return selected
