"""Bounded-memory numeric history decoding for the official DatasetV2 format.

No padded history cache is retained. Stochastic sampling uses the original
decoder so its RNG consumption and per-visit behavior are preserved.
"""
import ast
import csv
import re
import sys

_INTEGER = re.compile(r'\s*[+-]?\d+(?:\s*,\s*[+-]?\d+)*\s*,?\s*')
_NUMBER = re.compile(r'\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?:\s*,\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)*\s*,?\s*')


def _numeric_int_list(raw: str):
    """Decode the frozen sequence column without pandas/eval in the hot path."""

    import numpy as np

    value = str(raw).strip()
    if value[:1] in ('[', '(') and value[-1:] in (']', ')'):
        value = value[1:-1]
    if not _INTEGER.fullmatch(value):
        return None
    return np.fromstring(value.rstrip().rstrip(','), sep=',', dtype=np.int64)


def _numeric_int_array(raw, *, field):
    import numpy as np

    value = _numeric_int_list(raw)
    if value is None:
        parsed = ast.literal_eval(raw)
        value = np.asarray(
            parsed if isinstance(parsed, (list, tuple)) else [parsed], dtype=np.int64
        )
    if value.ndim != 1:
        raise ValueError(f'{field} must be one-dimensional')
    return value


def iter_numeric_rating_examples(
    path, *, max_history_length: int = 1024, include_timestamps: bool = False
):
    """Yield the same expanded prefixes as ``recbole_train_entry``.

    This is intentionally a separate iterator from ``NumericHistoryDataset``:
    the latter reproduces the official one-sample-per-user DatasetV2 path,
    while DeltaRec's Full-GDR contract expands every train prefix.  Both paths
    use the same numeric item IDs and keep the final validation item excluded.
    Malformed rows fall back to ``ast.literal_eval`` so old data remains
    readable without reintroducing unrestricted ``eval``.
    """

    import numpy as np

    if max_history_length < 1:
        raise ValueError('max_history_length must be positive')
    csv.field_size_limit(sys.maxsize)
    with open(path, newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        required = {
            'user_id', 'sequence_item_ids', 'sequence_ratings',
            'sequence_timestamps',
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f'rating CSV is missing columns: {sorted(missing)}')
        for row in reader:
            items = _numeric_int_array(row['sequence_item_ids'], field='sequence_item_ids')
            timestamps = None
            if include_timestamps:
                timestamps = _numeric_int_array(
                    row['sequence_timestamps'], field='sequence_timestamps'
                )
                if timestamps.shape != items.shape:
                    raise ValueError(
                        'sequence_item_ids and sequence_timestamps have different lengths'
                    )
            for target_index in range(1, len(items) - 1):
                start = max(0, target_index - max_history_length)
                history = items[start:target_index]
                if history.size:
                    result = {
                        'user_id': int(row['user_id']),
                        'history': history.tolist(),
                        'target': int(items[target_index]),
                    }
                    if timestamps is not None:
                        result['history_timestamps'] = timestamps[start:target_index].tolist()
                        result['target_timestamp'] = int(timestamps[target_index])
                    yield result


class NumericHistoryDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        frame = dataset.ratings_frame
        self.columns = {name: frame[name].to_numpy(copy=False) for name in
                        ('user_id', 'sequence_item_ids', 'sequence_ratings', 'sequence_timestamps')}

    def __len__(self):
        return len(self.dataset)

    def __setstate__(self, state):
        # Older launch packages pickle only ``dataset``. Workers may be spawned
        # at the next epoch after the shared compatibility class is upgraded.
        self.__dict__.update(state)
        if 'columns' not in state:
            self.__init__(state['dataset'])

    def __getitem__(self, index):
        import numpy as np
        import torch
        d = self.dataset
        if d._sample_ratio != 1.:
            return d.load_item(d.ratings_frame.iloc[index])
        arrays = []
        for name, floating in [('sequence_item_ids', False), ('sequence_ratings', True),
                               ('sequence_timestamps', False)]:
            raw = str(self.columns[name][index]).strip()
            if raw[:1] in ('[', '(') and raw[-1:] in (']', ')'):
                raw = raw[1:-1]
            if not (_NUMBER if floating else _INTEGER).fullmatch(raw):
                return d.load_item(d.ratings_frame.iloc[index])
            values = np.fromstring(raw.rstrip().rstrip(','), sep=',', dtype=np.float64 if floating else np.int64)
            if d._ignore_last_n > 0:
                values = values[:-d._ignore_last_n]
            arrays.append(values)
        if not len(arrays[0]) or len({len(v) for v in arrays}) != 1:
            return d.load_item(d.ratings_frame.iloc[index])
        if d._shift_id_by > 0:
            arrays[0] = arrays[0] + d._shift_id_by
        length = min(len(arrays[0]) - 1, d._padding_length - 1)
        result = {'user_id': self.columns['user_id'][index], 'history_lengths': length}
        for values, history, target in zip(arrays,
                ('historical_ids', 'historical_ratings', 'historical_timestamps'),
                ('target_ids', 'target_ratings', 'target_timestamps')):
            result[target] = values[-1].item()
            past = values[:-1]
            if not length:
                past = past[:0]
            elif d._chronological:
                past = past[-length:]
            else:
                past = past[::-1][:length]
            padded = np.zeros(d._padding_length - 1, dtype=np.int64)
            padded[:length] = past
            result[history] = torch.from_numpy(padded)
        return result
