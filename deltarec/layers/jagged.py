"""Packed/padded layout operations used at the FuXi boundary."""
import torch

def asynchronous_complete_cumsum(lengths):
    return torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))

def dense_to_jagged(dense, offsets, total_L=None):
    bounds = offsets[0]
    lengths = bounds[1:] - bounds[:-1]
    mask = torch.arange(dense.shape[1], device=dense.device)[None] < lengths[:, None]
    return dense[mask], offsets

def jagged_to_padded_dense(values, offsets, max_lengths, padding_value=0.):
    bounds, width = offsets[0], max_lengths[0]
    lengths = bounds[1:] - bounds[:-1]
    columns = torch.arange(width, device=values.device)[None]
    valid = columns < lengths[:, None]
    indices = bounds[:-1, None] + columns
    output = values.new_full((len(lengths), width, *values.shape[1:]), padding_value)
    output[valid] = values[indices[valid]]
    return output
