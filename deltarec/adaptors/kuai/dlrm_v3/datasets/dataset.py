# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

from typing import List, Tuple

import torch

from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

@dataclass
class Samples:
    """
    Container for batched samples with user interaction history and candidate features.

    Attributes:
        uih_features_kjt: User interaction history features as KeyedJaggedTensor.
        candidates_features_kjt: Candidate item features as KeyedJaggedTensor.
    """

    uih_features_kjt: KeyedJaggedTensor
    candidates_features_kjt: KeyedJaggedTensor

    def to(self, device: torch.device) -> None:
        """
        Move all tensors to the specified device.

        Args:
            device: Target device to move tensors to.
        """
        for attr in vars(self):
            setattr(self, attr, getattr(self, attr).to(device=device))

    def batch_size(self) -> int:
        """
        Get the batch size of the samples.

        Returns:
            Number of samples in the batch.
        """
        return self.uih_features_kjt.stride()

def collate_fn(
    samples: List[Tuple[KeyedJaggedTensor, KeyedJaggedTensor]],
) -> Samples:
    """
    Collate multiple samples into a batched Samples object.

    Args:
        samples: List of (uih_features, candidates_features) tuples.

    Returns:
        Batched Samples object with concatenated features.
    """
    (
        uih_features_kjt_list,
        candidates_features_kjt_list,
    ) = list(zip(*samples))

    return Samples(
        uih_features_kjt=kjt_batch_func(uih_features_kjt_list),
        candidates_features_kjt=kjt_batch_func(candidates_features_kjt_list),
    )

@torch.jit.script
def kjt_batch_func(
    kjt_list: List[KeyedJaggedTensor],
) -> KeyedJaggedTensor:
    """
    Batch multiple KeyedJaggedTensors into a single tensor.

    Uses FBGEMM operations for efficient batching and reordering of
    jagged tensor data.

    Args:
        kjt_list: List of KeyedJaggedTensors to batch.

    Returns:
        Batched KeyedJaggedTensor with reordered indices and lengths.
    """
    bs_list = [kjt.stride() for kjt in kjt_list]
    bs = sum(bs_list)
    batched_length = torch.cat([kjt.lengths() for kjt in kjt_list], dim=0)
    batched_indices = torch.cat([kjt.values() for kjt in kjt_list], dim=0)
    bs_offset = torch.ops.fbgemm.asynchronous_complete_cumsum(
        torch.tensor(bs_list)
    ).int()
    batched_offset = torch.ops.fbgemm.asynchronous_complete_cumsum(batched_length)
    reorder_length = torch.ops.fbgemm.reorder_batched_ad_lengths(
        batched_length, bs_offset, bs
    )
    reorder_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(reorder_length)
    reorder_indices = torch.ops.fbgemm.reorder_batched_ad_indices(
        batched_offset, batched_indices, reorder_offsets, bs_offset, bs
    )
    out = KeyedJaggedTensor(
        keys=kjt_list[0].keys(),
        lengths=reorder_length.long(),
        values=reorder_indices.long(),
    )
    return out

