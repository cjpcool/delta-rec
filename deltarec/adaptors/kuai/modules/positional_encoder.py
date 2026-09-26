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

#!/usr/bin/env python3

# pyre-strict

from math import sqrt
from typing import Optional

import torch
from deltarec.adaptors.kuai.common import HammerModule
from deltarec.adaptors.kuai.ops.position import add_timestamp_positional_embeddings
from deltarec.adaptors.kuai.ops.pytorch.pt_position import _get_col_indices


class HSTUPositionalEncoder(HammerModule):
    def __init__(
        self,
        num_position_buckets: int,
        num_time_buckets: int,
        embedding_dim: int,
        contextual_seq_len: int,
        is_inference: bool = True,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._embedding_dim: int = embedding_dim
        self._contextual_seq_len: int = contextual_seq_len
        self._position_embeddings_weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(num_position_buckets, embedding_dim).uniform_(
                -sqrt(1.0 / num_position_buckets),
                sqrt(1.0 / num_position_buckets),
            ),
        )
        self._timestamp_embeddings_weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(num_time_buckets + 1, embedding_dim).uniform_(
                -sqrt(1.0 / num_time_buckets),
                sqrt(1.0 / num_time_buckets),
            ),
        )

    def forward(
        self,
        max_seq_len: int,
        seq_lengths: torch.Tensor,
        seq_offsets: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_embeddings = add_timestamp_positional_embeddings(
            alpha=self._embedding_dim**0.5,
            max_seq_len=max_seq_len,
            max_contextual_seq_len=self._contextual_seq_len,
            position_embeddings_weight=self._position_embeddings_weight,
            timestamp_embeddings_weight=self._timestamp_embeddings_weight,
            seq_offsets=seq_offsets,
            seq_lengths=seq_lengths,
            seq_embeddings=seq_embeddings,
            timestamps=seq_timestamps,
            num_targets=num_targets,
            interleave_targets=False,
            kernel=self.hammer_kernel(),
        )
        if position_ids is not None:
            if position_ids.shape != (len(seq_embeddings),) or position_ids.dtype not in (
                torch.int32,
                torch.int64,
            ):
                raise ValueError("position_ids must be an integer packed token vector")
            current_dense = _get_col_indices(
                max_seq_len=max_seq_len,
                max_contextual_seq_len=self._contextual_seq_len,
                max_pos_ind=self._position_embeddings_weight.shape[0],
                seq_lengths=seq_lengths,
                num_targets=num_targets,
                interleave_targets=False,
            )
            current_rows = [
                current_dense[row, : int(length)]
                for row, length in enumerate(seq_lengths)
            ]
            current = (
                torch.cat(current_rows)
                if current_rows
                else seq_lengths.new_empty(0)
            )
            desired = position_ids.to(current.device).long().clamp(
                0, self._position_embeddings_weight.shape[0] - 1
            )
            correction = self._position_embeddings_weight.index_select(0, desired) - (
                self._position_embeddings_weight.index_select(0, current.long())
            )
            seq_embeddings = seq_embeddings + correction.to(seq_embeddings.dtype)
        return seq_embeddings

