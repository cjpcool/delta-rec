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


import logging
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
from deltarec.adaptors.kuai.common import (
    fx_infer_max_len,
    fx_mark_length_features,
    HammerKernel,
    HammerModule,
    init_mlp_weights_optional_bias,
    set_static_max_seq_lens,
)
from deltarec.adaptors.kuai.modules.hstu_transducer import HSTUTransducer
from deltarec.adaptors.kuai.modules.multitask_module import (
    DefaultMultitaskModule,
    MultitaskTaskType,
    TaskConfig,
)
from deltarec.adaptors.kuai.modules.positional_encoder import HSTUPositionalEncoder
from deltarec.adaptors.kuai.modules.postprocessors import (
    LayerNormPostprocessor,
    TimestampLayerNormPostprocessor,
)
from deltarec.adaptors.kuai.modules.preprocessors import ContextualPreprocessor
from deltarec.adaptors.kuai.modules.stu import STU, STULayer, STULayerConfig, STUStack
from deltarec.adaptors.kuai.ops.jagged_tensors import concat_2D_jagged
from deltarec.adaptors.kuai.ops.layer_norm import LayerNorm, SwishLayerNorm
from torch.autograd.profiler import record_function
from torchrec import KeyedJaggedTensor
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection

logger: logging.Logger = logging.getLogger(__name__)

torch.fx.wrap("fx_infer_max_len")
torch.fx.wrap("len")


class SequenceEmbedding(NamedTuple):
    lengths: torch.Tensor
    embedding: torch.Tensor


@dataclass
class DlrmHSTUConfig:
    max_seq_len: int = 16384
    max_num_candidates: int = 10
    max_num_candidates_inference: int = 5
    hstu_num_heads: int = 1
    hstu_attn_linear_dim: int = 256
    hstu_attn_qk_dim: int = 128
    hstu_attn_num_layers: int = 12
    hstu_embedding_table_dim: int = 192
    hstu_preprocessor_hidden_dim: int = 256
    hstu_transducer_embedding_dim: int = 0
    hstu_group_norm: bool = False
    hstu_input_dropout_ratio: float = 0.2
    hstu_linear_dropout_rate: float = 0.2
    contextual_feature_to_max_length: Dict[str, int] = field(default_factory=dict)
    contextual_feature_to_min_uih_length: Dict[str, int] = field(default_factory=dict)
    candidates_weight_feature_name: str = ""
    candidates_watchtime_feature_name: str = ""
    candidates_querytime_feature_name: str = ""
    causal_multitask_weights: float = 0.2
    multitask_configs: List[TaskConfig] = field(default_factory=list)
    user_embedding_feature_names: List[str] = field(default_factory=list)
    item_embedding_feature_names: List[str] = field(default_factory=list)
    uih_post_id_feature_name: str = ""
    uih_action_time_feature_name: str = ""
    uih_weight_feature_name: str = ""
    hstu_uih_feature_names: List[str] = field(default_factory=list)
    hstu_candidate_feature_names: List[str] = field(default_factory=list)
    merge_uih_candidate_feature_mapping: List[Tuple[str, str]] = field(
        default_factory=list
    )
    action_weights: Optional[List[int]] = None
    action_embedding_init_std: float = 0.1
    enable_postprocessor: bool = True
    use_layer_norm_postprocessor: bool = False


def _get_supervision_labels_and_weights(
    supervision_bitmasks: torch.Tensor,
    watchtime_sequence: torch.Tensor,
    task_configs: List[TaskConfig],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    supervision_labels: Dict[str, torch.Tensor] = {}
    supervision_weights: Dict[str, torch.Tensor] = {}
    for task in task_configs:
        if task.task_type == MultitaskTaskType.REGRESSION:
            supervision_labels[task.task_name] = watchtime_sequence.to(torch.float32)
        elif task.task_type == MultitaskTaskType.BINARY_CLASSIFICATION:
            supervision_labels[task.task_name] = (
                torch.bitwise_and(supervision_bitmasks, task.task_weight) > 0
            ).to(torch.float32)
        else:
            raise RuntimeError("Unsupported MultitaskTaskType")
    return supervision_labels, supervision_weights


class DlrmHSTU(HammerModule):
    def __init__(  # noqa C901
        self,
        hstu_configs: DlrmHSTUConfig,
        embedding_tables: Dict[str, EmbeddingConfig],
        is_inference: bool,
        is_dense: bool = False,
        bf16_training: bool = True,
    ) -> None:
        super().__init__(is_inference=is_inference)
        logger.info(f"Initialize HSTU module with configs {hstu_configs}")
        self._hstu_configs = hstu_configs
        self._bf16_training: bool = bf16_training
        set_static_max_seq_lens([self._hstu_configs.max_seq_len])

        if not is_dense:
            self._embedding_collection: EmbeddingCollection = EmbeddingCollection(
                tables=list(embedding_tables.values()),
                need_indices=False,
                device=torch.device("meta"),
            )

        # multitask configs must be sorted by task types
        self._multitask_configs: List[TaskConfig] = hstu_configs.multitask_configs
        self._multitask_module = DefaultMultitaskModule(
            task_configs=self._multitask_configs,
            embedding_dim=hstu_configs.hstu_transducer_embedding_dim,
            prediction_fn=lambda in_dim, num_tasks: torch.nn.Sequential(
                torch.nn.Linear(in_features=in_dim, out_features=512),
                SwishLayerNorm(512),
                torch.nn.Linear(in_features=512, out_features=num_tasks),
            ).apply(init_mlp_weights_optional_bias),
            causal_multitask_weights=hstu_configs.causal_multitask_weights,
            is_inference=self._is_inference,
        )
        self._additional_embedding_features: List[str] = [
            uih_feature_name
            for (
                uih_feature_name,
                candidate_feature_name,
            ) in self._hstu_configs.merge_uih_candidate_feature_mapping
            if (
                candidate_feature_name
                in self._hstu_configs.item_embedding_feature_names
            )
            and (uih_feature_name in self._hstu_configs.user_embedding_feature_names)
            and (uih_feature_name is not self._hstu_configs.uih_post_id_feature_name)
        ]

        # preprocessor setup
        preprocessor = ContextualPreprocessor(
            input_embedding_dim=hstu_configs.hstu_embedding_table_dim,
            hidden_dim=hstu_configs.hstu_preprocessor_hidden_dim,
            output_embedding_dim=hstu_configs.hstu_transducer_embedding_dim,
            contextual_feature_to_max_length=hstu_configs.contextual_feature_to_max_length,
            contextual_feature_to_min_uih_length=hstu_configs.contextual_feature_to_min_uih_length,
            action_embedding_dim=8,
            action_feature_name=self._hstu_configs.uih_weight_feature_name,
            action_weights=self._hstu_configs.action_weights,
            action_embedding_init_std=self._hstu_configs.action_embedding_init_std,
            additional_embedding_features=self._additional_embedding_features,
            is_inference=is_inference,
        )

        # positional encoder
        positional_encoder = HSTUPositionalEncoder(
            num_position_buckets=8192,
            num_time_buckets=2048,
            embedding_dim=hstu_configs.hstu_transducer_embedding_dim,
            contextual_seq_len=sum(
                dict(hstu_configs.contextual_feature_to_max_length).values()
            ),
            is_inference=self._is_inference,
        )

        if hstu_configs.enable_postprocessor:
            if hstu_configs.use_layer_norm_postprocessor:
                postprocessor = LayerNormPostprocessor(
                    embedding_dim=hstu_configs.hstu_transducer_embedding_dim,
                    eps=1e-5,
                    is_inference=self._is_inference,
                )
            else:
                postprocessor = TimestampLayerNormPostprocessor(
                    embedding_dim=hstu_configs.hstu_transducer_embedding_dim,
                    time_duration_features=[
                        (60 * 60, 24),  # hour of day
                        (24 * 60 * 60, 7),  # day of week
                        # (24 * 60 * 60, 365), # time of year (approximate)
                    ],
                    eps=1e-5,
                    is_inference=self._is_inference,
                )
        else:
            postprocessor = None

        # construct HSTU
        stu_module: STU = STUStack(
            stu_list=[
                STULayer(
                    config=STULayerConfig(
                        embedding_dim=hstu_configs.hstu_transducer_embedding_dim,
                        num_heads=hstu_configs.hstu_num_heads,
                        hidden_dim=hstu_configs.hstu_attn_linear_dim,
                        attention_dim=hstu_configs.hstu_attn_qk_dim,
                        output_dropout_ratio=hstu_configs.hstu_linear_dropout_rate,
                        use_group_norm=hstu_configs.hstu_group_norm,
                        causal=True,
                        target_aware=True,
                        max_attn_len=None,
                        attn_alpha=None,
                        recompute_normed_x=True,
                        recompute_uvqk=True,
                        recompute_y=True,
                        sort_by_length=True,
                        contextual_seq_len=0,
                    ),
                    is_inference=is_inference,
                )
                for _ in range(hstu_configs.hstu_attn_num_layers)
            ],
            is_inference=is_inference,
        )
        self._hstu_transducer: HSTUTransducer = HSTUTransducer(
            stu_module=stu_module,
            input_preprocessor=preprocessor,
            output_postprocessor=postprocessor,
            input_dropout_ratio=hstu_configs.hstu_input_dropout_ratio,
            positional_encoder=positional_encoder,
            is_inference=self._is_inference,
            return_full_embeddings=False,
            listwise=False,
        )

        # item embeddings
        self._item_embedding_mlp: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=hstu_configs.hstu_embedding_table_dim
                * len(self._hstu_configs.item_embedding_feature_names),
                out_features=512,
            ),
            SwishLayerNorm(512),
            torch.nn.Linear(
                in_features=512,
                out_features=hstu_configs.hstu_transducer_embedding_dim,
            ),
            LayerNorm(hstu_configs.hstu_transducer_embedding_dim),
        ).apply(init_mlp_weights_optional_bias)

    def _construct_payload(
        self,
        payload_features: Dict[str, torch.Tensor],
        seq_embeddings: Dict[str, SequenceEmbedding],
    ) -> Dict[str, torch.Tensor]:
        if len(self._hstu_configs.contextual_feature_to_max_length) > 0:
            contextual_offsets: List[torch.Tensor] = []
            for x in self._hstu_configs.contextual_feature_to_max_length.keys():
                contextual_offsets.append(
                    torch.ops.fbgemm.asynchronous_complete_cumsum(
                        seq_embeddings[x].lengths
                    )
                )
        else:
            # Dummy, offsets are unused
            contextual_offsets = torch.empty((0, 0))
        return {
            **payload_features,
            **{
                x: seq_embeddings[x].embedding
                for x in self._hstu_configs.contextual_feature_to_max_length.keys()
            },
            **{
                x + "_offsets": contextual_offsets[i]
                for i, x in enumerate(
                    list(self._hstu_configs.contextual_feature_to_max_length.keys())
                )
            },
            **{
                x: seq_embeddings[x].embedding
                for x in self._additional_embedding_features
            },
        }

    def _delta_rec_position_ids(
        self,
        source_lengths: torch.Tensor,
        num_candidates: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Build original chronological position IDs after KJT compaction."""

        stack = self._hstu_transducer._stu_module
        source_positions = getattr(stack, "last_history_source_positions", None)
        original_lengths = getattr(stack, "last_original_history_lengths", None)
        if source_positions is None or original_lengths is None:
            return None
        compact_lengths = source_lengths.long() - num_candidates.long()
        if bool((compact_lengths < 0).any()) or (
            original_lengths.shape != compact_lengths.shape
        ):
            raise RuntimeError("DeltaRec position metadata does not match the batch")
        offsets = torch.cat(
            (compact_lengths.new_zeros(1), compact_lengths.cumsum(0))
        )
        contextual = int(
            getattr(self._hstu_transducer._positional_encoder, "_contextual_seq_len", 0)
        )
        rows: list[torch.Tensor] = []
        for row in range(len(compact_lengths)):
            start = int(offsets[row])
            end = int(offsets[row + 1])
            selected_positions = source_positions[start:end].to(source_lengths.device)
            rows.append(
                torch.cat(
                    (
                        torch.arange(contextual, device=source_lengths.device),
                        original_lengths[row].to(source_lengths.device)
                        - selected_positions
                        + contextual,
                        torch.full(
                            (int(num_candidates[row]),),
                            contextual,
                            device=source_lengths.device,
                            dtype=torch.int64,
                        ),
                    )
                ).long()
            )
        stack.last_history_source_positions = None
        stack.last_original_history_lengths = None
        return torch.cat(rows) if rows else source_lengths.new_empty(0)

    def _user_forward(
        self,
        max_uih_len: int,
        max_candidates: int,
        seq_embeddings: Dict[str, SequenceEmbedding],
        payload_features: Dict[str, torch.Tensor],
        num_candidates: torch.Tensor,
    ) -> torch.Tensor:
        source_lengths = seq_embeddings[
            self._hstu_configs.uih_post_id_feature_name
        ].lengths
        if "_non_jagged_source_timestamps" in payload_features:
            source_timestamps = payload_features["_non_jagged_source_timestamps"]
        else:
            source_timestamps = concat_2D_jagged(
                max_seq_len=max_uih_len + max_candidates,
                max_len_left=max_uih_len,
                offsets_left=payload_features["uih_offsets"],
                values_left=payload_features[
                    self._hstu_configs.uih_action_time_feature_name
                ].unsqueeze(-1),
                max_len_right=max_candidates,
                offsets_right=payload_features["candidate_offsets"],
                values_right=payload_features[
                    self._hstu_configs.candidates_querytime_feature_name
                ].unsqueeze(-1),
                kernel=self.hammer_kernel(),
            ).squeeze(-1)
        total_targets = int(num_candidates.sum().item())
        embedding = seq_embeddings[
            self._hstu_configs.uih_post_id_feature_name
        ].embedding
        dtype = embedding.dtype
        if (not self.is_inference) and self._bf16_training:
            embedding = embedding.to(torch.bfloat16)
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=(not self.is_inference) and self._bf16_training,
        ):
            seq_payloads = self._construct_payload(
                payload_features=payload_features,
                seq_embeddings=seq_embeddings,
            )
            position_ids = self._delta_rec_position_ids(
                source_lengths,
                num_candidates,
            )
            if position_ids is not None:
                seq_payloads["_delta_rec_position_ids"] = position_ids
            candidate_symmetric = getattr(
                self._hstu_transducer._stu_module,
                "forward_candidate_symmetric",
                None,
            )
            group_shared = getattr(
                self._hstu_transducer._stu_module,
                "forward_group_shared",
                None,
            )
            if candidate_symmetric is not None and group_shared is not None:
                raise RuntimeError(
                    "an STU backend cannot expose both PC and GC execution"
                )
            if candidate_symmetric is not None or group_shared is not None:
                history_item_ids = payload_features.get("_deltarec_history_item_ids")
                candidate_item_ids = payload_features.get("_deltarec_candidate_item_ids")
                history_selector_embeddings = payload_features.get(
                    "_deltarec_history_selector_embeddings"
                )
                candidate_selector_embeddings = payload_features.get(
                    "_deltarec_candidate_selector_embeddings"
                )
                if (
                    history_item_ids is None
                    or candidate_item_ids is None
                    or history_selector_embeddings is None
                    or candidate_selector_embeddings is None
                ):
                    raise RuntimeError(
                        "DeltaRec DLRMv3 requires raw IDs and exact shared "
                        "history/candidate embedding tensors"
                    )
                execute = (
                    self._hstu_transducer.forward_candidate_symmetric
                    if candidate_symmetric is not None
                    else self._hstu_transducer.forward_group_shared
                )
                candidates_user_embeddings, _, _ = execute(
                        max_uih_len=max_uih_len,
                        max_targets=max_candidates,
                        total_uih_len=source_timestamps.numel() - total_targets,
                        total_targets=total_targets,
                        seq_embeddings=embedding,
                        seq_lengths=source_lengths,
                        seq_timestamps=source_timestamps,
                        seq_payloads=seq_payloads,
                        num_targets=num_candidates,
                        history_item_ids=history_item_ids,
                        candidate_item_ids=candidate_item_ids,
                        history_selector_embeddings=history_selector_embeddings,
                        candidate_selector_embeddings=candidate_selector_embeddings,
                    )
            else:
                candidates_user_embeddings, _ = self._hstu_transducer(
                    max_uih_len=max_uih_len,
                    max_targets=max_candidates,
                    total_uih_len=source_timestamps.numel() - total_targets,
                    total_targets=total_targets,
                    seq_embeddings=embedding,
                    seq_lengths=source_lengths,
                    seq_timestamps=source_timestamps,
                    seq_payloads=seq_payloads,
                    num_targets=num_candidates,
                )
        candidates_user_embeddings = candidates_user_embeddings.to(dtype)

        return candidates_user_embeddings

    def _item_forward(
        self,
        seq_embeddings: Dict[str, SequenceEmbedding],
    ) -> torch.Tensor:  # [L, D]
        all_embeddings = torch.cat(
            [
                seq_embeddings[name].embedding
                for name in self._hstu_configs.item_embedding_feature_names
            ],
            dim=-1,
        )
        item_embeddings = self._item_embedding_mlp(all_embeddings)
        return item_embeddings

    def preprocess(
        self,
        uih_features: KeyedJaggedTensor,
        candidates_features: KeyedJaggedTensor,
    ) -> Tuple[
        Dict[str, SequenceEmbedding],
        Dict[str, torch.Tensor],
        int,
        torch.Tensor,
        int,
        torch.Tensor,
    ]:
        # embedding lookup for uih and candidates
        merged_sparse_features = KeyedJaggedTensor.from_lengths_sync(
            keys=uih_features.keys() + candidates_features.keys(),
            values=torch.cat(
                [uih_features.values(), candidates_features.values()],
                dim=0,
            ),
            lengths=torch.cat(
                [uih_features.lengths(), candidates_features.lengths()],
                dim=0,
            ),
        )
        seq_embeddings_dict = self._embedding_collection(merged_sparse_features)
        num_candidates = fx_mark_length_features(
            candidates_features.lengths().view(len(candidates_features.keys()), -1)
        )[0]
        max_num_candidates = fx_infer_max_len(num_candidates)
        uih_seq_lengths = uih_features[
            self._hstu_configs.uih_post_id_feature_name
        ].lengths()
        max_uih_len = fx_infer_max_len(uih_seq_lengths)

        # prepare payload features
        payload_features: Dict[str, torch.Tensor] = {}
        for (
            uih_feature_name,
            candidate_feature_name,
        ) in self._hstu_configs.merge_uih_candidate_feature_mapping:
            if (
                candidate_feature_name
                not in self._hstu_configs.item_embedding_feature_names
                and uih_feature_name
                not in self._hstu_configs.user_embedding_feature_names
            ):
                values_left = uih_features[uih_feature_name].values()
                if self._is_inference and (
                    candidate_feature_name
                    == self._hstu_configs.candidates_weight_feature_name
                    or candidate_feature_name
                    == self._hstu_configs.candidates_watchtime_feature_name
                ):
                    total_candidates = torch.sum(num_candidates).item()
                    values_right = torch.zeros(
                        total_candidates,  # pyre-ignore
                        dtype=torch.int64,
                        device=values_left.device,
                    )
                else:
                    values_right = candidates_features[candidate_feature_name].values()
                payload_features[uih_feature_name] = values_left
                payload_features[candidate_feature_name] = values_right
        payload_features["uih_offsets"] = torch.ops.fbgemm.asynchronous_complete_cumsum(
            uih_seq_lengths
        )
        payload_features["candidate_offsets"] = (
            torch.ops.fbgemm.asynchronous_complete_cumsum(num_candidates)
        )
        # These fields are deliberately kept out of the public model API.  The
        # contextual preprocessor forwards payload dictionaries unchanged, so
        # the production candidate-symmetric STU can hash IDs after standard
        # embedding, preprocessing, and positional encoding.
        payload_features["_deltarec_history_item_ids"] = uih_features[
            self._hstu_configs.uih_post_id_feature_name
        ].values().to(torch.int64)
        candidate_id_name = next(
            (
                candidate_name
                for uih_name, candidate_name in self._hstu_configs.merge_uih_candidate_feature_mapping
                if uih_name == self._hstu_configs.uih_post_id_feature_name
                and candidate_name in candidates_features.keys()
            ),
            None,
        )
        if candidate_id_name is None:
            for candidate_name in self._hstu_configs.hstu_candidate_feature_names:
                if candidate_name in candidates_features.keys():
                    candidate_id_name = candidate_name
                    break
        if candidate_id_name is None:
            raise KeyError(
                "no configured candidate ID feature is present in candidates_features"
            )
        payload_features["_deltarec_candidate_item_ids"] = candidates_features[
            candidate_id_name
        ].values().to(torch.int64)

        seq_embeddings = {
            k: SequenceEmbedding(
                lengths=seq_embeddings_dict[k].lengths(),
                embedding=seq_embeddings_dict[k].values(),
            )
            for k in self._hstu_configs.user_embedding_feature_names
            + self._hstu_configs.item_embedding_feature_names
        }
        # Preserve the exact outputs of the shared TorchRec ``video_id``
        # table before history/candidate concatenation and contextual
        # projection.  PC/GC selectors consume these tensors directly, so no
        # copied embedding table or detached export can enter training.
        payload_features["_deltarec_history_selector_embeddings"] = (
            seq_embeddings[self._hstu_configs.uih_post_id_feature_name].embedding
        )
        payload_features["_deltarec_candidate_selector_embeddings"] = (
            seq_embeddings[candidate_id_name].embedding
        )

        return (
            seq_embeddings,
            payload_features,
            max_uih_len,
            uih_seq_lengths,
            max_num_candidates,
            num_candidates,
        )

    def main_forward(
        self,
        seq_embeddings: Dict[str, SequenceEmbedding],
        payload_features: Dict[str, torch.Tensor],
        max_uih_len: int,
        uih_seq_lengths: torch.Tensor,
        max_num_candidates: int,
        num_candidates: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        # merge uih and candidates embeddings
        if "_non_jagged_premerged_embeddings" not in payload_features:
            for (
                uih_feature_name,
                candidate_feature_name,
            ) in self._hstu_configs.merge_uih_candidate_feature_mapping:
                if uih_feature_name in seq_embeddings:
                    seq_embeddings[uih_feature_name] = SequenceEmbedding(
                        lengths=uih_seq_lengths + num_candidates,
                        embedding=concat_2D_jagged(
                            max_seq_len=max_uih_len + max_num_candidates,
                            max_len_left=max_uih_len,
                            offsets_left=torch.ops.fbgemm.asynchronous_complete_cumsum(
                                uih_seq_lengths
                            ),
                            values_left=seq_embeddings[uih_feature_name].embedding,
                            max_len_right=max_num_candidates,
                            offsets_right=torch.ops.fbgemm.asynchronous_complete_cumsum(
                                num_candidates
                            ),
                            values_right=seq_embeddings[
                                candidate_feature_name
                            ].embedding,
                            kernel=self.hammer_kernel(),
                        ),
                    )

        with record_function("## item_forward ##"):
            candidates_item_embeddings = self._item_forward(
                seq_embeddings,
            )
        with record_function("## user_forward ##"):
            candidates_user_embeddings = self._user_forward(
                max_uih_len=max_uih_len,
                max_candidates=max_num_candidates,
                seq_embeddings=seq_embeddings,
                payload_features=payload_features,
                num_candidates=num_candidates,
            )
        with record_function("## multitask_module ##"):
            supervision_labels, supervision_weights = (
                _get_supervision_labels_and_weights(
                    supervision_bitmasks=payload_features[
                        self._hstu_configs.candidates_weight_feature_name
                    ],
                    watchtime_sequence=payload_features[
                        self._hstu_configs.candidates_watchtime_feature_name
                    ],
                    task_configs=self._multitask_configs,
                )
            )
            mt_target_preds, mt_target_labels, mt_target_weights, mt_losses = (
                self._multitask_module(
                    encoded_user_embeddings=candidates_user_embeddings,
                    item_embeddings=candidates_item_embeddings,
                    supervision_labels=supervision_labels,
                    supervision_weights=supervision_weights,
                )
            )

        aux_losses: Dict[str, torch.Tensor] = {}
        if not self._is_inference and self.training:
            for i, task in enumerate(self._multitask_configs):
                aux_losses[task.task_name] = mt_losses[i]

        return (
            candidates_user_embeddings,
            candidates_item_embeddings,
            aux_losses,
            mt_target_preds,
            mt_target_labels,
            mt_target_weights,
        )

    def activate_attention_backend(
        self,
        backend: str,
        *,
        compactor: Optional[torch.nn.Module] = None,
        seed: int = 20260818,
        gdr_kernel: str = "triton",
        selector_checkpoint: Optional[str] = None,
        selector_embedding_table: Optional[str] = None,
        recent_floor: int = 32,
        candidate_chunk_size: int = 100,
    ) -> STU:
        """Activate a Subplan 6 backend after loading the stock checkpoint.

        This method intentionally does not run from ``__init__``: DLRMv3's
        dense checkpoint loader requires the exact original state-dict keys.
        The public model forward signature is therefore unchanged and backend
        parameters are introduced only after checkpoint compatibility has been
        established.
        """

        from deltarec.adaptors.kuai.modules.dlrm_delta_rec import (
            DLRMv3BackendConfig,
            activate_dlrmv3_backend,
        )

        return activate_dlrmv3_backend(
            self,
            DLRMv3BackendConfig(
                backend=backend,
                seed=seed,
                gdr_kernel=gdr_kernel,
                selector_checkpoint=selector_checkpoint,
                selector_embedding_table=selector_embedding_table,
                recent_floor=recent_floor,
                candidate_chunk_size=candidate_chunk_size,
            ),
            compactor=compactor,
        )

    @property
    def active_attention_backend(self) -> str:
        return getattr(self, "_active_dlrmv3_backend", "original_hstu")

    def prepare_delta_rec_selection(
        self,
        history: object,
        constraints: Optional[object] = None,
    ) -> None:
        """Attach selector fields for the next DeltaRec forward only."""

        stu_module = self._hstu_transducer._stu_module
        prepare = getattr(stu_module, "prepare_selection", None)
        if prepare is None:
            raise RuntimeError(
                "the active attention backend does not consume DeltaRec selection"
            )
        prepare(history, constraints)

    def forward(
        self,
        uih_features: KeyedJaggedTensor,
        candidates_features: KeyedJaggedTensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        precompact = getattr(
            self._hstu_transducer._stu_module,
            "precompact_history_features",
            None,
        )
        if precompact is not None:
            uih_features = precompact(
                uih_features,
                reference_key=self._hstu_configs.uih_post_id_feature_name,
            )
        with record_function("## preprocess ##"):
            (
                seq_embeddings,
                payload_features,
                max_uih_len,
                uih_seq_lengths,
                max_num_candidates,
                num_candidates,
            ) = self.preprocess(
                uih_features=uih_features,
                candidates_features=candidates_features,
            )

        with record_function("## main_forward ##"):
            return self.main_forward(
                seq_embeddings=seq_embeddings,
                payload_features=payload_features,
                max_uih_len=max_uih_len,
                uih_seq_lengths=uih_seq_lengths,
                max_num_candidates=max_num_candidates,
                num_candidates=num_candidates,
            )

