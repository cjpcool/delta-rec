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

# pyre-strict
"""
Configuration module for DLRMv3 model.

This module provides configuration functions for the HSTU model architecture and embedding table configurations.
"""

from typing import Dict

from deltarec.adaptors.kuai.modules.dlrm_hstu import DlrmHSTUConfig
from deltarec.adaptors.kuai.modules.multitask_module import (
    MultitaskTaskType,
    TaskConfig,
)
from torchrec.modules.embedding_configs import DataType, EmbeddingConfig

HSTU_EMBEDDING_DIM = 512  # final DLRMv3 model
HASH_SIZE = 10_000_000
HASH_SIZE_1B = 1_000_000_000


def get_hstu_configs(dataset: str = "kuairand-1k") -> DlrmHSTUConfig:
    """
    Create and return HSTU model configuration.

    Builds a complete DlrmHSTUConfig with default hyperparameters for the HSTU
    architecture including attention settings, embedding dimensions, dropout rates,
    and feature name mappings.

    Args:
        dataset: Dataset identifier (currently unused, reserved for dataset-specific configs).

    Returns:
        DlrmHSTUConfig: Complete configuration object for the HSTU model.
    """
    hstu_config = DlrmHSTUConfig(
        hstu_num_heads=4,
        hstu_attn_linear_dim=128,
        hstu_attn_qk_dim=128,
        hstu_attn_num_layers=5,
        hstu_embedding_table_dim=HSTU_EMBEDDING_DIM,
        hstu_preprocessor_hidden_dim=256,
        hstu_transducer_embedding_dim=512,
        hstu_group_norm=False,
        hstu_input_dropout_ratio=0.2,
        hstu_linear_dropout_rate=0.1,
        causal_multitask_weights=0.2,
    )
    hstu_config.user_embedding_feature_names = [
        "video_id",
        "user_id",
        "user_active_degree",
        "follow_user_num_range",
        "fans_user_num_range",
        "friend_user_num_range",
        "register_days_range",
    ]
    hstu_config.item_embedding_feature_names = [
        "item_video_id",
    ]
    hstu_config.uih_post_id_feature_name = "video_id"
    hstu_config.uih_action_time_feature_name = "action_timestamp"
    hstu_config.candidates_querytime_feature_name = "item_query_time"
    hstu_config.uih_weight_feature_name = "action_weight"
    hstu_config.candidates_weight_feature_name = "item_action_weight"
    hstu_config.candidates_watchtime_feature_name = "item_target_watchtime"
    # There are more contextual features in the dataset, see https://kuairand.com/ for details
    hstu_config.contextual_feature_to_max_length = {
        "user_id": 1,
        "user_active_degree": 1,
        "follow_user_num_range": 1,
        "fans_user_num_range": 1,
        "friend_user_num_range": 1,
        "register_days_range": 1,
    }
    hstu_config.merge_uih_candidate_feature_mapping = [
        ("video_id", "item_video_id"),
        ("action_timestamp", "item_query_time"),
        ("action_weight", "item_action_weight"),
        ("watch_time", "item_target_watchtime"),
    ]
    hstu_config.hstu_uih_feature_names = [
        "user_id",
        "user_active_degree",
        "follow_user_num_range",
        "fans_user_num_range",
        "friend_user_num_range",
        "register_days_range",
        "video_id",
        "action_timestamp",
        "action_weight",
        "watch_time",
    ]
    hstu_config.hstu_candidate_feature_names = [
        "item_video_id",
        "item_action_weight",
        "item_target_watchtime",
        "item_query_time",
    ]
    hstu_config.multitask_configs = [
        TaskConfig(
            task_name="is_click",
            task_weight=1,
            task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
        ),
        TaskConfig(
            task_name="is_like",
            task_weight=2,
            task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
        ),
        TaskConfig(
            task_name="is_follow",
            task_weight=4,
            task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
        ),
        TaskConfig(
            task_name="is_comment",
            task_weight=8,
            task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
        ),
        TaskConfig(
            task_name="is_forward",
            task_weight=16,
            task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
        ),
        TaskConfig(
            task_name="is_hate",
            task_weight=32,
            task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
        ),
        TaskConfig(
            task_name="long_view",
            task_weight=64,
            task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
        ),
        TaskConfig(
            task_name="is_profile_enter",
            task_weight=128,
            task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
        ),
    ]
    hstu_config.action_weights = [1, 2, 4, 8, 16, 32, 64, 128]
    return hstu_config


def get_embedding_table_config(dataset: str = "kuairand-1k") -> Dict[str, EmbeddingConfig]:
    """
    Create and return embedding table configurations.

    Defines the embedding table configurations for item IDs, category IDs, and user IDs
    with their respective dimensions and data types.

    Args:
        dataset: Dataset identifier (currently unused, reserved for dataset-specific configs).

    Returns:
        Dict mapping table names to their EmbeddingConfig objects.
    """
    return {
        "video_id": EmbeddingConfig(
            num_embeddings=HASH_SIZE,
            embedding_dim=HSTU_EMBEDDING_DIM,
            name="video_id",
            data_type=DataType.FP16,
            feature_names=["video_id", "item_video_id"],
        ),
        "user_id": EmbeddingConfig(
            num_embeddings=HASH_SIZE,
            embedding_dim=HSTU_EMBEDDING_DIM,
            name="user_id",
            data_type=DataType.FP16,
            feature_names=["user_id"],
        ),
        "user_active_degree": EmbeddingConfig(
            num_embeddings=8,
            embedding_dim=HSTU_EMBEDDING_DIM,
            name="user_active_degree",
            data_type=DataType.FP16,
            feature_names=["user_active_degree"],
        ),
        "follow_user_num_range": EmbeddingConfig(
            num_embeddings=9,
            embedding_dim=HSTU_EMBEDDING_DIM,
            name="follow_user_num_range",
            data_type=DataType.FP16,
            feature_names=["follow_user_num_range"],
        ),
        "fans_user_num_range": EmbeddingConfig(
            num_embeddings=9,
            embedding_dim=HSTU_EMBEDDING_DIM,
            name="fans_user_num_range",
            data_type=DataType.FP16,
            feature_names=["fans_user_num_range"],
        ),
        "friend_user_num_range": EmbeddingConfig(
            num_embeddings=8,
            embedding_dim=HSTU_EMBEDDING_DIM,
            name="friend_user_num_range",
            data_type=DataType.FP16,
            feature_names=["friend_user_num_range"],
        ),
        "register_days_range": EmbeddingConfig(
            num_embeddings=8,
            embedding_dim=HSTU_EMBEDDING_DIM,
            name="register_days_range",
            data_type=DataType.FP16,
            feature_names=["register_days_range"],
        ),
    }
