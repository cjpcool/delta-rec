from __future__ import annotations

import copy

from dataclasses import dataclass, field

import hashlib

import importlib

import json

from pathlib import Path

import sys

import types

from typing import Any, Mapping



FUXI_SOURCE_ID = "fuxi-linear"



KUAI_TASKS = (
    "click",
    "like",
    "follow",
    "comment",
    "forward",
    "hate",
    "long_view",
    "profile_enter",
)

class FuxiBridgeError(RuntimeError):
    pass

class FuxiDependencyError(FuxiBridgeError):
    pass

@dataclass(frozen=True)
class FuxiModelConfig:
    dataset: str
    source_dataset: str
    source_revision: str
    upstream_gin: Path
    max_history_length: int
    max_output_length: int
    item_embedding_dim: int
    dropout_rate: float
    user_embedding_norm: str
    item_l2_norm: bool
    l2_norm_eps: float
    architecture: Mapping[str, Any] = field(default_factory=dict)
    optimizer: Mapping[str, Any] = field(default_factory=dict)
    objective: Mapping[str, Any] = field(default_factory=dict)

    @property
    def total_sequence_length(self) -> int:
        return self.max_history_length + self.max_output_length

    @property
    def multitask(self) -> bool:
        return self.dataset == "kuairand-1k"

    def fingerprint(self) -> str:
        payload = {
            "dataset": self.dataset,
            "source_dataset": self.source_dataset,
            "source_revision": self.source_revision,
            "upstream_gin": str(self.upstream_gin),
            "max_history_length": self.max_history_length,
            "max_output_length": self.max_output_length,
            "item_embedding_dim": self.item_embedding_dim,
            "dropout_rate": self.dropout_rate,
            "user_embedding_norm": self.user_embedding_norm,
            "item_l2_norm": self.item_l2_norm,
            "l2_norm_eps": self.l2_norm_eps,
            "architecture": self.architecture,
            "optimizer": self.optimizer,
            "objective": self.objective,
            "kuai_tasks": KUAI_TASKS if self.multitask else (),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()




def build_upstream_model(
    model_config: FuxiModelConfig,
    *,
    max_item_id: int,
    source_checkout: Path | None = None,
) -> Any:
    """Construct the official FuXiLinear class with parsed publication values."""

    if max_item_id <= 0:
        raise ValueError("max_item_id must be positive")
    if model_config.user_embedding_norm != "l2_norm":
        raise FuxiBridgeError("published FuXi configs require l2_norm output")
    symbols = _load_upstream_symbols(source_checkout)
    embedding = symbols["LocalEmbeddingModule"](
        num_items=max_item_id,
        item_embedding_dim=model_config.item_embedding_dim,
    )
    preprocessor = symbols["LearnablePositionalEmbeddingInputFeaturesPreprocessor"](
        max_sequence_len=model_config.total_sequence_length,
        embedding_dim=model_config.item_embedding_dim,
        dropout_rate=model_config.dropout_rate,
    )
    postprocessor = symbols["L2NormEmbeddingPostprocessor"](
        embedding_dim=model_config.item_embedding_dim,
        eps=1e-6,
    )
    similarity = symbols["DotProductSimilarity"]()
    architecture = copy.deepcopy(dict(model_config.architecture))
    model = symbols["FuXiLinear"](
        max_sequence_len=model_config.max_history_length,
        max_output_len=model_config.max_output_length,
        embedding_dim=model_config.item_embedding_dim,
        num_blocks=int(architecture["num_blocks"]),
        num_heads=int(architecture["num_heads"]),
        linear_dim=int(architecture["dv"]),
        attention_dim=int(architecture["dqk"]),
        normalization=str(architecture["normalization"]),
        linear_activation=str(architecture["linear_activation"]),
        linear_dropout_rate=float(architecture["linear_dropout_rate"]),
        attn_dropout_rate=float(architecture["attn_dropout_rate"]),
        ffn_multiply=int(architecture["ffn_multiply"]),
        embedding_module=embedding,
        similarity_module=similarity,
        input_features_preproc_module=preprocessor,
        output_postproc_module=postprocessor,
        channel_t_config=architecture["channel_t_config"],
        channel_p_config=architecture["channel_p_config"],
        use_rope=bool(architecture["use_rope"]),
        enable_relative_attention_bias=bool(
            architecture["enable_relative_attention_bias"]
        ),
        chunk_size=architecture["chunk_size"],
        verbose=False,
    )
    apply_pinned_forward_compatibility(model)
    return model

def apply_pinned_forward_compatibility(model: Any) -> tuple[str, ...]:
    """Apply the two source-free shims required by pinned commit 5a70406.

    The pinned constructor logs and constructs the multi-head temporal path but
    neglects to create ``LinearTemporalChannel._no_multihead`` before forward
    reads it.  ``False`` selects that already-constructed multi-head path.  No
    parameter, tensor, or published configuration value is changed.
    """

    applied: list[str] = []
    layers = getattr(getattr(model, "_fuxi", None), "_attention_layers", ())
    for index, layer in enumerate(layers):
        channel = getattr(layer, "_channel_t", None)
        if channel is not None and not hasattr(channel, "_no_multihead"):
            channel._no_multihead = False
            applied.append(f"_fuxi._attention_layers.{index}._channel_t._no_multihead=False")
    similarity = getattr(model, "_ndp_module", None)
    if similarity is not None and not getattr(
        similarity, "_deltarec_consistent_tuple", False
    ):
        original_forward = similarity.forward

        def consistent_forward(this: Any, *args: Any, **kwargs: Any) -> Any:
            del this
            output = original_forward(*args, **kwargs)
            return output if isinstance(output, tuple) else (output, {})

        similarity.forward = types.MethodType(consistent_forward, similarity)
        similarity._deltarec_consistent_tuple = True
        applied.append("_ndp_module.forward=rowwise-tuple-contract")
    return tuple(applied)

def _load_upstream_symbols(source_checkout=None):
    from deltarec.adaptors.fuxi.modeling.sequential.embedding_modules import LocalEmbeddingModule
    from deltarec.adaptors.fuxi.modeling.sequential.input_features_preprocessors import LearnablePositionalEmbeddingInputFeaturesPreprocessor
    from deltarec.adaptors.fuxi.modeling.sequential.output_postprocessors import L2NormEmbeddingPostprocessor
    from deltarec.adaptors.fuxi.modeling.similarity.dot_product import DotProductSimilarity
    from deltarec.adaptors.fuxi.modeling.sequential.fuxi_linear import FuXiLinear
    import torch
    return locals()
