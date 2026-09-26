"""Research HSTU components used by DeltaRec on the rating datasets."""
from dataclasses import dataclass,field
from typing import Any,Mapping
from torch import nn
from deltarec.adaptors.hstu.modeling.sequential import embedding_modules as embedding_module
from deltarec.adaptors.hstu.modeling.sequential import input_features_preprocessors as preprocessing
from deltarec.adaptors.hstu.modeling.sequential import output_postprocessors as postprocessing
from deltarec.adaptors.hstu.modeling.sequential import hstu
from deltarec.adaptors.hstu.rails.similarities import dot_product_similarity_fn as similarity
from deltarec.layers.hstu_gdr import install_research_hstu_gdr
MetaBridgeError=ValueError

@dataclass(frozen=True)
class HSTUModelConfig:
    max_history_length:int
    max_output_length:int
    item_embedding_dim:int
    dropout_rate:float
    user_embedding_norm:str
    item_l2_norm:bool
    l2_norm_eps:float
    architecture:Mapping[str,Any]=field(default_factory=dict)
    @property
    def total_sequence_length(self):return self.max_history_length+self.max_output_length

def build_model(model_config,*,max_item_id,seed,kernel_backend,decay_timescale_range, install_gdr=True):
    embedding = embedding_module.LocalEmbeddingModule(
        num_items=max_item_id,
        item_embedding_dim=model_config.item_embedding_dim,
    )
    preprocessor = preprocessing.LearnablePositionalEmbeddingInputFeaturesPreprocessor(
        max_sequence_len=model_config.total_sequence_length,
        embedding_dim=model_config.item_embedding_dim,
        dropout_rate=model_config.dropout_rate,
    )
    if model_config.user_embedding_norm == "l2_norm":
        postprocessor = postprocessing.L2NormEmbeddingPostprocessor(
            embedding_dim=model_config.item_embedding_dim, eps=1e-6
        )
    elif model_config.user_embedding_norm == "layer_norm":
        postprocessor = postprocessing.LayerNormEmbeddingPostprocessor(
            embedding_dim=model_config.item_embedding_dim, eps=1e-6
        )
    else:
        raise MetaBridgeError("unsupported official output normalization")
    architecture = dict(model_config.architecture)
    model = hstu.HSTU(
        max_sequence_len=model_config.max_history_length,
        max_output_len=model_config.max_output_length,
        embedding_dim=model_config.item_embedding_dim,
        num_blocks=int(architecture["num_blocks"]),
        num_heads=int(architecture["num_heads"]),
        linear_dim=int(architecture["dv"]),
        attention_dim=int(architecture["dqk"]),
        normalization=str(architecture["normalization"]),
        linear_config=str(architecture["linear_config"]),
        linear_activation=str(architecture["linear_activation"]),
        linear_dropout_rate=float(architecture["linear_dropout_rate"]),
        attn_dropout_rate=float(architecture["attn_dropout_rate"]),
        embedding_module=embedding,
        similarity_module=similarity.DotProductSimilarity(),
        input_features_preproc_module=preprocessor,
        output_postproc_module=postprocessor,
        enable_relative_attention_bias=bool(
            architecture["enable_relative_attention_bias"]
        ),
        concat_ua=bool(architecture["concat_ua"]),
        verbose=False,
    )
    if install_gdr:
        install_research_hstu_gdr(model,seed=seed,kernel_backend=kernel_backend,
            decay_timescale_range=decay_timescale_range)
    return model
