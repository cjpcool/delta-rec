"""DLRMv3 components and DeltaRec state boundaries for KuaiRand."""
from types import SimpleNamespace
import random
import numpy as np
import torch
from deltarec.adaptors.kuai.modules.dlrm_hstu import DlrmHSTU
from deltarec.adaptors.kuai.dlrm_v3.configs import get_hstu_configs, get_embedding_table_config
from deltarec.adaptors.kuai.modules.dlrm_delta_rec import DLRMv3DeltaRecSTUStack
from deltarec.adaptors.kuai.modules.dlrm_candidate_symmetric import SharedEmbeddingCandidateSelector, DLRMv3CandidateSymmetricSTUStack
from deltarec.adaptors.kuai.modules.dlrm_group_shared import DLRMv3GroupSharedSTUStack, GroupSharedDeltaRecConfig
from deltarec.adaptors.kuai_runtime import configure_sparse_stack
from deltarec.data.kuai_loader import _validate_and_apply_protocol_shape


def build_model(config):
    seed=int(config['seed'])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    architecture=_validate_and_apply_protocol_shape(get_hstu_configs('kuairand-1k'))
    tables=get_embedding_table_config('kuairand-1k')
    return DlrmHSTU(hstu_configs=architecture,embedding_tables=tables,is_inference=False), architecture


def activate(model, config, *, mode, selector_state=None, grouping=None):
    original=model._hstu_transducer._stu_module
    contextual=model._hstu_transducer._positional_encoder._contextual_seq_len
    common=dict(seed=int(config['seed']),kernel_backend=config['kernel_backend'],contextual_seq_len=contextual)
    if mode=='full_gdr':
        stack=DLRMv3DeltaRecSTUStack(original,mode='full_gdr',**common)
        lower,upper=config['decay_timescale_range']
        with torch.no_grad():
            for index,layer in enumerate(stack.layers):
                generator=torch.Generator(device='cpu').manual_seed(config['seed']+index)
                dt=torch.exp(torch.rand(layer.base._num_heads,generator=generator)*(torch.log(torch.tensor(upper))-torch.log(torch.tensor(lower)))+torch.log(torch.tensor(lower))).clamp_min(1e-6)
                layer.gdr_decay_bias.copy_(dt+torch.log(-torch.expm1(-dt)))
    else:
        if selector_state is None:raise ValueError('A trained CWI selector is required')
        width=selector_state['input.weight'].shape[1]//3
        selector=SharedEmbeddingCandidateSelector(embedding_dim=width,hidden_dim=selector_state['input.weight'].shape[0],seed=config['seed'],utility_scale=1.)
        state={k:v for k,v in selector_state.items() if not k.startswith('_selector_embedding_snapshot.')}
        result=selector.mlp.load_state_dict(state,strict=False)
        if set(result.missing_keys)!={'utility_scale'} or result.unexpected_keys:raise ValueError('Selector state coverage changed')
        selector.requires_grad_(False)
        if mode=='delta_pc':
            stack=DLRMv3CandidateSymmetricSTUStack(original,selector=selector,recent_floor=32,retention_ratio=.25,candidate_chunk_size=32,**common)
        elif mode=='delta_gc':
            if grouping is None:raise ValueError('Bound category grouping is required')
            stack=DLRMv3GroupSharedSTUStack(original,selector=selector,
                config=GroupSharedDeltaRecConfig(candidate_group_count=config['group_count'],grouping_policy='fixed_category_prototype_v1',group_pool='logmeanexp',retention_ratio=.25,recent_floor=32,execution_chunk_size=32,projection_dtype='bfloat16',state_dtype='float32',projection_mode='packed_mm',validate_runtime=True),
                item_to_category_group=grouping['item_to_category_group'],category_group_prototypes=grouping['category_group_prototypes'],unknown_category_group=0,**common)
        else:raise ValueError('Only internal Full-GDR, PC selector validation and DeltaRec-GC are supported')
        configure_sparse_stack(stack,method=mode)
    model._hstu_transducer._stu_module=stack
    object.__setattr__(model,"_active_dlrmv3_backend",mode)
    return stack


def make_optimizer_and_shard(model, config, *, device, learning_rate_multiplier):
    import gin
    from deltarec.adaptors.kuai.dlrm_v3.train import utils
    common=dict(learning_rate=.001,momentum=0,weight_decay=0,eps=1e-8,betas=(.95,.999))
    for name,opt in [('dense_optimizer_factory_and_class','Adam'),('sparse_optimizer_factory_and_class','RowWiseAdagrad')]:
        for key,value in dict(common,optimizer_name=opt).items():gin.bind_parameter(name+'.'+key,value)
    return utils.make_optimizer_and_shard(model=model,device=device,world_size=1,learning_rate_multiplier=learning_rate_multiplier)
