from __future__ import annotations
import json
from pathlib import Path
import time
import hashlib
CANDIDATE_IDENTITY = 'slate-id,exposure-position,item-id,timestamp,play-time,duration,eight-labels'

def capture_forward(teacher, sample):
    from deltarec.data.kuai_loader import _find_multitask_module
    raw = []
    hook = _find_multitask_module(teacher.model)._prediction_module.register_forward_hook(lambda _m, _i, output: raw.append(output))
    try:
        output = teacher.model.forward(sample.uih_features_kjt, sample.candidates_features_kjt)
    finally:
        hook.remove()
    if len(raw) != 1:
        raise ValueError('expected one raw-logit forward')
    return (raw[0], output[4], output[5])

def generate_labels(teacher, output, *, sample_count=1024):
    import numpy as np
    import torch
    from deltarec.data.kuai_slates import HeadlineKuaiSlateDataset
    from deltarec.utils.kuai_training import write_immutable_json
    from deltarec.utils.hstu_selector_training import write_cwi_label_shard, load_cwi_label_shard, create_cwi_label_manifest
    from deltarec.layers.kuai_cwi import PackedCWIStack, candidate_cwi_losses
    from deltarec.adaptors.kuai.dlrm_v3.datasets.dataset import collate_fn
    identity = teacher.identity
    dataset = HeadlineKuaiSlateDataset(teacher.files['train'], teacher.files['user_features'])
    indices = sorted(np.random.default_rng(1).choice(len(dataset), sample_count, replace=False).tolist())
    write_immutable_json(output / 'cwi_sampling.json', {'split': 'train', 'seed': 1, 'sampling': 'uniform-without-replacement-from-frozen-training-slates', 'sample_count': sample_count, 'indices': indices, 'history_cap': 1024, 'candidate_count': 32, 'candidates_resampled': False, 'test_access': False})
    root = teacher.model.module
    full = teacher.active_stack
    stack = PackedCWIStack(full)
    root._hstu_transducer._stu_module = stack
    label_root = output / 'cwi_labels'
    label_root.mkdir(parents=True, exist_ok=True)
    shards = []
    started = time.perf_counter()
    try:
        for number, index in enumerate(indices):
            shard = label_root / f'train-{number:05d}.pt'
            shards.append(shard)
            if shard.exists():
                saved = load_cwi_label_shard(shard)
                if saved['parent_full_checkpoint_sha256'] != identity['file_tree_sha256']:
                    raise ValueError('CWI resume teacher changed')
                continue
            row = dataset._row(index)
            sample = collate_fn([dataset[index]])
            sample.to(teacher.device)
            raw, labels, weights = capture_forward(teacher, sample)
            losses = candidate_cwi_losses(raw, labels, weights, batch_size=1)
            cwi = -torch.autograd.grad(losses.sum(), stack.last_gate)[0]
            history = [int(v) for v in row['history_item_ids'].split(',')][-1024:]
            candidates = [int(v) for v in row['candidate_item_ids'].split(',')]
            if len(history) != int(stack.last_history_lengths[0]):
                raise ValueError('CWI source history does not match teacher preprocessing')
            write_cwi_label_shard(shard, dataset='kuairand-1k', seed=1, parent_full_checkpoint_sha256=identity['file_tree_sha256'], candidate_identity=CANDIDATE_IDENTITY, user_ids=torch.tensor([int(row['user_id'])]), history_item_ids=torch.tensor([history]), history_lengths=torch.tensor([len(history)]), candidate_item_ids=torch.tensor([candidates]), cwi1=cwi.detach().cpu())
            del raw, labels, weights, losses, cwi
            stack.last_gate = None
            if number % 16 == 0:
                print(f'CWI labels {number + 1}/{sample_count}; elapsed {time.perf_counter() - started:.1f}s', flush=True)
        protocol = {'protocol_sha256': teacher.config['config_sha256']}
        create_cwi_label_manifest(label_root / 'manifest.json', dataset='kuairand-1k', seed=1, protocol_sha256=protocol['protocol_sha256'], parent_full_checkpoint_sha256=identity['file_tree_sha256'], parent_full_manifest_content_sha256=identity['manifest_content_sha256'], split_manifest=teacher.files['split_manifest'], training_histories=teacher.files['train'], training_catalog=teacher.files['train_catalog'], item_id_map=teacher.files['item_map'], candidate_identity=CANDIDATE_IDENTITY, shard_paths=shards)
    finally:
        root._hstu_transducer._stu_module = full

def make_pc_stack(teacher, selector_state, ratio, *, optimized=True):
    from deltarec.adaptors.kuai.modules.stu import STUStack
    from deltarec.adaptors.kuai.modules.dlrm_candidate_symmetric import DLRMv3CandidateSymmetricSTUStack, SharedEmbeddingCandidateSelector
    from deltarec.adaptors.kuai_runtime import configure_sparse_stack
    full = teacher.active_stack
    live = SharedEmbeddingCandidateSelector(embedding_dim=512, hidden_dim=512, seed=1)
    result = live.mlp.load_state_dict(selector_state, strict=False)
    if set(result.missing_keys) != {'utility_scale'} or result.unexpected_keys:
        raise ValueError('CWI MLP weight coverage mismatch')
    for parameter in live.parameters():
        parameter.requires_grad_(False)
    stack = DLRMv3CandidateSymmetricSTUStack(STUStack([layer.base for layer in full.layers]), selector=live, seed=1, kernel_backend='fla', recent_floor=32, retention_ratio=ratio, candidate_chunk_size=32, contextual_seq_len=full.contextual_seq_len)
    stack.layers.load_state_dict(full.layers.state_dict(), strict=True)
    stack.to(teacher.device).eval()
    if optimized:
        configure_sparse_stack(stack, method='delta_pc')
    return stack
