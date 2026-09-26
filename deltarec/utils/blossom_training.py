from __future__ import annotations

from contextlib import nullcontext

import csv

import itertools

import json

import os

from pathlib import Path

import random

import socket

import time

import torch

import torch.nn.functional as F

from deltarec.models.blossom_gdr import BlossomDeltaRec

from deltarec.models.hstu_multitask import checkpoint_state_without_legacy_head, hstu_multitask_bce

from deltarec.utils.io import atomic_write_json

from deltarec.adaptors import recbole as bridge

from deltarec.data import training as data

from deltarec.utils.checkpoint import atomic_torch_save

from deltarec.utils.early_stopping import EarlyStoppingState

from deltarec.utils.protocols import AdapterOutput

from deltarec.metrics.evaluation import AtomicJsonlWriter, evaluate_rating_stream, evaluate_kuai_stream



def catalog(path, device):
    with Path(path).open() as f:
        ids = [int(row['item_id']) for row in csv.DictReader(f)]
    return torch.tensor(ids, dtype=torch.long, device=device)

def construct(c, run, files, *, device=None, fixture=False):
    device = device or c.get('device', 'cpu')
    torch.manual_seed(c['seed'])
    random.seed(c['seed'])
    g = torch.load(c['grouping'], map_location='cpu', weights_only=False)
    backbone = bridge.build_upstream_model(run.model,
        max_item_id=data._max_catalog_id(files['full_catalog']), 
        device='cpu', fixture_only=fixture)
    model = BlossomDeltaRec(backbone, g['item_to_category_group'], c['group_count'],
        backend='reference' if fixture else c['kernel'], multitask=run.model.multitask,
        retention_ratio=c['retention_ratio'])
    model.runtime_validation = bool(c.get('runtime_validation', True))
    del backbone, g
    model.to(device)
    return model

def batches(c, files, epoch, *, microbatch=None):
    rows = data._slate_examples(files['train'], role='train') if c['dataset'] == 'kuairand-1k' else data._rating_examples(files['train'])
    rows = data._bounded_shuffle(rows, seed=c['seed'] + epoch)
    size = microbatch or c['microbatch']
    while batch := list(itertools.islice(rows, size)):
        yield batch  # Include the tail with exact denominator weighting.

def sample_training_rows(c, files, limit, *, seed, min_history=1):
    """Bounded reservoir across the whole training split, independent of CSV order."""
    cache = None
    binding = None
    if 'output' in c:
        cache = Path(c['output']) / f'training-samples-s{seed}-n{limit}-l{min_history}.pt'
        binding = dict(data_binding_sha256=c['binding_sha256'], code=c['code'],
                       seed=seed, limit=limit, min_history=min_history)
        if cache.exists():
            payload = torch.load(cache, map_location='cpu', weights_only=False)
            if payload.get('binding') != binding:
                raise ValueError('training sample cache has a different source/data binding')
            return payload['rows']
    rng = random.Random(seed)
    def rating_rows():
        source = data.validate_sequence_input(files['train'])
        with source.open(newline='') as handle:
            for row in csv.DictReader(handle):
                items = data._parse_ints(row['sequence_item_ids'])
                # len-1 is validation and is never sampled as history or target.
                if len(items) - 2 < min_history:
                    continue
                cutoff = rng.randrange(min_history, len(items) - 1)
                yield dict(user_id=int(row['user_id']),
                    history=items[max(0, cutoff-1024):cutoff], target=items[cutoff])
    source = data._slate_examples(files['train'], role='train') if c['dataset'] == 'kuairand-1k' else rating_rows()
    result = []
    eligible = 0
    for row in source:
        if len(row['history']) < min_history:
            continue
        eligible += 1
        if len(result) < limit:
            result.append(row)
        else:
            index = rng.randrange(eligible)
            if index < limit:
                result[index] = row
    if not result:
        raise ValueError('no eligible training histories for the requested sampling stage')
    rng.shuffle(result)
    if cache is not None:
        atomic_torch_save(torch, dict(binding=binding, rows=result), cache)
    return result

def collate(c, rows, device):
    if c['dataset'] == 'kuairand-1k':
        return data._collate_slates(torch, rows, device)
    # Row lengths are already Python integers before the device transfer.
    # Use them to allocate the tight batch width without reading a CUDA
    # scalar back merely to crop the fixed protocol tensor.
    width = max(len(row['history']) for row in rows)
    return data._collate_rating(torch, rows, device, history_width=width)

def candidates_for_loss(c, batch, catalog_ids):
    if c['dataset'] == 'kuairand-1k':
        return batch['candidates']
    negatives = catalog_ids[torch.randint(catalog_ids.numel(),
        (batch['targets'].shape[0], c['negatives']), device=catalog_ids.device)]
    return torch.cat((batch['targets'][:, None], negatives), dim=1)

def recommendation_loss(c, model, history, batch, candidates):
    scores = model.read(history, candidates)
    if c['dataset'] == 'kuairand-1k':
        return hstu_multitask_bce(
            scores, batch['labels'], batch.get('label_weights')
        )
    accidental_hits = candidates[:, 1:] == batch['targets'][:, None]
    logits = scores.float() / c['temperature']
    logits = torch.cat((logits[:, :1], logits[:, 1:].masked_fill(accidental_hits, -torch.inf)), -1)
    return F.cross_entropy(logits, torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device))

def autocast(model):
    device = model.item_embedding.weight.device.type
    return torch.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' else nullcontext()

def emit(output, stage, **values):
    record = dict(stage=stage, time=time.time(), 
                   **values)
    atomic_write_json(output / 'progress.json', record)
    print(json.dumps(record, sort_keys=True), flush=True)

def save_local(path, c, model, optimizer, stage, **state):
    resume_semantics = state.pop('resume_semantics', 'true-resume')
    model_state = {
        key: value for key, value in model.state_dict().items()
        if key != 'selector_embedding'
    }
    atomic_torch_save(torch, dict(schema='blossom-deltarec-resume-v1', config=c,
        model=model_state, optimizer=optimizer.state_dict() if optimizer else None,
        stage=stage, rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
        python_rng=random.getstate(), resume_semantics=resume_semantics,
        scheduler_contract='none', scheduler_state=None,
        sampler_contract='deterministic-bounded-shuffle(seed+epoch)',
        **state), path)

def restore_local(path, c, model, catalog_ids=None, parent_path=None):
    if not path.is_file():
        return None
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['config'] != c:
        raise ValueError('resume configuration mismatch')
    compatible, fresh_head = checkpoint_state_without_legacy_head(
        model, saved['model']
    ) if model.task_head is not None else (saved['model'], set())
    result = model.load_state_dict(compatible, strict=False)
    missing = [key for key in result.missing_keys
               if key != 'selector_embedding' and not key.startswith('selector')
               and key != 'prototypes' and key not in fresh_head]
    if result.unexpected_keys or missing:
        raise ValueError(
            f'invalid local resume state: missing={missing}, '
            f'unexpected={result.unexpected_keys}'
        )
    if fresh_head:
        saved['optimizer'] = None
        saved['epoch'] = 0
        saved['early'] = None
        saved['cursor'] = {}
        saved['prediction_head_reinitialized'] = True
    if saved.get('stage') in {'selector', 'sparse'}:
        # Keep the pre-trajectory test/helper API usable.  Production
        # Trajectory-Late resumes always supply both arguments and therefore
        # reconstruct the frozen theta(T) feature space explicitly.
        if catalog_ids is None or parent_path is None:
            torch.set_rng_state(saved['rng'])
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
            random.setstate(saved['python_rng'])
            del saved['model']
            return saved
        if not parent_path.is_file():
            raise RuntimeError('aligned local resume needs the shared Full-GDR parent')
        parent = torch.load(parent_path, map_location='cpu', weights_only=False)
        parent_table = parent['model'].get('item_embedding.weight')
        if parent_table is None:
            raise ValueError('Full-GDR parent lacks the item embedding feature table')
        model.bind_selector_space(catalog_ids, feature_table=parent_table)
        del parent
    torch.set_rng_state(saved['rng'])
    torch.cuda.set_rng_state_all(saved['cuda_rng'])
    random.setstate(saved['python_rng'])
    del saved['model']
    return saved

def freeze_selector(model):
    model.selector.requires_grad_(False)
    model.selector.eval()
    for parameter in model.selector.parameters():
        parameter.grad = None

def cwi_selector_loss(scores, importance, valid):
    """Official PC-selector loss on the signed CWI1 labels."""

    if scores.shape != importance.shape or scores.shape != valid.shape:
        raise ValueError('CWI selector tensors must have the same shape')
    if not bool(valid.any()):
        raise ValueError('CWI selector batch has no valid history event')
    target = torch.asinh(importance.detach().float())
    return F.smooth_l1_loss(scores.float().masked_select(valid), target.masked_select(valid))

class Adapter:
    def __init__(self, model, *, dense_parent=False):
        self.model = model
        self.dense_parent = dense_parent

    @torch.inference_mode()
    def run(self, batch, mode):
        del mode
        self.model.eval()
        device = self.model.item_embedding.weight.device
        with autocast(self.model):
            histories = batch.history_item_ids.to(device)
            lengths = batch.history_lengths.to(device)
            candidates = batch.candidate_item_ids.to(device)
            history = self.model.prefill(
                histories, lengths, sparse=not self.dense_parent,
                single_group=self.dense_parent,
            )
            scores = self.model.read(history, candidates)
        return AdapterOutput(scores)

def evaluate(c, model, files, output, *, label, split='validation', candidate_manifest=None,
             slate_file=None, lock_hash=None, dense_parent=False):
    output.mkdir(parents=True, exist_ok=True)
    evidence = output / f'{label}-rows.jsonl'
    with AtomicJsonlWriter(evidence) as writer:
        common = dict(torch=torch, adapter=Adapter(model, dense_parent=dense_parent), split_role=split,
                      protocol_lock_hash=lock_hash, microbatch_size=c['eval_microbatch'], evidence=writer)
        if c['dataset'] == 'kuairand-1k':
            metrics, details, digest = evaluate_kuai_stream(**common,
                slate_file=slate_file or files['validation'], split_manifest=files['split_manifest'])
        else:
            metrics, details, digest = evaluate_rating_stream(**common, dataset=c['dataset'],
                candidate_manifest=candidate_manifest or files['validation_candidates'])
    result = dict(dataset=c['dataset'], method='DeltaRec-GC (BlossomRec backend)', split=split,
                  metrics=metrics, details=details, input_sha256=digest, evidence=str(evidence))
    atomic_write_json(output / f'{label}.json', result)
    return result

def base_model_state(model):
    """Return the recommender state without the separately bound selector."""

    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith('selector') and key != 'prototypes'
    }

def load_base_model_state(model, state):
    compatible, fresh_head = checkpoint_state_without_legacy_head(
        model, state
    ) if model.task_head is not None else (state, set())
    result = model.load_state_dict(compatible, strict=False)
    unexpected = list(result.unexpected_keys)
    if unexpected:
        raise ValueError(f'unexpected parent checkpoint tensors: {unexpected}')
    missing = [key for key in result.missing_keys
               if not key.startswith('selector') and key != 'prototypes'
               and key not in fresh_head]
    if missing:
        raise ValueError(f'parent checkpoint is missing recommender tensors: {missing}')

def make_optimizer(c, model):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.AdamW(parameters, lr=c['learning_rate'], weight_decay=c['weight_decay'])

def early_state(c, *, maximum_epochs=None, minimum_epochs=None, patience=None, multitask=False):
    return EarlyStoppingState(
        min_delta=c['min_delta'],
        patience=c['patience'] if patience is None else patience,
        minimum_epochs=c['min_epochs'] if minimum_epochs is None else minimum_epochs,
        maximum_epochs=c['max_epochs'] if maximum_epochs is None else maximum_epochs,
        tie_breaker_mode='min' if multitask else None,
    )

def primary_metric(run, metrics):
    return metrics['macro_gauc'] if run.model.multitask else metrics['ndcg_at_10']
