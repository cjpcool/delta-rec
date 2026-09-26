from __future__ import annotations

from contextlib import nullcontext

from dataclasses import replace

import csv

import itertools

import json

import logging

import os

from pathlib import Path

import random

import time

import numpy as np

import torch

from torch.nn import functional as F

from deltarec.utils.io import atomic_write_json, sha256_file

from deltarec.utils.trajectory import immutable_json

from deltarec.data.sampling import sample_training_rows

from deltarec.layers.utility_grouping import UtilityGroupingConfig, prepare_utility_groups

from deltarec.models.hstu_multitask import hstu_multitask_bce

from deltarec.models.linrec import FrozenUtilitySelector, LinRecDeltaRec, grouped_scores

from deltarec.adaptors import recbole as bridge
from deltarec.data import training as data

from deltarec.utils.checkpoint import atomic_torch_save

from deltarec.utils.early_stopping import EarlyStoppingState

from deltarec.utils.protocols import AdapterOutput

from deltarec.metrics.evaluation import AtomicJsonlWriter, evaluate_rating_stream, evaluate_kuai_stream

from deltarec.utils.numerics import nonfinite_gradient_names



STOP = False

def read(path):
    return json.loads(Path(path).read_text())

def bind_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])

def catalog(path, device):
    with Path(path).open() as handle:
        ids = [int(row['item_id']) for row in csv.DictReader(handle)]
    if not ids or min(ids) < 1 or len(set(ids)) != len(ids):
        raise ValueError('invalid training catalog')
    return torch.tensor(sorted(ids), device=device, dtype=torch.long)

def construct(c, run, files, *, device=None, fixture=False, sparse=False):
    device = device or c.get('device', 'cpu')
    bind_seed(c['seed'])
    config = run.model
    if fixture:
        width = 32 if str(device).startswith('cuda') else 8
        config = replace(config, hidden_size=width, inner_size=2 * width, n_heads=2,
                         hidden_dropout_prob=0., attn_dropout_prob=0.)
    upstream = bridge.build_upstream_model(config,
        max_item_id=73 if fixture else data._max_catalog_id(files['full_catalog']),
         device='cpu', fixture_only=fixture)
    model = LinRecDeltaRec(upstream, core=c['core'], multitask=run.model.multitask,
        backend=c.get('kernel', 'reference'), seed=c['seed']).to(device)
    model.selection_groups = c['group_count']
    model.mandatory_recent_suffix = c.get('mandatory_recent_suffix', True)
    if sparse and c.get('sparse_head_512', False):
        from deltarec.models.hstu_multitask import HSTUMultitaskHead
        model.task_head = HSTUMultitaskHead(model.width).to(model.device)
    return model

def autocast(model):
    return (torch.autocast('cuda', dtype=torch.bfloat16) if model.core == 'gdr' and model.device.type == 'cuda'
            else nullcontext())

def source_rows(c, files):
    return (data._slate_examples(files['train'], role='train') if c['dataset'] == 'kuairand-1k'
            else data._rating_examples(files['train']))

def windows(c, files, epoch):
    rows = data._bounded_shuffle(source_rows(c, files), seed=c['seed'] + epoch)
    while window := list(itertools.islice(rows, c['effective_batch'])):
        yield window

def microbatches(c, window, *, sparse):
    # Only reorder this optimizer window, preserving its exact supervision set.
    rows = sorted(window, key=lambda row: len(row['history']))
    pending, tokens = [], 0
    for row in rows:
        n = len(row['history'])
        cost = n if not sparse else c['group_count'] * max((n + 3) // 4, min(n, 32))
        if pending and (len(pending) == c['microbatch'] or tokens + cost > c['max_stream_tokens']):
            yield pending
            pending, tokens = [], 0
        pending.append(row)
        tokens += cost
    if pending:
        yield pending

def loss_for_rows(c, model, rows, catalog_ids, *, selector=None, grouping=None):
    histories = [row['history'] for row in rows]
    kuai = model.multitask
    if kuai:
        candidates = torch.tensor([row['candidates'] for row in rows], device=model.device)
    elif model.core == 'gdr':
        targets = torch.tensor([row['target'] for row in rows], device=model.device)
        negatives = catalog_ids[torch.randint(len(catalog_ids), (len(rows), c['negatives']), device=model.device)]
        candidates = torch.cat((targets[:, None], negatives), 1)
    else:
        candidates = None
    if selector is not None:
        logits, _ = grouped_scores(model, histories, candidates, selector, grouping['mapping'], grouping.get('native_table'))
    else:
        query = model.encode_full(histories)
        logits = model.catalog_logits(query) if candidates is None else model.pair_scores(query, candidates)
    if kuai:
        labels = torch.tensor([row['labels'] for row in rows], device=model.device, dtype=torch.float32)
        return hstu_multitask_bce(logits, labels), len(rows) * candidates.shape[1]
    if model.core == 'native':
        targets = torch.tensor([row['target'] for row in rows], device=model.device)
    else:
        collisions = candidates[:, 1:].eq(candidates[:, :1])
        logits = torch.cat((logits[:, :1], logits[:, 1:].masked_fill(collisions, -torch.inf)), 1)
        targets = torch.zeros(len(rows), dtype=torch.long, device=model.device)
    return F.cross_entropy(logits.float(), targets), len(rows)

def model_state(model):
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}

def save_resume(path, c, model, optimizer, stage, epoch, cursor, early=None):
    atomic_torch_save(torch, dict(schema='linrec-grouped-resume-v1', config_sha256=c['config_sha256'],
        model=model_state(model), optimizer=optimizer.state_dict(), stage=stage, epoch=epoch,
        cursor=cursor, rng=rng_state(), early=early.to_dict() if early else None), path)

def load_bound(path, c):
    value = torch.load(path, map_location='cpu', weights_only=False)
    if value.get('config_sha256') != c['config_sha256']:
        raise ValueError(f'checkpoint/config mismatch: {path}')
    return value

class EvaluationAdapter:
    def __init__(self, model, selector=None, grouping=None):
        self.model, self.selector, self.grouping = model, selector, grouping

    def run(self, batch, mode):
        if batch.metadata.get('split_role') != 'validation':
            raise ValueError('this training adapter evaluates validation only')
        histories = [row[:int(n)].tolist() for row, n in zip(batch.history_item_ids, batch.history_lengths)]
        candidates = batch.candidate_item_ids.to(self.model.device)
        with torch.inference_mode(), autocast(self.model):
            if self.selector is None:
                scores = self.model.pair_scores(self.model.encode_full(histories), candidates)
            else:
                scores, _ = grouped_scores(self.model, histories, candidates, self.selector,
                    self.grouping['mapping'], self.grouping.get('native_table'))
        return AdapterOutput(scores)

def validate(c, model, files, selector, grouping, epoch):
    output = Path(c['output'])
    model.eval()
    evidence = output / f'validation-{epoch:03d}.jsonl'
    # A crash may leave a complete evaluation before the optimizer cursor was
    # published. Re-evaluate that same epoch rather than accepting unbound data.
    evidence.unlink(missing_ok=True)
    adapter = EvaluationAdapter(model, selector, grouping)
    with AtomicJsonlWriter(evidence) as writer:
        common = dict(torch=torch, adapter=adapter, split_role='validation',
                      protocol_lock_hash=None, microbatch_size=8, evidence=writer)
        if model.multitask:
            metrics, _, _ = evaluate_kuai_stream(slate_file=files['validation'],
                split_manifest=files['split_manifest'], **common)
        else:
            metrics, _, _ = evaluate_rating_stream(candidate_manifest=files['validation_candidates'],
                                                   dataset=c['dataset'], **common)
    atomic_write_json(output / f'validation-{epoch:03d}.json', dict(metrics=metrics,
                      config_sha256=c['config_sha256'], epoch=epoch))
    return metrics, evidence

def train_stage(c, model, files, catalog_ids, *, sparse=False, selector=None, grouping=None):
    output = Path(c['output'])
    stage = 'sparse' if sparse else 'full-gdr'
    resume = output / f'{stage}-latest.pt'
    optimizer = (torch.optim.Adam if model.core == 'native' else torch.optim.AdamW)(
        model.parameters(), lr=c['learning_rate'], weight_decay=0.)
    early = EarlyStoppingState(minimum_epochs=c['minimum_epochs'], maximum_epochs=c['max_epochs'],
        patience=c['patience'], min_delta=c['min_delta'], tie_breaker_mode='min' if model.multitask else None)
    epoch_start, cursor = 0, dict(windows=0, step=0, numerator=0., denominator=0)
    if resume.exists():
        saved = load_bound(resume, c)
        if saved['stage'] != stage:
            raise ValueError('resume stage mismatch')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        epoch_start, cursor = saved['epoch'], saved['cursor']
        if saved['early'] is not None:
            early = EarlyStoppingState.from_dict(saved['early'], contract=early.contract())
        restore_rng(saved['rng'])
        del saved
    if sparse and early.stopped:
        atomic_write_json(output / 'completed.json', dict(config_sha256=c['config_sha256'], early=early.to_dict()))
        return True
    last_save = time.monotonic()
    epochs = c['max_epochs'] if sparse else c['trajectory_count']
    for epoch in range(epoch_start, epochs):
        model.train()
        iterator = itertools.islice(windows(c, files, epoch + 1), cursor['windows'], None)
        for window in iterator:
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            total = torch.zeros((), device=model.device, dtype=torch.float64)
            denominator = 0
            before = model.executed_tokens
            for rows in microbatches(c, window, sparse=sparse):
                with autocast(model):
                    loss, count = loss_for_rows(c, model, rows, catalog_ids, selector=selector, grouping=grouping)
                if not bool(torch.isfinite(loss.detach())):
                    raise FloatingPointError('nonfinite LinRec loss before optimizer update')
                (loss * count).backward()
                total += loss.detach().double() * count
                denominator += count
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(denominator)
            bad = nonfinite_gradient_names(model.named_parameters())
            if bad:
                raise FloatingPointError(f'nonfinite gradients: {bad}')
            if model.core == 'gdr':
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            cursor['step'] += 1
            cursor['windows'] += 1
            cursor['numerator'] += float(total)
            cursor['denominator'] += denominator
            if cursor['step'] == 1 or cursor['step'] % 10 == 0:
                progress = dict(stage=stage, epoch=epoch + 1, **cursor, loss=float(total) / denominator,
                    step_seconds=time.monotonic() - started, executed_tokens=model.executed_tokens - before,
                    cuda_peak_bytes=torch.cuda.max_memory_allocated() if model.device.type == 'cuda' else 0,
                     config_sha256=c['config_sha256'])
                atomic_write_json(output / 'progress.json', progress)
                logging.info('%s', json.dumps(progress))
            if cursor['step'] == 1 or STOP or time.monotonic() - last_save >= c['checkpoint_seconds']:
                save_resume(resume, c, model, optimizer, stage, epoch, cursor, early if sparse else None)
                last_save = time.monotonic()
            if STOP:
                return False
        if not sparse:
            path = output / 'full_gdr_trajectory' / f'epoch-{epoch + 1:03d}.pt'
            parent = path.with_name(f'epoch-{epoch:03d}.pt') if epoch else None
            checkpoint = dict(schema='linrec-full-teacher-v1', config_sha256=c['config_sha256'],
                model=model_state(model), epoch=epoch + 1, step=cursor['step'],
                teacher_weight=c['trajectory_weights'][epoch], parent_sha256=sha256_file(parent) if parent else None)
            if path.exists():
                old = load_bound(path, c)
                if old['epoch'] != epoch + 1 or any(not torch.equal(old['model'][k], v) for k, v in checkpoint['model'].items()):
                    raise ValueError('immutable teacher already differs')
            else:
                atomic_torch_save(torch, checkpoint, path, overwrite=False)
        else:
            metrics, evidence = validate(c, model, files, selector, grouping, epoch + 1)
            primary = metrics['macro_gauc' if model.multitask else 'ndcg_at_10']
            decision = early.observe(primary_metric=primary, epoch=epoch,
                tie_breaker=metrics['multitask_loss'] if model.multitask else None)
            if not decision['valid']:
                raise FloatingPointError('invalid validation metric')
            if decision['improved']:
                atomic_torch_save(torch, dict(schema='linrec-grouped-best-v1',
                    config_sha256=c['config_sha256'], model=model_state(model), metrics=metrics,
                    epoch=epoch + 1, selector_sha256=sha256_file(output / 'selector.pt'),
                    grouping_sha256=sha256_file(output / 'grouping.pt')), output / 'best.pt')
                os.replace(evidence, output / 'best-validation.jsonl')
            else:
                evidence.unlink()
        cursor = dict(windows=0, step=cursor['step'], numerator=0., denominator=0)
        save_resume(resume, c, model, optimizer, stage, epoch + 1, cursor, early if sparse else None)
        if sparse and early.stopped:
            break
    if sparse:
        atomic_write_json(output / 'completed.json', dict(config_sha256=c['config_sha256'], early=early.to_dict(),
                                                         best_sha256=sha256_file(output / 'best.pt')))
    else:
        immutable_json(output / 'trajectory_manifest.json', dict(config_sha256=c['config_sha256'],
            teachers=[dict(path=str(output / 'full_gdr_trajectory' / f'epoch-{i:03d}.pt'),
                sha256=sha256_file(output / 'full_gdr_trajectory' / f'epoch-{i:03d}.pt'), weight=c['trajectory_weights'][i - 1])
                for i in range(1, c['trajectory_count'] + 1)]))
    return True

def cwi_record(c, model, row, catalog_ids):
    """Frozen teacher; differentiate ONLY the scalar event interventions."""
    history = [row['history']]
    n = len(row['history'])
    gates = torch.ones(n, device=model.device, requires_grad=True)
    query = model.encode_full(history, gates)
    if model.multitask:
        candidates = torch.tensor([row['candidates']], device=model.device)
        logits = model.pair_scores(query, candidates)
        labels = torch.tensor([row['labels']], device=model.device, dtype=torch.float32)
        losses = .2 * F.binary_cross_entropy_with_logits(logits, labels, reduction='none').sum(-1)[0]
        utilities = [(-torch.autograd.grad(loss, gates, retain_graph=i + 1 < len(losses))[0]).detach().cpu()
                     for i, loss in enumerate(losses)]
        focal = list(row['candidates'])
    else:
        positive = torch.tensor([row['target']], device=model.device)
        if model.core == 'native':
            loss = F.cross_entropy(model.catalog_logits(query).float(), positive)
        else:
            negatives = catalog_ids[torch.randint(len(catalog_ids), (1, c['negatives']), device=model.device)]
            candidates = torch.cat((positive[:, None], negatives), 1)
            logits = model.pair_scores(query, candidates)
            logits = torch.cat((logits[:, :1], logits[:, 1:].masked_fill(negatives.eq(positive[:, None]), -torch.inf)), 1)
            loss = F.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device=model.device))
        utilities = [-torch.autograd.grad(loss, gates)[0].detach().cpu()]
        focal = [row['target']]
    values = torch.stack(utilities)
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError('nonfinite teacher labels')
    return dict(history=list(row['history']), candidates=focal, utility=values, user_id=row['user_id'])

def generate_cwi(c, model, files, catalog_ids):
    output = Path(c['output'])
    manifest_path = output / 'cwi_trajectory_manifest.json'
    if manifest_path.exists():
        manifest = read(manifest_path)
        if manifest['config_sha256'] != c['config_sha256']:
            raise ValueError('CWI manifest binding changed')
        for teacher in manifest['teachers']:
            if sha256_file(teacher['checkpoint']) != teacher['checkpoint_sha256']:
                raise ValueError('CWI teacher changed')
            for shard in teacher['shards']:
                if sha256_file(shard['path']) != shard['sha256']:
                    raise ValueError('CWI shard changed')
        return True
    rows = sample_training_rows(c, files, c['cwi_samples'], seed=1000, min_history=1)
    trajectory = read(output / 'trajectory_manifest.json')
    if trajectory['config_sha256'] != c['config_sha256']:
        raise ValueError('teacher trajectory binding changed')
    model.eval().requires_grad_(False)
    teachers = []
    for index, teacher in enumerate(trajectory['teachers']):
        if sha256_file(teacher['path']) != teacher['sha256']:
            raise ValueError('teacher content changed')
        saved = load_bound(teacher['path'], c)
        model.load_state_dict(saved['model'], strict=True)
        del saved
        shards = []
        for start in range(0, len(rows), 16):
            path = output / 'cwi_trajectory' / f'teacher-{index + 1:03d}-{start:05d}.pt'
            identity = dict(config_sha256=c['config_sha256'], teacher_sha256=teacher['sha256'],
                            offset=start, intervention=c['cwi_intervention'])
            if path.exists():
                payload = load_bound(path, c)
                if payload['identity'] != identity:
                    raise ValueError('CWI partial shard binding changed')
            else:
                records = []
                for offset, row in enumerate(rows[start:start + 16]):
                    # Every context has its own negative RNG identity so a
                    # partial-shard restart cannot change later teacher labels.
                    bind_seed(100000 * (index + 1) + start + offset)
                    records.append(cwi_record(c, model, row, catalog_ids))
                payload = dict(config_sha256=c['config_sha256'], identity=identity, records=records)
                atomic_torch_save(torch, payload, path, overwrite=False)
            shards.append(dict(path=str(path), sha256=sha256_file(path), rows=len(payload['records'])))
            del payload
            logging.info('CWI teacher=%s contexts=%s/%s', index + 1, min(start + 16, len(rows)), len(rows))
            if STOP:
                return False
        teachers.append(dict(checkpoint=teacher['path'], checkpoint_sha256=teacher['sha256'],
                             weight=c['trajectory_weights'][index], shards=shards))
    immutable_json(manifest_path, dict(config_sha256=c['config_sha256'], teachers=teachers,
        sampling='checkpoint-specific-observations-no-label-averaging', context_count=len(rows)))
    return True

def final_teacher_path(c):
    return Path(c['output']) / 'full_gdr_trajectory' / f"epoch-{c['trajectory_count']:03d}.pt"

def load_final_teacher(c, model):
    path = final_teacher_path(c)
    saved = load_bound(path, c)
    model.load_state_dict(saved['model'], strict=True)
    return sha256_file(path)

def selector_loss(selector, records):
    total, count = None, 0
    for record in records:
        events = torch.tensor(record['history'], device=selector.embedding.device)
        candidates = torch.tensor(record['candidates'], device=events.device)
        targets = record['utility'].to(events.device).T.asinh()
        prediction = selector(events[:, None], candidates[None])
        value = F.smooth_l1_loss(prediction, targets, reduction='sum')
        total = value if total is None else total + value
        count += targets.numel()
    return total / count

def fit_selector(c, model, files, catalog_ids):
    output = Path(c['output'])
    teacher_sha = load_final_teacher(c, model)
    bind_seed(c['seed'])
    selector = FrozenUtilitySelector(model.item_embedding.weight, catalog_ids)
    path, resume = output / 'selector.pt', output / 'selector-latest.pt'
    identity = dict(config_sha256=c['config_sha256'], teacher_sha256=teacher_sha,
                    cwi_manifest_sha256=sha256_file(output / 'cwi_trajectory_manifest.json'))
    if path.exists():
        saved = load_bound(path, c)
        if saved['identity'] != identity:
            raise ValueError('selector lineage mismatch')
        selector.load_state_dict(saved['selector'], strict=True)
        immutable_json(output / 'selector_binding.json', dict(**identity, selector_sha256=sha256_file(path),
            embedding_ownership='frozen-final-trajectory-snapshot', normalized_trajectory_weights=c['trajectory_weights']))
        return selector.requires_grad_(False).eval()
    manifest = read(output / 'cwi_trajectory_manifest.json')
    datasets = []
    for teacher in manifest['teachers']:
        records = []
        for shard in teacher['shards']:
            if sha256_file(shard['path']) != shard['sha256']:
                raise ValueError('selector CWI shard hash mismatch')
            records.extend(load_bound(shard['path'], c)['records'])
        datasets.append(records)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=c['selector_learning_rate'], weight_decay=0.)
    steps_per_epoch = math_ceil_div(len(datasets[0]), c['selector_batch'])
    first = 0
    if resume.exists():
        saved = load_bound(resume, c)
        if saved['identity'] != identity:
            raise ValueError('selector resume lineage mismatch')
        selector.load_state_dict(saved['selector'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        first = saved['step']
        restore_rng(saved['rng'])
    for step in range(first, c['selector_epochs'] * steps_per_epoch):
        teacher = random.choices(range(c['trajectory_count']), weights=c['trajectory_weights'], k=1)[0]
        records = random.choices(datasets[teacher], k=c['selector_batch'])
        optimizer.zero_grad(set_to_none=True)
        loss = selector_loss(selector, records)
        if not bool(torch.isfinite(loss.detach())):
            raise FloatingPointError('nonfinite selector loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(selector.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step % 25 == 0 or STOP:
            logging.info('Selector step=%s teacher=%s loss=%s', step + 1, teacher + 1, float(loss))
            atomic_torch_save(torch, dict(config_sha256=c['config_sha256'], identity=identity,
                selector=model_state(selector), optimizer=optimizer.state_dict(), step=step + 1,
                rng=rng_state()), resume)
        if STOP:
            return None
    atomic_torch_save(torch, dict(config_sha256=c['config_sha256'], identity=identity,
                     selector=model_state(selector)), path, overwrite=False)
    immutable_json(output / 'selector_binding.json', dict(**identity, selector_sha256=sha256_file(path),
        embedding_ownership='frozen-final-trajectory-snapshot', normalized_trajectory_weights=c['trajectory_weights']))
    return selector.requires_grad_(False).eval()

def math_ceil_div(value, divisor):
    return (value + divisor - 1) // divisor

@torch.no_grad()
def make_grouping(c, selector, files, catalog_ids):
    output = Path(c['output'])
    path = output / 'grouping.pt'
    identity = dict(config_sha256=c['config_sha256'], selector_sha256=sha256_file(output / 'selector.pt'),
                    teacher_sha256=sha256_file(final_teacher_path(c)))
    if path.exists():
        saved = load_bound(path, c)
        if saved['identity'] != identity:
            raise ValueError('grouping binding changed')
    else:
        rows = sample_training_rows(c, files, c['anchor_count'], seed=0, min_history=1)
        generator = random.Random(0)
        anchors = [row['history'][generator.randrange(len(row['history']))] for row in rows]
        events = torch.tensor(anchors, device=catalog_ids.device)
        profiles = []
        for start in range(0, len(catalog_ids), 256):
            ids = catalog_ids[start:start + 256]
            values = selector(events[None], ids[:, None]).sinh()
            if not bool(torch.isfinite(values).all()):
                raise FloatingPointError('nonfinite grouping profile')
            profiles.append(values.cpu())
            if start % 65536 == 0:
                logging.info('Grouping profiles %s/%s', start, len(catalog_ids))
        profiles = torch.cat(profiles)
        plan = prepare_utility_groups(profiles, UtilityGroupingConfig(groups=c['group_count']))
        catalog_cpu = catalog_ids.cpu()
        mapping = torch.zeros(len(selector.embedding), dtype=torch.long)
        mapping[catalog_cpu] = plan.candidate_group_ids
        representatives = []
        for group in range(c['group_count']):
            members = (plan.candidate_group_ids == group).nonzero(as_tuple=True)[0]
            distance = (profiles[members].double() - plan.group_utilities[group].double()).square().sum(1)
            representatives.append(int(catalog_cpu[members[distance.argmin()]]))
        saved = dict(config_sha256=c['config_sha256'], identity=identity, mapping=mapping,
            representatives=representatives, anchors=anchors, group_sizes=torch.bincount(plan.candidate_group_ids),
            group_centers=plan.group_utilities, unknown_item_policy='group-zero')
        if c['core'] == 'native':
            reps = torch.tensor(representatives, device=catalog_ids.device)
            table = []
            for start in range(0, len(selector.embedding), 4096):
                ids = torch.arange(start, min(start + 4096, len(selector.embedding)), device=catalog_ids.device)
                table.append(selector(ids[:, None], reps[None]).cpu())
            saved['native_table'] = torch.cat(table)
        atomic_torch_save(torch, saved, path, overwrite=False)
    result = {'mapping': saved['mapping'].to(catalog_ids.device)}
    if c['core'] == 'native':
        result['native_table'] = saved['native_table'].to(catalog_ids.device)
    return result


def main(argv=None):
    import argparse
    from types import SimpleNamespace
    from deltarec.utils.config import load_training_config
    parser = argparse.ArgumentParser(description='DeltaRec LinRec training and validation')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--start-stage', choices=('full-gdr','cwi','selector','sparse'), default='full-gdr')
    parser.add_argument('--stop-stage', choices=('full-gdr','cwi','selector','sparse'), default='sparse')
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    args=parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    c,files=load_training_config(args.config,args.data_root,args.output,args.device)
    run=SimpleNamespace(model=bridge.RecBoleModelConfig(**c['architecture']))
    model=construct(c,run,files,sparse=args.evaluate)
    catalog_ids=catalog(files['train_catalog'],model.device)
    output=Path(c['output']);output.mkdir(parents=True,exist_ok=True)
    if not args.resume and any(output.glob('*-latest.pt')):
        raise FileExistsError('Resume state exists; use --resume or a new output directory')
    if args.evaluate:
        if args.checkpoint is None: parser.error('--evaluate requires --checkpoint')
        saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model'],strict=True)
        selector,grouping=load_evaluation_assets(c,model,catalog_ids,output,saved)
        print(json.dumps(validate(c,model,files,selector,grouping,-1)[0]))
        return
    stages=('full-gdr','cwi','selector','sparse')
    if stages.index(args.start_stage)>stages.index(args.stop_stage): parser.error('Invalid stage order')
    if not (output/'trajectory_manifest.json').exists():
        if args.start_stage!='full-gdr':raise FileNotFoundError('Full-history teacher trajectory is required')
        if not train_stage(c,model,files,catalog_ids):return
    if args.stop_stage=='full-gdr':return
    if not generate_cwi(c,model,files,catalog_ids):return
    if args.stop_stage=='cwi':return
    selector=fit_selector(c,model,files,catalog_ids)
    if selector is None:return
    grouping=make_grouping(c,selector,files,catalog_ids)
    if args.stop_stage=='selector':return
    if c.get('sparse_head_512', False):
        model=construct(c,run,files,sparse=True)
        teacher=load_bound(final_teacher_path(c),c)['model']
        teacher={k:v for k,v in teacher.items() if not k.startswith('task_head.')}
        missing,unexpected=model.load_state_dict(teacher,strict=False)
        assert not unexpected and set(missing)=={k for k in model.state_dict() if k.startswith('task_head.')}
    else:
        load_final_teacher(c,model)
    model.requires_grad_(True)
    if model.core=='native' and not model.multitask:model.bind_catalog_groups(grouping['mapping'],c['group_count'])
    bind_seed(c['seed'])
    if not train_stage(c,model,files,catalog_ids,sparse=True,selector=selector,grouping=grouping):return
    saved=load_bound(output/'best.pt',c)
    model.load_state_dict(saved['model'],strict=True)
    atomic_torch_save(torch,dict(config=json.loads(args.config.read_text()),model=model_state(model),
        selector={key:value.cpu() for key,value in selector.state_dict().items()},
        selector_embedding=selector.embedding.cpu(),
        grouping={key:value.cpu() for key,value in grouping.items()}),output/'model.pt')


def load_evaluation_assets(c,model,catalog_ids,output,saved):
    if 'selector' in saved and 'selector_embedding' in saved and 'grouping' in saved:
        selector=FrozenUtilitySelector(saved['selector_embedding'].to(model.device),catalog_ids)
        selector.load_state_dict(saved['selector'],strict=True)
        selector.requires_grad_(False).eval()
        grouping={key:value.to(model.device) for key,value in saved['grouping'].items()}
        return selector,grouping
    teacher=load_bound(final_teacher_path(c),c)
    table=teacher['model']['item_embedding.weight'].to(model.device)
    selector=FrozenUtilitySelector(table,catalog_ids)
    selector.load_state_dict(load_bound(output/'selector.pt',c)['selector'],strict=True)
    selector.requires_grad_(False).eval()
    grouping=load_bound(output/'grouping.pt',c)
    return selector,{k:grouping[k].to(model.device) for k in ('mapping','native_table') if k in grouping}

if __name__=='__main__':
    main()
