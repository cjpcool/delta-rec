from __future__ import annotations
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import random
import time
import types
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from deltarec.utils.io import atomic_write_json, sha256_file
from deltarec.data.hstu_amazon import epoch_windows
from deltarec.models.hstu_amazon import AmazonScorer, AmazonLoss, IDOnlyLocalSampler
from deltarec.models.hstu_runtime import ResearchGDRStateCache, RUNTIME_SCHEMA
from deltarec.models.hstu_selector import RatingPCSelector
from deltarec.data.hstu_training import load_item_ids

def emit(**kw):
    print(json.dumps(dict(time=time.time(), **kw), allow_nan=False), flush=True)

def save_pt(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    os.replace(temp, path)

def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())

def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    torch.cuda.set_rng_state(state['cuda'])
_RESUME_EXECUTION_FIELDS = ('short_microbatch', 'medium_microbatch', 'long_microbatch', 'ranker_supervision_chunk', 'effective_batch', 'checkpoint_step_interval')

class Context:

    def consume_training_context(self):
        x = self.value
        self.value = None
        return x

def selector_for(loaded):
    emb = loaded.model._embedding_module._item_emb
    return RatingPCSelector(emb, expected_num_items=emb.num_embeddings - 1, expected_embedding_dim=emb.embedding_dim, seed=0).cuda()

def batch_ids(items, offsets, rows):
    lengths = np.asarray([offsets[i + 1] - offsets[i] - 1 for i in rows])
    dense = np.zeros((len(rows), int(lengths.max()) + 1), dtype=np.int64)
    for j, i in enumerate(rows):
        seq = items[offsets[i]:offsets[i + 1]]
        dense[j, :len(seq)] = seq
    return (torch.from_numpy(dense).cuda(), torch.from_numpy(lengths).cuda())

def packed_outputs(loaded, ids, lengths):
    """Official preprocessor/layers/postprocessor on actual packed tokens."""
    model = loaded.model
    _, x, _ = model._input_features_preproc(past_lengths=lengths, past_ids=ids, past_embeddings=model.get_item_embeddings(ids), past_payloads={'timestamps': torch.zeros_like(ids)})
    valid = torch.arange(ids.shape[1], device=ids.device)[None] < lengths[:, None]
    x = x[valid]
    offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
    cpu = offsets.cpu()
    for layer in _research_layers(model):
        x, _ = layer.forward_gdr_streams(x=x, x_offsets=offsets, x_offsets_cpu=cpu, event_gate=torch.ones(len(x), device=x.device), return_final_state=False)
    return model._output_postproc(x)

def parent_loss(loaded, ids, lengths, catalog):
    query = packed_outputs(loaded, ids, lengths)
    valid = torch.arange(ids.shape[1] - 1, device=ids.device)[None] < lengths[:, None]
    positives = ids[:, 1:][valid]
    negative = catalog[torch.randint(len(catalog), (len(positives), 512), device='cuda')]
    candidates = torch.cat((positives[:, None], negative), 1)
    values = []

    def block(q, c):
        e = loaded.model.get_item_embeddings(c)
        e = e / e.norm(dim=-1, keepdim=True).clamp_min(1e-06)
        logits = torch.einsum('bd,bkd->bk', q, e) / 0.05
        mask = c[:, 1:].eq(c[:, :1])
        logits = torch.cat((logits[:, :1], logits[:, 1:].masked_fill(mask, -50000.0)), 1)
        return -F.log_softmax(logits, dim=1)[:, 0]
    for start in range(0, len(positives), 128):
        values.append(checkpoint(block, query[start:start + 128], candidates[start:start + 128], use_reentrant=False))
    return torch.cat(values).mean()

def train_stage(loaded, run, stage, selector=None, stop_after_steps=None):
    cfg = execution_configuration(run)
    folder = run / stage
    folder.mkdir(exist_ok=True)
    if (folder / 'completed.json').exists():
        loaded.model.load_state_dict(torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)['model'])
        return
    model = loaded.model.requires_grad_(True)
    if selector is not None:
        selector.requires_grad_(False)
    scorer = scorer_for(loaded, run, selector)
    lr = cfg['full_gdr_lr'] if selector is None else cfg['ranker_lr']
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=2, threshold=0.0001, threshold_mode='abs', min_lr=3e-06)
    catalog = torch.tensor(load_item_ids(Path(cfg['data_files']['train_catalog'])), device='cuda')
    context = Context()
    if selector is not None:
        from deltarec.adaptors.hstu.modeling.sequential.autoregressive_losses import LocalNegativesSampler
        sampler = LocalNegativesSampler(num_items=model._embedding_module._item_emb.num_embeddings - 1, item_emb=model._embedding_module._item_emb, all_item_ids=catalog.cpu().tolist(), l2_norm=True, l2_norm_eps=1e-06).cuda()
        sampler.forward = types.MethodType(IDOnlyLocalSampler.forward, sampler)
        lossfn = AmazonLoss(num_to_sample=512, softmax_temperature=0.05, model=model, scorer=scorer, context_owner=context, train_catalog_ids=catalog.cpu().tolist(), supervision_chunk_size=cfg['ranker_supervision_chunk'], candidate_chunk_size=cfg['candidate_chunk'])
    items = np.load(run / 'data/items.npy', mmap_mode='r')
    offsets = np.load(run / 'data/offsets.npy', mmap_mode='r')
    cursor = dict(epoch=0, window=0, phase='train')
    step = 0
    best = -math.inf
    bad = 0
    history = []
    latest = folder / 'latest.pt'
    base_config_sha256 = sha256_file(run / 'experiment.json')
    execution_config_sha256 = sha256_file(run / 'resume_execution.json') if (run / 'resume_execution.json').is_file() else None
    compatible_execution_config_sha256 = json.loads((run / 'resume_execution.json').read_text()).get('compatible_checkpoint_execution_sha256') if (run / 'resume_execution.json').is_file() else None
    if not latest.exists():
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
    if latest.exists():
        state = torch.load(latest, map_location='cpu', weights_only=False)
        if state['config_sha256'] != base_config_sha256:
            raise RuntimeError('resume base config changed')
        observed_execution_sha256 = state.get('execution_config_sha256', execution_config_sha256)
        if observed_execution_sha256 != execution_config_sha256 and observed_execution_sha256 != compatible_execution_config_sha256:
            raise RuntimeError('resume execution config changed')
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        sched.load_state_dict(state['scheduler'])
        cursor = state['cursor']
        step = state['step']
        best = state['best']
        bad = state['bad']
        history = state['history']
        restore_rng(state['rng'])
        emit(stage=f'{stage}-resumed', checkpoint=str(latest), epoch=cursor['epoch'], window=cursor['window'], step=step, lr=opt.param_groups[0]['lr'], short_microbatch=cfg['short_microbatch'], medium_microbatch=cfg['medium_microbatch'], long_microbatch=cfg['long_microbatch'], supervision_chunk=cfg['ranker_supervision_chunk'], effective_batch=cfg['effective_batch'], checkpoint_step_interval=cfg['checkpoint_step_interval'])

    def persist():
        save_pt(latest, dict(model=cpu_state(model), optimizer=opt.state_dict(), scheduler=sched.state_dict(), cursor=cursor, step=step, best=best, bad=bad, history=history, rng=rng_state(), config_sha256=base_config_sha256, execution_config_sha256=execution_config_sha256))
        atomic_write_json(folder / 'resume.json', dict(cursor=cursor, step=step, best=None if not math.isfinite(best) else best, bad=bad, updated_at=time.time()))
    for epoch in range(cursor['epoch'], cfg['max_epochs']):
        if epoch >= cfg['min_epochs'] and bad >= cfg['early_stopping_patience']:
            break
        scorer.train()
        start_time = time.time()
        if cursor['phase'] == 'train':
            for wi, window in enumerate(epoch_windows(offsets, 0, epoch, size=cfg['effective_batch'])):
                if wi < cursor['window']:
                    continue
                denom = int(sum((offsets[i + 1] - offsets[i] - 1 for i in window)))
                opt.zero_grad(set_to_none=True)
                total = 0.0
                pos = 0
                while pos < len(window):
                    n = int(offsets[window[pos] + 1] - offsets[window[pos]] - 1)
                    size = cfg['short_microbatch'] if n <= 32 else cfg['medium_microbatch'] if n <= 128 else cfg['long_microbatch']
                    boundary = 32 if n <= 32 else 128 if n <= 128 else 1024
                    stop = pos
                    while stop < min(pos + size, len(window)) and offsets[window[stop] + 1] - offsets[window[stop]] - 1 <= boundary:
                        stop += 1
                    ids, lengths = batch_ids(items, offsets, window[pos:stop])
                    tokens = int(lengths.sum())
                    if selector is None:
                        loss = parent_loss(loaded, ids, lengths, catalog)
                    else:
                        with torch.no_grad():
                            packed_outputs(loaded, ids, lengths)
                        context.value = dict(past_ids=ids, past_lengths=lengths)
                        embeddings = model.get_item_embeddings(ids[:, 1:])
                        weights = (ids[:, 1:] != 0).float()
                        loss, _ = lossfn(lengths=lengths, output_embeddings=torch.zeros_like(embeddings), supervision_ids=ids[:, 1:], supervision_embeddings=embeddings, supervision_weights=weights, negatives_sampler=sampler)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f'nonfinite {stage} loss')
                    (loss * (tokens / denom)).backward()
                    total += float(loss.detach()) * tokens / denom
                    pos = stop
                if any((p.grad is not None and (not torch.isfinite(p.grad).all()) for p in model.parameters())):
                    raise RuntimeError('nonfinite gradients')
                opt.step()
                step += 1
                cursor = dict(epoch=epoch, window=wi + 1, phase='train')
                if step % 10 == 0 or step <= 3:
                    status = dict(stage=stage, epoch=epoch, step=step, window=wi + 1, loss=total, lr=opt.param_groups[0]['lr'], elapsed=time.time() - start_time)
                    emit(**status)
                    atomic_write_json(run / 'status.json', status)
                if step % cfg['checkpoint_step_interval'] == 0 or step == 1:
                    persist()
                if stop_after_steps and step >= stop_after_steps:
                    persist()
                    return
        cursor = dict(epoch=epoch, window=cursor['window'], phase='validation')
        persist()
        metrics = evaluate(scorer, run, f'{stage}/validation-e{epoch:03d}')
        primary = metrics['ndcg_at_10']
        improved = primary > best + cfg['min_delta']
        if improved:
            best = primary
            bad = 0
            save_pt(folder / 'best.pt', dict(model=cpu_state(model), epoch=epoch, metrics=metrics, config_sha256=sha256_file(run / 'experiment.json')))
        else:
            bad += 1
        sched.step(primary)
        history.append(dict(epoch=epoch, step=step, metrics=metrics, selected=improved, lr=opt.param_groups[0]['lr']))
        atomic_write_json(folder / 'history.json', history)
        cursor = dict(epoch=epoch + 1, window=0, phase='train')
        persist()
        emit(stage=stage, epoch=epoch, metrics=metrics, bad_epochs=bad)
    atomic_write_json(folder / 'completed.json', dict(best=best, epochs=len(history), steps=step))
    model.load_state_dict(torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)['model'])

def cwi_stage(loaded, run):
    cfg = execution_configuration(run)
    folder = run / 'cwi'
    folder.mkdir(exist_ok=True)
    parent_sha = sha256_file(run / 'full-gdr/best.pt')
    chosen = json.loads((run / 'data/training_prefixes.json').read_text())['rows']
    full = AmazonScorer(loaded, method='full-gdr').eval()
    loaded.model.requires_grad_(False)
    catalog = torch.tensor(load_item_ids(Path(cfg['data_files']['train_catalog'])), device='cuda')
    for i, row in enumerate(chosen):
        path = folder / f'{i:04d}.pt'
        if path.exists():
            old = torch.load(path, map_location='cpu', weights_only=False)
            if old['parent_sha256'] != parent_sha or old['sample'] != row:
                raise RuntimeError('CWI resume lineage changed')
            continue
        ids = torch.tensor([row['history']], device='cuda')
        lengths = torch.tensor([ids.shape[1]], device='cuda')
        gen = torch.Generator(device='cuda').manual_seed(13092026 + i)
        negative = catalog[torch.randint(len(catalog), (1, 512), device='cuda', generator=gen)]
        c = torch.cat((torch.tensor([[row['target']]], device='cuda'), negative), 1)
        gate = torch.ones(ids.shape[1], device='cuda', requires_grad=True)
        x = full._preprocess_history(ids, lengths).reshape(-1, 64)
        off = torch.tensor([0, len(x)], device='cuda')
        states = []
        for layer in _research_layers(loaded.model):
            prior = layer.fla_kernel.assume_binary_event_gate
            layer.fla_kernel.assume_binary_event_gate = False
            try:
                x, state = layer.forward_gdr_streams(x=x, x_offsets=off, event_gate=gate, return_final_state=True)
            finally:
                layer.fla_kernel.assume_binary_event_gate = prior
            states.append(state)
        cache = ResearchGDRStateCache(schema=RUNTIME_SCHEMA, method='full-gdr', state=torch.stack(states, 1).unsqueeze(1), history_lengths=lengths, stream_count=1, candidate_key=None, candidate_group_ids=None, selected_counts=lengths[:, None], realized_write_ratio=1.0, model_binding_sha256=full.model_binding_sha256, selector_binding_sha256=None, grouping_binding_sha256=None)
        scores = full.serve_cache_hit(candidate_item_ids=c, cache=cache)
        logits = scores / 0.05
        logits = torch.cat((logits[:, :1], logits[:, 1:].masked_fill(c[:, 1:] == c[:, :1], -50000.0)), 1)
        loss = -F.log_softmax(logits, 1)[0, 0]
        labels = -torch.autograd.grad(loss, gate)[0]
        if not torch.isfinite(labels).all():
            raise RuntimeError('nonfinite CWI')
        save_pt(path, dict(sample=row, cwi=labels.detach().cpu(), parent_sha256=parent_sha))
        if i % 16 == 0:
            emit(stage='cwi', completed=i + 1, total=len(chosen))
            atomic_write_json(run / 'status.json', dict(stage='cwi', completed=i + 1, total=len(chosen)))
    atomic_write_json(folder / 'completed.json', dict(samples=len(chosen), parent_sha256=parent_sha))

def selector_stage(loaded, run):
    folder = run / 'selector'
    folder.mkdir(exist_ok=True)
    selector = selector_for(loaded)
    loaded.model.eval().requires_grad_(False)
    if (folder / 'completed.json').exists():
        selector.load_state_dict(torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)['selector'])
        return selector
    opt = torch.optim.AdamW(selector.parameters(), lr=0.0001, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, min_lr=1e-06)
    labels = [torch.load(p, map_location='cpu', weights_only=False) for p in sorted((run / 'cwi').glob('*.pt'))]
    if len(labels) != execution_configuration(run)['cwi_count']:
        raise RuntimeError('selector needs all 4096 CWI prefixes')
    epoch0 = 0
    best = -math.inf
    history = []
    phase = 'train'
    epoch_loss = None
    if (folder / 'latest.pt').exists():
        state = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=False)
        selector.load_state_dict(state['selector'])
        opt.load_state_dict(state['optimizer'])
        sched.load_state_dict(state['scheduler'])
        epoch0 = state['epoch']
        best = state['best']
        history = state['history']
        phase = state.get('phase', 'train')
        epoch_loss = state.get('epoch_loss')
        restore_rng(state['rng'])
    for epoch in range(epoch0, execution_configuration(run)['selector_epochs']):
        if phase == 'train':
            selector.train().requires_grad_(True)
            total = count = 0
            order = np.random.default_rng(epoch).permutation(len(labels))
            for i in order:
                rec = labels[i]
                row = rec['sample']
                h = torch.tensor([row['history']], device='cuda')
                n = torch.tensor([h.shape[1]], device='cuda')
                c = torch.tensor([[row['target']]], device='cuda')
                target = rec['cwi'].cuda().asinh()
                score = selector.score_ids(h, c, n)[0, 0]
                loss = F.smooth_l1_loss(score, target)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(target)
                count += len(target)
            epoch_loss = total / count
            phase = 'validation'
            save_pt(folder / 'latest.pt', dict(selector=cpu_state(selector), optimizer=opt.state_dict(), scheduler=sched.state_dict(), epoch=epoch, phase=phase, epoch_loss=total / count, best=best, history=history, rng=rng_state()))
        scorer = scorer_for(loaded, run, selector)
        save_pt(folder / f'epoch-{epoch:02d}.pt', dict(selector=cpu_state(selector), epoch=epoch, loss=epoch_loss))
        metrics = evaluate(scorer, run, f'selector/validation-e{epoch:03d}')
        if metrics['ndcg_at_10'] > best:
            best = metrics['ndcg_at_10']
            save_pt(folder / 'best.pt', dict(selector=cpu_state(selector), epoch=epoch, metrics=metrics))
        sched.step(epoch_loss)
        history.append(dict(epoch=epoch, loss=epoch_loss, metrics=metrics))
        atomic_write_json(folder / 'history.json', history)
        save_pt(folder / 'latest.pt', dict(selector=cpu_state(selector), optimizer=opt.state_dict(), scheduler=sched.state_dict(), epoch=epoch + 1, best=best, history=history, phase='train', rng=rng_state()))
        phase = 'train'
        emit(stage='selector', epoch=epoch, loss=epoch_loss, metrics=metrics)
    atomic_write_json(folder / 'completed.json', dict(best=best, epochs=execution_configuration(run)['selector_epochs']))
    selector.load_state_dict(torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)['selector'])
    return selector
from types import SimpleNamespace
from deltarec.layers.hstu_gdr import _research_layers
from deltarec.models.hstu_selector import RatingGCSelector,VerifiedBoundGrouping
from deltarec.metrics.evaluation import AtomicJsonlWriter,evaluate_rating_stream
from deltarec.utils.protocols import AdapterOutput

def execution_configuration(run):
    cfg=json.loads((run/"experiment.json").read_text())
    path=run/"resume_execution.json"
    if not path.is_file(): return cfg
    resume=json.loads(path.read_text())
    if resume.get("base_experiment_sha256") != sha256_file(run/"experiment.json"):
        raise RuntimeError("resume execution base experiment changed")
    if set(resume["parameters"]) != set(_RESUME_EXECUTION_FIELDS):
        raise RuntimeError("resume execution parameter set changed")
    cfg.update(resume["parameters"])
    if cfg["effective_batch"] != 128 or any(cfg[name] < 1 for name in _RESUME_EXECUTION_FIELDS):
        raise RuntimeError("invalid resume execution configuration")
    return cfg

def scorer_for(loaded,run,selector=None):
    if selector is None:return AmazonScorer(loaded,method='full-gdr')
    grouping=torch.load(run/'grouping.pt',map_location='cpu',weights_only=True)
    evidence={'binding_sha256':sha256_file(run/'grouping.pt')}
    record=VerifiedBoundGrouping('amazon-books',len(grouping['prototypes']),grouping['mapping'],grouping['prototypes'],evidence)
    gc=RatingGCSelector(selector,record).to(next(loaded.model.parameters()).device)
    return AmazonScorer(loaded,method='deltarec-gc',retention_ratio=.25,selector=gc,
        selector_binding_sha256='utility-estimator',grouping_binding_sha256=evidence['binding_sha256'])

class EvaluationAdapter:
    def __init__(self,scorer):self.scorer=scorer
    def run(self,batch,mode):
        device=next(self.scorer.model.parameters()).device
        lengths=batch.history_lengths.to(device)
        histories=batch.history_item_ids[:,:int(lengths.max())].to(device)
        candidates=batch.candidate_item_ids.to(device)
        with torch.inference_mode():
            cache=self.scorer.build_state_cache(history_item_ids=histories,history_lengths=lengths,candidate_item_ids=candidates)
            scores=self.scorer.serve_cache_hit(candidate_item_ids=candidates,cache=cache)
        return AdapterOutput(scores)

def evaluate(scorer,run,stage):
    out=run/stage;out.mkdir(parents=True,exist_ok=True)
    cfg=execution_configuration(run)
    evidence=out/'per_request.jsonl'
    if evidence.exists():evidence.unlink()
    scorer.eval()
    with AtomicJsonlWriter(evidence) as writer:
        metrics,details,_=evaluate_rating_stream(torch=torch,adapter=EvaluationAdapter(scorer),
            candidate_manifest=Path(cfg['data_files']['validation_candidates']),dataset='amazon-books',
            split_role='validation',protocol_lock_hash=None,microbatch_size=cfg['eval_microbatch'],evidence=writer)
    atomic_write_json(out/'metrics.json',dict(metrics=metrics,details=details))
    return metrics
