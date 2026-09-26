import torch
from dataclasses import asdict
from deltarec.utils.io import atomic_write_json
from dataclasses import asdict
import logging
import os
from pathlib import Path
import time
PLATEAU = dict(kind='reduce_lr_on_plateau', mode='max', factor=0.5, patience=2, threshold=0.0001, threshold_mode='abs', cooldown=0, min_lr=3e-06, metric_name='NDCG@10')
EARLY = dict(primary_mode='max', tie_breaker_mode=None, min_delta=0.0001, patience=10, minimum_epochs=5, maximum_epochs=100)

def train_loop(owner, scoring, config, loader, sampler, catalog, work, manager, trainer, *, validation_scoring=None):
    import torch
    shared_backward = getattr(scoring, 'backward_batch', None)
    backward = scoring.backward_batch
    from deltarec.utils.hstu_accumulation import GradientAccumulationController
    from deltarec.utils.hstu_resume import ResumableWarmupScheduler
    from deltarec.utils.io import atomic_write_json
    model, c = (owner.model, owner.contract)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-05, betas=(0.9, 0.98), weight_decay=config['weight_decay'])
    scheduler = ResumableWarmupScheduler(optimizer, learning_rate=3e-05, warmup_steps=0, validation_scheduler_contract=PLATEAU, validation_history_path=work / 'lr_scheduler_history.json')
    resume = manager.restore(torch, model=model, optimizer=optimizer, scheduler=scheduler, sampler=sampler)
    accumulation = GradientAccumulationController(c['accumulation'])
    accumulation.optimizer_steps = resume.global_step
    completed = 0
    started = time.monotonic()
    checkpoint_args = dict(model=model, optimizer=optimizer, scheduler=scheduler, sampler=sampler, num_microbatches=len(loader), provenance=c)
    for epoch in range(resume.epoch, config.get('max_epochs', 100)):
        if manager.early_stopping_state.stopped:
            break
        sampler.set_epoch(epoch)
        model.train()
        skip = resume.next_microbatch if epoch == resume.epoch else 0
        if not 0 <= skip <= len(loader) or (skip != len(loader) and skip % c['accumulation']):
            raise ValueError('resume cursor is not an optimizer boundary')
        iterator = iter(loader)
        for _ in range(skip):
            next(iterator)
        if epoch == resume.epoch and resume.resumed_from is not None:
            manager.restore_rng_after_loader_position(torch)
        for index, row in enumerate(iterator, start=skip):
            should_step = accumulation.start_microbatch(model=model, optimizer=optimizer, microbatch_index=index, microbatches_in_epoch=len(loader))
            features, target_ids, _ = trainer.movielens_seq_features_from_row(row, device=torch.device('cpu' if shared_backward else 'cuda:0'), max_output_length=config['gr_output_length'] + 1)
            features.past_ids.scatter_(1, features.past_lengths[:, None], target_ids.reshape(-1, 1))
            numerator, tokens = backward(features.past_ids, features.past_lengths, catalog, config['num_negatives'], config['temperature'], supervision_block=c['supervision_chunk_size'])
            accumulation.numerator_loss(torch.tensor(numerator / tokens, dtype=torch.float64), tokens)
            if should_step:
                for parameter in model.parameters():
                    gradient = parameter.grad
                    if gradient is not None:
                        values = gradient._values() if gradient.is_sparse else gradient
                        if not bool(torch.isfinite(values).all()):
                            raise ValueError('non-finite sparse gradient')
                scheduler.apply_for_step(accumulation.optimizer_steps)
            stepped = accumulation.finish_microbatch(model, optimizer)
            completed += 1
            if stepped:
                scheduler.mark_step_completed(accumulation.optimizer_steps)
                manager.maybe_save_step(torch, global_step=accumulation.optimizer_steps, epoch=epoch, next_microbatch=index + 1, **checkpoint_args)
            if completed <= 3 or completed % 8 == 0 or stepped:
                progress = dict(epoch=epoch, next_microbatch=index + 1, completed_backward_microbatches=completed, work_counter_scope='current-process-since-resume', resumed_from=None if resume.resumed_from is None else Path(resume.resumed_from).name, physical_users=c.get('microbatch', 8), effective_users=c.get('effective_batch', 128), optimizer_steps=accumulation.optimizer_steps, last_history_loss=numerator / tokens, work_counts=asdict(scoring.counts), supervised_prefixes=scoring.counts.supervised_positions, elapsed_seconds=time.monotonic() - started, cuda_peak_bytes=torch.cuda.max_memory_allocated())
                atomic_write_json(work / 'training_progress.json', progress)
                logging.info('Sparse progress %s', progress)
        metrics = validate(owner, work / 'common_validation', epoch, scoring=validation_scoring)
        scheduler.observe_validation(metric=metrics['ndcg@10'], epoch=epoch, global_step=accumulation.optimizer_steps)
        decision = manager.observe_validation(primary_metric=metrics['ndcg@10'], epoch=epoch)
        checkpoint_path = manager.save(torch, kind='epoch', epoch=epoch + 1, next_microbatch=0, global_step=accumulation.optimizer_steps, **checkpoint_args)
        if decision['improved']:
            manager.pin_best(checkpoint_path)
        logging.info('Sparse validation epoch=%s metrics=%s decision=%s', epoch, metrics, decision)
        if decision['should_stop']:
            manager.pin_final(checkpoint_path)
            break
    if not (work / 'best.pt').is_file():
        manager.save_final_export(torch, work / 'best.pt', use_best=True)
    atomic_write_json(work / 'completion.json', dict(early_stopping_state=manager.early_stopping_state.to_dict(), test_access=False, diagnostic_only=False))
from deltarec.metrics.evaluation import AtomicJsonlWriter,evaluate_rating_stream
from deltarec.utils.protocols import AdapterOutput

class EvaluationAdapter:
    def __init__(self,scoring):self.scoring=scoring
    def run(self,batch,mode):
        device=next(self.scoring.model.parameters()).device
        lengths=batch.history_lengths.to(device)
        histories=batch.history_item_ids[:,:int(lengths.max())].to(device)
        candidates=batch.candidate_item_ids.to(device)
        with torch.inference_mode():scores=self.scoring.scores(histories,lengths,candidates)
        return AdapterOutput(scores)

def validate(owner,output,epoch,*,scoring):
    c=owner.contract;owner.model.eval();output.mkdir(parents=True,exist_ok=True)
    evidence=output/f'epoch-{epoch:04d}.jsonl'
    if evidence.exists():evidence.unlink()
    with AtomicJsonlWriter(evidence) as writer:
        metrics,details,_=evaluate_rating_stream(torch=__import__('torch'),adapter=EvaluationAdapter(scoring),
            candidate_manifest=Path(c['inputs']['candidates']['path']),dataset=c['dataset'],split_role='validation',
            protocol_lock_hash=None,microbatch_size=c['validation_microbatch'],evidence=writer)
    atomic_write_json(output/f'epoch-{epoch:04d}.json',dict(metrics=metrics,details=details))
    return {'hr@10':metrics['hr_at_10'],'ndcg@10':metrics['ndcg_at_10']}
