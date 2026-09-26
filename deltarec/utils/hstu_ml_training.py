"""HSTU ML-20M stage wiring; training operations follow the source trainer."""
import json,random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from deltarec.data.history import NumericHistoryDataset
from deltarec.data.hstu_training import load_item_ids
from deltarec.data.hstu_cwi import sample_hstu_cwi_contexts
from deltarec.adaptors.hstu.modeling.sequential.features import movielens_seq_features_from_row
from deltarec.adaptors.hstu.modeling.sequential.autoregressive_losses import LocalNegativesSampler
from deltarec.adaptors.hstu.modeling.sequential.losses.sampled_softmax import SampledSoftmaxLoss
from deltarec.utils.hstu_resume import ResumableCheckpointManager,ResumableWarmupScheduler
from deltarec.utils.hstu_sparse_training import EARLY,PLATEAU,train_loop
from deltarec.utils.hstu_partition import global_partition
from deltarec.utils.hstu_selector_training import (write_cwi_label_shard,create_cwi_label_manifest,
    create_trajectory_cwi_manifest,train_candidate_cwi_mlp_trajectory)
from deltarec.utils.hstu_cwi import label_contexts
from deltarec.models.hstu_selector import RatingPCSelector
from deltarec.models.hstu_runtime import OfficialResearchSparseScorer
from deltarec.utils.io import sha256_file,atomic_write_json
from deltarec.utils.checkpoint import atomic_torch_save


def training_dataset(c,files):
    from deltarec.adaptors.hstu.data.dataset import DatasetV2
    d=DatasetV2(str(files['train']),padding_length=c['model']['max_history_length']+1,
        ignore_last_n=1,shift_id_by=0,chronological=True,sample_ratio=1.)
    # Same zero-supervision filter as the original training entry.
    keep=d.ratings_frame['sequence_item_ids'].map(lambda s:len(str(s).split(','))>2)
    d.ratings_frame=d.ratings_frame.loc[keep].reset_index(drop=True)
    return NumericHistoryDataset(d)


def loader_for(dataset,batch):
    sampler=torch.utils.data.DistributedSampler(dataset,num_replicas=1,rank=0,shuffle=True,seed=0,drop_last=False)
    loader=torch.utils.data.DataLoader(dataset,batch_size=batch,sampler=sampler,num_workers=0,
        generator=torch.Generator().manual_seed(0))
    return loader,sampler


def loss_modules(c,loaded,catalog):
    table=loaded.model._embedding_module._item_emb
    sampler=LocalNegativesSampler(len(catalog),table,catalog,True,1e-6).to(c['device'])
    loss=SampledSoftmaxLoss(num_to_sample=c['num_negatives'],softmax_temperature=c['temperature'],model=loaded.model).to(c['device'])
    return sampler,loss


def warmup(c,files,loaded,training,out,args):
    from train_hstu import load_state,seed,construct,evaluate
    parent=Path(c['initializer'])
    paths=[]
    for index,epochs in enumerate(c['teacher_epochs'],1):
        dest=out/f'teacher-{index:02d}.pt';paths.append(dest)
        if dest.exists():parent=dest;continue
        seed(c['seed']);loaded=construct(c,files)
        loaded.model.load_state_dict(load_state(parent),strict=True)
        loaded.model.requires_grad_(True)
        work=out/f'warmup-{index:02d}';work.mkdir(exist_ok=True)
        loader,sampler=loader_for(training,128)
        opt=torch.optim.AdamW(loaded.model.parameters(),lr=3e-5,betas=(.9,.98),weight_decay=0.)
        scheduler=ResumableWarmupScheduler(opt,learning_rate=3e-5,warmup_steps=0,
            validation_scheduler_contract=PLATEAU,validation_history_path=work/'lr.json')
        identity={'config':c['config_sha256'],'stage':'warmup','segment':index,'parent':sha256_file(parent)}
        manager=ResumableCheckpointManager(work/'checkpoints',identity=identity,step_interval=5000,keep_last=3)
        resume=manager.restore(torch,model=loaded.model,optimizer=opt,scheduler=scheduler,sampler=sampler)
        step=resume.global_step;negatives,loss_fn=loss_modules(c,loaded,load_item_ids(files['train_catalog']))
        ck=dict(model=loaded.model,optimizer=opt,scheduler=scheduler,sampler=sampler,num_microbatches=len(loader),provenance=identity)
        for epoch in range(resume.epoch,epochs):
            sampler.set_epoch(epoch);loaded.model.train();it=iter(loader)
            skip=resume.next_microbatch if epoch==resume.epoch else 0
            for _ in range(skip):next(it)
            if epoch==resume.epoch and resume.resumed_from is not None:manager.restore_rng_after_loader_position(torch)
            for batch_index,row in enumerate(it,skip):
                f,target,_=movielens_seq_features_from_row(row,device=c['device'],max_output_length=c['gr_output_length']+1)
                f.past_ids.scatter_(1,f.past_lengths[:,None],target.reshape(-1,1))
                opt.zero_grad(set_to_none=True)
                embeddings=loaded.model.get_item_embeddings(f.past_ids)
                outputs=loaded.model(past_lengths=f.past_lengths,past_ids=f.past_ids,past_embeddings=embeddings,past_payloads=f.past_payloads)
                loss,_=loss_fn(lengths=f.past_lengths,output_embeddings=outputs[:,:-1],supervision_ids=f.past_ids[:,1:],
                    supervision_embeddings=embeddings[:,1:],supervision_weights=(f.past_ids[:,1:]!=0).float(),negatives_sampler=negatives,**f.past_payloads)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite full-history loss')
                loss.backward()
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in loaded.model.parameters()):raise FloatingPointError('nonfinite full-history gradients')
                scheduler.apply_for_step(step);opt.step();step+=1;scheduler.mark_step_completed(step)
                manager.maybe_save_step(torch,global_step=step,epoch=epoch,next_microbatch=batch_index+1,**ck)
            scorer=OfficialResearchSparseScorer(loaded,method='full-gdr')
            metrics=evaluate(c,files,scorer,work/f'validation-{epoch:02d}')
            scheduler.observe_validation(metric=metrics['ndcg_at_10'],epoch=epoch,global_step=step)
            manager.save(torch,kind='epoch',epoch=epoch+1,next_microbatch=0,global_step=step,**ck)
        atomic_torch_save(torch,{'model':{k:v.cpu() for k,v in loaded.model.state_dict().items()}},dest)
        parent=dest
    return paths


def run(c,files,args,loaded):
    from train_hstu import load_state,seed,construct,evaluate,export_model,scoring_for
    out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
    training=training_dataset(c,files);catalog=load_item_ids(files['train_catalog'])
    paths=[Path(p) for p in c['teacher_checkpoints']]
    if not all(p.is_file() for p in paths):
        if args.start_stage!='full-gdr':raise FileNotFoundError('Both full-history teachers are required')
        paths=warmup(c,files,loaded,training,out,args)
    if args.stop_stage=='full-gdr':return
    trajectory=out/'cwi_trajectory.json'
    if not trajectory.exists():
        sample_path=out/'cwi_contexts.pt'
        if sample_path.exists():contexts=torch.load(sample_path,weights_only=True)['contexts']
        else:
            contexts=sample_hstu_cwi_contexts(training,cwi_samples=c['cwi_samples'],seed=1000)
            atomic_torch_save(torch,{'contexts':contexts},sample_path)
        manifests=[]
        for index,path in enumerate(paths,1):
            loaded.model.load_state_dict(load_state(path),strict=True);loaded.model.eval().requires_grad_(False)
            torch.manual_seed(2000+index)
            scorer=OfficialResearchSparseScorer(loaded,method='full-gdr');negatives,loss_fn=loss_modules(c,loaded,catalog)
            def focal_loss(queries,positives):
                values=[]
                for query,positive in zip(queries,positives[:,0]):
                    positive=positive.reshape(1)
                    value,_=loss_fn.jagged_forward(output_embeddings=query.reshape(1,-1),supervision_ids=positive,
                        supervision_embeddings=loaded.model.get_item_embeddings(positive),supervision_weights=torch.ones(1,device=query.device),negatives_sampler=negatives)
                    values.append(value)
                return torch.stack(values)[:,None]
            folder=out/f'cwi-{index:02d}';folder.mkdir(exist_ok=True);shards=[]
            for start in range(0,len(contexts),2):
                shard=folder/f'{start:06d}.pt';shards.append(shard)
                if shard.exists():continue
                rows=contexts[start:start+2];h,n,p,labels=label_contexts(scorer,focal_loss,rows)
                write_cwi_label_shard(shard,dataset=c['dataset'],seed=0,parent_full_checkpoint_sha256=sha256_file(path),
                    candidate_identity='candidate-id',user_ids=torch.tensor([r['user_id'] for r in rows]),history_item_ids=h,history_lengths=n,candidate_item_ids=p,cwi1=labels)
            manifest=folder/'manifest.json';manifests.append(manifest)
            create_cwi_label_manifest(manifest,dataset=c['dataset'],seed=0,protocol_sha256=c['config_sha256'],
                parent_full_checkpoint_sha256=sha256_file(path),parent_full_manifest_content_sha256=sha256_file(path),
                split_manifest=files['split_manifest'],training_histories=files['train'],training_catalog=files['train_catalog'],item_id_map=files['item_map'],candidate_identity='candidate-id',shard_paths=shards)
        create_trajectory_cwi_manifest(trajectory,teacher_manifests=manifests,teacher_weights=c['teacher_weights'])
    if args.stop_stage=='cwi':return
    loaded.model.load_state_dict(load_state(paths[-1]),strict=True);loaded.model.eval().requires_grad_(False)
    table=loaded.model._embedding_module._item_emb
    selector=RatingPCSelector(table,expected_num_items=table.num_embeddings-1,expected_embedding_dim=table.embedding_dim,seed=0,freeze_embedding_snapshot=True).to(c['device'])
    selector_path=out/'selector.pt'
    if selector_path.exists():selector.load_state_dict(torch.load(selector_path,weights_only=True),strict=True)
    else:
        teachers=[torch.nn.Embedding.from_pretrained(load_state(path)['_embedding_module._item_emb.weight'],freeze=True,padding_idx=0) for path in paths]
        live=RatingPCSelector(table,expected_num_items=table.num_embeddings-1,expected_embedding_dim=table.embedding_dim,seed=0).to(c['device'])
        scorer=OfficialResearchSparseScorer(loaded,method='deltarec-pc',retention_ratio=.25,selector=live,selector_binding_sha256=sha256_file(paths[-1]))
        def validate(current,epoch):
            live.input.load_state_dict(current.input.state_dict());live.output.load_state_dict(current.output.state_dict())
            return evaluate(c,files,scorer,out/f'selector-validation-{epoch:02d}')
        fit=train_candidate_cwi_mlp_trajectory(selector,trajectory_cwi_manifest=trajectory,teacher_embeddings=teachers,
            validation_score_fn=validate,seed=0,device=c['device'],resume_path=out/'selector_resume.pt')
        atomic_write_json(out/'selector_fit.json',fit);atomic_torch_save(torch,selector.state_dict(),selector_path)
    selector.requires_grad_(False).eval()
    partition=global_partition(out/'grouping.pt',selector,training,catalog,identity={'selector':sha256_file(selector_path),'teacher':sha256_file(paths[-1])},groups=c['group_count'])
    grouping={'mapping':partition['item_to_group']}
    if args.stop_stage=='selector':return
    loaded.model.requires_grad_(True);seed(0)
    scoring=scoring_for(c,loaded,selector,grouping);loader,sampler=loader_for(training,c['microbatch'])
    contract=dict(dataset=c['dataset'],config_sha256=c['config_sha256'],accumulation=1,microbatch=c['microbatch'],effective_batch=c['effective_batch'],
        supervision_chunk_size=16,max_stream_tokens=16384,validation_microbatch=c['eval_microbatch'],method='deltarec-global-utility',
        inputs={'candidates':{'path':str(files['validation_candidates'])}},candidate_manifest_sha256=sha256_file(files['validation_candidates']))
    owner=SimpleNamespace(model=loaded.model,scorer=scoring.scorer,contract=contract)
    work=out/'sparse';work.mkdir(exist_ok=True)
    manager=ResumableCheckpointManager(work/'checkpoints',identity=contract,step_interval=10,keep_last=3,early_stopping_contract=EARLY)
    train_loop(owner,scoring,c,loader,sampler,torch.tensor(catalog,device=c['device']),work,manager,
        SimpleNamespace(movielens_seq_features_from_row=movielens_seq_features_from_row),validation_scoring=scoring)
    loaded.model.load_state_dict(load_state(work/'best.pt'),strict=True)
    export_model(out/'model.pt',c,loaded,selector,grouping)
