"""KuaiRand HSTU stage wiring using the original DLRMv3 training loop."""
from pathlib import Path
from types import SimpleNamespace
import json,socket,gc
import torch
from deltarec.adaptors.hstu_kuai import build_model,activate,make_optimizer_and_shard
from deltarec.adaptors.kuai.dlrm_v3.train import utils
from deltarec.adaptors.kuai.dlrm_v3 import checkpoint
from deltarec.data.kuai_loader import make_frozen_kuai_dataloaders,_kuai_video_embedding_state
from deltarec.metrics.kuai_evaluation import evaluate_user_gauc
from deltarec.utils.kuai_training import resumable_train_loop,resolve_latest_complete_checkpoint
from deltarec.utils.validation_lr_scheduler import ValidationReduceLROnPlateau
from deltarec.utils.checkpoint import atomic_torch_save
from deltarec.utils.io import sha256_file,atomic_write_json


class ProgressLogger:
    """Step state and loss logging; quality is computed by the full evaluator."""
    def __init__(self):
        self.global_step={'train':0,'eval':0}
        self.class_metrics={'train':[],'eval':[]};self.regression_metrics={'train':[],'eval':[]}
    def update(self,*,mode,**values):self.global_step[mode]+=1
    def compute_and_log(self,*,mode,additional_logs):
        print(json.dumps({'mode':mode,'step':self.global_step[mode],
            'losses':{k:float(v.detach()) for k,v in additional_logs.get('losses',{}).items()}}),flush=True)


def portable_state(model):
    result={}
    for key,value in model.state_dict().items():
        if hasattr(value,'local_shards'):
            shards=value.local_shards()
            if len(shards)!=1 or list(shards[0].metadata.shard_offsets)!=[0,0] or tuple(shards[0].tensor.shape)!=tuple(value.shape):
                raise ValueError('Portable export requires a complete world-size-one tensor')
            value=shards[0].tensor
        result[key]=value.detach().cpu()
    return result


def load_portable(model,state,*,allow_sparse_additions=False,embedding_only=False):
    own=model.state_dict();missing=set(own)-set(state);extra=set(state)-set(own)
    if not embedding_only and (extra or (missing and not allow_sparse_additions)):
        raise ValueError(f'Checkpoint coverage mismatch: missing={sorted(missing)} extra={sorted(extra)}')
    if allow_sparse_additions and any(not any(x in k for x in ['selector','category_group','item_to_category']) for k in missing):
        raise ValueError('Sparse initialization omitted non-selector parameters')
    with torch.no_grad():
        for key,target in own.items():
            if key not in state or (embedding_only and 'embedding_collection' not in key):continue
            value=state[key]
            if hasattr(target,'local_shards'):
                shards=target.local_shards()
                if len(shards)!=1 or list(shards[0].metadata.shard_offsets)!=[0,0]:raise ValueError('Only world size one is supported')
                target=shards[0].tensor
            if target.shape!=value.shape:raise ValueError('Checkpoint tensor shape changed: '+key)
            target.copy_(value)


def export_model(path,model,*,selector=None,grouping=None):
    atomic_torch_save(torch,dict(schema='deltarec-release-v1',model=portable_state(model),
        selector=selector,grouping=grouping),path)


def construct(c,*,mode,payload=None):
    model,_=build_model(c)
    activate(model,c,mode=mode,selector_state=None if payload is None else payload.get('selector'),
        grouping=None if payload is None else payload.get('grouping'))
    model,optimizer=make_optimizer_and_shard(model,c,device=torch.device(c['device']),
        learning_rate_multiplier=c['warmup_lr_multiplier'] if mode=='full_gdr' else c['sparse_lr_multiplier'])
    if payload is not None:load_portable(model,payload['model'],allow_sparse_additions=mode=='delta_gc')
    return model,optimizer


def loaders(c,files,*,training):
    return make_frozen_kuai_dataloaders(train_slates=files['train'] if training else None,
        evaluation_slates=files['validation'],user_features=files['user_features'],batch_size=c['microbatch'],
        num_workers=0,prefetch_factor=None,train_utils=utils,torch=torch)


def evaluate(c,model,loader,output):
    loader.metric_evidence_root=output;loader.metric_split_role='validation'
    result=evaluate_user_gauc(model=model,dataloader=loader,metric_logger=None,device=torch.device(c['device']),torch=torch)
    atomic_write_json(output/'metrics.json',result);return result


def train_stage(c,files,model,optimizer,out,stage):
    training,validation=loaders(c,files,training=True)
    ck=out/stage/'checkpoints';logger=ProgressLogger()
    contract=dict(kind='reduce_lr_on_plateau',mode='max',factor=.5,patience=2,threshold=1e-4,
        threshold_mode='abs',cooldown=0,min_lr=3e-6,metric_name='Macro-GAUC')
    scheduler=ValidationReduceLROnPlateau(optimizer,contract)
    def validate(epoch,cursor):return evaluate(c,model,validation,out/stage/f'validation-{epoch:03d}')
    binding=dict(schema='deltarec-kuai-training-v1',config_sha256=c['config_sha256'],data_sha256=c['binding_sha256'],stage=stage)
    resumable_train_loop(rank=0,model=model,dataloader=training,optimizer=optimizer,metric_logger=logger,
        device=torch.device(c['device']),checkpoint_module=checkpoint,checkpoint_root=ck,binding=binding,torch=torch,
        num_epochs=c['max_epochs'],checkpoint_frequency=2000,metric_log_frequency=100,output_trace=False,profiler_class=None,
        resume_checkpoint=resolve_latest_complete_checkpoint(ck),keep_last_complete=3,validation_callback=validate,
        early_stopping_contract=dict(primary_mode='max',tie_breaker_mode='min',min_delta=1e-4,patience=10,minimum_epochs=5,maximum_epochs=c['max_epochs']),
        early_stopping_state_path=out/stage/'early_stopping.json',validation_history_path=out/stage/'validation.json',
        lr_scheduler=scheduler,lr_scheduler_history_path=out/stage/'lr.json')
    best=ck/(ck/'BEST').read_text().strip()
    checkpoint.load_dmp_checkpoint(model=model,optimizer=None,metric_logger=None,device=torch.device(c['device']),path=str(best))


def bind_grouping(model,source,count):
    payload=torch.load(source,map_location='cpu',weights_only=True)
    ids=payload['training_catalog_item_ids'];positions=torch.searchsorted(payload['catalog_item_ids'],ids)
    ptr=payload['catalog_indptr'];active=ptr[positions+1]>ptr[positions]
    mapping=payload['item_to_category_group'];groups=mapping[ids];counts=torch.bincount(groups[active],minlength=count)
    if bool((counts==0).any()):raise ValueError('Category group has no categorized training members')
    _,table=_kuai_video_embedding_state(model);weight=table.local_shards()[0].tensor
    sums=torch.zeros(count,512)
    for start in range(0,len(ids),65536):
        mask=active[start:start+65536]
        selected=weight[ids[start:start+65536][mask].to(weight.device)].detach().float().cpu()
        sums.index_add_(0,groups[start:start+65536][mask],selected)
    return {'item_to_category_group':mapping,'category_group_prototypes':sums/counts[:,None].float()}


def fit_selector(teacher,out):
    from deltarec.models.hstu_selector import RatingPCSelector
    from deltarec.utils.hstu_selector_training import _training_batches,selector_validation_key
    from deltarec.utils.hstu_kuai_cwi import make_pc_stack
    c=teacher.config;dest=out/'selector.pt'
    if dest.exists():return torch.load(dest,map_location='cpu',weights_only=True)
    _,table=_kuai_video_embedding_state(teacher.model);weight=table.local_shards()[0].tensor
    embedding=torch.nn.Embedding.from_pretrained(weight.detach(),freeze=True)
    selector=RatingPCSelector(embedding,expected_num_items=embedding.num_embeddings-1,expected_embedding_dim=512,seed=1).to(teacher.device)
    optimizer=torch.optim.AdamW(selector.parameters(),lr=.001)
    resume=out/'selector_resume.pt';history=[];best=None;first=1
    if resume.exists():
        saved=torch.load(resume,map_location=teacher.device,weights_only=False)
        if saved['parent']!=teacher.identity['file_tree_sha256']:raise ValueError('Selector teacher changed')
        selector.load_state_dict(saved['selector'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        history=saved['history'];best=saved['best'];first=saved['epoch']+1
    _,validation=loaders(c,teacher.files,training=False)
    root=teacher.model.module
    try:
        for epoch in range(first,c['selector_epochs']+1):
            selector.train();total=count=0
            for histories,lengths,candidates,cwi in _training_batches(out/'cwi_labels/manifest.json',microbatch_size=1):
                histories,lengths,candidates,cwi=[x.to(teacher.device) for x in (histories,lengths,candidates,cwi)]
                optimizer.zero_grad(set_to_none=True);scores=selector.score_ids(histories,candidates,lengths)
                valid=(torch.arange(scores.shape[-1],device=teacher.device)[None,None,:]<lengths[:,None,None]).expand_as(scores)
                loss=torch.nn.functional.smooth_l1_loss(scores[valid],cwi.float().asinh()[valid])
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite selector loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in selector.parameters()):raise FloatingPointError('Nonfinite selector gradients')
                optimizer.step();n=int(valid.sum());total+=float(loss.detach())*n;count+=n
            current={k:v.detach().cpu().clone() for k,v in selector.state_dict().items()}
            stack=make_pc_stack(teacher,current,.25);root._hstu_transducer._stu_module=stack
            metrics=evaluate(c,teacher.model,validation,out/f'selector-validation-{epoch:02d}')
            key=(*selector_validation_key('kuairand-1k',metrics),-epoch)
            if best is None or key>tuple(best['key']):best={'key':key,'state':current,'metrics':metrics}
            root._hstu_transducer._stu_module=teacher.active_stack;del stack
            history.append(dict(epoch=epoch,training_loss=total/count,metrics=metrics))
            atomic_torch_save(torch,dict(parent=teacher.identity['file_tree_sha256'],epoch=epoch,selector=selector.state_dict(),
                optimizer=optimizer.state_dict(),history=history,best=best),resume)
        atomic_torch_save(torch,best['state'],dest);return best['state']
    finally:root._hstu_transducer._stu_module=teacher.active_stack


def run(c,files,args):
    if not c['device'].startswith('cuda'):raise ValueError('DLRMv3 HSTU requires Linux/CUDA')
    import fbgemm_gpu
    if c['device']=='cuda':c['device']='cuda:0'
    out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
    with socket.socket() as sock:sock.bind(('',0));port=sock.getsockname()[1]
    utils.setup(rank=0,world_size=1,master_port=port,device=torch.device(c['device']))
    try:
        if args.evaluate:
            saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True)
            model,opt=construct(c,mode='delta_gc',payload=saved)
            _,validation=loaders(c,files,training=False)
            return evaluate(c,model,validation,out)
        teacher_path=out/'teacher.pt'
        if not teacher_path.exists():teacher_path=Path(c['teacher'])
        if teacher_path.exists():
            saved=torch.load(teacher_path,map_location='cpu',weights_only=True,mmap=True)
            model,opt=construct(c,mode='full_gdr',payload=saved);del saved
        else:
            initializer=torch.load(c['initializer'],map_location='cpu',weights_only=True,mmap=True)
            model,opt=construct(c,mode='full_gdr')
            load_portable(model,initializer['model'],embedding_only=True);del initializer
            train_stage(c,files,model,opt,out,'warmup')
            teacher_path=out/'teacher.pt';export_model(teacher_path,model)
        if args.stop_stage=='full-gdr':return
        model.eval().requires_grad_(False)
        digest=sha256_file(teacher_path)
        teacher=SimpleNamespace(model=model,active_stack=model.module._hstu_transducer._stu_module,
            device=torch.device(c['device']),train_utils=utils,metric_logger=None,files=files,config=c,
            identity=dict(file_tree_sha256=digest,manifest_content_sha256=digest))
        from deltarec.utils.hstu_kuai_cwi import generate_labels
        generate_labels(teacher,out,sample_count=c['cwi_samples'])
        if args.stop_stage=='cwi':return
        selector=fit_selector(teacher,out)
        if args.stop_stage=='selector':return
        grouping=bind_grouping(model,files['grouping'],c['group_count'])
        del teacher,opt,model;gc.collect();torch.cuda.empty_cache()
        saved=torch.load(teacher_path,map_location='cpu',weights_only=True,mmap=True)
        saved['selector']=selector;saved['grouping']=grouping
        model,opt=construct(c,mode='delta_gc',payload=saved);del saved
        train_stage(c,files,model,opt,out,'sparse')
        export_model(out/'model.pt',model,selector=selector,grouping=grouping)
    finally:utils.cleanup()

