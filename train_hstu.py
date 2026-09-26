"""DeltaRec HSTU: rating and KuaiRand training/validation entry."""
import argparse,json,random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from deltarec.adaptors.hstu_model import HSTUModelConfig,build_model
from deltarec.models.hstu_selector import RatingPCSelector
from deltarec.models.hstu_grouped import GlobalUtilityScoring
from deltarec.models.hstu_runtime import OfficialResearchSparseScorer
from deltarec.models.hstu_amazon import AmazonScorer
from deltarec.models.hstu_selector import RatingGCSelector,VerifiedBoundGrouping
from deltarec.utils.config import load_training_config
from deltarec.utils.io import atomic_write_json,sha256_file
from deltarec.metrics.evaluation import AtomicJsonlWriter,evaluate_rating_stream
from deltarec.utils.protocols import AdapterOutput


def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(value)


def construct(c,files,*,install_gdr=True):
    from deltarec.data.training import _max_catalog_id
    seed(c['seed']);config=HSTUModelConfig(**c['model'])
    model=build_model(config,max_item_id=_max_catalog_id(files['full_catalog']),seed=c['seed'],
        kernel_backend=c['kernel'],decay_timescale_range=c['decay_timescale_range'],install_gdr=install_gdr).to(c['device'])
    return SimpleNamespace(model=model,model_config=config,evidence={'binding':{'binding_file_sha256':c['config_sha256']}})


def load_state(path):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    state=saved.get('model',saved.get('model_state_dict',saved.get('state_dict')))
    if state is None:raise ValueError('Checkpoint does not contain model tensors')
    return {k.removeprefix('module.'):v for k,v in state.items()}


def selector_for(loaded,saved,*,frozen):
    table=loaded.model._embedding_module._item_emb
    selector=RatingPCSelector(table,expected_num_items=table.num_embeddings-1,
        expected_embedding_dim=table.embedding_dim,seed=0,freeze_embedding_snapshot=frozen).to(table.weight.device)
    selector.load_state_dict(saved,strict=True)
    return selector.requires_grad_(False).eval()


def scoring_for(c,loaded,selector,grouping):
    if c['dataset']=='ml-20m':
        proxy=SimpleNamespace(model=loaded.model,model_config=loaded.model_config,method='full-gdr',retention_ratio=1.)
        return GlobalUtilityScoring(proxy,selector,grouping['mapping'],c['group_count'])
    evidence={'binding_sha256':'bound-configuration'}
    record=VerifiedBoundGrouping(c['dataset'],c['group_count'],grouping['mapping'],grouping['prototypes'],evidence)
    gc=RatingGCSelector(selector,record).to(c['device'])
    return AmazonScorer(loaded,method='deltarec-gc',retention_ratio=.25,selector=gc,
        selector_binding_sha256='bound-selector',grouping_binding_sha256=evidence['binding_sha256'])


class EvaluationAdapter:
    def __init__(self,scoring):self.scoring=scoring
    def run(self,batch,mode):
        device=next(self.scoring.model.parameters()).device
        lengths=batch.history_lengths.to(device)
        histories=batch.history_item_ids[:,:int(lengths.max())].to(device)
        candidates=batch.candidate_item_ids.to(device)
        with torch.inference_mode():
            if isinstance(self.scoring,GlobalUtilityScoring):scores=self.scoring.scores(histories,lengths,candidates)
            else:
                cache=self.scoring.build_state_cache(history_item_ids=histories,history_lengths=lengths,candidate_item_ids=candidates)
                scores=self.scoring.serve_cache_hit(candidate_item_ids=candidates,cache=cache)
        return AdapterOutput(scores)


def evaluate(c,files,scoring,output):
    output.mkdir(parents=True,exist_ok=True);scoring.model.eval()
    evidence=output/'per_request.jsonl'
    if evidence.exists():evidence.unlink()
    with AtomicJsonlWriter(evidence) as writer:
        metrics,details,_=evaluate_rating_stream(torch=torch,adapter=EvaluationAdapter(scoring),
            candidate_manifest=files['validation_candidates'],dataset=c['dataset'],split_role='validation',
            protocol_lock_hash=None,microbatch_size=c['eval_microbatch'],evidence=writer)
    atomic_write_json(output/'metrics.json',dict(metrics=metrics,details=details))
    print(json.dumps(metrics),flush=True);return metrics


def export_model(path,c,loaded,selector,grouping):
    from deltarec.utils.checkpoint import atomic_torch_save
    # Every required frozen tensor is inline; no parent checkpoint is consulted.
    atomic_torch_save(torch,dict(schema='deltarec-release-v1',model={k:v.cpu() for k,v in loaded.model.state_dict().items()},
        selector={k:v.cpu() for k,v in selector.state_dict().items()},grouping={k:v.cpu() for k,v in grouping.items()}),path)


def run_amazon(c,files,args,loaded):
    from deltarec.utils import hstu_amazon_training as stages
    from deltarec.layers.hstu_gdr import install_research_hstu_gdr
    from deltarec.data.hstu_amazon import prepare
    out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
    cfg=dict(c,data_files={k:str(v) for k,v in files.items()});atomic_write_json(out/'experiment.json',cfg)
    if not (out/'data/data.json').exists():prepare(files['train'],out/'data',count=c['cwi_count'])
    full=out/'full-gdr/best.pt'
    provided_teacher=Path(c['teacher_checkpoint']) if c.get('teacher_checkpoint') else None
    if args.start_stage=='full-gdr' and provided_teacher is not None and provided_teacher.is_file() and not full.exists():
        from deltarec.utils.checkpoint import atomic_torch_save
        atomic_torch_save(torch,{'model':load_state(provided_teacher)},full)
    if args.start_stage=='full-gdr' and not (provided_teacher is not None and provided_teacher.is_file()):
        if not full.exists():
            initializer=Path(c['initializer']);raw=construct(c,files,install_gdr=False)
            raw.model.load_state_dict(load_state(initializer),strict=True)
            install_research_hstu_gdr(raw.model,seed=c['seed'],kernel_backend=c['kernel'],decay_timescale_range=c['decay_timescale_range'])
            raw.model.to(c['device'])
            loaded=raw
        stages.train_stage(loaded,out,'full-gdr')
    if args.stop_stage=='full-gdr':return
    loaded.model.load_state_dict(load_state(full),strict=True)
    # Selector validation uses G=2; the published sparse run fixed G=16 after its
    # original training/validation search. No new hyperparameter search runs here.
    def bind_group(g):
        raw=torch.load(Path(args.data_root)/c['grouping_assets'][str(g)],map_location='cpu',weights_only=True)
        mapping=raw['mapping'];ids=torch.tensor(stages.load_item_ids(files['train_catalog']))
        table=load_state(full)['_embedding_module._item_emb.weight'].float()
        sums=torch.zeros(g,table.shape[1]);groups=mapping[ids]
        sums.index_add_(0,groups,table[ids]);counts=torch.bincount(groups,minlength=g)
        if bool((counts==0).any()):raise ValueError('empty training category group')
        grouping=dict(mapping=mapping,prototypes=sums/counts[:,None])
        torch.save(grouping,out/'grouping.pt');return grouping
    grouping=bind_group(c['selector_validation_groups'])
    if args.start_stage in ('full-gdr','cwi'):stages.cwi_stage(loaded,out)
    if args.stop_stage=='cwi':return
    selector=stages.selector_stage(loaded,out)
    if args.stop_stage=='selector':return
    grouping=bind_group(c['group_count'])
    if c.get('sparse_execution') and not (out/'resume_execution.json').exists():
        atomic_write_json(out/'resume_execution.json',dict(base_experiment_sha256=sha256_file(out/'experiment.json'),parameters=c['sparse_execution']))
    stages.train_stage(loaded,out,'ranker',selector=selector)
    loaded.model.load_state_dict(load_state(out/'ranker/best.pt'),strict=True)
    export_model(out/'model.pt',c,loaded,selector,grouping)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--data-root',type=Path,default=Path('data'))
    p.add_argument('--output',type=Path);p.add_argument('--device',default='cuda')
    p.add_argument('--evaluate',action='store_true');p.add_argument('--checkpoint',type=Path)
    stages=('full-gdr','cwi','selector','sparse')
    p.add_argument('--start-stage',choices=stages,default='full-gdr');p.add_argument('--stop-stage',choices=stages,default='sparse')
    args=p.parse_args(argv)
    c,files=load_training_config(args.config,args.data_root,args.output,args.device)
    if args.evaluate and args.checkpoint is None:p.error('--evaluate requires --checkpoint')
    if stages.index(args.start_stage)>stages.index(args.stop_stage):p.error('invalid stage order')
    if c['dataset']=='kuairand-1k':
        from deltarec.utils.hstu_kuai_training import run
        return run(c,files,args)
    if c['device'].startswith('cuda'):__import__('fbgemm_gpu')
    torch.backends.cuda.matmul.allow_tf32=c['enable_tf32'];torch.backends.cudnn.allow_tf32=c['enable_tf32']
    if c['dataset']=='amazon-books' and c['kernel']=='fla':
        from deltarec.utils.hstu_precision import ieee_gdr_solve
        ieee_gdr_solve()
    loaded=construct(c,files)
    if args.evaluate:
        saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        loaded.model.load_state_dict(saved['model'],strict=True)
        selector=selector_for(loaded,saved['selector'],frozen=c['dataset']=='ml-20m')
        scoring=scoring_for(c,loaded,selector,saved['grouping'])
        return evaluate(c,files,scoring,Path(c['output']))
    if c['dataset']=='amazon-books':return run_amazon(c,files,args,loaded)
    from deltarec.utils.hstu_ml_training import run
    return run(c,files,args,loaded)

if __name__=='__main__':main()
