"""Small synthetic engineering check; never a Table 1 measurement."""
import argparse,csv,json,os,subprocess,sys,tempfile
from pathlib import Path
import torch
from deltarec.utils.io import sha256_file
from deltarec.metrics.evaluation import RATING_MANIFEST_SCHEMA,RATING_SHARD_SCHEMA

def fixture(root):
    data=root/'data'/'ml-20m';(data/'splits').mkdir(parents=True)
    cand=data/'candidates'/'validation';cand.mkdir(parents=True)
    with (data/'splits'/'train_validation_sequences.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['user_id','sequence_item_ids','sequence_ratings','sequence_timestamps'])
        for user in (1,5):w.writerow([user,','.join(map(str,range(1,132))),','.join(['5']*131),','.join(map(str,range(1,132)))])
    for name in ['train_catalog','full_catalog']:
        with (data/'splits'/(name+'.csv')).open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['item_id']);w.writerows([[i] for i in range(1,232)])
    (data/'splits'/'split_manifest.json').write_text('{}')
    histories=torch.arange(1,131)[None];candidates=torch.arange(131,231)[None]
    labels=candidates.eq(131)
    torch.save(dict(schema=RATING_SHARD_SCHEMA,diagnostic_only=True,user_ids=torch.tensor([1]),
        history_item_ids=histories,history_lengths=torch.tensor([130]),target_item_ids=torch.tensor([131]),
        candidate_item_ids=candidates,target_indices=torch.tensor([0]),labels=labels),cand/'rows.pt')
    manifest=dict(schema=RATING_MANIFEST_SCHEMA,dataset='ml-20m',split_role='validation',method='hstu',top_k=100,
        positive_injection=False,target_used_by_retriever=False,tie_break='score-desc-item-id-asc',history_seen_item_mask=True,
        rows=1,shards=[dict(filename='rows.pt',rows=1,sha256=sha256_file(cand/'rows.pt'))],diagnostic_only=True)
    (cand/'manifest.json').write_text(json.dumps(manifest))
    torch.save(dict(item_to_category_group=torch.arange(232)%2),data/'grouping.pt')
    return data

def run(root,backend,device):
    fixture_root=root/'data'; output=root/backend
    config=json.loads(Path('configs',backend+'_ml20m.json').read_text())
    config['data_files'].pop('item_id_map',None);config['data_sha256']={};config['trajectory_checkpoint_paths']=[]
    config.update(trajectory_count=1,trajectory_weights=[1.0],max_epochs=1,min_epochs=1,minimum_epochs=1,
        effective_batch=64,microbatch=64,accumulation=1,selector_epochs=1,cwi_batches=2,cwi_microbatch=1,
        cwi_samples=2,selector_batch=2,anchor_count=4,eval_microbatch=1)
    if backend=='hstu':
        config.update(kernel='reference',teacher_epochs=[1,1],teacher_weights=[1/3,2/3],num_negatives=8,enable_tf32=False)
        config['model'].update(max_history_length=160,item_embedding_dim=8,dropout_rate=0.)
        config['model']['architecture'].update(num_blocks=1,num_heads=2,dv=4,dqk=4,linear_dropout_rate=0.)
        config['initializer']=str(root/'hstu_initializer.pt')
        config['teacher_checkpoints']=[str(root/'hstu'/'teacher-01.pt'),str(root/'hstu'/'teacher-02.pt')]
        mapping=root/'data/ml-20m/item_id_map.csv'
        mapping.write_text('item_id,raw_item_id\n'+''.join('%d,%d\n'%(i,i) for i in range(1,232)))
    elif backend=='linrec':
        config['group_count']=4
        config['architecture'].update(hidden_size=8,inner_size=16,n_heads=2,n_layers=1,hidden_dropout_prob=0.,attn_dropout_prob=0.)
    elif backend=='blossomrec':
        config['kernel']='reference'
        config['model'].update(hidden_size=8,inner_size=16,n_heads=2,n_layers=1,hidden_dropout_prob=0.,attn_dropout_prob=0.)
        config['architecture']=config['model']
    else:
        a=config['architecture'];a.update(num_blocks=1,num_heads=2,dqk=4,dv=4,linear_dropout_rate=0.)
        a['channel_p_config']['dim']=4;a['channel_t_config']['num_heads']=2
        config['model'].update(item_embedding_dim=8,dropout_rate=0.,architecture=a,max_history_length=160)
    cp=root/(backend+'.json');cp.write_text(json.dumps(config))
    if backend=='hstu':
        if not device.startswith('cuda'):raise ValueError('Full HSTU smoke requires Linux/CUDA for its original FBGEMM warm-up loss')
        from train_hstu import construct
        from deltarec.utils.config import load_training_config
        loaded,files=load_training_config(cp,fixture_root,output,device)
        owner=construct(loaded,files)
        torch.save({'model':owner.model.state_dict()},config['initializer'])
        del owner
    base=[sys.executable,'-B','train_'+backend+'.py','--config',str(cp),'--data-root',str(fixture_root),'--output',str(output),'--device',device]
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONNOUSERSITE='1',PYTHONPATH='')
    for args in [[],[],['--evaluate','--checkpoint',str(output/'model.pt')]]:
        r=subprocess.run(base+args,text=True,encoding='utf-8',capture_output=True,env=env)
        (root/(backend+('-eval' if args else '-train')+'.log')).write_text(r.stdout+r.stderr,encoding='utf-8')
        if r.returncode:raise RuntimeError(backend+' failed:\n'+r.stdout[-1400:]+r.stderr[-4500:])
    return dict(backend=backend,complete_training_chain=True,resume=True,evaluation=True,diagnostic_only=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backend',choices=('linrec','blossomrec','fuxilinear','hstu','all'),default='all');p.add_argument('--output',type=Path,default=Path('outputs/smoke'));p.add_argument('--device',default='cpu');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='run-',dir=a.output) as name:
        root=Path(name).resolve();fixture(root)
        records=[run(root,b,a.device) for b in (['linrec','blossomrec','fuxilinear','hstu'] if a.backend=='all' else [a.backend])]
        (a.output/'summary.json').write_text(json.dumps(dict(diagnostic_only=True,torch=torch.__version__,results=records),indent=2))
    print(json.dumps(records))
if __name__=='__main__':main()
