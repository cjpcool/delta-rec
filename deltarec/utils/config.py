"""Load explicit JSON settings and content-bound local data paths."""
import json
from pathlib import Path
from .io import sha256_file, protocol_hash

def load_training_config(path,data_root,output,device):
    config=json.loads(Path(path).read_text())
    files={key:Path(data_root)/relative for key,relative in config.pop('data_files').items()}
    missing=[key for key,file in files.items() if not file.is_file()]
    if missing:raise FileNotFoundError('Required prepared assets are absent: '+', '.join(missing))
    hashes={key:sha256_file(file) for key,file in files.items()}
    expected=config.get('data_sha256',{})
    for key,value in expected.items():
        if hashes.get(key)!=value:raise ValueError('Prepared data content mismatch: '+key)
    config['binding_sha256']=protocol_hash(hashes)
    config['config_sha256']=protocol_hash(config)
    config['code']={}
    config['output']=str(output or Path('outputs')/Path(path).stem)
    config['device']=device
    return config,files


def load_trajectory_config(args, backend):
    from types import SimpleNamespace
    config,files=load_training_config(args.config,args.data_root,args.output,args.device)
    model_settings=config.pop('model')
    if backend=='fuxilinear':
        from deltarec.adaptors.fuxi_model import FuxiModelConfig
        model=FuxiModelConfig(**model_settings)
    else:
        from deltarec.adaptors.recbole import RecBoleModelConfig
        model=RecBoleModelConfig(**model_settings)
    config['grouping']=str(files['grouping'])
    config['grouping_sha256']=sha256_file(files['grouping'])
    config['model_fingerprint']=model.fingerprint()
    config['config_content_sha256']=protocol_hash(config)
    config['run_fingerprint']=config['config_content_sha256']
    return config,SimpleNamespace(model=model),files
