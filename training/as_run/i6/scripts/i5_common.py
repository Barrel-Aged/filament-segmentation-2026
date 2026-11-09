"""Shared I5 contracts. Safe to import on CPU; never imports a GPU framework."""
from pathlib import Path
import os,sys,math,datetime
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from filament.common import read_json,write_json,sha256
from filament.f2_data import F2Index
OUT=Path('outputs/improvement-i5')
PROTOCOL='configs/improvement-i5.json'
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()

def index(fold,miner=None):
    p=read_json('manifests/improvement-i5-splits.json')['partitions'][str(fold)]
    idx=F2Index({k:p[k] for k in ['fit','inner','outer']},fold)
    idx.predict_names=[]
    if miner is not None:
        idx.train_names=p['miners'][miner]['fit'];idx.predict_names=p['miners'][miner]['predict']
    lookup={r['file_name']:r['group_id'] for r in read_json(idx.split_path)['observations']}
    sets=[{lookup[n] for n in names} for names in [idx.train_names,idx.inner_names,idx.validation_names,idx.predict_names]]
    assert all(sets[i].isdisjoint(sets[j]) for i in range(4) for j in range(i+1,4))
    assert all(idx.folds[n]!=4 for names in [idx.train_names,idx.inner_names,idx.validation_names,idx.predict_names] for n in names)
    return idx

def entry(cell):return next(e for e in read_json(PROTOCOL)['entries'] if e['cell']==cell)
def key(cell,stage,miner=None):return f'I5-{stage}-{cell}'+('' if miner is None else f'-m{miner}')
def directory(cell,stage,miner=None):
    e=entry(cell)
    if e['reused']:
        return Path('runs')/(f"G4-{stage}-f{e['fold']}-s{e['seed']}" if miner is None else f"G6-miner-{stage}-f{e['fold']}-m{miner}-s{e['seed']}")
    return Path('runs')/key(cell,stage,miner)

def spec(stage,miner=None):
    s=dict(read_json('configs/growth-4.json')[stage])
    if miner is not None:s['steps']//=2
    return s

def factor(step,s):
    if s['warmup'] and step<s['warmup']:return (step+1)/s['warmup']
    f=max(0,step-s['warmup'])/max(1,s['steps']-s['warmup'])
    return .01+.99*.5*(1+math.cos(math.pi*min(f,1)))

def verify():
    p=read_json(PROTOCOL)
    for f,h in {**p['inputs'],**read_json(OUT/'training-source-freeze.json')['files']}.items():assert sha256(f)==h,f
    return p

def storage(p):
    import shutil
    assert shutil.disk_usage(ROOT).free>=p['resources']['free_disk_floor_gib']*2**30,'I5 disk reserve'

def atomic(path,value):
    import tempfile,json
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',encoding='utf8',dir=path.parent,delete=False) as f:
        json.dump(value,f,indent=2,sort_keys=True);f.write('\n');tmp=Path(f.name)
    tmp.replace(path)
