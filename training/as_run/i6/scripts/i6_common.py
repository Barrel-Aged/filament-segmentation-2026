"""Explicit deployment-only I6 contracts; CPU-safe imports."""
from pathlib import Path
import os,sys,math,datetime,shutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from filament.common import read_json,write_json,sha256
from filament.s1_submission import SubmissionRefitIndex
from i5_common import atomic,factor
OUT=Path('outputs/improvement-i6')
PROTOCOL='configs/improvement-i6.json'
DATA='manifests/improvement-i6-data.json'
CELL='full-s20261014'
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()

def index(fold='production',miner=None):
    assert fold=='production','I6 cannot be used as a development index'
    idx=SubmissionRefitIndex(DATA);idx.predict_names=[]
    if miner is not None:
        assert miner in [0,1]
        part=read_json(DATA)['miners'][miner]
        idx.train_names=part['fit'];idx.predict_names=part['predict']
        groups={r['file_name']:r['group_id'] for r in read_json('manifests/split-v2.json')['observations']}
        assert {groups[n] for n in idx.train_names}.isdisjoint(groups[n] for n in idx.predict_names)
        assert set(idx.train_names)|set(idx.predict_names)==set(idx.images)
    assert not idx.inner_names and not idx.validation_names
    return idx

def entry(cell):
    assert cell==CELL
    return {'cell':CELL,'fold':'production','seed':20261014,'reused':False}

def directory(cell,stage,miner=None):
    entry(cell);assert stage in ['detector','refiner']
    return Path('runs')/('I6-'+stage+'-'+cell+('' if miner is None else '-m'+str(miner)))

def spec(stage,miner=None):
    s=dict(read_json('configs/growth-4.json')[stage])
    if miner is not None:s['steps']//=2
    return s

def verify():
    p=read_json(PROTOCOL)
    assert p['deployment_only'] and p['new_holdout_score'] is False
    for f,h in p['inputs'].items():assert sha256(f)==h,f
    for name in ['training-source-freeze.json','inference-source-freeze.json']:
        if (OUT/name).exists():
            for f,h in read_json(OUT/name)['files'].items():assert sha256(f)==h,f
    return p

def storage(p):
    assert shutil.disk_usage(ROOT).free>=p['resources']['free_disk_floor_gib']*2**30

def test_images():
    rows={r['file_name']:r for r in read_json('manifests/dataset-v1.json')['images'] if r['split']=='test'}
    assert sorted(rows)==read_json(DATA)['test_names'] and len(rows)==180
    return rows

def load_test(name):
    import numpy as np
    from PIL import Image
    row=test_images()[name];path=Path('data/raw')/row['relative_path']
    assert sha256(path)==row['sha256']
    with Image.open(path) as image:
        assert image.mode=='L' and image.size==(2048,2048)
        return np.asarray(image).copy()
