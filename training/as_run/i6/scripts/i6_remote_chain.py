"""Durable authorized continuation; waits for training, then runs frozen inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import subprocess,time
from i6_common import ROOT,OUT,verify,read_json,atomic,now,storage
from filament.compute import require_gpu_host
os.chdir(ROOT);require_gpu_host(ROOT/'configs/compute.json');p=verify();path=OUT/'chain.json';assert not path.exists()
state={'status':'waiting_for_training','started_utc':now(),'owned_pid':None};atomic(path,state)
try:
    while True:
        s=read_json(OUT/'training-execution.json')
        if s['status']=='complete':break
        assert s['status']=='running',s;storage(p);time.sleep(30)
    verify()
    with (OUT/'logs/chain-inference.log').open('x',encoding='utf8') as stream:
        child=subprocess.Popen(['.venv/bin/python','scripts/i6_inference_campaign.py'],stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        state.update(status='running',owned_pid=child.pid);atomic(path,state);code=child.wait()
    state['owned_pid']=None;assert code==0,code
    assert read_json(OUT/'inference-execution.json')['status']=='complete'
    state.update(status='awaiting_retrieval_and_actual_visual_review',ended_utc=now());atomic(path,state)
except BaseException as error:state.update(status='failed',error=repr(error),ended_utc=now());atomic(path,state);raise
