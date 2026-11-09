"""Owned frozen I6 production inference after all six fits and their audits."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,subprocess,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));os.chdir(ROOT)
from filament.compute import require_gpu_host
require_gpu_host(ROOT/'configs/compute.json')
from i6_common import OUT,PROTOCOL,read_json,verify,storage,atomic,now,sha256
p=verify();training=read_json(OUT/'training-execution.json');assert training['status']=='complete' and len(training['completed'])==18
path=OUT/'inference-execution.json';assert not path.exists()
state={'status':'running','started_utc':now(),'completed':[],'gpu_stage_seconds':training['gpu_stage_seconds'],'cpu_stage_seconds':training['cpu_stage_seconds'],
       'owned_pid':None,'protocol_sha256':sha256(PROTOCOL),'peak_child_rss_bytes':0};begin=time.monotonic()
def save():state['elapsed_seconds']=time.monotonic()-begin;atomic(path,state)
def invoke(script,args,label,gpu=False):
    state['active']=label;save();start=time.monotonic();field='gpu_stage_seconds' if gpu else 'cpu_stage_seconds';cap=p['resources']['gpu_stage_hours' if gpu else 'cpu_stage_hours']*3600
    with (OUT/'logs'/f'{label}.log').open('x',encoding='utf8') as stream:
        child=subprocess.Popen(['training/.venv/bin/python' if gpu else '.venv/bin/python',script,*args],stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0' if gpu else ''});state['owned_pid']=child.pid;save()
        try:
            while child.poll() is None:
                storage(p);assert state[field]+time.monotonic()-start<cap
                stat=Path(f'/proc/{child.pid}/status')
                if stat.exists():
                    rss=next((int(l.split()[1])*1024 for l in stat.read_text(encoding='utf8').splitlines() if l.startswith('VmRSS:')),0)
                    state['peak_child_rss_bytes']=max(state['peak_child_rss_bytes'],rss);assert rss<p['resources']['host_ram_gib']*2**30
                time.sleep(5)
        except BaseException:
            child.terminate()
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            state[field]+=time.monotonic()-start;state['owned_pid']=None;save();raise
    state[field]+=time.monotonic()-start;state['owned_pid']=None;assert child.returncode==0,(label,child.returncode)
    state['completed'].append({'label':label,'gpu':gpu,'seconds':time.monotonic()-start});save()
save()
try:
    invoke('scripts/i6_candidates_gpu.py',['proof'],'historical-parity',True)
    for miner in [0,1]:invoke('scripts/i6_candidates_gpu.py',['oof','--miner',str(miner)],'oof-m'+str(miner),True)
    for mode in ['fit','audit','freeze']:invoke('scripts/i6_selector.py',[mode],'selector-'+mode)
    invoke('scripts/i6_candidates_gpu.py',['test'],'test-inference',True)
    invoke('scripts/i6_candidates_gpu.py',['test','--repeat'],'test-exact-repeat',True)
    invoke('scripts/i6_export.py',['export'],'test-export-audit')
    invoke('scripts/review_i6_export.py',[],'export-panels')
    size=sum(f.stat().st_size for d in [OUT,*Path('runs').glob('I6-*')] for f in d.rglob('*') if f.is_file());assert size<p['resources']['new_remote_gib']*2**30
    state.update(status='complete',active=None,ended_utc=now());save()
except BaseException as error:state.update(status='failed',error=repr(error),ended_utc=now());save();raise
