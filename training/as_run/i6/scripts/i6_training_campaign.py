"""One-shot owned production fits and audits, bounded by the frozen I6 budget."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,subprocess,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));os.chdir(ROOT)
from filament.compute import require_gpu_host
require_gpu_host(ROOT/'configs/compute.json')
from i6_common import OUT,PROTOCOL,CELL,verify,read_json,sha256,atomic,now,storage,directory
p=verify();assert read_json(OUT/'preflight-linux.json')['passed'];assert read_json(OUT/'cross-host-proof.json')['passed']
path=OUT/'training-execution.json';assert not path.exists()
state={'status':'running','started_utc':now(),'completed':[],'gpu_stage_seconds':0.,'cpu_stage_seconds':0.,'owned_pid':None,'protocol_sha256':sha256(PROTOCOL),'peak_child_rss_bytes':0}
begin=time.monotonic();(OUT/'logs').mkdir(exist_ok=True)
def save():state['elapsed_seconds']=time.monotonic()-begin;atomic(path,state)
def invoke(script,args,label,gpu=False,tensor=False):
    state['active']=label;save();tick=time.monotonic();field='gpu_stage_seconds' if gpu else 'cpu_stage_seconds';cap=p['resources']['gpu_stage_hours' if gpu else 'cpu_stage_hours']*3600
    with (OUT/'logs'/(label+'.log')).open('x',encoding='utf8') as stream:
        child=subprocess.Popen(['training/.venv/bin/python' if gpu or tensor else '.venv/bin/python',script,*args],stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0' if gpu else ''});state['owned_pid']=child.pid;save()
        try:
            while child.poll() is None:
                storage(p);assert state[field]+time.monotonic()-tick<cap
                stat=Path(f'/proc/{child.pid}/status')
                if stat.exists():
                    rss=next((int(l.split()[1])*1024 for l in stat.read_text().splitlines() if l.startswith('VmRSS:')),0)
                    state['peak_child_rss_bytes']=max(state['peak_child_rss_bytes'],rss);assert rss<p['resources']['host_ram_gib']*2**30
                time.sleep(5)
        except BaseException:
            child.terminate()
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            state[field]+=time.monotonic()-tick;state['owned_pid']=None;save();raise
    seconds=time.monotonic()-tick;state[field]+=seconds;state['owned_pid']=None;assert child.returncode==0,(label,child.returncode)
    state['completed'].append({'label':label,'seconds':seconds,'gpu':gpu});save()
save()
try:
    for miner in [None,0,1]:
        for stage in ['detector','refiner']:
            opts=['--cell',CELL,'--stage',stage]+([] if miner is None else ['--miner',str(miner)]);label=directory(CELL,stage,miner).name
            invoke('scripts/i6_train_gpu.py',['train',*opts],label,True)
            invoke('scripts/audit_i6_fit.py',['checkpoint',*opts],label+'-checkpoint',tensor=True)
            invoke('scripts/audit_i6_fit.py',['fit',*opts],label+'-fit')
        total=sum(f.stat().st_size for d in [OUT,*Path('runs').glob('I6-*')] for f in d.rglob('*') if f.is_file());assert total<p['resources']['new_remote_gib']*2**30
    state.update(status='complete',active=None,ended_utc=now());save()
except BaseException as error:
    state.update(status='failed',error=repr(error),ended_utc=now());save();raise
