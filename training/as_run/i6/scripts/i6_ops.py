"""CPU transport, preflight and owned launch; credentials never enter payloads."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENCV_OPENCL_RUNTIME']='disabled'
from pathlib import Path
import sys,subprocess,tarfile,shlex,argparse,json,shutil,importlib.metadata
from i6_common import ROOT,OUT,PROTOCOL,verify,read_json,sha256,atomic,now
HOST='barrelaged@barrelagedworkflows-1';REMOTE='/home/barrelaged/projects/filament-segmentation-2026'
OPT=['-o','BatchMode=yes','-o','StrictHostKeyChecking=yes']
def remote(code):return subprocess.check_output(['ssh',*OPT,HOST,'cd '+shlex.quote(REMOTE)+' && '+code],text=True)
def py(code):return remote(shlex.join(['env','CUDA_VISIBLE_DEVICES=','.venv/bin/python','-c',code]))
def copy(name):subprocess.run(['scp','-q',*OPT,HOST+':'+REMOTE+'/'+(OUT/name).as_posix(),str(OUT/name)],check=True)

def main():
    os.chdir(ROOT);a=argparse.ArgumentParser();a.add_argument('mode',choices=['deliver','preflight','launch','status']);args=a.parse_args();p=verify()
    if args.mode=='deliver':
        files=set(read_json(OUT/'training-source-freeze.json')['files'])
        files.update(['scripts/i6_ops.py',(OUT/'training-source-freeze.json').as_posix(),(OUT/'cpu-proof-windows.json').as_posix()])
        files.update(f for f in p['inputs'] if f.startswith('outputs/improvement-i6/'))
        archive=OUT/'training-payload.tar';assert not archive.exists()
        with tarfile.open(archive,'w') as t:
            for f in sorted(files):t.add(f,arcname=f,recursive=False)
        subprocess.run(['scp','-q',*OPT,str(archive),HOST+':'+REMOTE+'/outputs/i6-training-payload.tar'],check=True)
        code="import hashlib,tarfile;from pathlib import Path;p=Path('outputs/i6-training-payload.tar');assert hashlib.sha256(p.read_bytes()).hexdigest()=="+repr(sha256(archive))+";t=tarfile.open(p);assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in t.getmembers());t.extractall('.');print('Payload hash verified')"
        print(py(code));print(remote(shlex.join(['env','CUDA_VISIBLE_DEVICES=','.venv/bin/python','scripts/i6_ops.py','preflight'])))
        for n in ['preflight-linux.json','cpu-proof-linux.json','cross-host-proof.json','tests-linux.json']:copy(n)
        atomic(OUT/'training-delivery.json',{'passed':True,'utc':now(),'archive_sha256':sha256(archive),'files':{f:sha256(f) for f in sorted(files)}})
    elif args.mode=='preflight':
        from filament.compute import require_gpu_host
        identity=require_gpu_host(ROOT/'configs/compute.json');assert not list(Path('runs').glob('I6-*'))
        assert shutil.disk_usage(ROOT).free>(p['resources']['free_disk_floor_gib']+p['resources']['new_remote_gib'])*2**30
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip();free,util=map(int,gpu.split(','));assert free>=7000 and util<=20
        subprocess.run([sys.executable,'scripts/prove_i6_cpu.py'],check=True)
        left=read_json(OUT/'cpu-proof-windows.json');right=read_json(OUT/'cpu-proof-linux.json')
        def canonical(x):return [{**{k:v for k,v in c.items() if k!='samples'},'samples':[{k:v for k,v in r.items() if k!='container_sha256'} for r in c['samples']]} for c in x['checks']]
        assert canonical(left)==canonical(right)
        atomic(OUT/'cross-host-proof.json',{'passed':True,'utc':now(),'decoded_tensors_exact':True,'windows_sha256':sha256(OUT/'cpu-proof-windows.json'),'linux_sha256':sha256(OUT/'cpu-proof-linux.json'),'containers_retained_separately':True})
        result=subprocess.run([sys.executable,'-m','pytest','tests/test_submission_s1.py','tests/test_h2_sets.py','tests/test_g4.py','tests/test_f2_data.py','-q'],capture_output=True,text=True)
        atomic(OUT/'tests-linux.json',{'passed':result.returncode==0,'stdout':result.stdout,'stderr':result.stderr});assert result.returncode==0,result.stdout+result.stderr
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
        atomic(OUT/'preflight-linux.json',{'passed':True,'utc':now(),'host':identity,'gpu':gpu,'processes':processes,'free_bytes':shutil.disk_usage(ROOT).free,'source_freeze_sha256':sha256(OUT/'training-source-freeze.json'),'packages':{x:importlib.metadata.version(x) for x in ['numpy','opencv-python-headless','pillow','pycocotools']}})
        print('I6 guarded preflight, cross-host tensors and Linux tests passed')
    elif args.mode=='launch':
        assert os.name=='nt' and read_json(OUT/'training-delivery.json')['passed'] and read_json(OUT/'cross-host-proof.json')['passed']
        assert read_json(OUT/'tests-windows.json')['passed']
        assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert subprocess.check_output(['git','ls-remote','linux-backup','refs/heads/codex/h8-phase-i'],text=True).split()[0]==commit
        assert not (OUT/'training-git.json').exists();atomic(OUT/'training-git.json',{'passed':True,'commit':commit,'utc':now(),'source_freeze_sha256':sha256(OUT/'training-source-freeze.json')})
        subprocess.run(['scp','-q',*OPT,str(OUT/'training-git.json'),HOST+':'+REMOTE+'/'+(OUT/'training-git.json').as_posix()],check=True)
        code="from pathlib import Path;import subprocess;path=Path('outputs/improvement-i6/coordinator.log');assert not path.exists();f=path.open('x');p=subprocess.Popen(['.venv/bin/python','scripts/i6_training_campaign.py'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);print(p.pid)"
        pid=int(py(code).strip());atomic(OUT/'launch.json',{'owned_pid':pid,'utc':now(),'authorized_quality_training_not_setup':True});print('Launched owned I6 training coordinator',pid)
    else:
        code="import json;from pathlib import Path;p=Path('outputs/improvement-i6/training-execution.json');s=json.loads(p.read_text());print(json.dumps({k:v for k,v in s.items() if k!='completed'}));print('completed stages',len(s['completed']));log=p.parent/'logs'/(str(s.get('active'))+'.log');print('\\n'.join(log.read_text(errors='replace').splitlines()[-1:]) if log.exists() else '')"
        print(py(code))

if __name__=='__main__':main()
