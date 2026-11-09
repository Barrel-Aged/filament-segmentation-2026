"""Hash-verified inference delivery and one-shot authorized continuation launch."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import subprocess,tarfile,argparse,shlex
from i6_common import ROOT,OUT,verify,read_json,sha256,atomic,now
from i6_ops import HOST,REMOTE,OPT,py
os.chdir(ROOT);a=argparse.ArgumentParser();a.add_argument('mode',choices=['deliver','launch']);args=a.parse_args();verify()
if args.mode=='deliver':
    files={f for f in read_json(OUT/'inference-source-freeze.json')['files'] if Path(f).stat().st_size<100*2**20}
    files.update((OUT/n).as_posix() for n in ['inference-source-freeze.json','inference-adaptation-audit.json'])
    archive=OUT/'inference-payload.tar';assert not archive.exists()
    with tarfile.open(archive,'w') as tar:
        for f in sorted(files):tar.add(f,arcname=f,recursive=False)
    subprocess.run(['scp','-q',*OPT,str(archive),HOST+':'+REMOTE+'/outputs/i6-inference-payload.tar'],check=True)
    code="""import hashlib,tarfile,json
from pathlib import Path
p=Path('outputs/i6-inference-payload.tar');assert hashlib.sha256(p.read_bytes()).hexdigest()==ARCHIVE_HASH
with tarfile.open(p) as t:
    ms=t.getmembers();assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in ms)
    for m in ms:
        target=Path(m.name)
        if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()==hashlib.sha256(t.extractfile(m).read()).hexdigest(),('Differing existing file',m.name)
    t.extractall('.')
import sys;sys.path.insert(0,'scripts');from i6_common import verify;verify()
print('All I6 inference source and historical weights verified')
""".replace('ARCHIVE_HASH',repr(sha256(archive)))
    result=py(code).strip();atomic(OUT/'inference-delivery.json',{'passed':True,'utc':now(),'archive_sha256':sha256(archive),'files':len(files),'linux_verification':result});print(result)
else:
    assert read_json(OUT/'inference-delivery.json')['passed'] and not (OUT/'inference-launch.json').exists()
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert subprocess.check_output(['git','ls-remote','linux-backup','refs/heads/codex/h8-phase-i'],text=True).split()[0]==commit
    atomic(OUT/'inference-git.json',{'passed':True,'utc':now(),'commit':commit,'source_freeze_sha256':sha256(OUT/'inference-source-freeze.json')})
    subprocess.run(['scp','-q',*OPT,str(OUT/'inference-git.json'),HOST+':'+REMOTE+'/'+(OUT/'inference-git.json').as_posix()],check=True)
    code="from pathlib import Path;import subprocess;p=Path('outputs/improvement-i6/chain.log');assert not p.exists();f=p.open('x');child=subprocess.Popen(['.venv/bin/python','scripts/i6_remote_chain.py'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);print(child.pid)"
    pid=int(py(code).strip());atomic(OUT/'inference-launch.json',{'owned_pid':pid,'utc':now(),'authorized_continuation':True});print('I6 continuation PID',pid)
