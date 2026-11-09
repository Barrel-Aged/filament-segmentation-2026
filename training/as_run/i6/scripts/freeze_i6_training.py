"""Freeze implementation before any I6 quality fit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from i6_common import ROOT,OUT,PROTOCOL,DATA,verify,atomic,now,sha256
os.chdir(ROOT);p=verify();assert not (OUT/'training-source-freeze.json').exists()
assert not list(Path('runs').glob('I6-*'))
files={f.as_posix() for f in Path('src').rglob('*.py')}
files.update(['scripts/'+f for f in ['i6_common.py','i5_common.py','i6_train_gpu.py','audit_i6_fit.py','prove_i6_cpu.py','prepare_i6.py','i6_training_campaign.py','freeze_i6_training.py']])
files.update([PROTOCOL,DATA,'docs/improvement-i6.md','configs/compute.json','configs/growth-4.json','training/uv.lock','uv.lock'])
atomic(OUT/'training-source-freeze.json',{'utc':now(),'protocol_sha256':sha256(PROTOCOL),'all_quality_fits_not_started':True,'files':{f:sha256(f) for f in sorted(files)}})
print('I6 training implementation frozen',len(files))
