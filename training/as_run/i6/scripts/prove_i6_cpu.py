"""Bounded production index/sample compatibility proof, with no framework import."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENCV_OPENCL_RUNTIME']='disabled'
import hashlib,sys
from i6_common import ROOT,OUT,PROTOCOL,DATA,read_json,sha256,index,atomic,now,verify
from filament.g4_data import G4Samples

def main():
    os.chdir(ROOT);p=verify();checks=[]
    for miner in [None,0,1]:
        idx=index(miner=miner)
        names=[next(n for n in idx.train_names if idx.folds[n]==f) for f in range(5)]
        for stage in ['detector','refiner']:
            data=G4Samples(idx,stage,seed=20261014,limit_names=names);rows=[]
            for item in [0,1,7,19]:
                x,y,m=data.sample(item);values=[y[k] for k in sorted(y)] if isinstance(y,dict) else [y]
                rows.append({'item':item,'metadata':{k:v for k,v in m.items() if k!='target_sha256'},
                    'container_sha256':m['target_sha256'],'input_sha256':hashlib.sha256(x.tobytes()).hexdigest(),
                    'target_sha256':hashlib.sha256(b''.join(v.tobytes() for v in values)).hexdigest()})
            checks.append({'miner':miner,'stage':stage,'limited_names':names,'fit_count':len(idx.train_names),'predict_count':len(idx.predict_names),'samples':rows})
            del data
    # The ordinary research index still denies fold4.
    from filament.dataset import ObservationIndex
    try:ObservationIndex(validation_fold=4)
    except ValueError:pass
    else:raise AssertionError('Research holdout guard was relaxed')
    atomic(OUT/('cpu-proof-windows.json' if os.name=='nt' else 'cpu-proof-linux.json'),
        {'passed':True,'utc':now(),'protocol_sha256':sha256(PROTOCOL),'checks':checks,'local_gpu_used':False,
         'scope':'24 deterministic samples from bounded five-image banks spanning every supplied fold; full fit sampling audited after training. No training/GPU framework imported.'})
    print('I6 CPU scope and bounded sample proof passed',flush=True)

if __name__=='__main__':main()
