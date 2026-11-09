"""Independent sample/schedule and guarded CPU-only tensor audits for I6."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENCV_OPENCL_RUNTIME']='disabled'
from pathlib import Path
import sys,argparse,gzip,json,math,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));os.chdir(ROOT)
from i6_common import OUT,PROTOCOL,index,entry,directory,spec,factor,read_json,sha256,atomic,verify

def main():
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['fit','checkpoint']);a.add_argument('--cell',required=True);a.add_argument('--stage',required=True);a.add_argument('--miner',type=int);args=a.parse_args()
    p=verify();e=entry(args.cell);s=spec(args.stage,args.miner);idx=index(e['fold'],args.miner);d=directory(args.cell,args.stage,args.miner);r=read_json(d/'run.json')
    assert r['status']=='complete' and not e['reused']
    assert r['protocol_sha256']==sha256(PROTOCOL) and r['source_freeze_sha256']==sha256(OUT/'training-source-freeze.json')
    for k,names in [('fit_names',idx.train_names),('inner_names',idx.inner_names),('outer_names',idx.validation_names),('predict_names',idx.predict_names)]:assert r[k]==names,k
    assert r['runtime']['host']['hostname'].lower()=='barrelagedworkflows'
    result={'passed':True,'cell':args.cell,'stage':args.stage,'miner':args.miner,'mode':args.mode,'run_sha256':sha256(d/'run.json')}
    if args.mode=='checkpoint':
        from filament.compute import require_gpu_host
        require_gpu_host(ROOT/'configs/compute.json')
        from filament.gpu.runtime import state_digest
        import torch
        assert not torch.cuda.is_initialized()
        for filename in ['weights.pt','checkpoint.pt']:
            expected=r['weights_sha256' if filename=='weights.pt' else 'checkpoint_sha256'];assert sha256(d/filename)==expected
            q=torch.load(d/filename,map_location='cpu',weights_only=True)
            assert q['step']==s['steps'] and q['protocol_sha256']==sha256(PROTOCOL)
            assert state_digest(q['model'])==r['model_digest'] and all(torch.isfinite(v).all() for v in q['model'].values())
            if filename=='checkpoint.pt':
                assert q['samples']==s['steps']*s['effective_batch'] and q['scheduler']['last_epoch']==s['steps']
                assert all(int(v['step'])==s['steps'] for v in q['optimizer']['state'].values())
            del q
        assert not torch.cuda.is_initialized();result.update(cuda_initialized=False,all_tensor_values_finite=True)
    else:
        from filament.g4_data import G4Samples
        import numpy as np
        data=G4Samples(idx,args.stage,seed=e['seed']);replay=set(np.linspace(0,r['samples']-1,32,dtype=int).tolist());count=0;seconds=0.;losses=[];tensors=[]
        with gzip.open(d/'events.jsonl.gz','rt',encoding='utf8') as stream:
            for step,line in enumerate(stream,1):
                event=json.loads(line);assert event['step']==step and len(event['samples'])==s['effective_batch']
                assert math.isfinite(event['loss']) and math.isfinite(event['gradient_norm']) and event['seconds']>0
                assert math.isclose(event['used_lr'],s['learning_rate']*factor(step-1,s),abs_tol=1e-14)
                for row in event['samples']:
                    assert row['item']==count and row['stage']==args.stage
                    n=row['file_name'];rid=row['annotator_image_id'];assert n in idx.train_names and rid in idx.images[n]['annotator_image_ids']
                    assert row['source_sha256']==idx.images[n]['sha256'] and row['fold']==idx.folds[n]
                    if row['selected_instance_id'] is not None:assert row['selected_instance_id'] in [v['id'] for v in idx.annotations[rid]]
                    if count in replay:
                        x,y,meta=data.sample(count)
                        assert {k:v for k,v in meta.items() if k!='target_sha256'}=={k:v for k,v in row.items() if k!='target_sha256'},('sample replay',count)
                        if os.name!='nt':assert meta['target_sha256']==row['target_sha256']
                        values=[y[k] for k in sorted(y)] if isinstance(y,dict) else [y]
                        tensors.append({'item':count,'input_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'target_sha256':hashlib.sha256(b''.join(v.tobytes() for v in values)).hexdigest()})
                    count+=1
                losses.append(event['loss']);seconds+=event['seconds']
        assert step==s['steps'] and count==r['samples']==s['steps']*s['effective_batch']
        assert math.isclose(seconds,r['training_seconds'],abs_tol=1e-6)
        result.update(updates=step,samples=count,replayed_sample_targets=len(replay),tensor_replay=tensors,training_seconds=seconds,production_only=True,miner_prediction_groups_excluded=args.miner is not None,first250_loss=float(np.mean(losses[:250])),last250_loss=float(np.mean(losses[-250:])),events_sha256=sha256(d/'events.jsonl.gz'))
        if os.name=='nt':assert tensors==read_json(OUT/'audits'/(d.name+'-fit.json'))['tensor_replay']
    dest=OUT/'audits'/(d.name+'-'+args.mode+('-windows' if os.name=='nt' else '')+'.json');assert not dest.exists();atomic(dest,result);print({k:v for k,v in result.items() if k!='tensor_replay'},flush=True)

if __name__=='__main__':main()
