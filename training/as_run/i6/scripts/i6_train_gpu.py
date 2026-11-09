"""Serial, host-guarded reproduction of the G4/G6 training recipes for I6."""
from pathlib import Path
import sys,os,argparse,time,math,gzip,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));os.chdir(ROOT)
from filament.compute import require_gpu_host

def main():
    require_gpu_host(ROOT/'configs/compute.json')
    from i6_common import OUT,PROTOCOL,read_json,sha256,index,entry,directory,spec,factor,verify,storage
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['proof','train']);a.add_argument('--cell',required=True);a.add_argument('--stage',choices=['detector','refiner'],required=True);a.add_argument('--miner',type=int,choices=[0,1]);args=a.parse_args()
    proof=args.mode=='proof';p=read_json(PROTOCOL) if proof else verify();e=entry(args.cell);s=spec(args.stage,args.miner);idx=index(e['fold'],args.miner)
    assert not e['reused'] or proof
    from filament.gpu.runtime import configure_gpu,atomic_json,state_digest,rng_state,utc_now
    from filament.gpu.model import ResNetUNet,prepare_weights,segmentation_loss
    from filament.g4_data import G4Samples
    import numpy as np
    import torch
    from torch import nn
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    runtime=configure_gpu(dict(p['runtime'],seed=e['seed']));torch.use_deterministic_algorithms(False)
    runtime.update(training_deterministic_algorithms=False,amp='bfloat16');assert torch.cuda.is_bf16_supported();storage(p)
    weights18,info18=prepare_weights();detector_path=Path('checkpoints/pretrained/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth')
    weights={str(weights18.relative_to(ROOT)).replace('\\','/'):info18['sha256'],detector_path.as_posix():sha256(detector_path)}
    assert all(p['inputs'][f]==h for f,h in weights.items())
    data=G4Samples(idx,args.stage,seed=e['seed'])
    if args.stage=='refiner':model=ResNetUNet(2,weights18)
    else:
        model=fasterrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None,num_classes=91,min_size=1024,max_size=1024,box_score_thresh=.01,box_nms_thresh=.7,box_detections_per_img=128)
        model.load_state_dict(torch.load(detector_path,map_location='cpu',weights_only=True))
        model.roi_heads.box_predictor=FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features,2)
        model.rpn.anchor_generator=AnchorGenerator(tuple((v,) for v in s['anchors']),tuple(tuple(s['aspect_ratios']) for _ in range(5)))
    model=model.cuda();model.train()
    if args.stage=='detector':
        for module in model.modules():
            if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()
    initial=state_digest(model.state_dict())
    if e['seed']==20261014:
        old=read_json(f'runs/G4-{args.stage}-f0-s20261014/run.json');assert initial==old['initial_model_digest'],'Generic initialization changed'
    optimizer=torch.optim.AdamW(model.parameters(),lr=s['learning_rate'],weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda t:factor(t,s))
    d=OUT/'proofs'/f"{args.stage}-{args.cell}" if proof else directory(args.cell,args.stage,args.miner)
    assert not d.exists(),'Never overwrite a fit or silently restart';d.mkdir(parents=True)
    run={'status':'running','stage':args.stage,'cell':args.cell,'fold':e['fold'],'seed':e['seed'],'miner':args.miner,'runtime':runtime,'protocol_sha256':sha256(PROTOCOL),'weights':weights,'initial_model_digest':initial,'fit_names':idx.train_names,'inner_names':idx.inner_names,'outer_names':idx.validation_names,'predict_names':idx.predict_names,'started_utc':utc_now(),'completed_steps':0,'training_seconds':0.,'engineering_only':proof}
    if not proof:run['source_freeze_sha256']=sha256(OUT/'training-source-freeze.json')
    atomic_json(d/'run.json',run)
    steps=12 if proof else s['steps'];durations=[];losses=[];start=time.monotonic()
    try:
        with gzip.open(d/'events.jsonl.gz','wt',encoding='utf8',compresslevel=3) as log:
            for step in range(1,steps+1):
                assert time.monotonic()-start<p['resources']['fit_minutes']*60
                tick=time.monotonic();optimizer.zero_grad(set_to_none=True);value=0.;metadata=[];pieces={};lr=optimizer.param_groups[0]['lr'];batch=s['effective_batch'];micro=s['micro_batch']
                for offset in range(0,batch,micro):
                    samples=[data.sample((step-1)*batch+offset+j) for j in range(micro)];metadata.extend(v[2] for v in samples)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        if args.stage=='detector':
                            x=[torch.from_numpy(v[0]).cuda() for v in samples];targets=[{k:torch.from_numpy(vv).cuda() for k,vv in v[1].items()} for v in samples];components=model(x,targets);loss=sum(components.values());pieces.update({k:pieces.get(k,0.)+float(v.detach())*micro/batch for k,v in components.items()})
                        else:
                            x=torch.from_numpy(np.stack([v[0] for v in samples])).cuda();target=torch.from_numpy(np.stack([v[1] for v in samples])).cuda();loss=segmentation_loss(model(x),target)
                    assert torch.isfinite(loss).item();(loss*micro/batch).backward();value+=float(loss.detach())*micro/batch
                norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));assert math.isfinite(norm);optimizer.step();scheduler.step();torch.cuda.synchronize();seconds=time.monotonic()-tick;durations.append(seconds);losses.append(value)
                log.write(json.dumps({'step':step,'loss':value,'components':pieces,'used_lr':lr,'gradient_norm':norm,'seconds':seconds,'samples':metadata},separators=(',',':'))+'\n')
                if step%250==0 or step==steps:
                    log.flush();run.update(completed_steps=step,training_seconds=sum(durations));atomic_json(d/'run.json',run);storage(p);print(json.dumps({'cell':args.cell,'stage':args.stage,'miner':args.miner,'step':step,'loss':value,'seconds':sum(durations)}),flush=True)
                if not proof and (step%1000==0 or step==steps):
                    tmp=d/'checkpoint.partial';torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':rng_state(),'step':step,'samples':step*batch,'protocol_sha256':sha256(PROTOCOL)},tmp);tmp.replace(d/'checkpoint.pt')
        if not proof:torch.save({'model':model.state_dict(),'step':steps,'protocol_sha256':sha256(PROTOCOL)},d/'weights.pt')
        run.update(status='complete',completed_steps=steps,samples=steps*batch,training_seconds=sum(durations),elapsed_seconds=time.monotonic()-start,peak_mib=torch.cuda.max_memory_allocated()/2**20,model_digest=state_digest(model.state_dict()),ended_utc=utc_now(),first_loss=losses[0],last_loss=losses[-1])
        if not proof:run.update(weights_sha256=sha256(d/'weights.pt'),checkpoint_sha256=sha256(d/'checkpoint.pt'))
        atomic_json(d/'run.json',run)
    except BaseException as error:
        run.update(status='failed',error=repr(error));atomic_json(d/'run.json',run);raise

if __name__=='__main__':main()
