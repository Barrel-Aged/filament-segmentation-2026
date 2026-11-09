"""Guarded I6 OOF/test ROI generation with exact full-test replay."""
import time
PROCESS_START=time.monotonic()
from pathlib import Path
import os,sys,argparse,hashlib,json,importlib.metadata
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));os.chdir(ROOT)
from filament.compute import require_gpu_host

def main():
    require_gpu_host(ROOT/'configs/compute.json')
    from i6_common import OUT,PROTOCOL,CELL,index,directory,read_json,sha256,verify,storage,test_images,load_test,now
    a=argparse.ArgumentParser();a.add_argument('scope',choices=['proof','oof','test']);a.add_argument('--miner',type=int,choices=[0,1]);a.add_argument('--repeat',action='store_true');args=a.parse_args()
    p=verify();assert (OUT/'inference-source-freeze.json').exists();started_utc=now()
    assert not args.repeat or args.scope=='test'
    assert args.scope=='oof' or args.miner is None
    if args.scope=='proof':
        from i5_common import index as oldindex,directory as olddir
        idx=oldindex(0);names=idx.inner_names[:2];paths={s:olddir('f0-s20261014',s) for s in ['detector','refiner']};mask=.5
    else:
        idx=index(miner=args.miner if args.scope=='oof' else None)
        assert args.scope!='oof' or args.miner is not None
        names=idx.predict_names if args.scope=='oof' else sorted(test_images());paths={s:directory(CELL,s,args.miner if args.scope=='oof' else None) for s in ['detector','refiner']};mask=p['selection']['mask_threshold']
    selected=None
    if args.scope=='test':
        freeze=read_json(OUT/'selector-freeze.json');assert freeze['before_test_inference']
        assert sha256(OUT/'selector.json')==freeze['selector_sha256'] and freeze['protocol_sha256']==sha256(PROTOCOL)
        assert freeze['inference_source_freeze_sha256']==sha256(OUT/'inference-source-freeze.json')
        selected=read_json(OUT/'selector.json')['model']
    from filament.gpu.runtime import configure_gpu,atomic_json,state_digest
    from filament.gpu.model import ResNetUNet,prepare_weights
    from filament.g4_data import nms,roi_input
    from filament.g6_quality import candidates,FEATURES
    from filament.h2_sets import decode
    import numpy as np
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    runtime=configure_gpu(dict(p['runtime'],seed=20261014));torch.use_deterministic_algorithms(False);storage(p)
    weights18,_=prepare_weights();lineage={};models={};tick=time.monotonic()
    for stage,d in paths.items():
        run=read_json(d/'run.json');assert run['status']=='complete' and sha256(d/'weights.pt')==run['weights_sha256']
        assert run['fit_names']==idx.train_names
        if args.scope!='proof':
            assert run['protocol_sha256']==sha256(PROTOCOL)
            for kind in ['fit','checkpoint']:assert read_json(OUT/'audits'/(d.name+'-'+kind+'.json'))['passed']
        if stage=='refiner':model=ResNetUNet(2,weights18)
        else:
            model=fasterrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None,num_classes=91,min_size=1024,max_size=1024,box_score_thresh=.01,box_nms_thresh=.7,box_detections_per_img=128)
            model.roi_heads.box_predictor=FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features,2)
            s=read_json('configs/growth-4.json')['detector'];model.rpn.anchor_generator=AnchorGenerator(tuple((v,) for v in s['anchors']),tuple(tuple(s['aspect_ratios']) for _ in range(5)))
        payload=torch.load(d/'weights.pt',map_location='cpu',weights_only=True);assert payload['protocol_sha256']==run['protocol_sha256'];model.load_state_dict(payload['model']);del payload
        assert state_digest(model.state_dict())==run['model_digest'];models[stage]=model.cuda().eval()
        lineage[stage]={'run':d.as_posix(),'run_sha256':sha256(d/'run.json'),'weights_sha256':run['weights_sha256'],'model_digest':run['model_digest'],'fit_names':run['fit_names']}
    loads=time.monotonic()-tick
    @torch.inference_mode()
    def infer(raw):
        boxes=[];scores=[]
        for top in [0,768,1024]:
            for left in [0,768,1024]:
                tile=np.repeat((raw[top:top+1024,left:left+1024].astype(np.float32)/255)[None],3,axis=0)
                with torch.autocast('cuda',dtype=torch.bfloat16):r=models['detector']([torch.from_numpy(tile).cuda()])[0]
                b=r['boxes'].float().cpu().numpy();s=r['scores'].float().cpu().numpy();b+=np.array([left,top,left,top],np.float32);boxes.extend(b.tolist());scores.extend(s.tolist())
        boxes=np.asarray(boxes,np.float32).reshape(-1,4);scores=np.asarray(scores,np.float32);keep=nms(boxes,scores,.7,128);boxes,scores=boxes[keep],scores[keep];probs=[]
        for start in range(0,len(boxes),2):
            batch=np.stack([roi_input(raw,b)[0] for b in boxes[start:start+2]])
            with torch.autocast('cuda',dtype=torch.bfloat16):prob=models['refiner'](torch.from_numpy(batch).cuda()).float().sigmoid().cpu().numpy()[:,0]
            probs.extend(prob)
        return boxes,scores,np.asarray(probs,np.float32).reshape(-1,512,512)
    key='oof-m'+str(args.miner) if args.scope=='oof' else args.scope;dest=OUT/'candidates'/key
    if args.repeat:
        original=read_json(dest/'manifest.json');assert original['names']==names and not (dest/'repeat.json').exists()
    else:assert not dest.exists();dest.mkdir(parents=True)
    old={r['file_name']:r for r in read_json('outputs/growth-6/features/inner-f0/manifest.json')['records']} if args.scope=='proof' else {}
    records=[];start=time.monotonic();cold={}
    for i,name in enumerate(names):
        tick=time.monotonic();raw=load_test(name) if args.scope=='test' else idx.load_image(name)
        source=test_images()[name] if args.scope=='test' else idx.images[name]
        boxes,scores,prob=infer(raw);digest=hashlib.sha256(boxes.tobytes()+scores.tobytes()+prob.tobytes()).hexdigest()
        views={str(mask):candidates(raw,boxes,scores,prob,mask)}
        assert all(np.isfinite(r['features']).all() and len(r['features'])==len(FEATURES) for r in views[str(mask)])
        base=time.monotonic()-tick;predictions={};timing={}
        if selected is not None:
            for arm in ['confidence_pq','set_selector_pq']:
                dt=time.monotonic();predictions[arm]=decode(views[str(mask)],p['selection']['confidence_threshold'],selected if arm=='set_selector_pq' else None,6,.02)[0]
                json.dumps(predictions[arm],separators=(',',':'));timing[arm]=base+time.monotonic()-dt
                if i==0:cold[arm]=time.monotonic()-PROCESS_START
        record={'file_name':name,'source_sha256':source['sha256'],'input_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),'raw_predictions_sha256':digest,'views':views,'predictions':predictions}
        file=dest/f'{i:04d}.json'
        if args.repeat:assert record==read_json(file),name
        else:atomic_json(file,record)
        if old:assert digest==old[name]['raw_predictions_sha256'],('historical parity',name)
        records.append({'file_name':name,'file':file.name,'sha256':sha256(file),'input_sha256':record['input_sha256'],'raw_predictions_sha256':digest,'base_seconds':base,'complete_seconds':timing})
        if i%20==0:storage(p);print(args.scope,args.miner,'repeat' if args.repeat else '',i+1,'/',len(names),flush=True)
    result={'passed':True,'scope':args.scope,'miner':args.miner,'names':names,'records':records,'lineage':lineage,'runtime':runtime,
        'started_utc':started_utc,'ended_utc':now(),'pid':os.getpid(),
        'mask_threshold':mask,'model_load_seconds':loads,'cold_first_image_seconds':cold,'seconds':time.monotonic()-start,
        'process_seconds':time.monotonic()-PROCESS_START,'peak_mib':torch.cuda.max_memory_allocated()/2**20,
        'protocol_sha256':sha256(PROTOCOL),'source_sha256':sha256(__file__),'inference_source_freeze_sha256':sha256(OUT/'inference-source-freeze.json'),
        'selector_sha256':sha256(OUT/'selector.json') if selected is not None else None,'all_image_exact_repeat':args.repeat,
        'scored_test_labels':False,'timing_note':'Same timing convention as I5: per-image mean includes first image; selector cold includes preceding confidence decode. Script-entry startup excludes OS process creation; caches not flushed.',
        'versions':{k:importlib.metadata.version(k) for k in ['numpy','opencv-python-headless','pillow','pycocotools']}}
    atomic_json(dest/('repeat.json' if args.repeat else 'manifest.json'),result)

if __name__=='__main__':main()
