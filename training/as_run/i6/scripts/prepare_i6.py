"""Freeze deployment rules before reading the accepted inner-only choices."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,hashlib,collections,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));os.chdir(ROOT)
from i6_common import OUT,PROTOCOL,DATA,CELL,atomic,read_json,sha256,now,index

def main():
    assert not Path(PROTOCOL).exists() and not (OUT/'deployment-rule.json').exists()
    accepted=read_json('manifests/improvement-i5.json');assert accepted['status']=='complete'
    assert 'set_selector_pq' in accepted['pq_confirmed'] and accepted['balanced_incumbent']=='H7'
    rule={'utc':now(),'seed':20261014,'model':'single full-data detector/refiner plus own group-excluded learned selector',
          'mask':'mode of eight accepted I5 inner-selected mask thresholds; ties nearest .5 then lower',
          'confidence':'mode of eight accepted I5 inner-selected set-selector confidence thresholds; ties lower',
          'schedule':'unchanged fixed successful-update schedules; no exposure scaling or checkpoint selection',
          'outer_test_or_public_selection':False,'authorization':'User requested I6 completion including submission of current best',
          'benchmark_only':'PQ-focused tradeoff; H7 remains balanced reference, further research continues'}
    atomic(OUT/'deployment-rule.json',rule)
    inputs={f:sha256(f) for f in ['manifests/dataset-v1.json','manifests/split-v2.json',
        'data/annotations/MAGFiLO_1.0_Annotations_kaggle2026_train.json','configs/compute.json','configs/growth-4.json',
        'configs/improvement-i5.json','manifests/improvement-i5.json','outputs/improvement-i5/outer-freeze.json',
        'outputs/improvement-i5/final-delivery.json','uv.lock','training/uv.lock',
        'checkpoints/pretrained/resnet18-f37072fd.pth','checkpoints/pretrained/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth',
        'runs/G4-detector-f0-s20261014/run.json','runs/G4-refiner-f0-s20261014/run.json']}
    outer=read_json('outputs/improvement-i5/outer-freeze.json');masks=[];thresholds=[];cells=[]
    for e in read_json('configs/improvement-i5.json')['entries']:
        cell=e['cell'];f=f'outputs/improvement-i5/{cell}/mask-choice.json';inputs[f]=sha256(f)
        assert inputs[f]==outer['mask_choice_sha256'][cell]
        m=read_json(f)['mask_threshold'];t=outer['choices'][cell]['set_selector_pq']['threshold']
        masks.append(m);thresholds.append(t);cells.append({'cell':cell,'mask_threshold':m,'confidence_threshold':t})
    count=collections.Counter(masks);mask=min(count,key=lambda v:(-count[v],abs(v-.5),v))
    count=collections.Counter(thresholds);threshold=min(count,key=lambda v:(-count[v],v))
    selection={'rule_sha256':sha256(OUT/'deployment-rule.json'),'cells':cells,'mask_threshold':mask,'confidence_threshold':threshold,
               'minimum_area':64,'cluster_limit':6,'margin':.02,'utility_reference_pq':.35,'ridge_penalty':.01,
               'no_new_generalization_measurement':True,'transfer_limit':'Consensus of inner choices, not recalibration on test or full-fit training predictions.'}
    atomic(OUT/'decoder-selection.json',selection)
    dataset=read_json('manifests/dataset-v1.json');train=sorted(r['file_name'] for r in dataset['images'] if r['split']=='train');test=sorted(r['file_name'] for r in dataset['images'] if r['split']=='test')
    groups={r['file_name']:r['group_id'] for r in read_json('manifests/split-v2.json')['observations']}
    ordered=sorted(set(groups.values()),key=lambda g:hashlib.sha256(('I6-production-mining-v1|'+g).encode()).hexdigest())
    half={g:i%2 for i,g in enumerate(ordered)}
    miners=[{'fit':sorted(n for n in train if half[groups[n]]!=m),'predict':sorted(n for n in train if half[groups[n]]==m)} for m in [0,1]]
    atomic(DATA,{'purpose':'authorized_full_supplied_training_refit_no_validation','fit_names':train,'test_names':test,
        'input_sha256':{f:inputs[f] for f in ['manifests/dataset-v1.json','manifests/split-v2.json','data/annotations/MAGFiLO_1.0_Annotations_kaggle2026_train.json']},
        'miners':miners,'group_rule':'I6-production-mining-v1 SHA256 observation-day ordering, alternating halves, no label-dependent allocation',
        'policy':'All supplied folds including prior holdout enter production. No new holdout score. I6 weights excluded from development initialization/scoring.'})
    assert len(train)==707 and len(test)==180 and set(train).isdisjoint(test)
    assert len(index().records)==1154
    for m in [0,1]:index(miner=m)
    inputs[DATA]=sha256(DATA)
    for f in [OUT/'deployment-rule.json',OUT/'decoder-selection.json']:inputs[f.as_posix()]=sha256(f)
    p={'milestone':'I6','created_utc':now(),'deployment_only':True,'new_holdout_score':False,'cell':CELL,'seed':20261014,
       'inputs':inputs,'runtime':read_json('configs/growth-4.json')['runtime'],'selection':selection,
       'training':{'fits':6,'successful_updates':90000,'sample_occurrences':600000,'detector_steps':15000,'refiner_steps':30000,
           'half_detector_steps':7500,'half_refiner_steps':15000,'initialization':'generic COCO detector and ImageNet refiner; no filament warm-start',
           'exposure_limit':'Same update budget as validation, lower per-image exposure in larger production pool; not claimed exposure-matched.'},
       'resources':{'gpu_stage_hours':10,'cpu_stage_hours':6,'fit_minutes':150,'free_disk_floor_gib':85,'new_remote_gib':20,'new_windows_gib':25,'host_ram_gib':24},
       'selector':{'utility_reference_pq':.35,'ridge_penalty':.01,'cluster_limit':6,'margin':.02,'fit_only':'707 supplied train images from own group-excluded half-fit miners; no full-fit or test labels used for selector fitting'},
       'test':{'images':180,'all_image_separate_process_repeat':True,'visual_export_names':sorted(test,key=lambda n:hashlib.sha256(('I6-export-review-v1|'+n).encode()).hexdigest())[:3],
           'no_test_tuning':True,'submit_recipe':'set_selector_pq','confidence_control':'Export and audit alongside selected arm; not submitted or selected using test appearance'},
       'freeze_policy':'Training sources freeze before quality fitting. Inference/selector/export sources freeze before OOF prediction or test processing. Selector hash freezes before test. No scientific setting changes from production outputs or public feedback.',
       'user_authorized_one_submission':True,'git_parent':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    atomic(PROTOCOL,p)
    print({'protocol_sha256':sha256(PROTOCOL),'selected':selection,'miner_sizes':[(len(m['fit']),len(m['predict'])) for m in miners]})

if __name__=='__main__':main()
