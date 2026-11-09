"""Freeze I6 inference implementation and immutable proof parents before processing."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import ast,py_compile
from pathlib import Path
from i6_common import ROOT,OUT,PROTOCOL,verify,read_json,sha256,atomic,now
os.chdir(ROOT);p=verify();assert not (OUT/'inference-source-freeze.json').exists() and not (OUT/'candidates').exists()
files=set(read_json(OUT/'training-source-freeze.json')['files'])
pending=['scripts/'+f for f in ['i6_candidates_gpu.py','i6_selector.py','i6_export.py','review_i6_export.py','i6_inference_campaign.py','i6_remote_chain.py','freeze_i6_inference.py']]
seen=set()
while pending:
    path=pending.pop()
    if path in seen:continue
    seen.add(path);files.add(path);tree=ast.parse(Path(path).read_text(encoding='utf8'));py_compile.compile(path,doraise=True)
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom) and node.module:
            candidate=Path('scripts')/(node.module.split('.')[0]+'.py')
            if candidate.exists():pending.append(candidate.as_posix())
        elif isinstance(node,ast.Import):
            for name in node.names:
                candidate=Path('scripts')/(name.name.split('.')[0]+'.py')
                if candidate.exists():pending.append(candidate.as_posix())
files.update(['configs/enhancement-h2.json','manifests/improvement-i5-splits.json','outputs/growth-6/features/inner-f0/manifest.json'])
for stage in ['detector','refiner']:files.update(f'runs/G4-{stage}-f0-s20261014/{name}' for name in ['run.json','weights.pt'])
def infer(path):return ast.dump(next(n for n in ast.walk(ast.parse(Path(path).read_text(encoding='utf8'))) if isinstance(n,ast.FunctionDef) and n.name=='infer'),include_attributes=False)
assert infer('scripts/i6_candidates_gpu.py')==infer('scripts/i5_candidates_gpu.py')
assert read_json('configs/enhancement-h2.json')['utility_reference_pq']==p['selection']['utility_reference_pq']==.35
atomic(OUT/'inference-adaptation-audit.json',{'passed':True,'utc':now(),'native_inference_function_ast_exact_i5':True,'utility_reference_unchanged':True,'scripts_compiled':sorted(seen)})
atomic(OUT/'inference-source-freeze.json',{'utc':now(),'protocol_sha256':sha256(PROTOCOL),'all_i6_quality_predictions_not_started':True,
    'files':{f:sha256(f) for f in sorted(files)},'scope':'Additive implementation and immutable historical parity parents; science unchanged.'})
print('I6 inference frozen',len(files),sha256(OUT/'inference-source-freeze.json'))
