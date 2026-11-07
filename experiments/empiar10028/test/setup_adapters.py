"""Reuse validated coordinate parsers and ANN inference, with frozen test membership."""
import ast,copy,json,shutil
from pathlib import Path
from common_test import D,E,R,M,sha,dump,names
BASE=Path('/root/autodl-tmp/cryoem_baselines/experiments/empiar10028_pretrained_val_v3')
ANN=D/'experiments/ann_yolo26m_cryo_tiles_v3/code'

def main():
    if (E/'frozen_config.json').exists():raise FileExistsError('Adapters already frozen')
    E.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
    t=ast.parse((BASE/'run_val_inference.py').read_text());funcs=[];proof={}
    for name in ['mrc_shape','sha','cryolo_log_confirms_zero_candidates','parse_cryolo_cbox','cryolo_cbox_to_csv','topaz_csv_to_canonical']:
        old=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name);new=copy.deepcopy(old)
        for n in ast.walk(new):
            if isinstance(n,ast.Constant) and n.value=='val':n.value='test'
        back=copy.deepcopy(new)
        for n in ast.walk(back):
            if isinstance(n,ast.Constant) and n.value=='test':n.value='val'
        proof[name]=ast.dump(old,include_attributes=False)==ast.dump(back,include_attributes=False)
        funcs.append(ast.unparse(new))
    assert all(proof.values())
    helper='import csv,hashlib,struct\nfrom pathlib import Path\nIMAGE_WIDTH=IMAGE_HEIGHT=4096\n\n'+'\n\n'.join(funcs)
    compile(helper,'baseline_helpers.py','exec');(E/'code/baseline_helpers.py').write_text(helper)
    (E/'topaz_runtime_compat').mkdir(exist_ok=True)
    bridge=(BASE/'topaz_runtime_compat/sitecustomize.py').read_text()
    bridge+='\nimport torch\ntorch.backends.cuda.matmul.allow_tf32=False\ntorch.backends.cudnn.allow_tf32=False\n'
    (E/'topaz_runtime_compat/sitecustomize.py').write_text(bridge)
    (E/'topaz_preprocess_repro').mkdir(exist_ok=True)
    (E/'topaz_preprocess_repro/sitecustomize.py').write_text('import numpy as np\nnp.random.seed(10028)\n')
    for fn in ['cryolo_inference_config.json','topaz_inference_config.json']:
        cfg=json.loads((BASE/fn).read_text());cfg['input_split']='manifests/test_100_v1/test.txt only'
        cfg['execution_device']='GPU:0; FP32; no TensorFloat32'
        cfg['threshold_selection']='Previously fixed validation threshold; no test selection or scan'
        if fn.startswith('topaz'):
            cfg['score_map']['device']=0
            cfg['preprocess']['random_seed']=10028
            cfg['segment_compatibility_bridge']=str(E/'topaz_runtime_compat/sitecustomize.py')
            cfg['segment_compatibility_bridge_sha256']=sha(E/'topaz_runtime_compat/sitecustomize.py')
            cfg['outlier_exception_not_used']='080.mrc is not a test100 member; all100 use original MRC sources'
            assert '080.mrc' not in names()
        else:cfg['gpu_id']=0
        dump(E/fn,cfg)
    src=(ANN/'infer_eval.py').read_text();t=ast.parse(src)
    write=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='write_csv')
    fn=copy.deepcopy(next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='infer'));fn.name='infer_ann_test'
    cut=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='smoke')
    fn.body=fn.body[:cut]
    for n in fn.body:
        if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='ns' for v in n.targets):n.value=ast.Name(id='TEST_NAMES',ctx=ast.Load())
    for n in ast.walk(fn):
        if isinstance(n,ast.Constant) and n.value=='val':n.value='test'
        if isinstance(n,ast.Dict):
            for i,k in enumerate(n.keys):
                if isinstance(k,ast.Constant) and k.value=='test_accessed':n.values[i]=ast.Constant(value=True)
    body=ast.unparse(ast.fix_missing_locations(fn));assert 'threshold_sweep' not in body and 'load_gt' not in body
    prefix=f'''import sys,json
from pathlib import Path
sys.path.insert(0,{str(ANN)!r})
from common import *
from ultralytics import YOLO
sys.path.insert(0,str(DATA/'metadata/three_method_evaluation'))
from evaluate_empiar10028_v3_pretrained_val import Point
from common_test import frozen,sha as file_sha,dump as save_json,E as TEST_EXP,M as TEST_MAN,names as test_names,prediction_rows
TEST_NAMES=test_names()
'''
    tail='''
if __name__=='__main__':
    f=frozen();weight=Path(f['methods']['ANN-YOLO26']['checkpoint']);out=TEST_EXP/'ann_test_predictions'
    torch.set_num_threads(4)
    infer_ann_test(weight,out)
    info=json.loads((out/'inference.json').read_text())
    if [r['micrograph_name'] for r in info['micrographs']]!=TEST_NAMES:raise ValueError('ANN test coverage mismatch')
    with (TEST_MAN/'test_manifest.csv').open(newline='') as h:gold={r['micrograph_name']:r['mrc_sha256'] for r in csv.DictReader(h)}
    for r in info['micrographs']:
        r['source_sha256']=file_sha(DATA/'micrographs'/r['micrograph_name'])
        if r['source_sha256']!=gold[r['micrograph_name']]:raise ValueError('ANN source checksum changed')
    save_json(out/'inference.json',info)
    rows=prediction_rows(out/'predictions.csv','ANN-YOLO26')
    save_json(out/'done.json',{'status':'complete','method':'ANN-YOLO26','split':'test','images':100,'predictions':len(rows),'predictions_sha256':file_sha(out/'predictions.csv'),'inference_sha256':file_sha(out/'inference.json'),'frozen_config_sha256':file_sha(TEST_EXP/'frozen_config.json')})
'''
    code=prefix+'\n'+ast.unparse(write)+'\n'+body+'\n'+tail;compile(code,'ann_test_inference.py','exec')
    (E/'code/ann_test_inference.py').write_text(code)
    dump(E/'adapter_equivalence.json',{'status':'PASS','baseline_coordinate_helpers_identical_except_split_tag':proof,
        'ann_source':str(ANN/'infer_eval.py'),'ann_source_sha256':sha(ANN/'infer_eval.py'),
        'ann_change_scope':'test100 membership/split; GT and all threshold selection branches removed; input checksum metadata added',
        'GPU_validation_compatibility':str(Path('/root/autodl-tmp/cryoem_baselines/gpu_validation_smoke/gpu_cpu_compatibility.json')),
        'test_inference_started':False})
    print('GPU_TEST_ADAPTER_SETUP_PASS')

if __name__=='__main__':main()
