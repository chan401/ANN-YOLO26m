"""Seal weights, validation thresholds, source code and test inputs before any test forward."""
from common_test import *
B=Path('/root/autodl-tmp/cryoem_baselines')
A=D/'experiments/ann_yolo26m_cryo_tiles_v3'

def main():
    if FREEZE.exists():raise FileExistsError('Frozen configuration already exists')
    assert not any(E.glob('*_test_predictions')),'Test inference already started'
    selected=D/'reports/ann_yolo26_cryo_adaptation_v3/selected_model_config.json'
    thresholds=D/'reports/pretrained_baseline_validation_v3/validation_thresholds.json'
    s=json.loads(selected.read_text());t=json.loads(thresholds.read_text())
    assert json.loads((A/'workflow_status.json').read_text())['status']=='baseline_complete'
    assert json.loads((A/'finalizer_status.json').read_text())['status']=='complete'
    assert s['test_accessed'] is False and t['test_used_for_selection'] is False
    for name in ['common.py','infer_eval.py']:
        assert sha(A/'code'/name)==s['code_hashes'][name]
    proof=B/'gpu_validation_smoke/gpu_cpu_compatibility.json';assert json.loads(proof.read_text())['status']=='PASS'
    methods={
      'ANN-YOLO26':{'checkpoint':s['checkpoint'],'checkpoint_sha256':s['checkpoint_sha256'],
        'threshold':s['single_global_validation_threshold'],'validation_source':str(selected),
        'parameters':{'device':0,'precision':'FP32','normalization':s['normalization'],'orientation':'flipud before tiling',
          'tile_size':1536,'tile_starts':[0,1280,2560],'ownership_bounds':[0,1408,2688,4096],'max_det':2000,
          'score_floor':1e-5,'center_merge_radius_px':28,'heads':'P3/P4/P5','TTA':False}},
      'crYOLO':{'checkpoint':str(B/'cryolo/weights/gmodel_phosnet_202005_N63_c17.h5'),
        'checkpoint_sha256':'17b374db3b51046511a280ac00f7f0dd5bbf70a16c905f6d8d4e4e2b514faa89',
        'threshold':t['methods']['crYOLO'],'validation_source':str(thresholds),
        'parameters':{'device':0,'precision':'FP32; TF32 disabled','cryolo_version':'1.9.9','tensorflow':'1.15.5+nv22.12',
          'input_size':1024,'anchor_px':224,'lowpass':0.1,'normalization':'STANDARD','native_nms_iou':0.3,
          'score_floor':0,'prediction_batch_size':1,'on_the_fly':True,'coordinate_adapter':'x=X+w/2; y=4096-(Y+h/2)'}},
      'Topaz':{'checkpoint':str(B/'topaz/topaz/pretrained/detector/resnet16_u64.sav'),
        'checkpoint_sha256':'00d224629a5dc0c2dc122cca11e47c25f5f850d08f9e7f71f54bcf8037d52c46',
        'threshold':t['methods']['Topaz'],'validation_source':str(thresholds),
        'parameters':{'device':0,'precision':'FP32; TF32 disabled','topaz_version':'0.3.21','model':'resnet16_u64',
          'preprocessing':'same CPU GMM normalization and Fourier downsample8 as validation','segment_patch_size':144,
          'preprocessing_random_seed':10028,
          'nms_radius_scaled_px':14,'nms_device':'CPU','native_score_floor':'-inf','up_scale':8,
          'coordinate_adapter':'x=x; y=4095-y','outlier_exception_used':False}}}
    for item in methods.values():assert sha(item['checkpoint'])==item['checkpoint_sha256']
    contract=json.loads((M/'evaluation_contract.json').read_text())
    gt=Path(contract['shared_gt']);matcher=Path(contract['matching_implementation'])
    assert sha(gt)==contract['shared_gt_sha256']
    assert sha(matcher)==contract['matching_implementation_sha256']
    assert sha(M/'test.txt')==contract['test_manifest_sha256']
    ns=set(names())
    for split,n in [('train',132),('val',44)]:
        others={Path(x.strip()).name for x in (M/(split+'.txt')).read_text().splitlines() if x.strip()}
        assert len(others)==n and ns.isdisjoint(others)
    with (M/'test_manifest.csv').open(newline='') as h:manifest=list(csv.DictReader(h))
    assert len(manifest)==100 and sum(int(r['gt_count']) for r in manifest)==8830
    for r in manifest:assert sha(r['cryoppp_csv_path'])==r['cryoppp_csv_sha256']
    protected=list((E/'code').glob('*.py'))+[E/'topaz_runtime_compat/sitecustomize.py',E/'topaz_preprocess_repro/sitecustomize.py',E/'adapter_equivalence.json',
        E/'cryolo_inference_config.json',E/'topaz_inference_config.json',A/'code/common.py',A/'code/infer_eval.py',
        selected,thresholds,M/'test.txt',M/'test_manifest.csv',M/'train.txt',M/'val.txt',gt,matcher,proof,
        B/'cryolo/env-gpu-py38/bin/cryolo_predict_gpu.sh',B/'cryolo/pretrained_baseline/config_lowpass_0p1_box224.json']
    f={'frozen':True,'frozen_utc':now(),'configuration':'test_100_v1','test_images':100,'test_gt':8830,
       'matching_radius_px':56.0,'matching_implementation':str(matcher),'shared_gt':str(gt),'methods':methods,
       'test_manifest':str(M/'test.txt'),'test_manifest_csv':str(M/'test_manifest.csv'),
       'threshold_selection':'validation only; fixed global threshold per method; no test scan or tuning',
       'protected_files':{str(p):sha(p) for p in protected},'GPU_compatibility_proof':str(proof),
       'pretraining_provenance_caveat':'Topaz packaged pretrained model may include EMPIAR-10028; exact image overlap unknown. crYOLO exact pretraining accession/image overlap unknown. These are pretrained baseline comparisons, not verified fully independent pretraining.'}
    dump(FREEZE,f);(E/'frozen_config.sha256').write_text(sha(FREEZE)+'\n')
    dump(R/'frozen_config.json',f);(R/'frozen_config.sha256').write_text(sha(FREEZE)+'\n')
    frozen();print('FROZEN_BEFORE_TEST',sha(FREEZE));print(json.dumps({k:v['threshold'] for k,v in methods.items()}))

if __name__=='__main__':main()
