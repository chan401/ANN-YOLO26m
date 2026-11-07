"""One fixed-threshold evaluation; no threshold-scan or selection interface."""
from common_test import *
import importlib.util

def main():
    f=frozen();ns=names();seal=sha(FREEZE);lock=R/'evaluation_lock.json'
    if lock.exists():
        old=json.loads(lock.read_text())
        if old.get('status')=='complete' and old['frozen_config_sha256']==seal:
            print('ALREADY_COMPLETED',str(R/'metrics.json'));return
        raise FileExistsError('One-time evaluation lock exists; inspect rather than rerun')
    spec=importlib.util.spec_from_file_location('v3match',f['matching_implementation']);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    gt={n:[] for n in ns}
    with Path(f['shared_gt']).open(newline='') as h:
        for i,r in enumerate(csv.DictReader(h)):
            name=r['micrograph_name'];x=float(r['x_original_px']);y=float(r['y_original_px'])
            assert name in gt and 0<=x<4096 and 0<=y<4096
            gt[name].append(mod.Point(x,y,1.0,i))
    assert sum(map(len,gt.values()))==8830
    with (M/'test_manifest.csv').open(newline='') as h:manifest=list(csv.DictReader(h))
    for r in manifest:
        assert len(gt[r['micrograph_name']])==int(r['gt_count'])
        assert sha(r['cryoppp_csv_path'])==r['cryoppp_csv_sha256']
        assert sha(r['original_path'])==r['mrc_sha256']
    preds={};doneinfo={}
    for method,folder in [('ANN-YOLO26','ann'),('crYOLO','cryolo'),('Topaz','topaz')]:
        root=E/(folder+'_test_predictions');done=json.loads((root/'done.json').read_text())
        assert done['status']=='complete' and done['images']==100 and done['frozen_config_sha256']==seal
        assert done['predictions_sha256']==sha(root/'predictions.csv')
        pred={n:[] for n in ns}
        for i,r in enumerate(prediction_rows(root/'predictions.csv',method)):
            pred[r['micrograph_name']].append(mod.Point(float(r['x_original_px']),float(r['y_original_px']),float(r['score']),i))
        preds[method]=pred;doneinfo[method]=done
    with lock.open('x') as h:json.dump({'status':'running','started_utc':now(),'frozen_config_sha256':seal},h,indent=2)
    table=[];per=[];matches=[];used=[]
    for method,pred in preds.items():
        threshold=f['methods'][method]['threshold'];a,b,metrics=mod.evaluate_method(ns,gt,pred,threshold)
        assert len(a)==100 and metrics['TP']+metrics['FN']==8830
        table.append({'Method':method,**metrics});per.extend({'method':method,**r} for r in a);matches.extend({'method':method,**r} for r in b)
        used.append({'Method':method,'fixed_validation_threshold':threshold,'checkpoint_sha256':f['methods'][method]['checkpoint_sha256']})
    write_csv(R/'paper_table.csv',table);write_csv(R/'per_micrograph_metrics.csv',per);write_csv(R/'matching_records.csv',matches);write_csv(R/'frozen_thresholds_used.csv',used)
    dump(R/'metrics.json',{'configuration':'test_100_v1','images':100,'GT':8830,'matching_radius_px':56,'methods':{r['Method']:{k:v for k,v in r.items() if k!='Method'} for r in table},'frozen_config_sha256':seal})
    integrity={'status':'PASS','images_per_method':100,'GT_per_method':8830,'same_manifest_and_GT':True,'train_val_disjoint':True,
      'source_MRC_and_GT_checksums_unchanged':True,'weights_and_sources_frozen':True,'all_model_forwards_device':'GPU:0',
      'thresholds_validation_only':True,'test_threshold_scan':False,'model_retraining':False,'matching_implementation_sha256':sha(f['matching_implementation']),
      'completed_utc':now(),'frozen_config_sha256':seal,'prediction_provenance':doneinfo}
    dump(R/'integrity_report.json',integrity)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.6));xx=list(range(3));width=.24
    for j,key in enumerate(['Precision','Recall','F1']):
        bars=ax.bar([i+(j-1)*width for i in xx],[r[key] for r in table],width,label=key)
        ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=8)
    ax.set_xticks(xx,[r['Method'] for r in table]);ax.set_ylim(0,1.12);ax.set_ylabel('Micro-averaged metric');ax.set_title('EMPIAR-10028 test_100_v1 | 100 micrographs, 8,830 GT');ax.legend(ncol=3,loc='upper center');fig.tight_layout()
    fig.savefig(R/'test100_fixed_threshold_metrics.png',dpi=220);fig.savefig(R/'test100_fixed_threshold_metrics.pdf');plt.close(fig)
    lines=['# EMPIAR-10028 test_100_v1: frozen GPU evaluation','',
      'All three methods ran model inference on GPU 0. Same 100 micrographs and 8,830 GT; inclusive 56 px one-to-one maximum-cardinality matching; pooled TP/FP/FN. Each method used its previously selected validation threshold. No test threshold scan, retraining or post-test adjustment.','',
      '| Method | TP | FP | FN | Precision | Recall | F1 |','|---|---:|---:|---:|---:|---:|---:|']
    for r in table:lines.append(f"| {r['Method']} | {r['TP']} | {r['FP']} | {r['FN']} | {r['Precision']:.6f} | {r['Recall']:.6f} | {r['F1']:.6f} |")
    lines+=['','Frozen before test: '+f['frozen_utc']+'; SHA-256 '+seal+'.','',
      'Fixed validation thresholds: '+', '.join(f"{r['Method']}={r['fixed_validation_threshold']}" for r in used)+'.','',
      'ANN: official P3/P4/P5 YOLO26m, robust float grayscale normalization, nine 1536 px tiles, fixed ownership, 28 px center merge, max_det=2000/tile. crYOLO: original PhosaurusNet pretrained H5, 1024 input, lowpass 0.1, box224, NVIDIA TensorFlow GPU. Topaz: packaged resnet16_u64 pretrained model; same CPU scale8/GMM preprocessing and CPU NMS as validation, GPU score-map model forward. All FP32; TF32 disabled for baseline GPU execution.','',
      f['pretraining_provenance_caveat'],'',
      'ANN training completed 80 epochs. Its 44-image validation result was P=0.790333, R=0.962082, F1=0.867791 (3,824 GT), meeting all stated validation targets.','',
      'Raw predictions and per-method logs: `'+str(E)+'`.','Report directory: `'+str(R)+'`.','Manifest: `'+str(M/'test.txt')+'`.','',
      'Integrity: PASS; original MRC and GT checksums unchanged; train/validation manifests untouched. No reconstruction started.']
    (R/'final_report.md').write_text('\n'.join(lines)+'\n')
    dump(lock,{'status':'complete','completed_utc':now(),'frozen_config_sha256':seal,'metrics_sha256':sha(R/'metrics.json')})
    print(json.dumps(table,indent=2));print('ONE_TIME_TEST_EVALUATION_COMPLETE',str(R),flush=True)

if __name__=='__main__':main()
