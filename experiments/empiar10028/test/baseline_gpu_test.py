"""Batched GPU forwards with the validated per-image preprocessing and coordinate parsers."""
import argparse,csv,json,os,subprocess,sys
from pathlib import Path
from common_test import D,E,M,FREEZE,FIELDS,frozen,sha,dump,names,prediction_rows,write_csv,now
from baseline_helpers import cryolo_cbox_to_csv,cryolo_log_confirms_zero_candidates,topaz_csv_to_canonical,mrc_shape
B=Path('/root/autodl-tmp/cryoem_baselines')

def run(cmd,log,cwd=None,segment=False):
    log.parent.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',NVIDIA_TF32_OVERRIDE='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    if segment:env['PYTHONPATH']=str(E/'topaz_runtime_compat')
    elif 'preprocess' in list(map(str,cmd)):env['PYTHONPATH']=str(E/'topaz_preprocess_repro')
    with log.open('w') as f:
        f.write('COMMAND '+json.dumps(list(map(str,cmd)))+'\n');f.flush()
        r=subprocess.run(list(map(str,cmd)),cwd=cwd or log.parent,stdout=f,stderr=subprocess.STDOUT,env=env)
    if r.returncode:raise RuntimeError(f'GPU command failed ({r.returncode}): {log}')

def main():
    p=argparse.ArgumentParser();p.add_argument('method',choices=['cryolo','topaz']);a=p.parse_args();f=frozen();ns=names()
    label='crYOLO' if a.method=='cryolo' else 'Topaz';out=E/(a.method+'_test_predictions');out.mkdir(parents=True,exist_ok=True)
    done=out/'done.json'
    if done.exists():
        d=json.loads(done.read_text());assert d['frozen_config_sha256']==sha(FREEZE) and d['predictions_sha256']==sha(out/'predictions.csv');print('SKIP_VERIFIED',label);return
    paths=[D/'micrographs'/n for n in ns];per=out/'per_image';per.mkdir(exist_ok=True)
    if a.method=='cryolo':
        raw=out/'raw';cboxes=[raw/'CBOX'/(Path(n).stem+'.cbox') for n in ns]
        if not all(p.exists() and p.stat().st_size>0 for p in cboxes):
            cmd=[B/'cryolo/env-gpu-py38/bin/cryolo_predict_gpu.sh','-c',B/'cryolo/pretrained_baseline/config_lowpass_0p1_box224.json','-w',f['methods'][label]['checkpoint'],'-i',*paths,'-o',raw,'-t','0','-nc','1','-pbs','1','-g','0','--otf','--write_empty']
            run(cmd,out/'gpu_predict.log')
        for name,p in zip(ns,cboxes):
            allow=p.exists() and p.stat().st_size==0 and cryolo_log_confirms_zero_candidates(out/'gpu_predict.log')
            n=cryolo_cbox_to_csv(p,per/(Path(name).stem+'.csv'),name,allow_empty=allow);print('CRYOLO_PARSED',name,n,flush=True)
        artifacts={'raw_cbox_sha256':{p.name:sha(p) for p in cboxes},'gpu_log':str(out/'gpu_predict.log')}
    else:
        cli=B/'topaz/.venv-gpu/bin/topaz';pre=out/'preprocessed';scores=out/'scores';pre.mkdir(exist_ok=True);scores.mkdir(exist_ok=True)
        missing=[n for n in ns if not (pre/n).exists() or not(pre/(Path(n).stem+'.metadata.json')).exists()]
        if missing:run([cli,'preprocess','--scale','8','--metadata','--device','-1','--num-workers','0','--num-threads','1','--destdir',pre,*[D/'micrographs'/n for n in missing]],out/'preprocess.log')
        for n in ns:
            if mrc_shape(pre/n)!=(512,512,1,2):raise ValueError(f'Topaz normalized input dimensions: {n}')
        missing=[n for n in ns if not(scores/(Path(n).stem+'.tiff')).exists()]
        if missing:run([cli,'segment','--model','resnet16_u64','--device','0','--num-threads','1','--patch-size','144','--destdir',scores,*[pre/n for n in missing]],out/'gpu_segment.log',segment=True)
        for n in ns:
            if not(scores/(Path(n).stem+'.tiff')).is_file():raise FileNotFoundError(n)
        raw=out/'raw_candidates.csv'
        if not raw.exists():
            tmp=out/'raw_candidates.csv.partial'
            run([cli,'extract','--model','none','--radius','14','--threshold=-inf','--up-scale','8','--device','-1','--num-threads','1','--num-workers','0','--format','csv','--output',tmp,*[scores/(Path(n).stem+'.tiff') for n in ns]],out/'extract.log')
            os.replace(tmp,raw)
        with raw.open(newline='') as h:
            first=h.readline();h.seek(0);reader=csv.DictReader(h,delimiter='\t' if '\t' in first else ',');cols=reader.fieldnames;groups={n:[] for n in ns}
            for r in reader:
                name=Path(r['image_name']).stem+'.mrc'
                if name not in groups:raise ValueError(f'Topaz emitted non-test member {name}')
                groups[name].append(r)
        for name in ns:
            native=per/(Path(name).stem+'_native.csv');write_csv(native,groups[name],cols)
            n=topaz_csv_to_canonical(native,per/(Path(name).stem+'.csv'),name);print('TOPAZ_PARSED',name,n,flush=True)
        artifacts={'raw_candidates_sha256':sha(raw),'preprocessed_sha256':{n:sha(pre/n) for n in ns},'scoremap_sha256':{n:sha(scores/(Path(n).stem+'.tiff')) for n in ns},'gpu_log':str(out/'gpu_segment.log')}
    allrows=[];counts={}
    for name in ns:
        rows=prediction_rows(per/(Path(name).stem+'.csv'),label);allrows.extend(rows);counts[name]=len(rows)
    write_csv(out/'predictions.csv',allrows,FIELDS)
    with (M/'test_manifest.csv').open(newline='') as h:gold={r['micrograph_name']:r['mrc_sha256'] for r in csv.DictReader(h)}
    for p in paths:
        if sha(p)!=gold[p.name]:raise ValueError(f'Source MRC modified: {p}')
    dump(done,{'status':'complete','method':label,'split':'test','images':100,'prediction_counts':counts,'predictions':len(allrows),'predictions_sha256':sha(out/'predictions.csv'),
        'frozen_config_sha256':sha(FREEZE),'checkpoint_sha256':f['methods'][label]['checkpoint_sha256'],'execution_device':'GPU:0','artifacts':artifacts,'completed_utc':now()})
    print('GPU_BASELINE_COMPLETE',label,len(allrows),flush=True)

if __name__=='__main__':main()
