"""Sequential GPU inference followed by the one-time fixed-threshold comparison."""
from common_test import *
import subprocess,traceback
PY='/root/autodl-tmp/cryoem_annyolo26/.venv/bin/python'

def main():
    status=E/'workflow_status.json'
    try:
        frozen()
        stages=[('ANN-YOLO26','ann','ann_test_inference.py',[]),('crYOLO','cryolo','baseline_gpu_test.py',['cryolo']),('Topaz','topaz','baseline_gpu_test.py',['topaz'])]
        for label,folder,script,args in stages:
            frozen();done=E/(folder+'_test_predictions')/'done.json'
            if done.exists():
                d=json.loads(done.read_text());assert d['images']==100 and d['frozen_config_sha256']==sha(FREEZE)
                assert d['predictions_sha256']==sha(done.parent/'predictions.csv');continue
            dump(status,{'status':'running','stage':label,'updated_utc':now(),'device':'GPU:0'})
            with (E/(folder+'_inference.log')).open('w') as h:
                subprocess.run([PY,str(E/'code'/script),*args],stdout=h,stderr=subprocess.STDOUT,check=True,cwd=E/'code')
        dump(status,{'status':'evaluating_fixed_thresholds','updated_utc':now()})
        with (E/'evaluation.log').open('w') as h:
            subprocess.run([PY,str(E/'code/evaluate_test100.py')],stdout=h,stderr=subprocess.STDOUT,check=True,cwd=E/'code')
        dump(status,{'status':'complete','updated_utc':now(),'metrics':str(R/'metrics.json'),'device':'GPU:0 for all three models'})
    except Exception as exc:
        dump(status,{'status':'failed','updated_utc':now(),'error':repr(exc),'traceback':traceback.format_exc()});raise

if __name__=='__main__':main()
