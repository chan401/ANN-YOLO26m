"""Finalize only after prospective downloads and source selection have passed."""
from pathlib import Path
import json,subprocess,time,traceback
G=Path('/root/autodl-tmp/cryoem_generalization/v3_low_snr')
PY='/root/autodl-tmp/cryoem_annyolo26/.venv/bin/python'
def status(s,**kw):(G/'status.json').write_text(json.dumps(dict(status=s,updated_unix=time.time(),**kw),indent=2)+'\n')
def main():
 try:
  assert json.loads((G/'development_gate.json').read_text())['passed']
  assert json.loads((G/'target_integrity.json').read_text())['status']=='PASS'
  status('running',stage='frozen_final_evaluation')
  with (G/'final_evaluation.log').open('a') as f:subprocess.run([PY,'-u',str(G/'code/select_evaluate.py'),'--finalize'],stdout=f,stderr=subprocess.STDOUT,check=True,cwd=G/'code')
  report=json.loads((G/'final_report.json').read_text());assert (G/'export_manifest.json').exists()
  status('complete',stage='final_evaluation_and_export',TRPV1_F1_exceeds_both_baselines=report['ANN_F1_exceeds_both_baselines'],same_v3_model_both_datasets_exceed_baselines=report['same_v3_model_both_datasets_exceed_baselines'])
 except Exception as e:status('failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
