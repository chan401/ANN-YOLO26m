"""Continue source-only training and development; preserve all earlier results."""
from pathlib import Path
import json,subprocess,time,traceback
G=Path('/root/autodl-tmp/cryoem_generalization/v3_low_snr')
PY='/root/autodl-tmp/cryoem_annyolo26/.venv/bin/python'
def status(s,**kw):(G/'status.json').write_text(json.dumps(dict(status=s,updated_unix=time.time(),**kw),indent=2))
def run(stage,args):
 status('running',stage=stage)
 with (G/(stage+'.log')).open('a') as f:subprocess.run([PY,'-u',*map(str,args)],cwd=G/'code',stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
 try:
  if not (G/'runs/smoke_low_snr20/completion.json').exists():run('smoke_training',[G/'code/train_low_snr.py','--smoke'])
  if not (G/'runs/low_snr20/completion.json').exists():run('formal_training',[G/'code/train_low_snr.py'])
  run('source_validation',[G/'code/select_evaluate.py'])
  status('complete' if (G/'final_report.json').exists() else 'development_complete')
 except Exception as e:status('failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
