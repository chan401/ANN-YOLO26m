"""Use idle GPU for source-only scale adaptation while extra domains download."""
import subprocess,time,traceback
from common import *
PY='/root/autodl-tmp/cryoem_annyolo26/.venv/bin/python'
CODE=Path('/root/autodl-tmp/cryoem_generalization/v2/code')

def stage(name,args):
 dump(G/'status.json',dict(status='running',stage=name,updated_unix=time.time()))
 with (G/(name+'.log')).open('a') as f:subprocess.run([PY,'-u',*map(str,args)],cwd=CODE,stdout=f,stderr=subprocess.STDOUT,check=True)

def main():
 G.mkdir(parents=True,exist_ok=True)
 try:
  assert DOMAINS==['10028','10017','10077','10291']
  if not(G/'training_data/preparation.json').exists():stage('prepare_source_only',[CODE/'prepare_training.py'])
  if not(G/'runs/smoke_v2/completion.json').exists():stage('source_smoke',[CODE/'train.py','--smoke'])
  if not(G/'runs/canonical64_source4/completion.json').exists():stage('source_scale_training',[CODE/'train.py','--epochs','20','--name','canonical64_source4'])
  dump(G/'status.json',dict(status='complete',epochs=20,target_images_used=False,updated_unix=time.time()))
 except Exception as e:
  dump(G/'status.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc(),updated_unix=time.time()));raise

if __name__=='__main__':main()
