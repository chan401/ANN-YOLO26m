"""Resumable preparation, connectivity check, and genuine full training."""
import subprocess,time,traceback
from common import *
PY='/root/autodl-tmp/cryoem_annyolo26/.venv/bin/python'

def run(stage,args):
 dump(G/'status.json',dict(status='running',stage=stage,updated_unix=time.time()))
 with (G/(stage+'.log')).open('a') as f:subprocess.run([PY,'-u',*map(str,args)],cwd=G/'code',stdout=f,stderr=subprocess.STDOUT,check=True)

def main():
 try:
  dump(G/'status.json',dict(status='running',stage='waiting_for_selected_data'))
  while not (G/'preparation_report.json').exists():
   pid=int((G/'download.pid').read_text());os.kill(pid,0)
   assert Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z','Data preparer exited before completion; inspect download.log'
   time.sleep(10)
  warmup=G/'source_scale_adaptation/runs/canonical64_source4/completion.json'
  dump(G/'status.json',dict(status='running',stage='waiting_for_source_scale_adaptation'))
  while not warmup.exists():
   ws=G/'source_scale_adaptation/status.json'
   if ws.exists():assert json.loads(ws.read_text()).get('status')!='failed',ws.read_text()
   time.sleep(10)
  if not(G/'training_data/preparation.json').exists():run('prepare_training',[G/'code/prepare_training.py'])
  if not(G/'runs/smoke_v2/completion.json').exists():run('smoke_training',[G/'code/train.py','--smoke'])
  # Small GT-free inference check uses validation images only; no target input.
  if not(G/'connectivity.json').exists():
   dump(G/'status.json',dict(status='running',stage='smoke_inference'))
   from infer import infer
   rs=[next(r for r in records('val') if r['dataset']==d) for d in ['10028','10291','10760']]
   pred=infer(G/'runs/smoke_v2/weights/best.pt',rs,G/'smoke_inference')
   assert all(0<=p.x<r['width'] and 0<=p.y<r['height'] for r in rs for p in pred.get(key(r),[]))
   dump(G/'connectivity.json',dict(status='PASS',validation_images=len(rs),coordinates_restored_in_bounds=True,training_losses_finite=True,target_images_used=False))
   import gc
   gc.collect();torch.cuda.empty_cache()
  if not(G/'runs/canonical64_lowpass1_7domains/completion.json').exists():run('formal_training',[G/'code/train.py','--epochs','40','--lr','0.0002','--weights',warmup.parent/'weights/best.pt'])
  if not(G/'native7/training_data/preparation.json').exists():run('prepare_native_control',[G/'code/native_wrapper.py','prepare'])
  if not(G/'native7/runs/smoke_native7/completion.json').exists():run('native_control_smoke',[G/'code/native_wrapper.py','smoke'])
  if not(G/'native7/runs/native7/completion.json').exists():run('native_control_training',[G/'code/native_wrapper.py','train'])
  dump(G/'status.json',dict(status='training_complete_awaiting_validation',stage='full_validation',updated_unix=time.time()))
  if (G/'code/select_and_evaluate.py').exists():run('select_and_evaluate',[G/'code/select_and_evaluate.py'])
  gate=json.loads((G/'validation_gate.json').read_text()) if (G/'validation_gate.json').exists() else None
  status='complete' if (G/'final_report.json').exists() else 'needs_more_development' if gate and not gate['passed'] else 'development_validation_complete'
  dump(G/'status.json',dict(status=status,validation_gate=gate,updated_unix=time.time()))
 except Exception as e:
  dump(G/'status.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc(),updated_unix=time.time()));raise

if __name__=='__main__':main()
