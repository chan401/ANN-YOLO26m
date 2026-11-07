"""Cache new-source GPU references with the existing allow-growth runtime."""
import time,subprocess
from common import *
from select_and_evaluate import baseline_engine

def main():
 top=G/'early_topaz_validation_status.json'
 while not top.exists() or json.loads(top.read_text()).get('status')!='complete':time.sleep(10)
 # Keep a conservative margin for the continuing ANN training process.
 while True:
  free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0])
  if free>=8192:break
  time.sleep(15)
 bc=baseline_engine()
 for ds in DOMAINS[-3:]:
  dump(G/'early_cryolo_validation_status.json',dict(status='running',dataset=ds,target_images_used=False))
  bc.one(ds,'crYOLO','val')
 dump(G/'early_cryolo_validation_status.json',dict(status='complete',images=24,GPU=0,target_images_used=False))
 print('CRYOLO_NEW_SOURCE_VALIDATION_CACHED',flush=True)

if __name__=='__main__':main()
