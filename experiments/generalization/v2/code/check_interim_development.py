"""Target-free interim diagnostic; keeps a stable copy of the current best weights."""
import shutil,time,gc
from common import *
from infer import infer
from select_and_evaluate import choose,ANN_GRID

def main():
 src=G/'runs/canonical64_lowpass1_7domains/weights/best.pt';out=G/'development_interim_A';out.mkdir(exist_ok=True);weight=out/'snapshot.pt'
 if not weight.exists():
  for attempt in range(10):
   if time.time()-src.stat().st_mtime<5:time.sleep(5)
   before=sha(src);shutil.copy2(src,weight)
   if before==sha(src)==sha(weight):break
   weight.unlink();time.sleep(2)
  else:raise RuntimeError('Could not take a stable checkpoint snapshot')
 rs=records('val');assert len(rs)==92
 pred=infer(weight,rs,out/'predictions');best,grid,metrics=choose(rs,pred,ANN_GRID)
 dump(out/'metrics.json',metrics);csvout(out/'threshold_grid.csv',grid)
 dump(out/'selection.json',dict(checkpoint_sha256=sha(weight),selected_threshold=best['threshold'],validation_macro=metrics['macro'],validation_datasets=metrics['datasets'],target_images_used=False,diagnostic_only=True))
 print('INTERIM_DEVELOPMENT_ONLY',json.dumps({k:v for k,v in metrics.items() if k!='per_image'}),flush=True)

if __name__=='__main__':main()
