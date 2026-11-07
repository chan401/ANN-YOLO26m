"""Cache the two retained scale-branch checkpoints on source validation only."""
import gc
from common import *
from infer import infer
from select_and_evaluate import choose,ANN_GRID

def main():
 rs=records('val');assert len(rs)==92
 for tag in ['best','last']:
  weight=G/'runs/canonical64_lowpass1_7domains/weights'/(tag+'.pt')
  out=G/'validation'/('canonical64_lowpass1_'+tag)
  dump(G/'early_A_validation_status.json',dict(status='running',checkpoint=tag,target_images_used=False))
  pred=infer(weight,rs,out);best,grid,metrics=choose(rs,pred,ANN_GRID)
  csvout(out/'threshold_grid.csv',grid);dump(out/'metrics.json',metrics)
  gc.collect();torch.cuda.empty_cache()
 dump(G/'early_A_validation_status.json',dict(status='complete',checkpoints=2,validation_images=92,target_images_used=False))
 print('CANDIDATE_A_VALIDATION_CACHED',flush=True)

if __name__=='__main__':main()
