"""Check the scale-adaptation stage only on the original source validation domains."""
import time
from common import *
from infer import infer
from select_and_evaluate import choose,ANN_GRID

def main():
 assert DOMAINS==['10028','10017','10077','10291']
 done=G/'runs/canonical64_source4/completion.json'
 while not done.exists():
  s=json.loads((G/'status.json').read_text());assert s.get('status')!='failed',s;time.sleep(5)
 rs=records('val');assert len(rs)==68
 out=G/'full_source_validation';weight=G/'runs/canonical64_source4/weights/best.pt'
 pred=infer(weight,rs,out);selected,grid,metrics=choose(rs,pred,ANN_GRID)
 csvout(out/'threshold_grid.csv',grid);dump(out/'metrics.json',metrics)
 dump(out/'selected.json',dict(checkpoint=str(weight),checkpoint_sha256=sha(weight),threshold=selected['threshold'],validation_macro=metrics['macro'],validation_datasets=metrics['datasets'],target_images_used=False))
 print('SOURCE_STAGE_FULL_VALIDATION',json.dumps({k:v for k,v in metrics.items() if k!='per_image'}),flush=True)

if __name__=='__main__':main()
