"""Calibrate both reference thresholds on the same92 source images, no target labels."""
import time
from common import *
from select_and_evaluate import choose,CRYO_GRID,TOP_GRID

def main():
 for method in ['topaz','cryolo']:
  p=G/('early_'+method+'_validation_status.json')
  while not p.exists() or json.loads(p.read_text()).get('status')!='complete':time.sleep(10)
 rs=records('val');assert len(rs)==92;baseline={}
 for method,grid in [('crYOLO',CRYO_GRID),('Topaz',TOP_GRID)]:
  pred={};reused={}
  for ds in DOMAINS:
   if ds in ['10028','10017','10077','10291']:
    p=V1/'results/baseline_val'/method/ds/'predictions.csv';reused[str(p)]=sha(p)
   else:
    p=G/'results/baseline_val'/method/ds/'predictions.csv';done=json.loads(p.with_name('done.json').read_text());assert sha(p)==done['predictions_sha256']
   pred.update(load_predictions(p))
  best,sweep,metrics=choose(rs,pred,grid);csvout(G/'reports'/(method+'_validation_grid.csv'),sweep);dump(G/'reports'/(method+'_validation_metrics.json'),metrics)
  baseline[method]=dict(threshold=best['threshold'],validation_macro=metrics['macro'],validation_datasets=metrics['datasets'],same_validation_images=92,reused_prediction_sha256=reused)
 dump(G/'baseline_selection.json',baseline)
 print('SOURCE_ONLY_BASELINE_CALIBRATION_COMPLETE',json.dumps(baseline),flush=True)

if __name__=='__main__':main()
