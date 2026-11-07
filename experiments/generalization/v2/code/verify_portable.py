"""Compare a GT-free exported picker against frozen source-validation predictions."""
import gc,time
from common import *
import predict_particles as portable
from ultralytics import YOLO

def main():
 cases=[]
 native=json.loads((V1/'validation_selection/selected_general_model.json').read_text())
 cases.append(('native_multiscale',Path(native['checkpoint']),Path(native['validation_prediction_dir'])/'predictions.csv',['10028','10291','10017']))
 root=G/'validation/canonical64_lowpass1_best'
 while not(root/'done.json').exists():time.sleep(10)
 cases.append(('canonical64_lowpass1',G/'runs/canonical64_lowpass1_7domains/weights/best.pt',root/'predictions.csv',['10028','10291','10760']))
 proof=[];rs=records('val');torch.set_num_threads(4);cv2.setNumThreads(1)
 for profile,weight,reference,dslist in cases:
  ref=load_predictions(reference);model=YOLO(str(weight));model.model.model[-1].max_det=2000
  for ds in dslist:
   r=next(r for r in rs if r['dataset']==ds);floor=.001 if profile=='canonical64_lowpass1' else 1e-5
   rows,meta=portable.predict(model,Path(r['path']),floor,False,0,profile,r['diameter'])
   actual=sorted((p['x_cryoppp_px'],p['y_cryoppp_px'],p['score']) for p in rows)
   expected=sorted((p.x,p.y,p.score) for p in ref.get(key(r),[]) if p.score>=floor)
   assert len(actual)==len(expected),(profile,ds,len(actual),len(expected))
   delta=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected)))) if actual else 0.
   assert delta<=1e-5,(profile,ds,delta)
   proof.append(dict(profile=profile,dataset=ds,micrograph=r['micrograph_name'],checkpoint_sha256=sha(weight),candidates=len(actual),max_abs_coordinate_score_difference=delta))
   print('PORTABLE_EQUIVALENCE_PASS',profile,ds,len(actual),delta,flush=True)
  del model;gc.collect();torch.cuda.empty_cache()
 dump(G/'deployment_equivalence_v2.json',dict(status='PASS',comparisons=proof,target_images_used=False,standalone_requires_GT=False,standalone_requires_EMPIAR10028_project=False,picker_sha256=sha(G/'code/predict_particles.py')))

if __name__=='__main__':main()
