"""Frozen ANN image inference with arbitrary micrograph dimensions; no GT read during forward."""
import argparse,time
from general_common import *
from ultralytics import YOLO

def infer(weight,rs,mode,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);done=out/'done.json';target=out/'predictions.csv'
 if done.exists():
  d=json.loads(done.read_text());assert d['checkpoint_sha256']==sha(weight) and d['predictions_sha256']==sha(target);return load_predictions(target)
 assert not target.exists(),'Existing unfinished predictions require explicit inspection'
 model=YOLO(str(weight));model.model.model[-1].max_det=2000;torch.set_num_threads(4)
 rows=[];info=[];start=time.monotonic()
 for i,r in enumerate(rs):
  image,meta=normalize(r,mode);xs=starts(image.shape[1]);ys=starts(image.shape[0]);xb=ownership(xs,image.shape[1]);yb=ownership(ys,image.shape[0]);candidates=[];tileinfo=[]
  for iy,y0 in enumerate(ys):
   for ix,x0 in enumerate(xs):
    tile=np.ascontiguousarray(image[y0:y0+SIZE,x0:x0+SIZE]);features=tile_channels(tile,mode)
    if mode=='native_multiscale':features=np.ascontiguousarray(features[::-1]) # match training Format(bgr=0) channel reversal
    tensor=torch.from_numpy(features[None]).to('cuda')
    prediction=model.predict(tensor,imgsz=SIZE,device=0,conf=1e-5,max_det=2000,half=False,augment=False,verbose=False)[0]
    xywh=prediction.boxes.xywh.detach().cpu().numpy();scores=prediction.boxes.conf.detach().cpu().numpy()
    tileinfo.append({'x0':x0,'y0':y0,'candidates':len(scores),'at_max_det':len(scores)>=2000})
    for (x,y,w,h),s in zip(xywh,scores):
     xx,yy=float(x)+x0,float(y)+y0;ox,oy=to_original(xx,yy,meta)
     if xb[ix]<=xx<xb[ix+1] and yb[iy]<=yy<yb[iy+1] and 0<=ox<r['width'] and 0<=oy<r['height']:candidates.append((xx,yy,float(s)))
  kept=center_merge(candidates)
  for x,y,s in kept:
   ox,oy=to_original(x,y,meta);assert math.isfinite(ox+oy+s)
   rows.append({'method':'ANN-YOLO26','dataset':r['dataset'],'split':r['split'],'micrograph_name':r['micrograph_name'],'image_key':key(r),'x_original_px':ox,'y_original_px':oy,'score':s})
  info.append({'dataset':r['dataset'],'micrograph_name':r['micrograph_name'],'normalization':meta,'before_center_merge':len(candidates),'after_center_merge':len(kept),'tiles':tileinfo})
  print('ANN_INFERENCE',mode,i+1,'/',len(rs),r['dataset'],r['micrograph_name'],len(kept),'tiles',len(tileinfo),flush=True)
 csvout(target,rows,['method','dataset','split','micrograph_name','image_key','x_original_px','y_original_px','score'])
 dump(out/'inference.json',{'mode':mode,'checkpoint':str(weight),'checkpoint_sha256':sha(weight),'score_floor':1e-5,'max_det_per_tile':2000,'center_merge_scaled_px':28,'tile_size':SIZE,'minimum_overlap':OVERLAP,'padding':PAD,'micrographs':info,'GT_read_during_model_forward':False,'GPU':0,'elapsed_seconds':time.monotonic()-start})
 dump(done,{'status':'complete','images':len(rs),'mode':mode,'checkpoint_sha256':sha(weight),'predictions_sha256':sha(target),'protocol_sha256':sha(G/'protocol.json')})
 return load_predictions(target)

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--mode',choices=['native','native_multiscale','diameter','diameter_postnorm'],required=True);p.add_argument('--split',choices=['val','probe','heldout'],required=True);p.add_argument('--include-source',action='store_true');p.add_argument('--out',required=True);p.add_argument('--evaluate',action='store_true');a=p.parse_args()
 f=protocol();weight=Path(a.checkpoint or f['base_checkpoint'])
 if not a.checkpoint:assert sha(weight)==f['base_checkpoint_sha256']
 ds=HELDOUT if a.split=='heldout' else DEVELOPMENT
 if a.include_source:assert a.split=='val';ds=['10028']+ds
 rs=records(a.split,ds);pred=infer(weight,rs,a.mode,Path(a.out))
 if a.evaluate:
  assert a.split!='heldout','Heldout metrics use final frozen evaluator only'
  result=evaluate(rs,pred,f['source_fixed_threshold']);dump(Path(a.out)/'fixed_source_threshold_metrics.json',result);print('FIXED_SOURCE_THRESHOLD_METRICS',json.dumps({k:v for k,v in result.items() if k!='per_image'}),flush=True)
if __name__=='__main__':main()
