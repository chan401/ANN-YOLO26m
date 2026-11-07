"""GT-free forward pass using exactly the training-time float preprocessing."""
import time
from common import *
from ultralytics import YOLO
FIELDS=['method','dataset','split','micrograph_name','image_key','x_original_px','y_original_px','score']

def infer(weight,rs,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 if (out/'done.json').exists():
  d=json.loads((out/'done.json').read_text());assert d['checkpoint_sha256']==sha(weight) and d['prediction_sha256']==sha(out/'predictions.csv')
  return load_predictions(out/'predictions.csv')
 assert not(out/'predictions.csv').exists(),'Incomplete prediction output requires inspection'
 model=YOLO(str(weight));model.model.model[-1].max_det=2000;torch.set_num_threads(4);cv2.setNumThreads(1)
 rows=[];details=[];beg=time.monotonic()
 for j,r in enumerate(rs):
  img,g=normalize(r);xs,ys=starts(img.shape[1]),starts(img.shape[0]);xb,yb=ownership(xs,img.shape[1]),ownership(ys,img.shape[0]);cand=[];tileinfo=[]
  for iy,y in enumerate(ys):
   for ix,x in enumerate(xs):
    tile=img[y:y+SIZE,x:x+SIZE];tensor=torch.from_numpy(np.ascontiguousarray(np.repeat(tile[None,None],3,axis=1))).to('cuda')
    pp=model.predict(tensor,imgsz=SIZE,device=0,conf=.001,max_det=2000,half=False,augment=False,verbose=False)[0]
    xy=pp.boxes.xywh.detach().cpu().numpy();sc=pp.boxes.conf.detach().cpu().numpy();tileinfo.append(dict(x0=x,y0=y,candidates=len(sc),at_max_det=len(sc)>=2000))
    for (xx,yy,w,h),score in zip(xy,sc):
     xx,yy=float(xx)+x,float(yy)+y;ox,oy=to_original(xx,yy,g)
     if xb[ix]<=xx<xb[ix+1] and yb[iy]<=yy<yb[iy+1] and 0<=ox<r['width'] and 0<=oy<r['height']:cand.append((xx,yy,float(score)))
  kept=[];grid={};rad=PARTICLE/8
  for x,y,s in sorted(cand,key=lambda p:(-p[2],p[0],p[1])):
   gx,gy=int(x//rad),int(y//rad);near=[p for xx in range(gx-1,gx+2) for yy in range(gy-1,gy+2) for p in grid.get((xx,yy),[])]
   if all((x-a)**2+(y-b)**2>rad**2 for a,b in near):kept.append((x,y,s));grid.setdefault((gx,gy),[]).append((x,y))
  for x,y,s in kept:
   ox,oy=to_original(x,y,g);assert math.isfinite(ox+oy+s)
   rows.append(dict(method='ANN_v2',dataset=r['dataset'],split=r['split'],micrograph_name=r['micrograph_name'],image_key=key(r),x_original_px=ox,y_original_px=oy,score=s))
  details.append(dict(dataset=r['dataset'],micrograph_name=r['micrograph_name'],preprocess=g,before_merge=len(cand),after_merge=len(kept),tiles=tileinfo))
  print('ANN_V2_FORWARD',j+1,'/',len(rs),r['dataset'],r['micrograph_name'],len(kept),flush=True)
 csvout(out/'predictions.csv',rows,FIELDS)
 dump(out/'inference.json',dict(profile=PROFILE,checkpoint_sha256=sha(weight),GT_read_during_forward=False,score_floor=.001,max_det=2000,canonical_merge_radius=PARTICLE/8,tile_size=SIZE,overlap=OVERLAP,pad=PAD,elapsed_seconds=time.monotonic()-beg,micrographs=details))
 dump(out/'done.json',dict(status='PASS',images=len(rs),checkpoint_sha256=sha(weight),prediction_sha256=sha(out/'predictions.csv')))
 return load_predictions(out/'predictions.csv')
