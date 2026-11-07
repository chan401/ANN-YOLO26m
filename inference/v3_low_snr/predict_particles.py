"""Standalone native-MRC particle picking. No labels, GT or benchmark manifest required.

Example: python predict_particles.py --config model_config.json --weights best.pt --input /path/to/micrographs --output new_picks
Uses the Ultralytics runtime/source version saved with the model package.
"""
from pathlib import Path
import argparse,csv,hashlib,json,math,time
import cv2,mrcfile,numpy as np,torch
from ultralytics import YOLO
SIZE=1536;OVERLAP=256;PAD=128;MERGE=28;SCORE_FLOOR=1e-5

def configure(profile):
 global SIZE,PAD,MERGE,SCORE_FLOOR
 if profile=='canonical64_lowpass1':SIZE,PAD,MERGE,SCORE_FLOOR=1024,96,8,.001
 elif profile in ['native','native_multiscale']:SIZE,PAD,MERGE,SCORE_FLOOR=1536,128,28,1e-5
 else:raise ValueError('Unsupported input profile: '+str(profile))

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def normalize(path,unsigned_mode0=False,profile='native_multiscale',diameter=None):
 with mrcfile.open(path,permissive=True) as m:
  a=np.array(m.data,copy=True)
  if a.ndim==3 and a.shape[0]==1:a=a[0]
  if a.ndim!=2:raise ValueError('Expected one2D micrograph, not a movie/volume: '+str(path))
  if int(m.header.mode)==0 and unsigned_mode0:a=a.view(np.uint8)
  a=a.astype(np.float32)
 if not np.isfinite(a).all():raise ValueError('Nonfinite input pixels: '+str(path))
 h,w=a.shape;q16,median,q84=np.percentile(a,[16,50,84]);sigma=float((q84-q16)/2)
 if sigma<=0:raise ValueError('Degenerate input intensity spread: '+str(path))
 sx=sy=1.
 if profile=='canonical64_lowpass1':
  if diameter is None or diameter<=0:raise ValueError('Canonical64 inference requires the previously specified native particle diameter via --diameter')
  a=np.clip(a,median-8*sigma,median+8*sigma);a=np.ascontiguousarray(np.flipud(a))
  sw,sh=max(1,round(w*64/diameter)),max(1,round(h*64/diameter));sx,sy=sw/w,sh/h
  a=cv2.resize(a,(sw,sh),interpolation=cv2.INTER_AREA if sx<=1 and sy<=1 else cv2.INTER_CUBIC)
  a=cv2.GaussianBlur(a,(0,0),sigmaX=1.,sigmaY=1.,borderType=cv2.BORDER_REFLECT_101)
  q16,median,q84=np.percentile(a,[16,50,84]);sigma=float((q84-q16)/2)
  if sigma<=0:raise ValueError('Degenerate filtered intensity spread')
  a=np.clip((a-np.float32(median))/(10*np.float32(sigma))+.5,0,1).astype(np.float32)
 else:
  a=np.clip((a-np.float32(median))/(10*np.float32(sigma))+.5,0,1).astype(np.float32);a=np.ascontiguousarray(np.flipud(a))
 ah,aw=a.shape;a=cv2.copyMakeBorder(a,PAD,max(SIZE,ah+2*PAD)-ah-PAD,PAD,max(SIZE,aw+2*PAD)-aw-PAD,cv2.BORDER_REFLECT_101)
 return a,{'width':w,'height':h,'sx':sx,'sy':sy,'median':float(median),'robust_sigma':sigma,'unsigned_mode0':unsigned_mode0,'profile':profile}

def restore(x,y,meta):return ((x-PAD+.5)/meta['sx']-.5,(y-PAD+.5)/meta['sy']-.5)
def starts(length):
 n=max(1,math.ceil((length-SIZE)/(SIZE-OVERLAP))+1)
 return [round(float(v)) for v in np.linspace(0,max(0,length-SIZE),n)]
def bounds(ss,length):return [0.]+[(a+SIZE+b)/2 for a,b in zip(ss[:-1],ss[1:])]+[float(length)]
def merge(candidates):
 keep=[];grid={}
 for x,y,s in sorted(candidates,key=lambda v:(-v[2],v[0],v[1])):
  gx,gy=int(x//MERGE),int(y//MERGE)
  near=[p for xx in range(gx-1,gx+2) for yy in range(gy-1,gy+2) for p in grid.get((xx,yy),[])]
  if all((x-a)**2+(y-b)**2>MERGE**2 for a,b in near):keep.append((x,y,s));grid.setdefault((gx,gy),[]).append((x,y))
 return keep
def channels(tile,profile):
 if profile!='native_multiscale':return np.repeat(tile[None],3,axis=0)
 values=[tile]
 for sigma in [2.,4.]:
  filtered=cv2.GaussianBlur(tile,(0,0),sigmaX=sigma,sigmaY=sigma,borderType=cv2.BORDER_REFLECT_101);q16,med,q84=np.percentile(filtered,[16,50,84]);spread=float((q84-q16)/2)
  if spread<=0:raise ValueError('Degenerate low-frequency channel')
  values.append(np.clip((filtered-np.float32(med))/(10*np.float32(spread))+.5,0,1).astype(np.float32))
 return np.ascontiguousarray(np.stack(values,axis=0)[::-1]) # training Format(bgr=0) delivers LP4,LP2,raw
def predict(model,path,threshold,unsigned_mode0=False,device=0,profile='native_multiscale',diameter=None):
 configure(profile)
 image,meta=normalize(path,unsigned_mode0,profile,diameter);xs=starts(image.shape[1]);ys=starts(image.shape[0]);xb=bounds(xs,image.shape[1]);yb=bounds(ys,image.shape[0]);candidates=[];tile_counts=[]
 for iy,y0 in enumerate(ys):
  for ix,x0 in enumerate(xs):
   tile=np.ascontiguousarray(image[y0:y0+SIZE,x0:x0+SIZE]);tensor=torch.from_numpy(channels(tile,profile)[None]).to('cuda:'+str(device))
   r=model.predict(tensor,imgsz=SIZE,device=device,conf=SCORE_FLOOR,max_det=2000,half=False,augment=False,verbose=False)[0]
   xywh=r.boxes.xywh.detach().cpu().numpy();scores=r.boxes.conf.detach().cpu().numpy();tile_counts.append(len(scores))
   for (x,y,w,h),s in zip(xywh,scores):
    xx,yy=float(x)+x0,float(y)+y0;ox,oy=restore(xx,yy,meta)
    if xb[ix]<=xx<xb[ix+1] and yb[iy]<=yy<yb[iy+1] and 0<=ox<meta['width'] and 0<=oy<meta['height']:candidates.append((xx,yy,float(s)))
 kept=merge(candidates);rows=[]
 for x,y,s in kept:
  if s<threshold:continue
  cx,cy=restore(x,y,meta)
  rows.append({'micrograph_name':path.name,'x_mrc_array_px':cx,'y_mrc_array_px':meta['height']-1-cy,'score':s,'x_cryoppp_px':cx,'y_cryoppp_px':cy})
 return rows,{**meta,'tiles':len(tile_counts),'tiles_at_max_det':sum(n>=2000 for n in tile_counts),'raw_candidates':len(candidates),'after_center_merge':len(kept),'retained_picks':len(rows)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--weights',required=True);p.add_argument('--input',nargs='+',required=True);p.add_argument('--output',required=True);p.add_argument('--device',type=int,default=0);p.add_argument('--unsigned-mode0',action='store_true');p.add_argument('--diameter',type=float);a=p.parse_args()
 cfg=json.loads(Path(a.config).read_text(encoding='utf-8'));weight=Path(a.weights)
 if sha(weight)!=cfg['checkpoint_sha256']:raise ValueError('Checkpoint/config SHA256 mismatch')
 threshold=float(cfg.get('global_threshold',cfg.get('threshold')));profile=cfg.get('input_profile',cfg.get('profile'));configure(profile);paths=[]
 for value in a.input:
  q=Path(value)
  if q.is_dir():paths.extend(sorted(x for x in q.iterdir() if x.is_file() and x.suffix.lower()=='.mrc'))
  elif q.is_file():paths.append(q)
  else:raise FileNotFoundError(q)
 if not paths:raise ValueError('No MRC input files')
 if len({p.name for p in paths})!=len(paths):raise ValueError('Duplicate input basenames; use separate output directories')
 out=Path(a.output)
 if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new empty output directory')
 out.mkdir(parents=True,exist_ok=True);(out/'coordinates_mrc_array').mkdir(exist_ok=True)
 if not torch.cuda.is_available():raise RuntimeError('This frozen inference configuration requires an available CUDA GPU')
 torch.set_num_threads(4);model=YOLO(str(weight));model.model.model[-1].max_det=2000;allrows=[];infos=[];start=time.monotonic()
 for i,path in enumerate(paths):
  rows,meta=predict(model,path,threshold,a.unsigned_mode0,a.device,profile,a.diameter);allrows.extend(rows);infos.append({'micrograph_name':path.name,'source_path':str(path.resolve()),'source_sha256':sha(path),**meta})
  with (out/'coordinates_mrc_array'/(path.stem+'.txt')).open('w') as h:
   for r in rows:h.write(f'{r["x_mrc_array_px"]}\t{r["y_mrc_array_px"]}\n')
  print('PICKED',i+1,'/',len(paths),path.name,len(rows),flush=True)
 fields=['micrograph_name','x_mrc_array_px','y_mrc_array_px','score','x_cryoppp_px','y_cryoppp_px']
 with (out/'predictions.csv').open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(allrows)
 report={'images':len(paths),'picks':len(allrows),'checkpoint_sha256':sha(weight),'global_threshold':threshold,'input_profile':profile,'GT_required':False,'micrographs':infos,'elapsed_seconds':time.monotonic()-start,
  'coordinates':'MRC-array TXT: zero-based x=column,y=row; source MRC unmodified. CSV also includes CryoPPP-y coordinates for traceability.',
  'preprocessing':{'profile':profile,'float32':True,'tile_size':SIZE,'overlap':OVERLAP,'padding':PAD,'merge_radius':MERGE,'native_diameter_hint':a.diameter,'max_det':2000,'score_floor':SCORE_FLOOR,'precision':'GPU FP32'}}
 (out/'inference_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 print('COMPLETE',out,flush=True)
if __name__=='__main__':main()
