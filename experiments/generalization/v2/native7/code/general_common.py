"""Variable geometry/diameter adapters. Published diameter and pixel geometry only; no GT-based adaptation."""
from pathlib import Path
import csv,json,hashlib,math,os,sys,functools
import cv2,mrcfile,numpy as np,torch
G=Path('/root/autodl-tmp/cryoem_generalization/v1');D=Path('/root/autodl-tmp/cryoem_datasets/EMPIAR-10028')
ULTRA=Path('/root/autodl-tmp/cryoem_annyolo26/SNNYOLO26/ultralytics-snn-yolo26');sys.path.insert(0,str(ULTRA))
SIZE=1536;OVERLAP=256;PAD=128
DEVELOPMENT=['10017','10077','10291'];HELDOUT=['10075','10081','10059']
sys.path.insert(0,str(D/'metadata/three_method_evaluation'))
from evaluate_empiar10028_v3_pretrained_val import Point,match_one_image,metric_counts

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');os.replace(tmp,p)
def csvout(p,rows,fields=None):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def protocol():
 p=G/'protocol.json';assert sha(p)==(G/'protocol.sha256').read_text().strip();return json.loads(p.read_text())
def key(r):return r['dataset']+'__'+hashlib.sha256(r['micrograph_name'].encode()).hexdigest()[:20]
def records(split,datasets=None):
 datasets=datasets or DEVELOPMENT
 if split=='heldout':assert (G/'heldout_frozen_config.json').exists(),'Heldout access requires final checkpoint/threshold freeze'
 if (G/'all_images.csv').exists():allrows=list(csv.DictReader((G/'all_images.csv').open()))
 else:
  allrows=[]
  for r in protocol()['files']:
   if r['split']==split and r['dataset'] in datasets:
    p=G/'datasets'/r['dataset']/'file_records'/(r['micrograph_name']+'.json');assert p.exists(),f'Data not validated yet: {p}';allrows.append(json.loads(p.read_text()))
 result=[r for r in allrows if r['split']==split and r['dataset'] in datasets]
 for r in result:
  for field in ['width','height','mode','GT_count','diameter','bytes']:r[field]=int(r[field])
 if '10028' in datasets:
  assert split in ['train','val'];ns=[Path(x.strip()).name for x in (D/'manifests/split_v3_paper60_20_20'/f'{split}.txt').read_text().splitlines() if x.strip()]
  for n in ns:
   path=D/'micrographs'/n;gt=D/'annotations_original/cryoppp_10028/10028/ground_truth/particle_coordinates'/(Path(n).stem+'.csv')
   result.append({'dataset':'10028','split':split,'micrograph_name':n,'path':str(path),'GT_path':str(gt),'width':4096,'height':4096,'mode':2,'diameter':224,'GT_count':sum(1 for _ in csv.DictReader(gt.open()))})
 assert result and len({key(r) for r in result})==len(result)
 if split in ['train','val','probe']:assert not {r['dataset'] for r in result}&set(HELDOUT)
 return result
def gt_points(r,allow_heldout=False):
 if r['dataset'] in HELDOUT:assert allow_heldout and (G/'heldout_frozen_config.json').exists()
 points=[]
 with Path(r['GT_path']).open(newline='',encoding='utf-8-sig') as h:
  for i,row in enumerate(csv.DictReader(h)):
   x,y=float(row['X-Coordinate']),float(row['Y-Coordinate']);assert 0<=x<r['width'] and 0<=y<r['height'];points.append(Point(x,y,1.,i))
 assert len(points)==r['GT_count'];return points
def geometry(r,mode):
 nominal=224.0/r['diameter'] if mode.startswith('diameter') else 1.
 w=max(1,round(r['width']*nominal));h=max(1,round(r['height']*nominal))
 return {'scaled_width':w,'scaled_height':h,'sx':w/r['width'],'sy':h/r['height'],'padded_width':max(SIZE,w+2*PAD),'padded_height':max(SIZE,h+2*PAD)}
def starts(length):
 n=max(1,math.ceil((length-SIZE)/(SIZE-OVERLAP))+1)
 return [round(float(v)) for v in np.linspace(0,max(0,length-SIZE),n)]
def ownership(ss,length):return [0.]+[(a+SIZE+b)/2 for a,b in zip(ss[:-1],ss[1:])]+[float(length)]
def normalize(r,mode):
 with mrcfile.open(r['path'],permissive=True) as m:
  a=np.array(m.data,copy=True)
  unsigned=r['dataset']=='10059' and int(m.header.mode)==0
  if unsigned:a=a.view(np.uint8)
  a=a.astype(np.float32)
 assert a.shape==(r['height'],r['width']) and np.isfinite(a).all()
 q16,median,q84=np.percentile(a,[16,50,84]);sigma=float((q84-q16)/2);assert sigma>0
 a=np.clip((a-np.float32(median))/(10*np.float32(sigma))+.5,0,1).astype(np.float32)
 a=np.ascontiguousarray(np.flipud(a));g=geometry(r,mode)
 resized=(g['scaled_height'],g['scaled_width'])!=a.shape;postnorm=None
 if resized:a=cv2.resize(a,(g['scaled_width'],g['scaled_height']),interpolation=cv2.INTER_LINEAR)
 if mode=='diameter_postnorm' and resized:
  q16b,medb,q84b=np.percentile(a,[16,50,84]);sigmab=float((q84b-q16b)/2);assert sigmab>0
  a=np.clip((a-np.float32(medb))/(10*np.float32(sigmab))+.5,0,1).astype(np.float32)
  postnorm={'median':float(medb),'robust_sigma':sigmab,'reason':'resampling alters noise/intensity spread; restore the same robust standardization after resize; source scale1 remains exactly unchanged'}
 a=cv2.copyMakeBorder(a,PAD,g['padded_height']-a.shape[0]-PAD,PAD,g['padded_width']-a.shape[1]-PAD,cv2.BORDER_REFLECT_101)
 assert a.shape==(g['padded_height'],g['padded_width']) and a.dtype==np.float32
 return a,{**g,'median':float(median),'robust_sigma':sigma,'post_resize_normalization':postnorm,'unsigned_mode0_reinterpreted_per_EMPIAR_metadata':unsigned,'orientation':'flipud to CryoPPP before resize; pixel-center resize x_new=(x+.5)*sx-.5; reflect pad128','normalization':'source median,(P84-P16)/2 sigma; clip+/-5sigma; float32'}
def to_scaled(x,y,g):return ((x+.5)*g['sx']-.5+PAD,(y+.5)*g['sy']-.5+PAD)
def to_original(x,y,g):return ((x-PAD+.5)/g['sx']-.5,(y-PAD+.5)/g['sy']-.5)
def center_merge(candidates):
 keep=[];grid={}
 for x,y,s in sorted(candidates,key=lambda v:(-v[2],v[0],v[1])):
  gx,gy=int(x//28),int(y//28);near=[p for xx in range(gx-1,gx+2) for yy in range(gy-1,gy+2) for p in grid.get((xx,yy),[])]
  if all((x-a)**2+(y-b)**2>28**2 for a,b in near):keep.append((x,y,s));grid.setdefault((gx,gy),[]).append((x,y))
 return keep
def tile_channels(tile,mode):
 if mode!='native_multiscale':return np.repeat(tile[None],3,axis=0)
 channels=[tile]
 for sigma in [2.,4.]:
  filtered=cv2.GaussianBlur(tile,(0,0),sigmaX=sigma,sigmaY=sigma,borderType=cv2.BORDER_REFLECT_101)
  q16,med,q84=np.percentile(filtered,[16,50,84]);spread=float((q84-q16)/2)
  assert spread>0
  channels.append(np.clip((filtered-np.float32(med))/(10*np.float32(spread))+.5,0,1).astype(np.float32))
 return np.stack(channels,axis=0)
def evaluate(rs,predictions,threshold,allow_heldout=False):
 per=[];summed={}
 for r in rs:
  name=key(r);gt=gt_points(r,allow_heldout);pp=[p for p in predictions.get(name,[]) if p.score>=threshold]
  pair,_=match_one_image(gt,pp,radius=r['diameter']/4.)
  tp=len(pair);fp=len(pp)-tp;fn=len(gt)-tp;m=metric_counts(tp,fp,fn)
  per.append({'dataset':r['dataset'],'micrograph_name':r['micrograph_name'],'GT':len(gt),'Predictions':len(pp),'radius_px':r['diameter']/4.,**m})
  sums=summed.setdefault(r['dataset'],[0,0,0]);sums[0]+=tp;sums[1]+=fp;sums[2]+=fn
 summary={d:metric_counts(*v) for d,v in summed.items()};macro={k:float(np.mean([m[k] for m in summary.values()])) for k in ['Precision','Recall','F1']}
 pooled=metric_counts(*[sum(v[i] for v in summed.values()) for i in range(3)])
 return {'per_image':per,'datasets':summary,'macro':macro,'pooled':pooled,'threshold':threshold}
def load_predictions(p):
 out={}
 with Path(p).open(newline='') as h:
  for i,r in enumerate(csv.DictReader(h)):
   out.setdefault(r['image_key'],[]).append(Point(float(r['x_original_px']),float(r['y_original_px']),float(r['score']),i))
 return out
