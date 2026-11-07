"""V2 float32, particle-scale adapters and unchanged one-to-one evaluation."""
from pathlib import Path
import csv,functools,hashlib,json,math,os,sys
import cv2,mrcfile,numpy as np,torch
G=Path(os.environ.get('ANN_GENERALIZATION_RUN_ROOT','/root/autodl-tmp/cryoem_generalization/v2'))
V1=Path('/root/autodl-tmp/cryoem_generalization/v1')
D=Path('/root/autodl-tmp/cryoem_datasets/EMPIAR-10028')
ULTRA=Path('/root/autodl-tmp/cryoem_annyolo26/SNNYOLO26/ultralytics-snn-yolo26')
sys.path.insert(0,str(ULTRA));sys.path.insert(0,str(D/'metadata/three_method_evaluation'))
from evaluate_empiar10028_v3_pretrained_val import Point,match_one_image,metric_counts,_edges
SIZE=1024;PAD=96;OVERLAP=256;PARTICLE=64.;SIGMA=1.
DOMAINS=os.environ.get('ANN_GENERALIZATION_DOMAINS','10028,10017,10077,10291,10289,10240,10760').split(',');TARGETS=['10081','10059']
PROFILE='canonical64_lowpass1'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def dump(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');os.replace(tmp,p)

def csvout(p,rows,fields=None):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)

def key(r):return r['dataset']+'__'+hashlib.sha256(r['micrograph_name'].encode()).hexdigest()[:20]

def records(split):
 rows=json.loads((G/'image_registry.json').read_text())
 rs=[r for r in rows if r['split']==split]
 assert rs
 if split in ['train','val']:assert set(r['dataset'] for r in rs)<=set(DOMAINS) and not set(r['dataset'] for r in rs)&set(TARGETS)
 if split=='final_check':assert (G/'final_frozen_config.json').exists(),'Freeze before target inference'
 return rs

def points(r):
 if r['dataset'] in TARGETS:assert (G/'final_frozen_config.json').exists(),'No target-GT evaluation before freeze'
 with Path(r['GT_path']).open(encoding='utf-8-sig') as f:
  pp=[Point(float(v['X-Coordinate']),float(v['Y-Coordinate']),1.,i) for i,v in enumerate(csv.DictReader(f))]
 assert len(pp)==r['GT_count'];assert all(0<=p.x<r['width'] and 0<=p.y<r['height'] for p in pp)
 return pp

def starts(length):
 return [round(float(x)) for x in np.linspace(0,max(0,length-SIZE),max(1,math.ceil((length-SIZE)/(SIZE-OVERLAP))+1))]

def ownership(ss,length):return [0.]+[(a+SIZE+b)/2 for a,b in zip(ss[:-1],ss[1:])]+[float(length)]

def geometry(r):
 w,h=max(1,round(r['width']*PARTICLE/r['diameter'])),max(1,round(r['height']*PARTICLE/r['diameter']))
 return dict(scaled_width=w,scaled_height=h,sx=w/r['width'],sy=h/r['height'],padded_width=max(SIZE,w+2*PAD),padded_height=max(SIZE,h+2*PAD))

def to_scaled(x,y,g):return ((x+.5)*g['sx']-.5+PAD,(y+.5)*g['sy']-.5+PAD)
def to_original(x,y,g):return ((x-PAD+.5)/g['sx']-.5,(y-PAD+.5)/g['sy']-.5)

def normalize(r):
 with mrcfile.open(r['path'],permissive=True) as m:
  a=np.array(m.data,copy=True)
  if r['dataset']=='10059' and int(m.header.mode)==0:a=a.view(np.uint8)
  a=a.astype(np.float32)
 assert a.shape==(r['height'],r['width']) and np.isfinite(a).all()
 # Protect interpolation from corrupt finite spikes using a fixed image-only rule.
 q16,med,q84=np.percentile(a,[16,50,84]);spread=float((q84-q16)/2);assert spread>0
 a=np.clip(a,med-8*spread,med+8*spread);a=np.ascontiguousarray(np.flipud(a));g=geometry(r)
 interpolation=cv2.INTER_AREA if g['sx']<=1 and g['sy']<=1 else cv2.INTER_CUBIC
 a=cv2.resize(a,(g['scaled_width'],g['scaled_height']),interpolation=interpolation)
 a=cv2.GaussianBlur(a,(0,0),sigmaX=SIGMA,sigmaY=SIGMA,borderType=cv2.BORDER_REFLECT_101)
 q16,med,q84=np.percentile(a,[16,50,84]);spread=float((q84-q16)/2);assert spread>0
 a=np.clip((a-np.float32(med))/(10*np.float32(spread))+.5,0,1).astype(np.float32)
 a=cv2.copyMakeBorder(a,PAD,g['padded_height']-a.shape[0]-PAD,PAD,g['padded_width']-a.shape[1]-PAD,cv2.BORDER_REFLECT_101)
 assert a.shape==(g['padded_height'],g['padded_width']) and a.dtype==np.float32
 return a,{**g,'post_resampling_median':float(med),'post_resampling_sigma':spread,'profile':PROFILE}

def evaluate(rs,pred,threshold):
 per=[];counts={}
 for r in rs:
  gt=points(r);pp=[p for p in pred.get(key(r),[]) if p.score>=threshold]
  pairs,_=match_one_image(gt,pp,radius=r['diameter']/4.)
  tp=len(pairs);fp=len(pp)-tp;fn=len(gt)-tp
  per.append(dict(dataset=r['dataset'],micrograph_name=r['micrograph_name'],GT=len(gt),Predictions=len(pp),radius_px=r['diameter']/4.,**metric_counts(tp,fp,fn)))
  c=counts.setdefault(r['dataset'],[0,0,0]);c[0]+=tp;c[1]+=fp;c[2]+=fn
 ds={d:metric_counts(*c) for d,c in counts.items()}
 return dict(threshold=threshold,datasets=ds,macro={k:float(np.mean([v[k] for v in ds.values()])) for k in ['Precision','Recall','F1']},pooled=metric_counts(*[sum(c[i] for c in counts.values()) for i in range(3)]),per_image=per)

def load_predictions(path):
 out={}
 with Path(path).open() as f:
  for i,r in enumerate(csv.DictReader(f)):out.setdefault(r['image_key'],[]).append(Point(float(r['x_original_px']),float(r['y_original_px']),float(r['score']),i))
 return out
