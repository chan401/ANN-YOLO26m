"""Balanced seven-domain scale-normalized float32 training with official YOLO26m architecture."""
import argparse,importlib.util,random,shutil
from common import *
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import Compose,Format,RandomFlip,RandomPerspective
from ultralytics.models.yolo.detect import DetectionTrainer
spec=importlib.util.spec_from_file_location('source_adapters',D/'experiments/ann_yolo26m_cryo_tiles_v3/code/common.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
REGISTRY={};TILES={};SHARED={}

class Photometric:
 def __call__(self,labels):
  x=labels['img']/255.;x=(x-.5)*random.uniform(.85,1.15)+.5+random.uniform(-.025,.025)
  if random.random()<.5:x=x+np.random.normal(0,random.uniform(0,.01),x.shape[:2]+(1,)).astype(np.float32)
  labels['img']=np.clip(x,0,1).astype(np.float32)*255.;return labels

class Dataset(YOLODataset):
 def load_image(self,i,rect_mode=True):
  t=TILES[Path(self.im_files[i]).stem];a=SHARED[t['image_key']];x,y=t['x0'],t['y0'];tile=a[y:y+SIZE,x:x+SIZE]
  assert tile.shape==(SIZE,SIZE) and tile.dtype==np.float32
  return np.ascontiguousarray(np.repeat((tile*255)[...,None],3,axis=2)),(SIZE,SIZE),(SIZE,SIZE)
 def build_transforms(self,hyp=None):
  ts=[old.QuarterTurn(),RandomPerspective(degrees=0,translate=.04,scale=.20,shear=0,perspective=0,border=(0,0)),RandomFlip(.5,'horizontal'),RandomFlip(.5,'vertical'),Photometric()] if self.augment else []
  ts.append(Format(bbox_format='xywh',normalize=True,batch_idx=True,bgr=0));return Compose(ts)

class Trainer(DetectionTrainer):
 def build_dataset(self,img_path,mode='train',batch=None):
  return Dataset(img_path=img_path,imgsz=SIZE,batch_size=batch,augment=mode=='train',hyp=self.args,rect=False,cache=False,single_cls=False,stride=32,pad=0,prefix=mode+': ',task='detect',classes=None,data=self.data,fraction=1)
 def get_model(self,cfg=None,weights=None,verbose=True):
  m=super().get_model(cfg,weights,verbose);m.model[-1].max_det=2000;return m

def main():
 global REGISTRY,TILES,SHARED
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--epochs',type=int,default=40);p.add_argument('--name',default='canonical64_lowpass1_7domains');p.add_argument('--weights');p.add_argument('--lr',type=float,default=.0003);a=p.parse_args()
 root=G/'training_data';REGISTRY={key(r):r for r in records('train')+records('val')};TILES=json.loads((root/'tile_registry.json').read_text())
 name='smoke_v2' if a.smoke else a.name;out=G/'runs'/name;assert not out.exists()
 weight=Path(a.weights) if a.weights else V1/'runs/balanced_four_accession_v1/weights/best.pt'
 if not a.weights:assert sha(weight)=='8b0a5d2fadea5010202b3392cb6c740dd980f55075457a28ac69b3d167609039'
 needed={TILES[Path(p).stem]['image_key'] for split in ['train','val'] for p in (root/('smoke_'+split+'.txt' if a.smoke else split+'.txt')).read_text().splitlines()}
 estimate=sum(geometry(REGISTRY[k])['padded_width']*geometry(REGISTRY[k])['padded_height']*4 for k in needed)
 limit=int(Path('/sys/fs/cgroup/memory.max').read_text());assert estimate<limit*.55,(estimate,limit)
 torch.set_num_threads(8);cv2.setNumThreads(1);torch.multiprocessing.set_start_method('fork',force=True)
 for i,k in enumerate(sorted(needed)):
  arr,meta=normalize(REGISTRY[k]);arr.flags.writeable=False;SHARED[k]=arr
  if (i+1)%40==0:print('SHARED_FLOAT_CACHE',i+1,'/',len(needed),flush=True)
 print('FLOAT_CACHE_READY',len(SHARED),'GiB',round(estimate/1024**3,2),flush=True)
 cfg=dict(data=str(root/('smoke.yaml' if a.smoke else 'data.yaml')),imgsz=SIZE,epochs=1 if a.smoke else a.epochs,batch=7 if a.smoke else 8,device=0,workers=0 if a.smoke else 4,project=str(G/'runs'),name=name,exist_ok=False,
  pretrained=True,optimizer='AdamW',lr0=a.lr,lrf=.05,weight_decay=.0005,warmup_epochs=.1 if a.smoke else 3,warmup_bias_lr=.001,cos_lr=True,nbs=16,patience=20,amp=True,cache=False,rect=False,multi_scale=0,max_det=2000,conf=.0001,
  seed=100282,deterministic=True,save=True,save_period=-1,plots=False,mosaic=0,close_mosaic=0,mixup=0,cutmix=0,copy_paste=0,hsv_h=0,hsv_s=0,hsv_v=0,degrees=0,translate=0,scale=0,shear=0,perspective=0,fliplr=0,flipud=0,val=True)
 snapshot=G/'code_versions'/name;snapshot.mkdir(parents=True,exist_ok=True)
 for p in (G/'code').glob('*.py'):shutil.copy2(p,snapshot/p.name)
 dump(G/(name+'_config.json'),dict(training=cfg,initial_checkpoint=str(weight),initial_sha256=sha(weight),profile=PROFILE,canonical_diameter=PARTICLE,lowpass_sigma=SIGMA,training_accessions=DOMAINS,runtime_run_root=str(G),
  rationale='Seven source domains, one normalized 64px particle scale and low-pass signal enhancement; float post-resize normalization consistent in train/inference. Combined experiment; do not attribute gains to a single component.',
  target_accessions_not_trained=TARGETS,code_sha256={p.name:sha(p) for p in (G/'code').glob('*.py')},registry_sha256=sha(G/'image_registry.json'),tile_registry_sha256=sha(root/'tile_registry.json'),cache_bytes=estimate))
 os.chdir(ULTRA);model=YOLO(str(weight));assert model.model.model[-1].stride.tolist()==[8.,16.,32.]
 result=model.train(trainer=Trainer,**cfg);save=Path(result.save_dir);rows=list(csv.DictReader((save/'results.csv').open()))
 assert rows
 for row in rows:
  for k,v in row.items():
   if 'loss' in k:assert math.isfinite(float(v))
 dump(save/'completion.json',dict(status='PASS',epochs=len(rows),best_sha256=sha(save/'weights/best.pt'),last_sha256=sha(save/'weights/last.pt'),smoke=a.smoke))
 print('V2_TRAINING_COMPLETE',name,len(rows),flush=True)

if __name__=='__main__':main()
