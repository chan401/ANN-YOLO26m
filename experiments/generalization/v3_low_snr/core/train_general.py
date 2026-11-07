"""Low-LR multi-accession fine-tuning with lossless float grayscale virtual tiles."""
import argparse,functools
from general_common import *
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import Compose,Format,RandomFlip
from ultralytics.models.yolo.detect import DetectionTrainer
import importlib.util
SOURCE_CODE=D/'experiments/ann_yolo26m_cryo_tiles_v3/code/common.py'
spec=importlib.util.spec_from_file_location('source_adapters',SOURCE_CODE);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
REGISTRY={};TILES={};SHARED_IMAGES={};TRAIN_MODE='native'

@functools.lru_cache(maxsize=2)
def cached_image(image_key):
 if image_key in SHARED_IMAGES:return SHARED_IMAGES[image_key]
 return normalize(REGISTRY[image_key],TRAIN_MODE)[0]

class GeneralDataset(YOLODataset):
 def load_image(self,i,rect_mode=True):
  t=TILES[Path(self.im_files[i]).stem];a=cached_image(t['image_key']);x0,y0=t['x0'],t['y0'];tile=a[y0:y0+SIZE,x0:x0+SIZE]
  assert tile.shape==(SIZE,SIZE) and tile.dtype==np.float32
  if TRAIN_MODE=='native_multiscale' and self.augment:
   raw_rgb=np.repeat((tile*255.)[...,None],3,axis=2);tile=old.GrayPhotometric()({'img':raw_rgb})['img'][...,0]/255.
  x=np.ascontiguousarray(tile_channels(tile,TRAIN_MODE).transpose(1,2,0)*255.);return x,(SIZE,SIZE),(SIZE,SIZE)
 def build_transforms(self,hyp=None):
  ts=[old.QuarterTurn(),RandomFlip(.5,'horizontal'),RandomFlip(.5,'vertical')] if self.augment else []
  if self.augment and TRAIN_MODE!='native_multiscale':ts.append(old.GrayPhotometric())
  ts.append(Format(bbox_format='xywh',normalize=True,batch_idx=True,bgr=0.));return Compose(ts)

class GeneralTrainer(DetectionTrainer):
 def build_dataset(self,img_path,mode='train',batch=None):
  c=self.args
  return GeneralDataset(img_path=img_path,imgsz=SIZE,batch_size=batch,augment=mode=='train',hyp=c,rect=False,cache=False,single_cls=False,stride=32,pad=0.,prefix=mode+': ',task='detect',classes=None,data=self.data,fraction=1.)
 def get_model(self,cfg=None,weights=None,verbose=True):
  m=super().get_model(cfg,weights,verbose);m.model[-1].max_det=2000;return m

def main():
 global REGISTRY,TILES,SHARED_IMAGES,TRAIN_MODE
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--epochs',type=int,default=30);p.add_argument('--lr',type=float,default=.0002);p.add_argument('--name',default='balanced_four_accession_v1');p.add_argument('--weights');p.add_argument('--mode',choices=['native','native_multiscale']);a=p.parse_args()
 f=protocol();root=G/'training_data';TRAIN_MODE=a.mode or json.loads((root/'preparation.json').read_text())['mode'];REGISTRY=json.loads((root/'image_registry.json').read_text());TILES=json.loads((root/'tile_registry.json').read_text());assert not{r['dataset'] for r in REGISTRY.values()}&set(HELDOUT)
 assert sha(f['base_checkpoint'])==f['base_checkpoint_sha256'];torch.set_num_threads(8);assert torch.cuda.is_available()
 name=('smoke_general' if a.name=='balanced_four_accession_v1' else 'smoke_'+a.name) if a.smoke else a.name;out=G/'runs'/name;assert not out.exists();initial=a.weights or f['base_checkpoint']
 # Confirm loader returns varying original float pixels, not metadata placeholder pixels.
 first=next(iter(TILES.values()));pixels=cached_image(first['image_key']);assert pixels.dtype==np.float32 and float(pixels.std())>.001
 cached_image.cache_clear()
 # Read-only parent arrays are shared by Linux fork workers; avoid recomputing whole-MRC normalization per tile.
 needed={t['image_key'] for t in TILES.values()};estimate=sum(geometry(REGISTRY[k],TRAIN_MODE)['padded_width']*geometry(REGISTRY[k],TRAIN_MODE)['padded_height']*4 for k in needed)
 limit_path=Path('/sys/fs/cgroup/memory.max');limit_text=limit_path.read_text().strip() if limit_path.exists() else 'max';limit=int(limit_text) if limit_text.isdigit() else os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')
 if not a.smoke and estimate<.60*limit:
  cv2.setNumThreads(1);torch.multiprocessing.set_start_method('fork',force=True)
  for i,k in enumerate(sorted(needed)):
   array=normalize(REGISTRY[k],TRAIN_MODE)[0];array.flags.writeable=False;SHARED_IMAGES[k]=array
   if (i+1)%20==0:print('SHARED_FLOAT_CACHE',i+1,'/',len(needed),'estimated_GiB',round(estimate/1024**3,2),flush=True)
  print('SHARED_FLOAT_CACHE_READY',len(SHARED_IMAGES),flush=True)
 os.chdir(ULTRA);model=YOLO(initial);assert model.model.model[-1].stride.tolist()==[8.,16.,32.]
 cfg=dict(data=str(root/('smoke.yaml' if a.smoke else 'data.yaml')),imgsz=SIZE,epochs=1 if a.smoke else a.epochs,batch=4,device=0,workers=0 if a.smoke else 4,
  project=str(G/'runs'),name=name,exist_ok=False,pretrained=True,optimizer='AdamW',lr0=a.lr,lrf=.1,weight_decay=.0005,warmup_epochs=.1 if a.smoke else 2,warmup_bias_lr=.001,cos_lr=True,nbs=16,
  patience=30,amp=True,cache=False,rect=False,multi_scale=0.,max_det=2000,conf=.0001,seed=10028,deterministic=True,save=True,save_period=-1,plots=False,
  mosaic=0.,close_mosaic=0,mixup=0.,cutmix=0.,copy_paste=0.,hsv_h=0.,hsv_s=0.,hsv_v=0.,degrees=0.,translate=0.,scale=0.,shear=0.,perspective=0.,fliplr=0.,flipud=0.,val=True)
 dump(G/(name+'_training_config.json'),{'training':cfg,'initial_checkpoint':initial,'initial_checkpoint_sha256':sha(initial),'source_adapter_sha256':sha(SOURCE_CODE),
  'training_accessions':['10028']+DEVELOPMENT,'heldout_accessions':HELDOUT,'heldout_used':False,'mode':TRAIN_MODE,'strategy':'balanced multi-accession training; original float grayscale and augmentation retained; variable native image geometry; boxes use each published particle diameter',
  'shared_readonly_float_cache_images':len(SHARED_IMAGES),'shared_cache_estimated_bytes':estimate,
  'source_code_sha256':{p.name:sha(p) for p in (G/'code').glob('*.py')},'training_registries_sha256':{p.name:sha(p) for p in [root/'image_registry.json',root/'tile_registry.json']}})
 result=model.train(trainer=GeneralTrainer,**cfg);save=Path(result.save_dir);rows=list(csv.DictReader((save/'results.csv').open()))
 for row in rows:
  for k,v in row.items():
   if 'loss' in k:assert math.isfinite(float(v))
 dump(save/'completion.json',{'status':'PASS','epochs':len(rows),'best_sha256':sha(save/'weights/best.pt'),'last_sha256':sha(save/'weights/last.pt'),'smoke':a.smoke})
 print('GENERAL_TRAINING_COMPLETE',str(save),flush=True)
if __name__=='__main__':main()
