"""Float grayscale and tile adapters for ANN-YOLO26, isolated from other experiments."""
import csv, hashlib, json, random, sys
from pathlib import Path
import cv2
import mrcfile
import numpy as np
import torch

DATA = Path('/root/autodl-tmp/cryoem_datasets/EMPIAR-10028')
ULTRA = Path('/root/autodl-tmp/cryoem_annyolo26/SNNYOLO26/ultralytics-snn-yolo26')
EXP = DATA/'experiments/ann_yolo26m_cryo_tiles_v3'
SPLIT = DATA/'manifests/split_v3_paper60_20_20'
COORD = DATA/'annotations_standardized/split_v3_paper60_20_20/ann_yolo26/coords'
WEIGHT = ULTRA/'yolo26m.pt'
SIZE = 1536
STARTS = [0, 1280, 2560]
BOUNDS = [0, 1408, 2688, 4096]
BOX = 224.0
sys.path.insert(0, str(ULTRA))
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import Compose, Format, RandomFlip
from ultralytics.models.yolo.detect import DetectionTrainer

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def names(split):
    if split not in ('train','val'): raise ValueError('This adapter supports Train/Validation only')
    return [Path(s.strip()).name for s in (SPLIT/f'{split}.txt').read_text().splitlines() if s.strip()]

def points(split, name):
    p=COORD/split/(Path(name).stem+'.coord')
    return np.loadtxt(p, dtype=np.float64, ndmin=2)[:,:2]

def normalize_mrc(path):
    """Same float32, image-only preprocessing in preparation and inference."""
    with mrcfile.open(path, permissive=True) as m:
        a=np.asarray(m.data, dtype=np.float32).squeeze().copy()
    if a.shape!=(4096,4096) or not np.isfinite(a).all(): raise ValueError(f'Invalid source {path}')
    q16, med, q84=np.percentile(a,[16,50,84])
    sigma=float((q84-q16)/2)
    if sigma<=0: raise ValueError('Degenerate robust intensity range')
    # Fixed +/-5 robust sigma; suppresses corrupt pixel spikes without 8-bit quantization.
    a=np.clip((a-np.float32(med))/(10*np.float32(sigma))+.5,0,1).astype(np.float32)
    # Existing v3 baseline convention: CryoPPP y = 4095 - MRC-array y.
    a=np.ascontiguousarray(np.flipud(a))
    return a, {'median':float(med),'robust_sigma':sigma,'clip_sigma':5,
               'orientation':'flipud MRC array; x_CryoPPP=x_array, y_CryoPPP=4095-y_array',
               'dtype':'float32','output_range':[0,1],'quantization':'none'}

def tile_rows(split):
    with (EXP/'tiles.csv').open(newline='') as f:
        return [r for r in csv.DictReader(f) if r['split']==split]

class QuarterTurn:
    def __init__(self, k=None): self.k=k
    def __call__(self, labels):
        k=random.randrange(4) if self.k is None else self.k
        inst=labels['instances']; inst.convert_bbox('xywh')
        h,w=labels['img'].shape[:2]
        inst.normalize(w,h)
        b=inst.bboxes.copy()
        for _ in range(k): b=b[:,[1,0,3,2]]; b[:,1]=1-b[:,1]
        inst.update(bboxes=b)
        labels['img']=np.ascontiguousarray(np.rot90(labels['img'],k))
        return labels

class GrayPhotometric:
    def __call__(self, labels):
        # Shared noise in all repeated channels, preserving grayscale.
        x=labels['img']/255.0
        gain=random.uniform(.9,1.1); bias=random.uniform(-.02,.02)
        x=(x-.5)*gain+.5+bias
        if random.random()<.5:
            noise=np.random.normal(0, random.uniform(0,.01), x.shape[:2]+(1,)).astype(np.float32)
            x=x+noise
        labels['img']=np.clip(x,0,1).astype(np.float32)*255.0
        return labels

class CryoDataset(YOLODataset):
    def load_image(self,i,rect_mode=True):
        x=cv2.imread(self.im_files[i],cv2.IMREAD_UNCHANGED)
        if x is None or x.dtype!=np.float32 or x.shape!=(SIZE,SIZE):
            raise ValueError(f'Float32 tile loading failed: {self.im_files[i]}')
        # Keep all pretrained RGB first-convolution weights: repeat the same float channel.
        x=np.repeat((x*255.0)[...,None],3,axis=2)
        return x,(SIZE,SIZE),(SIZE,SIZE)
    def build_transforms(self,hyp=None):
        ts=[QuarterTurn(),RandomFlip(.5,'horizontal'),RandomFlip(.5,'vertical'),GrayPhotometric()] if self.augment else []
        ts.append(Format(bbox_format='xywh',normalize=True,batch_idx=True,bgr=0.0))
        return Compose(ts)

class CryoTrainer(DetectionTrainer):
    def build_dataset(self,img_path,mode='train',batch=None):
        c=self.args
        return CryoDataset(img_path=img_path,imgsz=c.imgsz,batch_size=batch,augment=mode=='train',
            hyp=c,rect=False,cache=False,single_cls=False,stride=32,pad=0.0,prefix=mode+': ',
            task='detect',classes=None,data=self.data,fraction=c.fraction if mode=='train' else 1.0)
    def get_model(self,cfg=None,weights=None,verbose=True):
        m=super().get_model(cfg,weights,verbose)
        m.model[-1].max_det=2000
        return m

def write_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2)+'\n')
