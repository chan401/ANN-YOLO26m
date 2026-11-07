"""One-module source-only lower-SNR augmentation; native inference unchanged."""
from pathlib import Path
import argparse, hashlib, importlib, json, random, sys
import numpy as np

G=Path('/root/autodl-tmp/cryoem_generalization/v3_low_snr')
V2=G.parent/'v2'
sys.path.insert(0,str(G/'core'))
import general_common as ref
ref.G=G;ref.DEVELOPMENT=['10017','10077','10291','10289','10240','10760'];ref.HELDOUT=['10059','10081']
def protocol():
 p=G/'protocol.json';assert ref.sha(p)==(G/'protocol.sha256').read_text().strip()
 d=json.loads(p.read_text());d.update(base_checkpoint=d['initial_checkpoint'],base_checkpoint_sha256=d['initial_checkpoint_sha256']);return d
ref.protocol=protocol

def degrade(tile,rng):
 """Labels unchanged: add independent noise, without geometric transformations."""
 contrast=rng.uniform(.35,.8);sigma=rng.uniform(.02,.10)
 noise=rng.normal(0,sigma,tile.shape).astype(np.float32)
 return np.clip((tile-.5)*np.float32(contrast)+.5+noise,0,1).astype(np.float32)

def main():
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args()
 train=importlib.import_module('train_general');Parent=train.GeneralDataset
 class LowSNRDataset(Parent):
  def load_image(self,i,rect_mode=True):
   t=train.TILES[Path(self.im_files[i]).stem];arr=train.cached_image(t['image_key']);x,y=t['x0'],t['y0'];tile=arr[y:y+ref.SIZE,x:x+ref.SIZE]
   assert tile.shape==(ref.SIZE,ref.SIZE) and tile.dtype==np.float32
   if self.augment:
    raw_rgb=np.repeat((tile*255)[...,None],3,axis=2)
    tile=train.old.GrayPhotometric()({'img':raw_rgb})['img'][...,0]/255.
    if random.random()<.65:tile=degrade(tile,np.random)
   features=np.ascontiguousarray(ref.tile_channels(tile,'native_multiscale').transpose(1,2,0)*255)
   return features,(ref.SIZE,ref.SIZE),(ref.SIZE,ref.SIZE)
 train.GeneralDataset=LowSNRDataset
 # Reproducible augmentation connectivity check, exclusively a source validation tile.
 r=next(v for v in json.loads((G/'training_data/image_registry.json').read_text()).values() if v['split']=='val')
 tile=ref.normalize(r,'native_multiscale')[0][:ref.SIZE,:ref.SIZE]
 before=ref.sha(r['path']);x=degrade(tile,np.random.RandomState(100283));y=degrade(tile,np.random.RandomState(100283))
 assert np.array_equal(x,y) and np.isfinite(x).all() and x.shape==tile.shape
 assert not np.array_equal(x,tile);assert ref.sha(r['path'])==before
 ref.dump(G/'augmentation_check.json',dict(status='PASS',source_image=r['micrograph_name'],target_used=False,
  deterministic_with_fixed_seed=True,geometry_changed=False,raw_MRC_modified=False,channels=3,
  parameter_source='predeclared source-only stress augmentation; no target-GT optimization',input_std=float(tile.std()),output_std=float(x.std())))
 sys.argv=['train_general.py','--mode','native_multiscale','--name','low_snr20','--epochs','20','--lr','0.00005','--weights',protocol()['base_checkpoint']]+(['--smoke'] if a.smoke else [])
 train.main()

if __name__=='__main__':main()
