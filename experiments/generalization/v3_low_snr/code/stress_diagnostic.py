"""Fixed source-only stress diagnostic, no target data or threshold selection."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
G=Path('/root/autodl-tmp/cryoem_generalization/v3_low_snr');sys.path.insert(0,str(G/'core'))
import general_common as ref
ref.G=G;ref.HELDOUT=['10059','10081']

def main():
 p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 original=ref.normalize
 def stressed(r,mode):
  image,meta=original(r,mode);seed=int(hashlib.sha256(r['micrograph_name'].encode()).hexdigest()[:8],16)^100283
  rng=np.random.RandomState(seed);image=np.clip((image-.5)*.5+.5+rng.normal(0,.07,image.shape).astype(np.float32),0,1).astype(np.float32)
  return image,{**meta,'source_only_stress':dict(contrast=.5,noise_sigma=.07,seed=seed)}
 ref.normalize=stressed
 from infer_ann import infer
 rows=json.loads((G/'image_registry.json').read_text());source=[r for r in rows if r['split']=='val'];domains=sorted({r['dataset'] for r in source})
 selected=[sorted([r for r in source if r['dataset']==ds],key=lambda r:r['micrograph_name'])[0] for ds in domains]
 assert len(selected)==7 and not {r['dataset'] for r in selected}&{'10059','10081'}
 pred=infer(Path(a.weights),selected,'native_multiscale',Path(a.out));m=ref.evaluate(selected,pred,.22)
 ref.dump(Path(a.out)/'fixed_threshold_metrics.json',dict(diagnostic_only=True,selection_use=False,threshold=.22,source_images=7,target_images_used=False,metrics=m))
 print('STRESS_DIAGNOSTIC',json.dumps(m['macro']),flush=True)
if __name__=='__main__':main()
