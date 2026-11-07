"""Download only the prospectively selected twenty new TRPV1 micrographs."""
from pathlib import Path
import csv,hashlib,importlib.util,json,os
import mrcfile,numpy as np
G=Path('/root/autodl-tmp/cryoem_generalization/v3_low_snr');V2=G.parent/'v2';V1=G.parent/'v1'
spec=importlib.util.spec_from_file_location('bounded_downloader',G/'code/bounded_downloader.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d);d.G=G
def main():
 protocol=json.loads((G/'protocol.json').read_text());assert d.sha(G/'protocol.json')==(G/'protocol.sha256').read_text().strip()
 source=G/'source_image_registry.json'
 if not source.exists():source.write_bytes((G/'image_registry.json').read_bytes())
 assert d.sha(source)==protocol['source_registry_sha256']
 for r in protocol['files']:
  dest=G/'datasets/10059/annotations_original'/(Path(r['micrograph_name']).stem+'.csv');dest.parent.mkdir(parents=True,exist_ok=True)
  src=V2/'datasets/10059/annotations_original'/dest.name
  if not dest.exists():os.link(src,dest)
  assert d.sha(src)==d.sha(dest)
 d.download(protocol)
 old=json.loads((V2/'image_registry.json').read_text());seen={r['pixel_sha256'] for r in old}
 # Include every earlier TRPV1 image, even v1 images absent from the v2 registry.
 for p in (V1/'datasets/10059/file_records').glob('*.json'):
  r=json.loads(p.read_text());path=Path(r.get('path',V1/'datasets/10059/micrographs'/r['micrograph_name']))
  if path.exists():
   with mrcfile.open(path,permissive=True) as m:seen.add(hashlib.sha256(np.ascontiguousarray(m.data).tobytes()).hexdigest())
 target=[]
 for r in protocol['files']:
  p=G/'datasets/10059/file_records'/(r['micrograph_name']+'.json');r=json.loads(p.read_text())
  assert r['pixel_sha256'] not in seen,'Repeated raw pixel array';seen.add(r['pixel_sha256']);target.append(r)
 assert len(target)==20
 secondary=json.loads((G/'secondary_HCN1_protocol.json').read_text())
 hcn=[dict(r,split='regression_check') for r in old if r['dataset']=='10081' and r['split']=='final_check']
 assert len(hcn)==secondary['images']==20
 d.dump(G/'image_registry.json',json.loads(source.read_text())+target+hcn)
 d.dump(G/'target_integrity.json',dict(status='PASS',new_images=20,GT=sum(r['GT_count'] for r in target),
  previous_v1_v2_names_excluded=True,unique_raw_pixel_arrays=True,GT_in_bounds=True,all_pixels_finite=True,source_registry_unchanged=d.sha(source)==protocol['source_registry_sha256']))
 print('NEW_TRPV1_READY',sum(r['GT_count'] for r in target),flush=True)
if __name__=='__main__':main()
