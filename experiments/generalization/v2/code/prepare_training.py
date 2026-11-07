"""Reuse v1 train/validation only; append three domains; no target GT for training."""
import random
from PIL import Image
from common import *

def registry():
 rows=[]
 for p in [V1/'all_images.csv',G/'new_images.csv']:
  if not p.exists():
   assert len(DOMAINS)==4,'Seven-domain preparation requires all new source data'
   continue
  for r in csv.DictReader(p.open()):
   if p.parent==V1 and r['split'] not in ['train','val']:continue
   for k in ['width','height','diameter','GT_count','bytes']:r[k]=int(r[k])
   rows.append(r)
 for split in ['train','val']:
  for name in (D/'manifests/split_v3_paper60_20_20'/f'{split}.txt').read_text().splitlines():
   n=Path(name.strip()).name;p=D/'micrographs'/n
   gt=D/'annotations_original/cryoppp_10028/10028/ground_truth/particle_coordinates'/(Path(n).stem+'.csv')
   rows.append(dict(dataset='10028',split=split,micrograph_name=n,path=str(p),GT_path=str(gt),width=4096,height=4096,diameter=224,GT_count=sum(1 for _ in csv.DictReader(gt.open())),bytes=p.stat().st_size,sha256=sha(p),GT_sha256=sha(gt)))
 assert len({key(r) for r in rows})==len(rows)
 for split in ['train','val']:
  assert {r['dataset'] for r in rows if r['split']==split}==set(DOMAINS)
 # File hashes, then pixel hashes when available, guard cross-role duplicates.
 by_hash={}
 for r in rows:
  assert Path(r['path']).exists() and Path(r['GT_path']).exists()
  assert r['sha256']==sha(r['path']) and r['GT_sha256']==sha(r['GT_path'])
  if not r.get('pixel_sha256'):
   with mrcfile.open(r['path'],permissive=True) as m:r['pixel_sha256']=hashlib.sha256(np.ascontiguousarray(m.data).tobytes()).hexdigest()
  for field in ['sha256','pixel_sha256']:
   if r.get(field):
    k=(field,r[field]);previous=by_hash.setdefault(k,r)
    assert previous['split']==r['split'],('Cross-role duplicate',previous,r)
 final_pixels={r['pixel_sha256'] for r in rows if r['split']=='final_check'}
 final_bytes={r['sha256'] for r in rows if r['split']=='final_check'}
 old_target_count=0
 for r in csv.DictReader((V1/'all_images.csv').open()):
  if r['split']=='heldout' and r['dataset'] in TARGETS:
   assert r['sha256'] not in final_bytes,('Previously evaluated file duplicated in final check',r['micrograph_name'])
   with mrcfile.open(r['path'],permissive=True) as m:h=hashlib.sha256(np.ascontiguousarray(m.data).tobytes()).hexdigest()
   assert h not in final_pixels,('Previously evaluated pixel array duplicated in final check',r['micrograph_name'])
   old_target_count+=1
 dump(G/'input_integrity.json',dict(status='PASS',records=len(rows),no_cross_role_file_or_pixel_duplicates=True,previous_target_micrographs_checked=old_target_count,no_previous_target_pixel_overlap=True))
 dump(G/'image_registry.json',rows)
 return rows

def main():
 rows=registry();rng=random.Random(100282);root=G/'training_data';root.mkdir(exist_ok=True)
 sentinel=root/'metadata_only_blank.png'
 if not sentinel.exists():Image.new('L',(SIZE,SIZE),128).save(sentinel)
 tiles={};statistics={}
 for split,count in [('train',192),('val',32)]:
  paths=[]
  for ds in DOMAINS:
   rs=[r for r in rows if r['split']==split and r['dataset']==ds];available=[];mandatory=[]
   for r in rs:
    g=geometry(r)
    ts=[dict(image_key=key(r),dataset=ds,x0=x,y0=y,geometry=g) for y in starts(g['padded_height']) for x in starts(g['padded_width'])]
    rng.shuffle(ts)
    if split=='train':mandatory.append(ts.pop())
    available.extend(ts)
   rng.shuffle(available);chosen=mandatory+available[:count-len(mandatory)]
   if len(chosen)<count:
    pool=mandatory+available
    while len(chosen)<count:chosen.append(rng.choice(pool).copy())
   assert len(chosen)==count
   statistics[split+'/'+ds]=dict(images=len(rs),tiles=count,unique_geometry=len({(t['image_key'],t['x0'],t['y0']) for t in chosen}))
   by={key(r):r for r in rs}
   for j,t in enumerate(chosen):
    tid=f'{ds}__{split}__{j:04}';r=by[t['image_key']];g=t['geometry'];image=root/'images'/split/(tid+'.png');label=root/'labels'/split/(tid+'.txt')
    image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
    if not image.exists():os.link(sentinel,image)
    lines=[]
    for p in points(r):
     sx,sy=to_scaled(p.x,p.y,g);ox,oy=to_original(sx,sy,g);assert abs(ox-p.x)<1e-8 and abs(oy-p.y)<1e-8
     x,y=sx-t['x0'],sy-t['y0']
     if not(0<=x<SIZE and 0<=y<SIZE):continue
     bw,bh=r['diameter']*g['sx'],r['diameter']*g['sy'];x1,y1=max(0,x-bw/2),max(0,y-bh/2);x2,y2=min(SIZE,x+bw/2),min(SIZE,y+bh/2)
     lines.append(f'0 {(x1+x2)/2/SIZE:.10f} {(y1+y2)/2/SIZE:.10f} {(x2-x1)/SIZE:.10f} {(y2-y1)/SIZE:.10f}')
    label.write_text('\n'.join(lines)+'\n' if lines else '');t['labels']=len(lines);tiles[tid]=t;paths.append(str(image))
  (root/(split+'.txt')).write_text('\n'.join(paths)+'\n')
 dump(root/'tile_registry.json',tiles)
 (root/'data.yaml').write_text(f'path: {root}\ntrain: train.txt\nval: val.txt\nnc: 1\nnames: [particle]\n')
 for split in ['train','val']:
  paths=(root/(split+'.txt')).read_text().splitlines();small=[next(p for p in paths if tiles[Path(p).stem]['dataset']==ds) for ds in DOMAINS]
  (root/('smoke_'+split+'.txt')).write_text('\n'.join(small)+'\n')
 (root/'smoke.yaml').write_text(f'path: {root}\ntrain: smoke_train.txt\nval: smoke_val.txt\nnc: 1\nnames: [particle]\n')
 dump(root/'preparation.json',dict(status='PASS',profile=PROFILE,tile_size=SIZE,canonical_particle_diameter=PARTICLE,statistics=statistics,source_MRC_copies_created=0,labels_roundtrip='PASS',target_accessions_in_training=False))
 print('V2_TRAINING_DATA_READY',json.dumps(statistics),flush=True)

if __name__=='__main__':main()
