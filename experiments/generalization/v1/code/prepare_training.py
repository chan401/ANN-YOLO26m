"""Balanced virtual tile dataset. PNG headers are metadata placeholders; loader uses original float MRC pixels."""
import random,argparse
from PIL import Image
from general_common import *

def candidates(r,mode):
 g=geometry(r,mode);out=[]
 for y0 in starts(g['padded_height']):
  for x0 in starts(g['padded_width']):out.append({'image_key':key(r),'dataset':r['dataset'],'micrograph_name':r['micrograph_name'],'x0':x0,'y0':y0,'mode':mode,'geometry':g})
 return out

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['native','diameter','diameter_postnorm'],default='native');args=parser.parse_args()
 root=G/'training_data';root.mkdir(exist_ok=True);seed=random.Random(10028);image_map={};tilemap={};statistics={}
 sentinel=root/'metadata_only_blank.png'
 if not sentinel.exists():Image.new('L',(SIZE,SIZE),128).save(sentinel)
 for split,target in [('train',192),('val',32)]:
  rs=records(split,['10028']+DEVELOPMENT);by={d:[r for r in rs if r['dataset']==d] for d in ['10028']+DEVELOPMENT};selected=[]
  for ds,images in by.items():
   available=[];mandatory=[]
   for r in images:
    image_map[key(r)]=r;ts=candidates(r,args.mode);seed.shuffle(ts)
    if split=='train':mandatory.append(ts.pop())
    available.extend(ts)
   seed.shuffle(available);assert len(mandatory)<=target
   chosen=mandatory+available[:target-len(mandatory)];assert len(chosen)==target
   statistics[split+'/'+ds]={'source_micrographs':len(images),'selected_tiles':len(chosen),'selection':'one random tile per source training micrograph then uniform shuffled tiles; balanced192/domain; validation proxy32/domain; independent of GT counts'}
   selected.extend(chosen)
  (root/'images'/split).mkdir(parents=True,exist_ok=True);(root/'labels'/split).mkdir(parents=True,exist_ok=True)
  paths=[]
  for t in selected:
   r=image_map[t['image_key']];g=t['geometry'];tid=f'{t["image_key"]}_x{t["x0"]}_y{t["y0"]}';tilemap[tid]=t
   image=root/'images'/split/(tid+'.png');label=root/'labels'/split/(tid+'.txt')
   if not image.exists():os.link(sentinel,image)
   lines=[]
   for p in gt_points(r):
    sx,sy=to_scaled(p.x,p.y,g);x,y=sx-t['x0'],sy-t['y0']
    if not(0<=x<SIZE and 0<=y<SIZE):continue
    bw=r['diameter']*g['sx'];bh=r['diameter']*g['sy'];x1,y1=max(0,x-bw/2),max(0,y-bh/2);x2,y2=min(SIZE,x+bw/2),min(SIZE,y+bh/2)
    lines.append(f'0 {(x1+x2)/2/SIZE:.10f} {(y1+y2)/2/SIZE:.10f} {(x2-x1)/SIZE:.10f} {(y2-y1)/SIZE:.10f}')
    ox,oy=to_original(x+t['x0'],y+t['y0'],g);assert abs(ox-p.x)<1e-8 and abs(oy-p.y)<1e-8
   label.write_text('\n'.join(lines)+'\n' if lines else '');paths.append(str(image));t['labels']=len(lines)
  (root/(split+'.txt')).write_text('\n'.join(paths)+'\n')
 dump(root/'image_registry.json',image_map);dump(root/'tile_registry.json',tilemap)
 (root/'data.yaml').write_text(f'path: {root}\ntrain: train.txt\nval: val.txt\nnc: 1\nnames: [particle]\n')
 for split in ['train','val']:
  paths=(root/(split+'.txt')).read_text().splitlines();chosen=[next(p for p in paths if tilemap[Path(p).stem]['dataset']==ds) for ds in ['10028']+DEVELOPMENT];(root/('smoke_'+split+'.txt')).write_text('\n'.join(chosen)+'\n')
 (root/'smoke.yaml').write_text(f'path: {root}\ntrain: smoke_train.txt\nval: smoke_val.txt\nnc: 1\nnames: [particle]\n')
 assert not {r['dataset'] for r in image_map.values()}&set(HELDOUT)
 dump(root/'preparation.json',{'status':'PASS','tiles_per_split':{'train':768,'val':128},'statistics':statistics,'heldout_used':False,
  'image_storage':'small hardlinked PNG headers for Ultralytics metadata only; custom load_image reads original MRC->float normalize->diameter resize->reflect pad->tile->repeat gray3; PNG pixels never used for model inputs',
  'mode':args.mode,'source_GT_and_MRC_modified':False,'target_particle_diameter_px':None if args.mode=='native' else 224,'native_label_diameters':'published per-dataset diameters; boxes transformed by exact geometry','tile_size':SIZE,'minimum_overlap':OVERLAP,'context_pad':PAD,'coordinate_roundtrip':'PASS',
  'protocol_sha256':sha(G/'protocol.json'),'registries_sha256':{f:sha(root/f) for f in ['image_registry.json','tile_registry.json']}})
 print('BALANCED_TRAINING_READY',json.dumps(statistics),flush=True)
if __name__=='__main__':main()
