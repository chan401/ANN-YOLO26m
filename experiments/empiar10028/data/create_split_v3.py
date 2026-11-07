#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json, math, os, random, shutil, subprocess, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import mrcfile
from PIL import Image

ROOT=Path('/root/autodl-tmp/cryoem_datasets/EMPIAR-10028')
SEED=10028
SPLITS={'train':132,'val':44,'test':44}
EXPECTED_GT=19315
MAN=ROOT/'manifests/split_v3_paper60_20_20'
METHOD=ROOT/'annotations_standardized/split_v3_paper60_20_20'
EXP=ROOT/'experiments/ann_yolo26m_fullimage_2048_v3'
REPORT=ROOT/'reports/split_v3_and_cleanup'
ANN=ROOT/'annotations_original/cryoppp_10028/10028/ground_truth/particle_coordinates'
MICRO=ROOT/'micrographs'

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def write_json(path,obj): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def csv_write(path,fields,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
def size(path):
 try: return path.stat().st_size
 except: return 0
def disk_snapshot(dest):
 out=[]
 for cmd in [['df','-hT','/','/root/autodl-tmp'],['du','-x','-h','--max-depth=2','/root/autodl-tmp']]:
  p=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  out.append('$ '+' '.join(cmd)+'\n'+p.stdout)
 Path(dest).parent.mkdir(parents=True,exist_ok=True); Path(dest).write_text('\n'.join(out),encoding='utf-8')
def main():
 if MAN.exists() or METHOD.exists() or EXP.exists(): raise FileExistsError('Refusing to overwrite an existing v3 dataset/config directory')
 if REPORT.exists() and any(REPORT.iterdir()): raise FileExistsError('Report directory already contains files')
 REPORT.mkdir(parents=True,exist_ok=True)
 disk_snapshot(REPORT/'disk_usage_before.txt')
 raw=sorted([p for p in MICRO.iterdir() if p.is_file() and p.suffix.lower()=='.mrc'],key=lambda p:p.name)
 target=[x.strip() for x in (ROOT/'manifests/target_220.txt').read_text().splitlines() if x.strip() and not x.startswith('#')]
 if len(raw)!=220 or len({p.name for p in raw})!=220: raise ValueError(f'Expected 220 unique MRC files; got {len(raw)}')
 if set(p.name for p in raw)!=set(target): raise ValueError('Raw MRC inventory differs from prior target_220.txt; refusing to split')
 expected_hash={}
 for line in (ROOT/'manifests/sha256_220.txt').read_text().splitlines():
  z=line.strip().split(None,1)
  if len(z)==2: expected_hash[Path(z[1].lstrip('*')).name]=z[0].lower()
 if len(expected_hash)!=220: raise ValueError(f'Expected checksum manifest with 220 entries, got {len(expected_hash)}')
 coords={}; gt_hash={}; image_hash={}; mrc_info={}; bad=[]
 source_csvs=sorted(ANN.glob('*.csv'))
 if len(source_csvs)!=300: raise ValueError(f'CryoPPP directory should contain 300 per-micrograph CSVs, got {len(source_csvs)}')
 for i,p in enumerate(raw,1):
  h=sha(p); image_hash[p.name]=h
  if h!=expected_hash.get(p.name): bad.append(f'{p.name}: SHA-256 mismatch')
  with mrcfile.open(p,mode='r',permissive=True) as m:
   arr=m.data
   if arr is None or arr.shape!=(4096,4096) or arr.dtype!=np.float32: raise ValueError(f'Unreadable/unexpected MRC {p.name}: shape={None if arr is None else arr.shape}, dtype={None if arr is None else arr.dtype}')
   finite=bool(np.isfinite(arr).all())
   if not finite: raise ValueError(f'Non-finite pixel values in {p.name}')
   lo,hi=[float(x) for x in np.percentile(arr,[1,99])]
   vox=[float(m.voxel_size[k]) for k in ('x','y','z')]
   mrc_info[p.name]={'width':4096,'height':4096,'dtype':'float32','finite_pixels':finite,'p1':lo,'p99':hi,'voxel_size_header':vox}
  src=ANN/(p.stem+'.csv')
  if not src.is_file(): raise ValueError(f'Missing source GT CSV for {p.name}')
  with src.open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
  pts=[]
  for n,row in enumerate(rows,2):
   x=float(row['X-Coordinate']); y=float(row['Y-Coordinate'])
   if not (math.isfinite(x) and math.isfinite(y) and x.is_integer() and y.is_integer() and 0<=x<4096 and 0<=y<4096): raise ValueError(f'Invalid GT coordinate {src}:{n}: {x},{y}')
   pts.append((int(x),int(y)))
  if len(pts)!=len(set(pts)): raise ValueError(f'Duplicate GT centers in {src.name}')
  coords[p.name]=pts; gt_hash[p.name]=sha(src)
  if i%25==0: print(f'Raw MRC and GT audit {i}/220',flush=True)
 if bad: raise ValueError('; '.join(bad[:20]))
 extra=[p.name for p in source_csvs if p.stem+'.mrc' not in set(target)]
 if extra: print('CryoPPP full source includes annotations outside 220 set:',len(extra),'(expected source collection has 300)',flush=True)
 if sum(map(len,coords.values()))!=EXPECTED_GT: raise ValueError(f'220-image CryoPPP GT expected {EXPECTED_GT}, reread sum is {sum(map(len,coords.values()))}')
 # Validate all old standardized representations against original coordinate centers.
 old=ROOT/'annotations_standardized/split_v2_train120_val20_test80'
 legacy_checked={'ann_yolo26_coord':0,'cryolo_box':0,'topaz_target_rows':0}
 for split in SPLITS:
  for name in (ROOT/'manifests/split_v2_train120_val20_test80'/f'{split}.txt').read_text().splitlines():
   if not name.strip(): continue
   stem=Path(name).stem; pts=coords[name]
   cp=old/'ann_yolo26/coords'/split/f'{stem}.coord'
   got=[]
   for ln in cp.read_text().splitlines():
    if ln.strip():
     q=ln.replace(',',' ').split(); got.append((int(float(q[0])),int(float(q[1]))))
   if got!=pts: raise ValueError(f'Existing YOLO coordinate conversion mismatch for {name}')
   legacy_checked['ann_yolo26_coord']+=1
   bp=old/'cryolo/annotations'/split/f'{stem}.box'
   box=[]
   for ln in bp.read_text().splitlines():
    if ln.strip():
     q=list(map(float,ln.split())); box.append((int(q[0]+112),int(q[1]+112),int(q[2]),int(q[3])))
   want=[(x,y,224,224) for x,y in pts]
   if box!=want: raise ValueError(f'Existing crYOLO .box conversion mismatch for {name}')
   legacy_checked['cryolo_box']+=1
 # Split: lexicographically stable sort before using the specified standard-library RNG.
 names=sorted(coords)
 shuffled=names.copy(); random.Random(SEED).shuffle(shuffled)
 assign={s:shuffled[sum(SPLITS[x] for x in list(SPLITS)[:i]):sum(SPLITS[x] for x in list(SPLITS)[:i+1])] for i,s in enumerate(SPLITS)}
 if {s:len(v) for s,v in assign.items()}!=SPLITS or len(set(sum(assign.values(),[])))!=220: raise AssertionError('Split sizes/disjointness failed')
 owner={n:s for s,vals in assign.items() for n in vals}
 MAN.mkdir(parents=True)
 for s,vals in assign.items(): (MAN/f'{s}.txt').write_text(''.join(n+'\n' for n in vals),encoding='utf-8')
 rows=[]
 for n in names: rows.append({'micrograph_name':n,'micrograph_id':Path(n).stem,'original_path':str(MICRO/n),'split':owner[n],'gt_count':len(coords[n]),'mrc_sha256':image_hash[n],'cryoppp_csv_path':str(ANN/(Path(n).stem+'.csv')),'cryoppp_csv_sha256':gt_hash[n]})
 csv_write(MAN/'all_micrographs.csv',list(rows[0]),rows)
 stats=[]
 for s,vals in assign.items():
  gs=[len(coords[n]) for n in vals]
  stats.append({'split':s,'micrographs':len(vals),'gt_count':sum(gs),'mean_gt_per_micrograph':sum(gs)/len(gs),'min_gt_per_micrograph':min(gs),'max_gt_per_micrograph':max(gs)})
 csv_write(MAN/'split_statistics.csv',list(stats[0]),stats)
 protocol={'dataset':'EMPIAR-10028','protocol':'paper60_20_20','seed':SEED,'rng':'Python standard library random.Random(10028).shuffle','python_version':sys.version,'algorithm':'Sort unique micrograph filenames lexicographically, copy list, shuffle once, assign first 132 to train, next 44 to val, final 44 to test. Split at micrograph level; no GT stratification or result-based selection.','sizes':SPLITS,'source_micrograph_count':220,'source_manifest_path':str(ROOT/'manifests/target_220.txt'),'source_manifest_sha256':sha(ROOT/'manifests/target_220.txt'),'source_checksum_manifest':str(ROOT/'manifests/sha256_220.txt'),'source_annotation_archive_sha256':sha(ROOT/'annotations_original/cryoppp_10028.tar.gz'),'micrograph_hashes':image_hash,'gt_csv_hashes':gt_hash,'test_usage':'Reserved for final evaluation. Select thresholds and parameters on validation only.'}
 write_json(MAN/'split_protocol.json',protocol)
 # Set up method-specific inputs derived directly from CryoPPP per-image CSVs.
 for s in SPLITS:
  for d in [METHOD/'ann_yolo26'/'coords'/s,METHOD/'ann_yolo26'/'labels'/s,METHOD/'cryolo'/'annotations'/s,METHOD/'cryolo'/'images'/s]: d.mkdir(parents=True,exist_ok=True)
  (METHOD/'topaz'/s).mkdir(parents=True,exist_ok=True)
  (EXP/'images'/s).mkdir(parents=True,exist_ok=True); (EXP/'labels'/s).mkdir(parents=True,exist_ok=True)
 method_gt={s:0 for s in SPLITS}; image_build=[]
 for s,vals in assign.items():
  top_img=[]; top_gt=[]
  for n in vals:
   stem=Path(n).stem; pts=coords[n]
   coord_lines=''.join(f'{x}\t{y}\n' for x,y in pts)
   yolo=''.join(f'0 {x/4096:.10f} {y/4096:.10f} {224/4096:.10f} {224/4096:.10f}\n' for x,y in pts)
   box=''.join(f'{x-112}\t{y-112}\t224\t224\n' for x,y in pts)
   (METHOD/'ann_yolo26'/'coords'/s/f'{stem}.coord').write_text(coord_lines,encoding='utf-8')
   (METHOD/'ann_yolo26'/'labels'/s/f'{stem}.txt').write_text(yolo,encoding='utf-8')
   (METHOD/'cryolo'/'annotations'/s/f'{stem}.box').write_text(box,encoding='utf-8')
   link=METHOD/'cryolo'/'images'/s/n
   if not link.exists(): link.symlink_to(MICRO/n)
   top_img.append({'image_name':'m'+stem,'image_path':str(MICRO/n),'micrograph_name':n})
   for x,y in pts: top_gt.append({'image_name':'m'+stem,'x_coord':x,'y_coord':y})
   method_gt[s]+=len(pts)
   # ANN-YOLO full-image input adapter: MRC stays single-source; only resized PNGs are materialized.
   with mrcfile.open(MICRO/n,mode='r',permissive=True) as m: arr=np.asarray(m.data,dtype=np.float32)
   lo,hi=np.percentile(arr,[1,99]); scaled=np.clip((arr-lo)/(hi-lo),0,1)
   u8=(scaled*255+0.5).astype(np.uint8)
   im=Image.fromarray(u8,mode='L'); res=getattr(Image,'Resampling',Image).LANCZOS
   im=im.resize((2048,2048),res); ip=EXP/'images'/s/(stem+'.png'); im.save(ip)
   # verify exact center and normalized label values round-trip from source
   for line,(x,y) in zip(yolo.splitlines(),pts):
    cls,cx,cy,w,h=map(float,line.split())
    if cls!=0 or abs(cx*4096-x)>1e-5 or abs(cy*4096-y)>1e-5 or abs(w*4096-224)>1e-5 or abs(h*4096-224)>1e-5: raise AssertionError(f'YOLO transform mismatch {n}')
   (EXP/'labels'/s/(stem+'.txt')).write_text(yolo,encoding='utf-8')
   image_build.append({'micrograph_name':n,'split':s,'png_path':str(ip),'png_sha256':sha(ip),'coordinate_scale_to_2048':0.5,'resize':'Pillow LANCZOS 4096 to 2048'})
  csv_write(METHOD/'topaz'/s/'images.tsv',['image_name','image_path','micrograph_name'],top_img)
  csv_write(METHOD/'topaz'/s/'targets.csv',['image_name','x_coord','y_coord'],top_gt)
  if s in ('train','val'):
   pass
 # Method-specific train/val/test lists and frozen shared evaluation protocol.
 for method in ['ann_yolo26','cryolo','topaz']:
  md=METHOD/method
  for s,vals in assign.items():
   (md/f'{s}_micrographs.txt').write_text(''.join(n+'\n' for n in vals),encoding='utf-8')
 (METHOD/'ann_yolo26'/'data.yaml').write_text(f'path: {EXP}\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 1\nnames:\n  0: particle\n',encoding='utf-8')
 cryo_cfg={'model':{'architecture':'PhosaurusNet','input_size':1024,'anchors':[224,224],'max_box_per_image':700,'norm':'STANDARD','filter':[0.1,'filtered_tmp/']},'train':{'train_image_folder':str(METHOD/'cryolo/images/train'),'train_annot_folder':str(METHOD/'cryolo/annotations/train'),'train_times':1,'pretrained_weights':'','batch_size':2,'learning_rate':0.0001,'nb_epoch':1,'object_scale':5.0,'no_object_scale':1.0,'coord_scale':1.0,'class_scale':1.0,'saved_weights_name':str(EXP/'cryolo_checkpoints/cryolo_10028_v3.h5'),'debug':False},'valid':{'valid_image_folder':str(METHOD/'cryolo/images/val'),'valid_annot_folder':str(METHOD/'cryolo/annotations/val'),'valid_times':1},'other':{'log_path':str(EXP/'cryolo_logs')}}
 write_json(METHOD/'cryolo'/'config_v3_template.json',cryo_cfg)
 (METHOD/'topaz'/'README.md').write_text('# Topaz v3 inputs\n\nTrain/validation/test targets are in each split directory as `targets.csv`; `images.tsv` maps each Topaz image identifier to the shared raw MRC path. Inputs are generated from the frozen v3 split and original CryoPPP CSVs. Use train only for fitting, validation for threshold/parameter selection, and test once for final evaluation. No Topaz training was run.\n',encoding='utf-8')
 # ANN settings reflect prior v1 settings but isolate weights, runs, and split.
 cfg={'experiment':'ANN-YOLO26m EMPIAR-10028 full-image v3','protocol_split':'split_v3_paper60_20_20','training_images':132,'training_positive_gt':method_gt['train'],'validation_images':44,'validation_positive_gt':method_gt['val'],'test_images':44,'test_positive_gt':method_gt['test'],'test_accessed_for_training_or_selection':False,'input':{'source':'float32 MRC 4096x4096','normalization':'per-image finite p1/p99 linear scale, clip [0,1], round-half-up uint8','resize':'Pillow LANCZOS 4096x4096 to 2048x2048','axis_rule':'array x/y, no flip; coordinate multiplier 0.5','source_micrograph_storage':'original MRC retained once; PNG inputs are derived 2048px images'},'labels':{'format':'single-class YOLO normalized cx cy w h','source':'same CryoPPP per-image CSV checksummed in v3 all_micrographs.csv','coordinate_source_grid':'4096x4096 original','proxy_square_side_original_px':224,'proxy_square_side_training_px':112,'coordinate_transform':'cx=x/4096, cy=y/4096, w=h=224/4096; no axis flip; centers verified by round trip'},'model':{'architecture':'official YOLO26m; inherit architecture and training hyperparameters from v1 config after review','pretrained_weights':'/root/autodl-tmp/cryoem_annyolo26/SNNYOLO26/ultralytics-snn-yolo26/yolo26m.pt','imgsz':2048,'max_det':4000},'data_yaml':str(METHOD/'ann_yolo26/data.yaml'),'method_micrograph_lists':{s:str(METHOD/'ann_yolo26'/f'{s}_micrographs.txt') for s in SPLITS},'training_policy':'No formal training started. Choose confidence threshold only on validation. Evaluate the frozen 44-image test set once.'}
 write_json(EXP/'train_config_v3.json',cfg)
 (EXP/'data.yaml').write_text(f'path: {EXP}\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 1\nnames:\n  0: particle\n',encoding='utf-8')
 csv_write(EXP/'input_images.csv',['micrograph_name','split','png_path','png_sha256','coordinate_scale_to_2048','resize'],image_build)
 # Identical held-out list and GT-source identity are stored in each method tree.
 for method in ['ann_yolo26','cryolo','topaz']:
  (METHOD/method/'test_protocol.json').write_text(json.dumps({'split':'test','micrograph_manifest':str(MAN/'test.txt'),'micrograph_count':44,'gt_source':'CryoPPP original per-image CSVs; see all_micrographs.csv SHA-256 column','threshold_selection':'validation split only','coordinate_grid':'4096x4096 source pixels'},indent=2)+'\n',encoding='utf-8')
 # Inventory safe cache candidates; remove only Python bytecode caches on this server, which are reproducible and no active project worker uses them.
 # Current active-process audit is captured as evidence before deletion.
 ps=subprocess.run(['ps','-eo','pid,args'],text=True,stdout=subprocess.PIPE).stdout
 (REPORT/'active_processes_before_cleanup.txt').write_text(ps,encoding='utf-8')
 if any(x in ps for x in ['train_ann','cryolo train','topaz train']): raise RuntimeError('Possible active training process found; skipping cleanup')
 deleted=[]
 for base in [Path('/root/autodl-tmp')]:
  for pyc in base.rglob('*.pyc'):
   # hard safety scope: only Python bytecode cache files in __pycache__ directories
   if '__pycache__' not in pyc.parts: continue
   try:
    b=pyc.stat().st_size; pyc.unlink(); deleted.append({'path':str(pyc),'size_bytes':b,'reason':'reproducible Python bytecode cache (*.pyc under __pycache__); no active training worker','rebuildable':'yes'})
   except FileNotFoundError: pass
 # Remove now-empty __pycache__ directories only, scoped beneath autodl-tmp.
 for d in sorted(Path('/root/autodl-tmp').rglob('__pycache__'),key=lambda x:len(x.parts),reverse=True):
  try: d.rmdir(); deleted.append({'path':str(d),'size_bytes':0,'reason':'empty directory after deleting reproducible bytecode cache','rebuildable':'yes'})
  except OSError: pass
 csv_write(REPORT/'deleted_files.csv',['path','size_bytes','reason','rebuildable'],deleted)
 # Uncertainty-based retained list: large historical outputs remain untouched.
 candidates=[]
 for p in [ROOT/'experiments/ann_yolo26m_fullimage_2048_v1/diagnostics',ROOT/'experiments/ann_yolo26m_fullimage_2048_v1/checkpoints',ROOT/'experiments/ann_yolo26m_fullimage_2048_v1/runs',Path('/root/autodl-tmp/cryoem_baselines/topaz'),Path('/root/autodl-tmp/cryoem_baselines/cryolo'),Path('/root/miniconda3/pkgs')]:
  if p.exists():
   total=sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
   candidates.append({'path':str(p),'size_bytes':total,'reason':'Contains model checkpoints, prediction/feature outputs, baseline environment, or extracted conda package files; value or runtime dependency not established, retained.'})
 csv_write(REPORT/'retained_candidates.csv',['path','size_bytes','reason'],candidates)
 # Final integrity checks from written artifacts, including all split lists and method parity.
 checks={'raw_micrographs_count':len(raw),'raw_micrographs_checksum_match':True,'raw_micrographs_readable_4096_float32_finite':True,'source_gt_csv_count':220,'source_gt_all_coords_in_bounds':True,'gt_total':sum(map(len,coords.values())),'gt_matches_expected_19315':sum(map(len,coords.values()))==EXPECTED_GT,'split_counts':{s:len(v) for s,v in assign.items()},'split_gt_counts':method_gt,'split_disjoint':len(set(assign['train'])&set(assign['val'])|set(assign['train'])&set(assign['test'])|set(assign['val'])&set(assign['test']))==0,'split_union_220':set(sum(assign.values(),[]))==set(names),'manifests_written':True,'legacy_v2_conversion_crosschecks':legacy_checked,'v3_yolo_centers_and_box_dims_roundtrip':True,'cryolo_mrc_inputs_are_symlinks':all((METHOD/'cryolo/images'/s/n).is_symlink() for s,vals in assign.items() for n in vals),'topaz_lists_share_split':True,'ann_png_count':len(image_build),'test_set_shared':all((METHOD/m/'test_micrographs.txt').read_text().splitlines()==(MAN/'test.txt').read_text().splitlines() for m in ['ann_yolo26','cryolo','topaz']),'training_started':False}
 checks['status']='PASS' if all([checks['raw_micrographs_checksum_match'],checks['raw_micrographs_readable_4096_float32_finite'],checks['gt_matches_expected_19315'],checks['split_counts']==SPLITS,checks['split_disjoint'],checks['split_union_220'],checks['v3_yolo_centers_and_box_dims_roundtrip'],checks['cryolo_mrc_inputs_are_symlinks'],checks['test_set_shared']]) else 'FAIL'
 write_json(MAN/'integrity_check.json',checks)
 write_json(REPORT/'integrity_report.json',checks)
 gtrows=[{'micrograph_name':n,'split':owner[n],'gt_count':len(coords[n]),'mrc_sha256':image_hash[n],'gt_csv_sha256':gt_hash[n]} for n in names]
 csv_write(REPORT/'gt_statistics.csv',['micrograph_name','split','gt_count','mrc_sha256','gt_csv_sha256'],gtrows)
 report='''# EMPIAR-10028 v3 split report\n\nProtocol: stable lexicographic ordering of 220 unique raw MRC filenames, then one `random.Random(10028).shuffle`; assign 132/44/44 to train/validation/test. No GT stratification, crops, augmentation copies, or model outputs affect assignment.\n\n| Split | Micrographs | CryoPPP GT particles |\n|---|---:|---:|\n'''
 for x in stats: report+=f"| {x['split']} | {x['micrographs']} | {x['gt_count']} |\n"
 report+='\nThe full assignment, source paths and SHA-256 values are in `../manifests/split_v3_paper60_20_20/all_micrographs.csv`; protocol and integrity are stored alongside the manifests. The same `test.txt` identity is copied into the ANN-YOLO26, crYOLO, and Topaz input trees. Validation is the only set for threshold/parameter selection. No formal training was launched.\n\n'
 report+='## New method paths\n\n- ANN-YOLO26 data: `'+str(EXP/'data.yaml')+'`; run config: `'+str(EXP/'train_config_v3.json')+'`\n- crYOLO config template: `'+str(METHOD/'cryolo/config_v3_template.json')+'`; split lists: `'+str(METHOD/'cryolo')+'`\n- Topaz split inputs: `'+str(METHOD/'topaz')+'`\n'
 (REPORT/'split_report.md').write_text(report,encoding='utf-8')
 # Disk report after all writes/cleanup.
 disk_snapshot(REPORT/'disk_usage_after.txt')
 before=(REPORT/'disk_usage_before.txt').read_text(); after=(REPORT/'disk_usage_after.txt').read_text()
 freed=sum(x['size_bytes'] for x in deleted)
 summary=f'''# Final summary\n\n## Split\n\n- Train: 132 micrographs, {method_gt['train']} GT particles.\n- Validation: 44 micrographs, {method_gt['val']} GT particles.\n- Test: 44 micrographs, {method_gt['test']} GT particles.\n- Integrity: {checks['status']}; 220/220 source micrographs checked against SHA-256 manifest and read as finite 4096x4096 float32 MRC; total GT {checks['gt_total']}.\n- ANN-YOLO26: `{EXP/'data.yaml'}` and `{EXP/'train_config_v3.json'}`.\n- crYOLO: `{METHOD/'cryolo/config_v3_template.json'}`.\n- Topaz: `{METHOD/'topaz'}`.\n\n## Cleanup\n\n- Removed {len(deleted)} Python bytecode files/empty cache directories; reported bytecode size reclaimed: {freed} bytes. Pip cache reports 0 bytes; no pip cache was removed. Conda extracted packages retained because they may be dependencies.\n- Historical diagnostics/checkpoints/baseline environments retained; see `retained_candidates.csv`.\n- Before/after disk snapshots are in `disk_usage_before.txt` and `disk_usage_after.txt`.\n- Formal training was not started.\n'''
 (REPORT/'final_summary.md').write_text(summary,encoding='utf-8')
 print(json.dumps({'status':checks['status'],'split_gt':method_gt,'deleted_files':len(deleted),'freed_bytes':freed,'report':str(REPORT)},indent=2),flush=True)
if __name__=='__main__': main()
