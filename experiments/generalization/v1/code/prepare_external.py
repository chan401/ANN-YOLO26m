"""Seeded cross-accession development/held-out data preparation, with bounded MRC-only downloads."""
from pathlib import Path
from datetime import datetime,timezone
import concurrent.futures,csv,hashlib,json,math,os,random,re,struct,time,urllib.request,urllib.parse,shutil
import mrcfile,numpy as np
G=Path('/root/autodl-tmp/cryoem_generalization/v1');SEED=10028
CONFIG={
 '10017':{'role':'development','diameter':108,'directory':'data','protein':'beta-galactosidase'},
 '10077':{'role':'development','diameter':216,'directory':'data/Micrographs_part2','protein':'E.coli70S ribosome-SelB'},
 '10291':{'role':'development','diameter':130,'directory':'data','protein':'INX6 hemichannel'},
 '10075':{'role':'heldout','diameter':233,'directory':'data/MS2_summed_images','protein':'MS2 bacteriophage'},
 '10081':{'role':'heldout','diameter':154,'directory':'data/micrographs','protein':'human HCN1 channel'},
 '10059':{'role':'heldout','diameter':132,'directory':'data/micrographs','protein':'TRPV1 channel in nanodisc'}}
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');os.replace(t,p)
def csvout(p,rows):
 with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def url_for(ds,name):return f'https://ftp.ebi.ac.uk/empiar/world_availability/{ds}/{CONFIG[ds]["directory"]}/'+urllib.parse.quote(name)
def header(ds,name):
 u=url_for(ds,name)
 for trial in range(5):
  try:
   req=urllib.request.Request(u,headers={'Range':'bytes=0-1023','Accept-Encoding':'identity','User-Agent':'cryo-generalization/1.0'})
   with urllib.request.urlopen(req,timeout=40) as r:
    data=r.read(1024);cr=r.headers.get('Content-Range','');total=int(cr.split('/')[-1]) if cr else int(r.headers.get('Content-Length',0))
   nx,ny,nz,mode=struct.unpack('<4i',data[:16]);assert nz==1 and nx>1000 and ny>1000 and mode in [0,1,2,6,12],(ds,name,nx,ny,nz,mode)
   return {'width':nx,'height':ny,'nz':nz,'mode':mode,'bytes':total,'header_sha256':hashlib.sha256(data).hexdigest()}
  except Exception:
   if trial==4:raise
   time.sleep(2*(trial+1))
def select():
 p=G/'protocol.json'
 if p.exists():return json.loads(p.read_text())
 chosen={};inputs=[]
 for ds,cfg in CONFIG.items():
  text=(G/'catalogs'/(ds+'_index.html')).read_text();available={urllib.parse.unquote(x) for x in re.findall(r'href="([^"]+)"',text) if x.lower().endswith('.mrc')}
  ann={p.stem+'.mrc' for p in (G/'datasets'/ds/'annotations_original').glob('*.csv')};candidates=sorted(ann&available)
  assert len(candidates)>=40,(ds,len(candidates));random.Random(SEED).shuffle(candidates)
  groups={'train':candidates[:24],'val':candidates[24:32],'probe':candidates[32:40]} if cfg['role']=='development' else {'heldout':candidates[:12]}
  chosen[ds]={'metadata':cfg,'candidate_count':len(candidates),'candidate_sorted_sha256':hashlib.sha256('\n'.join(sorted(ann&available)).encode()).hexdigest(),'groups':groups}
  for split,ns in groups.items():
   for name in ns:inputs.append({'dataset':ds,'split':split,'micrograph_name':name,'url':url_for(ds,name)})
 A=Path('/root/autodl-tmp/cryoem_datasets/EMPIAR-10028/experiments/ann_yolo26m_cryo_tiles_v3')
 s=json.loads((A.parent.parent/'reports/ann_yolo26_cryo_adaptation_v3/selected_model_config.json').read_text())
 protocol={'protocol_id':'cryo-generalization-v1','created_utc':now(),'seed':SEED,'selection':'stable filename sort, random.Random(10028).shuffle per accession; first24train,next8val,next8development_probe; first12heldout on other accessions',
  'datasets':chosen,'source_training_accession':'10028','base_checkpoint':s['checkpoint'],'base_checkpoint_sha256':s['checkpoint_sha256'],'source_fixed_threshold':s['single_global_validation_threshold'],
  'source_train_manifest':str(A.parent.parent/'manifests/split_v3_paper60_20_20/train.txt'),'source_val_manifest':str(A.parent.parent/'manifests/split_v3_paper60_20_20/val.txt'),
  'heldout_accessions_excluded_from_training_and_parameter_selection':True,'micrographs_per_development_accession':40,'micrographs_per_heldout_accession':12,
  'selection_uses_GT_counts_or_model_performance':False,'initial_native_zero_shot':'same base checkpoint and original global validation threshold; generic image geometry only; no external finetuning',
  'diameter_informed_variant':'rescale known published particle diameter to224px using dataset metadata, not target center labels; restore predictions to native pixels',
  'matching':'same deterministic maximum-cardinality one-to-one primitives; inclusive radius=published_particle_diameter/4 in native pixels;10028 stays56px',
  'metrics':'TP FP FN Precision Recall F1 per dataset, macro-average across datasets and pooled counts; operational failures remain in denominators',
  'improvement_policy':'develop only on10017/10077/10291 train,val,probe; heldout10075/10081/10059 evaluated once after checkpoint/global threshold/inference config freeze; do not adapt based on heldout results',
  'pretraining_provenance':'ANN selected model uses natural-image YOLO official initialization plus10028 fine-tuning; crYOLO/Topaz official pretrained sets may overlap external datasets and cannot be claimed independently trained.',
  'scope':'finite sampled external-accession benchmark, not a guarantee of universal generalization','files':inputs}
 dump(p,protocol);(G/'protocol.sha256').write_text(sha(p)+'\n');print('DATASET_ROLES_AND_SELECTION_FROZEN',len(inputs),sha(p),flush=True)
 return protocol
def download_one(r):
 ds,name=r['dataset'],r['micrograph_name'];root=G/'datasets'/ds;raw=root/'micrographs';raw.mkdir(exist_ok=True);p=raw/name;meta=header(ds,name)
 if not(p.exists() and p.stat().st_size==meta['bytes']):
  tmp=p.with_suffix('.mrc.partial');have=tmp.stat().st_size if tmp.exists() else 0
  if have>meta['bytes']:raise ValueError('Owned partial oversized')
  for trial in range(8):
   try:
    have=tmp.stat().st_size if tmp.exists() else 0
    if have==meta['bytes']:break
    end=meta['bytes']-1;req=urllib.request.Request(r['url'],headers={'Range':f'bytes={have}-{end}','Accept-Encoding':'identity','User-Agent':'cryo-generalization/1.0'})
    with urllib.request.urlopen(req,timeout=90) as response:
     cr=response.headers.get('Content-Range','')
     if response.status!=206 or cr!=f'bytes {have}-{end}/{meta["bytes"]}':raise ValueError(f'Unexpected range response {response.status} {cr}')
     with tmp.open('ab') as h:
      for b in iter(lambda:response.read(1024*1024),b''):h.write(b)
    if tmp.stat().st_size!=meta['bytes']:raise ValueError('Incomplete MRC body')
    break
   except Exception as e:
    print('DOWNLOAD_RETRY',ds,name,trial,repr(e),flush=True)
    if trial==7:raise
    time.sleep(2*(trial+1))
  assert tmp.stat().st_size==meta['bytes'];os.replace(tmp,p)
 with mrcfile.open(p,permissive=True) as m:
  a=m.data;assert a.ndim==2 and np.isfinite(a).all();assert a.shape==(meta['height'],meta['width'])
 annotation=root/'annotations_original'/(Path(name).stem+'.csv')
 rr=list(csv.DictReader(annotation.open(newline='',encoding='utf-8-sig')));points=[(float(v['X-Coordinate']),float(v['Y-Coordinate'])) for v in rr]
 assert points and len(points)==len(set(points)),('empty_or_duplicate_GT',ds,name)
 assert all(0<=x<meta['width'] and 0<=y<meta['height'] and math.isfinite(x+y) for x,y in points)
 assert all(float(v['Diameter'])==CONFIG[ds]['diameter'] for v in rr)
 out={**r,**meta,'path':str(p),'sha256':sha(p),'GT_path':str(annotation),'GT_sha256':sha(annotation),'GT_count':len(points),'diameter':CONFIG[ds]['diameter']}
 dump(root/'file_records'/(name+'.json'),out);print('MRC_READY',ds,r['split'],name,meta['bytes'],len(points),flush=True);return out
def main():
 protocol=select();records=G/'all_images.csv'
 # At most~9GB from156 selected images; reserve2GB for outputs before starting.
 projected=sum((4096*4096*4+1024) if r['dataset'] in ['10017','10077','10075'] else (3838*3710*4+1024) for r in protocol['files'])
 existing=sum(p.stat().st_size for p in (G/'datasets').glob('*/micrographs/*.mrc'))
 free=shutil.disk_usage(G).free;assert free+existing>projected+2*1024**3,('insufficient_capacity',free,projected,existing)
 dump(G/'capacity_plan.json',{'free_bytes_before':free,'conservative_projected_MRC_bytes':projected,'existing_selected_bytes':existing,'reserve_bytes':2*1024**3,'old_test100_deleted':False})
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:out=list(pool.map(download_one,protocol['files']))
 csvout(records,out);counts={}
 for r in out:
  key=r['dataset']+'/'+r['split'];counts.setdefault(key,{'images':0,'GT':0,'bytes':0});counts[key]['images']+=1;counts[key]['GT']+=r['GT_count'];counts[key]['bytes']+=r['bytes']
 dump(G/'preparation_report.json',{'status':'PASS','completed_utc':now(),'images':len(out),'counts':counts,'protocol_sha256':sha(G/'protocol.json'),'all_selected_files_complete_and_finite':True,'old_data_modified_or_deleted':False})
 print('EXTERNAL_DATA_PREPARATION_COMPLETE',json.dumps(counts),flush=True)
if __name__=='__main__':main()
