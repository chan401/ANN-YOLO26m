"""Bounded, reproducible v2 downloads; target accessions never enter training."""
from pathlib import Path
from datetime import datetime, timezone
import concurrent.futures, csv, hashlib, json, math, os, random, re, shutil, struct, tarfile, time
import urllib.request, urllib.parse,threading
import requests
import mrcfile
import numpy as np

G=Path('/root/autodl-tmp/cryoem_generalization/v2')
V1=G.parent/'v1'
CONFIG={
 '10289':dict(diameter=162,directory='data',role='development'),
 '10240':dict(diameter=156,directory='data/ceramide_micrographs',role='development'),
 '10760':dict(diameter=106,directory='data',role='development'),
 '10081':dict(diameter=154,directory='data/micrographs',role='final_check'),
 '10059':dict(diameter=132,directory='data/micrographs',role='final_check')}
SEED=100282
HTTP_LOCAL=threading.local()

def session():
 if not hasattr(HTTP_LOCAL,'session'):
  HTTP_LOCAL.session=requests.Session()
  HTTP_LOCAL.session.headers.update({'User-Agent':'ANN-cryo-generalization-v2/1.0','Accept-Encoding':'identity'})
 return HTTP_LOCAL.session

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def dump(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');os.replace(tmp,p)

def get(url):
 for i in range(5):
  try:
   with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'ANN-cryo-generalization-v2/1.0'}),timeout=60) as r:return r.read()
  except Exception:
   if i==4:raise
   time.sleep(2*(i+1))

def archive_bytes(ds,entry):
 if entry['size']<=10*1024*1024:return get(entry['links']['self'])
 url=f'https://zenodo.org/records/7934683/files/{ds}.tar.gz?download=1'
 size=entry['size'];step=1024*1024
 def part(start):
  end=min(size-1,start+step-1)
  for trial in range(5):
   try:
    req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
    with urllib.request.urlopen(req,timeout=60) as r:
     assert r.status==206 and r.headers.get('Content-Range')==f'bytes {start}-{end}/{size}'
     b=r.read();assert len(b)==end-start+1;return b
   except Exception:
    if trial==4:raise
    time.sleep(trial+1)
 with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:return b''.join(pool.map(part,range(0,size,step)))

def catalogs_and_protocol():
 p=G/'protocol.json'
 if p.exists():
  d=json.loads(p.read_text());assert sha(p)==(G/'protocol.sha256').read_text().strip();return d
 G.mkdir(parents=True,exist_ok=True)
 z=json.loads((V1/'catalogs/zenodo_7934683.json').read_text());zf={e['key']:e for e in z['files']}
 old=list(csv.DictReader((V1/'all_images.csv').open()))
 files=[];metadata={}
 for ds,cfg in CONFIG.items():
  print('CATALOG_START',ds,flush=True)
  base=G/'datasets'/ds;ann=base/'annotations_original';ann.mkdir(parents=True,exist_ok=True)
  if (V1/'datasets'/ds/'annotations_original').exists():
   for a in (V1/'datasets'/ds/'annotations_original').glob('*.csv'):
    dst=ann/a.name
    if not dst.exists():os.link(a,dst)
  else:
   entry=zf[ds+'.tar.gz'];archive=base/(ds+'.tar.gz')
   if not archive.exists():archive.write_bytes(archive_bytes(ds,entry))
   assert archive.stat().st_size==entry['size']
   assert hashlib.md5(archive.read_bytes()).hexdigest()==entry['checksum'].split(':',1)[1]
   with tarfile.open(archive,'r:gz') as t:
    for m in t:
     if m.isfile() and '/ground_truth/particle_coordinates/' in '/'+m.name and m.name.endswith('.csv'):
      (ann/Path(m.name).name).write_bytes(t.extractfile(m).read())
  meta=json.loads(get('https://www.ebi.ac.uk/empiar/api/entry/'+ds))['EMPIAR-'+ds]
  dump(base/'empiar_metadata.json',meta)
  directory=f'https://ftp.ebi.ac.uk/empiar/world_availability/{ds}/{cfg["directory"]}/'
  index=get(directory).decode();(base/'index.html').write_text(index)
  available={urllib.parse.unquote(x) for x in re.findall(r'href="([^"]+)"',index) if x.lower().endswith('.mrc')}
  annotated={a.stem+'.mrc' for a in ann.glob('*.csv')}
  excluded={r['micrograph_name'] for r in old if r['dataset']==ds}
  candidates=sorted((available&annotated)-excluded)
  assert len(candidates)>=32,(ds,len(candidates))
  sorted_sha=hashlib.sha256('\n'.join(candidates).encode()).hexdigest()
  random.Random(SEED).shuffle(candidates)
  groups={'train':candidates[:24],'val':candidates[24:32]} if cfg['role']=='development' else {'final_check':candidates[:20]}
  metadata[ds]={**cfg,'title':meta['title'],'candidate_count':len(candidates),'excluded_v1_count':len(excluded),'candidate_sorted_sha256':sorted_sha,'groups':groups}
  print('CATALOG_READY',ds,len(candidates),flush=True)
  for split,names in groups.items():
   for n in names:files.append(dict(dataset=ds,split=split,micrograph_name=n,url=directory+urllib.parse.quote(n),diameter=cfg['diameter']))
 protocol=dict(protocol_id='cryo-generalization-v2',created_utc=datetime.now(timezone.utc).isoformat(),seed=SEED,
  selection='stable filename sort; exclude every v1 image from new downloads; independent random.Random(100282).shuffle per accession; new domains first24 train,next8 val; target first20 final_check',
  existing_development_domains=['10028','10017','10077','10291'],additional_development_domains=['10289','10240','10760'],
  target_accessions=['10081','10059'],datasets=metadata,files=files,
  target_images_in_training=False,target_GT_used_for_checkpoint_or_threshold_selection=False,
  historical_target_results_already_known=True,
  interpretation='New micrograph-level final check on known target accessions. Target proteins/accessions excluded from ANN supervised training, but v1 target results informed development; not a pristine unseen-domain blind claim.',
  matching='unchanged maximum-cardinality one-to-one; native published diameter/4; HCN1 38.5px, TRPV1 33px; identical for all methods',
  final_check_policy='freeze checkpoint, global validation threshold and inference settings before opening final-check predictions; evaluate once; no target threshold scan; all failures retained')
 dump(p,protocol);(G/'protocol.sha256').write_text(sha(p)+'\n')
 print('PROTOCOL_FROZEN',sha(p),{ds:{s:len(ns) for s,ns in m['groups'].items()} for ds,m in metadata.items()},flush=True)
 return protocol

def header(r):
 cache=G/'datasets'/r['dataset']/'header_records'/(r['micrograph_name']+'.json')
 if cache.exists():return json.loads(cache.read_text())
 record=G/'datasets'/r['dataset']/'file_records'/(r['micrograph_name']+'.json')
 if record.exists():
  known=json.loads(record.read_text());return {k:known[k] for k in ['width','height','mode','bytes','header_sha256']}
 for i in range(7):
  try:
   with session().get(r['url'],headers={'Range':'bytes=0-1023'},timeout=(30,60)) as q:
    q.raise_for_status();b=q.content;assert len(b)==1024;cr=q.headers.get('Content-Range');total=int(cr.split('/')[-1]) if cr else int(q.headers['Content-Length'])
   w,h,z,mode=struct.unpack('<4i',b[:16]);assert z==1 and w>1000 and h>1000 and mode in [0,1,2,6,12]
   assert total>=1024+w*h*{0:1,1:2,2:4,6:2,12:2}[mode]
   meta=dict(width=w,height=h,mode=mode,bytes=total,header_sha256=hashlib.sha256(b).hexdigest());dump(cache,meta)
   print('HEADER_READY',r['dataset'],r['micrograph_name'],flush=True);return meta
  except Exception as e:
   print('HEADER_RETRY',r['dataset'],r['micrograph_name'],i,repr(e),flush=True)
   if i==6:raise
   time.sleep(min(60,10*(i+1)))

def validate(r,m,p):
 with mrcfile.open(p,permissive=True) as f:
  a=f.data;assert a.shape==(m['height'],m['width']) and np.isfinite(a).all()
  pixels=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
 gt=G/'datasets'/r['dataset']/'annotations_original'/(p.stem+'.csv')
 rr=list(csv.DictReader(gt.open(encoding='utf-8-sig')));xy=[(float(v['X-Coordinate']),float(v['Y-Coordinate'])) for v in rr]
 assert xy and len(xy)==len(set(xy))
 assert all(math.isfinite(x+y) and 0<=x<m['width'] and 0<=y<m['height'] for x,y in xy)
 assert all(float(v['Diameter'])==r['diameter'] for v in rr)
 d={**r,**m,'path':str(p),'sha256':sha(p),'pixel_sha256':pixels,'GT_path':str(gt),'GT_sha256':sha(gt),'GT_count':len(xy)}
 dump(p.parent.parent/'file_records'/(p.name+'.json'),d);return d

def download(protocol):
 selected=sorted(protocol['files'],key=lambda r:({'val':0,'train':1,'final_check':2}[r['split']],r['dataset'],r['micrograph_name']))
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:metas=list(pool.map(header,selected))
 missing_complete_bytes=sum(m['bytes'] for r,m in zip(selected,metas) if not (G/'datasets'/r['dataset']/'micrographs'/r['micrograph_name']).exists())
 reusable_partial_bytes=0
 for r,m in zip(selected,metas):
  if (G/'datasets'/r['dataset']/'micrographs'/r['micrograph_name']).exists():continue
  folder=G/'download_parts'/hashlib.sha256((r['dataset']+'\0'+r['micrograph_name']).encode()).hexdigest()[:24]
  for i in range((m['bytes']+4*1024*1024-1)//(4*1024*1024)):
   p=folder/f'{i:04}.part';expected=min(4*1024*1024,m['bytes']-i*4*1024*1024)
   have=p.stat().st_size if p.exists() else 0;assert 0<=have<=expected
   reusable_partial_bytes+=have
 missing=missing_complete_bytes-reusable_partial_bytes
 free=shutil.disk_usage(G).free;assert free>missing+1024**3,('Insufficient capacity',free,missing)
 dump(G/'capacity_plan.json',dict(free_bytes=free,additional_bytes=missing,missing_complete_file_bytes=missing_complete_bytes,reusable_partial_bytes=reusable_partial_bytes,reserve_bytes=1024**3))
 print('HEADERS_READY',len(metas),'additional_GiB',round(missing/1024**3,2),'free_GiB',round(free/1024**3,2),flush=True)
 chunk=4*1024*1024;jobs=[];states=[];out=[]
 for r,m in zip(selected,metas):
  p=G/'datasets'/r['dataset']/'micrographs'/r['micrograph_name'];p.parent.mkdir(exist_ok=True)
  if p.exists():
   assert p.stat().st_size==m['bytes'];out.append(validate(r,m,p));continue
  folder=G/'download_parts'/hashlib.sha256((r['dataset']+'\0'+p.name).encode()).hexdigest()[:24];folder.mkdir(parents=True,exist_ok=True)
  parts=[folder/f'{i:04}.part' for i in range((m['bytes']+chunk-1)//chunk)]
  si=len(states);states.append(dict(r=r,m=m,p=p,parts=parts,remaining=len(parts),etags=set()))
  jobs.extend((si,i,part) for i,part in enumerate(parts))
 def fetch(job):
  si,i,p=job;s=states[si];start=i*chunk;end=min(s['m']['bytes']-1,start+chunk-1);expected=end-start+1
  for attempt in range(7):
   try:
    have=p.stat().st_size if p.exists() else 0;assert have<=expected
    if have==expected:return si,None
    with session().get(s['r']['url'],headers={'Range':f'bytes={start+have}-{end}'},timeout=(30,60),stream=True) as q:
     assert q.status_code==206 and q.headers.get('Content-Range')==f'bytes {start+have}-{end}/{s["m"]["bytes"]}'
     etag=q.headers.get('ETag')
     with p.open('ab') as f:
      for b in q.iter_content(1024*1024):f.write(b)
    assert p.stat().st_size==expected;return si,etag
   except Exception as e:
    print('DOWNLOAD_RETRY',s['r']['dataset'],s['r']['micrograph_name'],'part',i,'attempt',attempt,repr(e),flush=True)
    if attempt==6:raise
    time.sleep(min(8,attempt+1))
 completed_parts=0;last_progress=time.monotonic()
 with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
  futures=[pool.submit(fetch,j) for j in jobs]
  for f in concurrent.futures.as_completed(futures):
   si,etag=f.result();s=states[si];s['remaining']-=1;completed_parts+=1
   if time.monotonic()-last_progress>30:
    print('DOWNLOAD_PARTS_PROGRESS',completed_parts,'/',len(jobs),flush=True);last_progress=time.monotonic()
   if etag:s['etags'].add(etag)
   assert len(s['etags'])<=1,'Remote file changed during download'
   if not s['remaining']:
    tmp=s['p'].with_suffix('.assembling')
    with tmp.open('wb') as h:
     for part in s['parts']:
      with part.open('rb') as q:shutil.copyfileobj(q,h,chunk)
    assert tmp.stat().st_size==s['m']['bytes']
    with tmp.open('rb') as q:assert hashlib.sha256(q.read(1024)).hexdigest()==s['m']['header_sha256']
    os.replace(tmp,s['p'])
    for part in s['parts']:part.unlink()
    s['parts'][0].parent.rmdir();out.append(validate(s['r'],s['m'],s['p']))
    print('MRC_READY',len(out),'/',len(selected),s['r']['dataset'],s['r']['split'],flush=True)
 assert len(out)==len(selected)
 with (G/'new_images.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 counts={}
 for r in out:
  k=r['dataset']+'/'+r['split'];v=counts.setdefault(k,dict(images=0,GT=0,bytes=0));v['images']+=1;v['GT']+=r['GT_count'];v['bytes']+=r['bytes']
 dump(G/'preparation_report.json',dict(status='PASS',images=len(out),counts=counts,protocol_sha256=sha(G/'protocol.json')))
 print('V2_DATA_READY',json.dumps(counts),flush=True)

if __name__=='__main__':download(catalogs_and_protocol())
