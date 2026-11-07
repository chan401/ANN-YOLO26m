"""Resume only selected MRCs with bounded parallel HTTP ranges; assemble and validate."""
from pathlib import Path
import concurrent.futures,hashlib,json,os,time,urllib.request,shutil
from prepare_external import G,header,download_one,csvout,dump,now
CHUNK=4*1024*1024

def main():
 protocol=json.loads((G/'protocol.json').read_text());selected=sorted(protocol['files'],key=lambda r:({'probe':0,'val':1,'train':2,'heldout':3}[r['split']],r['dataset']));(G/'download_parts').mkdir(exist_ok=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
  metas=list(pool.map(lambda r:header(r['dataset'],r['micrograph_name']),selected))
 print('ALL_SELECTED_HEADERS_VERIFIED',len(metas),sum(m['bytes'] for m in metas),flush=True)
 states=[];jobs=[]
 for r,m in zip(selected,metas):
  raw=G/'datasets'/r['dataset']/'micrographs';raw.mkdir(exist_ok=True);dst=raw/r['micrograph_name']
  if dst.exists() and dst.stat().st_size==m['bytes']:continue
  key=hashlib.sha256((r['dataset']+'\0'+r['micrograph_name']).encode()).hexdigest()[:24];folder=G/'download_parts'/key;folder.mkdir(exist_ok=True)
  parts=[folder/f'{i:04d}.part' for i in range((m['bytes']+CHUNK-1)//CHUNK)]
  prefix=dst.with_suffix('.mrc.partial')
  if prefix.exists():
   assert prefix.stat().st_size<=m['bytes']
   with prefix.open('rb') as h:
    for i,p in enumerate(parts):
     b=h.read(CHUNK)
     if not b:break
     if not p.exists() or p.stat().st_size<len(b):p.write_bytes(b)
   prefix.unlink() # explicit owned interrupted download only
  index=len(states);states.append({'r':r,'m':m,'dst':dst,'parts':parts,'remaining':len(parts),'etags':set()})
  for i,p in enumerate(parts):jobs.append((index,i,p))
 def fetch(job):
  si,i,p=job;s=states[si];start=i*CHUNK;end=min(s['m']['bytes']-1,start+CHUNK-1);expected=end-start+1
  for attempt in range(7):
   try:
    have=p.stat().st_size if p.exists() else 0;assert have<=expected
    if have==expected:return si,None
    req=urllib.request.Request(s['r']['url'],headers={'Range':f'bytes={start+have}-{end}','Accept-Encoding':'identity','User-Agent':'cryo-generalization-bounded-ranges/1.0'})
    with urllib.request.urlopen(req,timeout=70) as response:
     assert response.status==206 and response.headers.get('Content-Range')==f'bytes {start+have}-{end}/{s["m"]["bytes"]}'
     etag=response.headers.get('ETag')
     with p.open('ab') as h:
      for b in iter(lambda:response.read(1024*1024),b''):h.write(b)
    if p.stat().st_size!=expected:raise ValueError('Range length mismatch')
    return si,etag
   except Exception as e:
    if attempt==6:raise
    time.sleep(min(10,2*(attempt+1)))
 complete=0;beg=time.monotonic();last=0;validation=[]
 verify_pool=concurrent.futures.ThreadPoolExecutor(max_workers=2)
 for r in selected:
  dst=G/'datasets'/r['dataset']/'micrographs'/r['micrograph_name']
  if dst.exists():validation.append(verify_pool.submit(download_one,r))
 with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
  futures=[pool.submit(fetch,j) for j in jobs]
  for f in concurrent.futures.as_completed(futures):
   si,etag=f.result();s=states[si];s['remaining']-=1
   if etag:s['etags'].add(etag)
   assert len(s['etags'])<=1,'Remote source changed between ranges'
   if not s['remaining']:
    tmp=s['dst'].with_suffix('.assembling')
    with tmp.open('wb') as h:
     for p in s['parts']:
      with p.open('rb') as q:shutil.copyfileobj(q,h,4*1024*1024)
    assert tmp.stat().st_size==s['m']['bytes']
    with tmp.open('rb') as h:assert hashlib.sha256(h.read(1024)).hexdigest()==s['m']['header_sha256']
    os.replace(tmp,s['dst'])
    for p in s['parts']:p.unlink()
    s['parts'][0].parent.rmdir();complete+=1
    print('ASSEMBLED',complete,'/',len(states),s['r']['dataset'],s['r']['split'],s['r']['micrograph_name'],flush=True)
    validation.append(verify_pool.submit(download_one,s['r']))
   if time.monotonic()-last>30:
    done=sum(1 for p in (G/'datasets').glob('*/micrographs/*.mrc'));print('DOWNLOAD_PROGRESS',done,'/',len(selected),'elapsed',round(time.monotonic()-beg),flush=True);last=time.monotonic()
 out=[f.result() for f in validation];verify_pool.shutdown();assert len(out)==len(selected)
 csvout(G/'all_images.csv',out);counts={}
 for r in out:
  key=r['dataset']+'/'+r['split'];counts.setdefault(key,{'images':0,'GT':0,'bytes':0});counts[key]['images']+=1;counts[key]['GT']+=r['GT_count'];counts[key]['bytes']+=r['bytes']
 dump(G/'preparation_report.json',{'status':'PASS','completed_utc':now(),'images':len(out),'counts':counts,'all_selected_files_complete_and_finite':True,'protocol_sha256':hashlib.sha256((G/'protocol.json').read_bytes()).hexdigest()})
 print('EXTERNAL_PREPARATION_COMPLETE',json.dumps(counts),flush=True)
if __name__=='__main__':main()
