import concurrent.futures,hashlib,os,subprocess,sys,time,urllib.request,threading
D="/root/autodl-tmp/cryoem_datasets/EMPIAR-10028"
BASE="https://ftp.ebi.ac.uk/empiar/world_availability/10028/data/Micrographs/Micrographs_part1"
SIZE=67109888
CHUNK=16*1024*1024
FLOOR=15_000_000_000
target=[x.strip().removesuffix(".mrc") for x in open(D+"/manifests/target_220.txt") if x.strip()]
os.makedirs(D+"/metadata/.download_parts",exist_ok=True)
existing={n for n in target if os.path.exists(D+"/micrographs/"+n+".mrc") and os.path.getsize(D+"/micrographs/"+n+".mrc")==SIZE}
need=[n for n in target if n not in existing]
stat=os.statvfs("/root/autodl-tmp")
avail=stat.f_bavail*stat.f_frsize
remaining=len(need)*SIZE
print("preflight target",len(target),"complete",len(existing),"need",len(need),"required_bytes",remaining,"available_bytes",avail,"post_free_bytes",avail-remaining,flush=True)
if avail-remaining<FLOOR: raise SystemExit("STOP: projected free space under 15 GB")
lock=threading.Lock()
def get_part(arg):
    name,index=arg
    start=index*CHUNK
    end=min(SIZE-1,start+CHUNK-1)
    expected=end-start+1
    path=f"{D}/metadata/.download_parts/{name}.chunk{index}"
    url=f"{BASE}/{name}.mrc"
    for attempt in range(1,9):
        have=os.path.getsize(path) if os.path.exists(path) else 0
        if have>expected:
            os.remove(path);have=0
        if have==expected: return path
        req=urllib.request.Request(url,headers={"Range":f"bytes={start+have}-{end}","User-Agent":"cryoem-dataset-audit/1.0"})
        try:
            with urllib.request.urlopen(req,timeout=120) as response:
                if response.status!=206: raise RuntimeError(f"range status {response.status}")
                cr=response.headers.get("Content-Range","")
                if not cr.startswith(f"bytes {start+have}-"): raise RuntimeError(f"bad Content-Range {cr}")
                with open(path,"ab") as out:
                    while True:
                        block=response.read(1024*1024)
                        if not block: break
                        out.write(block)
            if os.path.getsize(path)!=expected: raise RuntimeError(f"short chunk {os.path.getsize(path)}/{expected}")
            return path
        except Exception as e:
            print(f"retry {name} chunk{index} attempt{attempt}: {e}",flush=True)
            time.sleep(min(30,attempt*2))
    raise RuntimeError(f"failed {name} chunk {index}")
def one(name):
    n_parts=(SIZE+CHUNK-1)//CHUNK
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_parts) as pool:
        parts=list(pool.map(get_part,[(name,i) for i in range(n_parts)]))
    tmp=D+"/micrographs/"+name+".mrc.assembling"
    with open(tmp,"wb") as out:
        for p in parts:
            with open(p,"rb") as part:
                while True:
                    b=part.read(4*1024*1024)
                    if not b:break
                    out.write(b)
    actual=os.path.getsize(tmp)
    if actual!=SIZE:
        os.remove(tmp);raise RuntimeError(f"{name} size {actual} != {SIZE}")
    os.replace(tmp,D+"/micrographs/"+name+".mrc")
    h=hashlib.sha256()
    with open(D+"/micrographs/"+name+".mrc","rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    for p in parts: os.remove(p)
    with lock:
        with open(D+"/manifests/downloaded_sha256.txt","a") as f:f.write(f"{h.hexdigest()}  micrographs/{name}.mrc\n")
    return name
completed=0
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures={pool.submit(one,n):n for n in need}
    for fut in concurrent.futures.as_completed(futures):
        n=fut.result();completed+=1
        print(f"complete {completed}/{len(need)} {n}.mrc",flush=True)
print("DOWNLOAD_COMPLETE total",len(target),"added",len(need),flush=True)
