"""Shared frozen-input contract for the one-time test_100_v1 evaluation."""
import csv,hashlib,json,math,os,sys
from pathlib import Path
from datetime import datetime,timezone
D=Path('/root/autodl-tmp/cryoem_datasets/EMPIAR-10028')
E=D/'experiments/test_100_v1_three_methods'
R=D/'reports/test_100_v1_evaluation'
M=D/'manifests/test_100_v1'
FREEZE=E/'frozen_config.json'
FIELDS=['method','split','micrograph_name','x_original_px','y_original_px','score']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def dump(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');os.replace(tmp,p)
def names():
    out=[Path(x.strip()).name for x in (M/'test.txt').read_text().splitlines() if x.strip()]
    if len(out)!=100 or len(set(out))!=100:raise ValueError('test_100_v1 must have 100 unique images')
    return out
def frozen():
    seal=(E/'frozen_config.sha256').read_text().strip()
    if sha(FREEZE)!=seal:raise ValueError('Frozen configuration seal mismatch')
    f=json.loads(FREEZE.read_text())
    if not f.get('frozen') or f.get('matching_radius_px')!=56.0:raise ValueError('Invalid frozen test contract')
    for p,h in f['protected_files'].items():
        if sha(p)!=h:raise ValueError(f'Frozen source/config/input changed: {p}')
    for method,item in f['methods'].items():
        if sha(item['checkpoint'])!=item['checkpoint_sha256']:raise ValueError(f'{method}: frozen weight mismatch')
    return f
def write_csv(p,rows,fields=None):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def prediction_rows(p,method):
    allowed=set(names());rows=[]
    with Path(p).open(newline='') as f:
        reader=csv.DictReader(f)
        if not set(FIELDS).issubset(reader.fieldnames or []):raise ValueError(f'Bad prediction schema: {p}')
        for row in reader:
            x,y,s=[float(row[k]) for k in ['x_original_px','y_original_px','score']]
            if row['method']!=method or row['split']!='test' or row['micrograph_name'] not in allowed:raise ValueError('Wrong method, split or member')
            if not all(map(math.isfinite,[x,y,s])) or not(0<=x<4096 and 0<=y<4096):raise ValueError('Nonfinite/out-of-range prediction')
            rows.append(row)
    return rows
