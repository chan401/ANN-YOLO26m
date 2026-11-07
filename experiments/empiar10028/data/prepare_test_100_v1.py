"""Prepare the immutable EMPIAR-10028 test_100_v1 extension; no model execution."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,concurrent.futures,csv,hashlib,json,math,os,random,re,shutil,time,urllib.request
import mrcfile
import numpy as np

D=Path('/root/autodl-tmp/cryoem_datasets/EMPIAR-10028')
OLD=D/'manifests/split_v3_paper60_20_20'
M=D/'manifests/test_100_v1'
A=D/'annotations_standardized/test_100_v1'
R=D/'reports/test_100_v1'
G=D/'annotations_original/cryoppp_10028/10028/ground_truth/particle_coordinates'
PART=D/'metadata/test_100_v1_download_parts'
BASE='https://ftp.ebi.ac.uk/empiar/world_availability/10028/data/Micrographs/Micrographs_part1'
SEED=10028;SIZE=67109888;CHUNK=16*1024*1024

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def dump(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');os.replace(tmp,p)

def csvout(p,rows,fields=None):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)

def readnames(p):return [Path(x.strip()).name for x in p.read_text().splitlines() if x.strip() and not x.startswith('#')]
def textlist(p,names):p.write_text('\n'.join(names)+'\n')
def link(source,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.is_symlink():
        if dest.resolve()!=source.resolve():raise RuntimeError(f'Unexpected link target {dest}')
    elif dest.exists():raise FileExistsError(dest)
    else:dest.symlink_to(source)

def training_state():
    E=D/'experiments/ann_yolo26m_cryo_tiles_v3'
    state=json.loads((E/'workflow_status.json').read_text())
    p=E/'runs/standard_m_p3p5/results.csv';rows=list(csv.DictReader(p.open()))
    return {'status':state['status'],'completed_epochs':len(rows)}

def protected():
    paths=list(OLD.glob('*'))
    E=D/'experiments/ann_yolo26m_cryo_tiles_v3'
    paths += [E/'data.yaml',E/'tiles.csv',E/'standard_m_p3p5_config.json',E/'code/common.py',E/'code/train.py',E/'code/infer_eval.py']
    return {str(p):sha(p) for p in paths if p.is_file()}

def select():
    s={k:readnames(OLD/f'{k}.txt') for k in ['train','val','test']}
    assert {k:len(v) for k,v in s.items()}=={'train':132,'val':44,'test':44}
    allold=sum(s.values(),[]);assert len(allold)==len(set(allold))==220
    catalog=list(csv.DictReader((D/'manifests/candidates_300.csv').open()))
    catalog_names={r['mrc'] for r in catalog};assert len(catalog_names)==300
    assert catalog_names=={p.stem+'.mrc' for p in G.glob('*.csv')}
    # Only historical membership lists count as use; candidate catalogs do not.
    historical=set(allold)
    for root in [D/'manifests']:
        for p in root.rglob('*.txt'):
            if M in p.parents:continue
            if p.name in ['train.txt','val.txt','test.txt','target_220.txt','split_train.txt','split_validation.txt','split_test.txt']:
                historical.update(n for n in readnames(p) if n in catalog_names)
    assert historical==set(allold),'Historical used images outside the frozen 220 require explicit resolution'
    candidates=sorted(catalog_names-historical)
    assert len(candidates)==80
    shuffled=candidates.copy();random.Random(SEED).shuffle(shuffled);added=shuffled[:56]
    raw={p.name for p in (D/'micrographs').glob('*.mrc')}
    # Existing eligible downloads are reused if a previous compatible run partially completed.
    existing_eligible=sorted(raw-set(allold))
    if set(existing_eligible)-set(added):
        raise RuntimeError('Existing eligible files differ from the seeded 56; preserve them and resolve inclusion before downloading')
    M.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
    p=M/'selection_protocol.json'
    if p.exists():
        protocol=json.loads(p.read_text());assert protocol['seed']==SEED and protocol['selected_new56']==added
    else:
        protocol={'configuration':'test_100_v1','created_utc':datetime.now(timezone.utc).isoformat(),
            'seed':SEED,'algorithm':'Python random.Random(10028); lexicographically sort the 80 annotation-backed identifiers, shuffle once, select first 56',
            'selection_uses_gt_counts':False,'selection_uses_model_results':False,'original_test44':s['test'],'selected_new56':added,
            'candidate_count':80,'candidate_order_before_shuffle':candidates,'candidate_order_after_shuffle':shuffled,
            'excluded_previously_used220':sorted(historical),'eligible_preexisting_downloads_reused':existing_eligible,
            'selected_before_reading_gt_contents':True,'original_v3_and_training_input_hashes':protected(),'training_state_before':training_state(),
            'GT_download_policy':'All 300 original per-image CryoPPP positive GT CSVs already exist locally. Reuse only the selected files; no GT redownload.',
            'micrograph_source_base':BASE,'status':'selection_locked'}
        dump(p,protocol)
        textlist(M/'candidates80_sorted.txt',candidates);textlist(M/'candidates80_shuffled.txt',shuffled)
        textlist(M/'added56.txt',added);shutil.copy2(OLD/'test.txt',M/'original_test44.txt')
        textlist(M/'test.txt',s['test']+added)
        link(OLD/'train.txt',M/'train.txt');link(OLD/'val.txt',M/'val.txt')
    assert not(set(added)&set(allold));assert len(set(s['test']+added))==100
    print('SELECTION_LOCKED',len(candidates),len(added),'seed',SEED,flush=True)
    print('ADDED56',' '.join(added),flush=True)
    return s,added,protocol

def getgt(name):
    src=G/(Path(name).stem+'.csv')
    rows=list(csv.DictReader(src.open(newline='',encoding='utf-8-sig')));pts=[]
    for row in rows:
        x,y,d=map(float,[row['X-Coordinate'],row['Y-Coordinate'],row['Diameter']])
        if not(math.isfinite(x) and math.isfinite(y) and 0<=x<4096 and 0<=y<4096 and d==224):raise ValueError(f'Invalid positive GT in {src}')
        pts.append((x,y))
    assert len(pts)==len(set(pts)) and pts
    return pts,sha(src)

def getpart(name,i):
    start=i*CHUNK;end=min(SIZE-1,start+CHUNK-1);expected=end-start+1
    path=PART/f'{Path(name).stem}.chunk{i}.part';url=BASE+'/'+name
    for attempt in range(1,9):
        have=path.stat().st_size if path.exists() else 0
        if have>expected:raise RuntimeError(f'Oversized owned partial {path}')
        if have==expected:return path,{'resumed_complete_part':True}
        req=urllib.request.Request(url,headers={'Range':f'bytes={start+have}-{end}','Accept-Encoding':'identity','User-Agent':'EMPIAR-10028-test100-preparation/1.0'})
        try:
            with urllib.request.urlopen(req,timeout=120) as r:
                cr=r.headers.get('Content-Range','')
                if r.status!=206 or cr!=f'bytes {start+have}-{end}/{SIZE}':raise RuntimeError(f'Unexpected range response {r.status}, {cr}')
                meta={'content_range':cr,'etag':r.headers.get('ETag'),'last_modified':r.headers.get('Last-Modified')}
                with path.open('ab') as f:
                    for b in iter(lambda:r.read(1024*1024),b''):f.write(b)
            if path.stat().st_size!=expected:raise RuntimeError('Incomplete HTTP range body')
            return path,meta
        except Exception as e:
            print('RETRY',name,i,attempt,str(e),flush=True);time.sleep(min(20,attempt*2))
    raise RuntimeError(f'Failed downloading {name} part {i}')

def validate_mrc(path):
    if path.stat().st_size!=SIZE:raise ValueError(f'Unexpected size {path}')
    with mrcfile.open(path,mode='r',permissive=True) as m:
        a=m.data
        if a.shape!=(4096,4096) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError(f'Invalid MRC {path}')
        return {'shape':[4096,4096],'dtype':'float32','all_pixels_finite':True,'min':float(a.min()),'max':float(a.max())}

def download(name):
    dst=D/'micrographs'/name
    if dst.exists():return {'micrograph_name':name,'path':str(dst),'bytes':dst.stat().st_size,'sha256':sha(dst),'reused_existing':True,'mrc':validate_mrc(dst)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        parts=list(pool.map(lambda i:getpart(name,i),range((SIZE+CHUNK-1)//CHUNK)))
    etags={meta['etag'] for _,meta in parts if meta.get('etag')}
    if len(etags)>1:raise RuntimeError(f'ETag disagreement for {name}')
    tmp=PART/(name+'.assembling')
    with tmp.open('wb') as f:
        for p,_ in parts:
            with p.open('rb') as g:shutil.copyfileobj(g,f,4*1024*1024)
    check=validate_mrc(tmp);h=sha(tmp)
    if dst.exists():raise FileExistsError(f'Refusing to overwrite {dst}')
    os.replace(tmp,dst)
    for p,_ in parts:p.unlink()  # explicit owned selected-file chunk paths only
    return {'micrograph_name':name,'path':str(dst),'bytes':SIZE,'sha256':h,'reused_existing':False,'source_url':BASE+'/'+name,'http_ranges':[m for _,m in parts],'mrc':check}

def organize(s,added,protocol,gt,gtsha,downloads):
    test=s['test']+added;oldrows=list(csv.DictReader((OLD/'all_micrographs.csv').open()));oldby={r['micrograph_name']:r for r in oldrows}
    known={r['mrc_sha256']:r['micrograph_name'] for r in oldrows};newhash={}
    for name in added:
        h=downloads[name]['sha256']
        if h in known or h in newhash:raise RuntimeError(f'Duplicate MRC contents {name}')
        newhash[h]=name
    manifest=[];centers=[];topimages=[];toptargets=[]
    for name in test:
        stem=Path(name).stem;isnew=name in set(added);mrc=D/'micrographs'/name;src=G/(stem+'.csv')
        if not mrc.is_file() or mrc.stat().st_size!=SIZE:raise ValueError(f'Missing/incomplete MRC {name}')
        with mrc.open('rb') as f:
            import struct
            if struct.unpack('<4i',f.read(16))!=(4096,4096,1,2):raise ValueError(f'Invalid header {name}')
        if not isnew:
            assert len(gt[name])==int(oldby[name]['gt_count']) and gtsha[name]==oldby[name]['cryoppp_csv_sha256']
        coord=A/'ann_yolo26/coords/test'/(stem+'.coord');box=A/'cryolo/annotations/test'/(stem+'.box')
        if isnew:
            coord.parent.mkdir(parents=True,exist_ok=True);box.parent.mkdir(parents=True,exist_ok=True)
            coord.write_text(''.join(f'{x:g} {y:g}\n' for x,y in gt[name]))
            box.write_text(''.join(f'{x-112:g}\t{y-112:g}\t224\t224\n' for x,y in gt[name]))
        else:
            oldann=D/'annotations_standardized/split_v3_paper60_20_20'
            if not coord.exists():link(oldann/'ann_yolo26/coords/test'/(stem+'.coord'),coord)
            if not box.exists():link(oldann/'cryolo/annotations/test'/(stem+'.box'),box)
        for method in ['ann_yolo26','cryolo','topaz']:
            dest=A/method/'micrographs/test'/name
            if not dest.exists():link(mrc,dest)
        coords=[tuple(map(float,l.split())) for l in coord.read_text().splitlines() if l.strip()]
        boxes=[tuple(map(float,l.split())) for l in box.read_text().splitlines() if l.strip()]
        assert coords==gt[name] and [(x+112,y+112) for x,y,w,h in boxes]==gt[name]
        assert all(w==h==224 for x,y,w,h in boxes)
        mh=downloads[name]['sha256'] if isnew else oldby[name]['mrc_sha256']
        manifest.append({'micrograph_name':name,'micrograph_id':stem,'original_path':str(mrc),'membership':'new56' if isnew else 'original44',
            'gt_count':len(gt[name]),'mrc_sha256':mh,'cryoppp_csv_path':str(src),'cryoppp_csv_sha256':gtsha[name],
            'canonical_coord_path':str(coord),'cryolo_box_path':str(box)})
        topimages.append({'image_name':stem,'path':str(mrc),'micrograph_name':name})
        for x,y in gt[name]:
            centers.append({'micrograph_name':name,'x_original_px':x,'y_original_px':y,'diameter_px':224})
            toptargets.append({'image_name':stem,'x_coord':x,'y_coord':y})
    for method in ['ann_yolo26','cryolo','topaz']:
        p=A/method/'test_micrographs.txt';p.parent.mkdir(parents=True,exist_ok=True);textlist(p,[str(D/'micrographs'/n) for n in test])
    tp=A/'topaz/test';tp.mkdir(parents=True,exist_ok=True)
    with (tp/'images.tsv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(topimages[0]),delimiter='\t');w.writeheader();w.writerows(topimages)
    csvout(tp/'targets.csv',toptargets)
    csvout(M/'test_manifest.csv',manifest);csvout(M/'added56_gt_statistics.csv',[r for r in manifest if r['membership']=='new56'])
    csvout(A/'ground_truth/test100_centers.csv',centers)
    allrows=[]
    for r in oldrows:allrows.append({**r,'configuration':'test_100_v1' if r['split']=='test' else 'original_v3'})
    for r in manifest:
        if r['membership']=='new56':allrows.append({**{k:r.get(k,'') for k in oldrows[0]},'split':'test','configuration':'test_100_v1'})
    csvout(M/'all_micrographs.csv',allrows,list(oldrows[0])+['configuration'])
    stats={'train':{'images':132,'gt':11592},'val':{'images':44,'gt':3824},'test_original44':{'images':44,'gt':sum(len(gt[n]) for n in s['test'])},
        'test_added56':{'images':56,'gt':sum(len(gt[n]) for n in added)},'test100':{'images':100,'gt':len(centers)}}
    assert stats['test_original44']['gt']==3899 and len(set(test))==100 and len(allrows)==276
    assert not(set(test)&set(s['train']+s['val']))
    assert protocol['original_v3_and_training_input_hashes']==protected(),'Protected v3 or training inputs changed'
    contract={'configuration':'test_100_v1','shared_test_manifest':str(M/'test.txt'),'shared_gt':str(A/'ground_truth/test100_centers.csv'),
        'original_gt_directory':str(G),'test_images':100,'test_gt':len(centers),'methods':['ANN-YOLO26m','crYOLO','Topaz'],
        'matching_radius_original_px':56.0,'matching_implementation':str(D/'metadata/three_method_evaluation/evaluate_empiar10028_v3_pretrained_val.py'),
        'matching_functions':['match_one_image','evaluate_method','metric_counts'],'evaluation_counts':['TP','FP','FN','Precision','Recall','F1'],
        'threshold_policy':'Use each method\'s fixed validation-selected global threshold; test threshold scanning prohibited',
        'required_gate':'ANN formal training finished; final checkpoint, validation threshold, preprocessing and all inference/postprocessing parameters frozen for all three methods',
        'current_phase':'data preparation only; no inference, evaluation, threshold selection or reconstruction scheduled',
        'coordinate_convention':'All GT is in original CryoPPP x/y. ANN uses existing shared normalize_mrc flipud before tiled inference; crYOLO CBOX and Topaz raw-prediction Y adapters retain existing validated transformations.',
        'GT_reference_files_only':'Topaz targets.csv and crYOLO boxes are canonical GT references, not test-time picking or parameter-selection inputs.',
        'train_and_val_manifests':{'train':str(OLD/'train.txt'),'val':str(OLD/'val.txt')},'model_or_training_modified':False,'test_inference_started':False}
    dump(M/'evaluation_contract.json',contract);dump(M/'statistics.json',stats)
    integrity={'status':'PASS','test_count':100,'added_count':56,'original_test_retained':True,'test_unique':True,'test_train_val_overlap':[],
        'all_mrc_present_and_headers_readable':True,'new56_all_pixels_finite':True,'new56_sha256_unique_and_not_in_original220':True,
        'all_positive_gt_loaded_and_coordinates_in_range':True,'derived_coords_and_boxes_equal_source_gt':True,'all_three_methods_same_100_names':True,
        'original_v3_and_training_inputs_unchanged':True,'test_inference_started':False,'training_state_after':training_state(),
        'gt_sources_reused_existing':100,'gt_files_downloaded':0,'mrc_files_downloaded_this_run':sum(not x['reused_existing'] for x in downloads.values()),
        'mrc_bytes_downloaded_this_run':sum(x['bytes'] for x in downloads.values() if not x['reused_existing']),
        'statistics':stats,'protected_hashes_after':protected()}
    dump(M/'integrity_check.json',integrity);dump(R/'integrity_report.json',integrity);dump(R/'statistics.json',stats)
    csvout(R/'per_micrograph_gt.csv',manifest)
    protocol.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),statistics=stats);dump(M/'selection_protocol.json',protocol)
    lines=['# EMPIAR-10028 test_100_v1 数据准备完成','',f"新增56张GT：{stats['test_added56']['gt']}；原44张GT：3899；最终100张GT：{len(centers)}。",'',
        '固定seed=10028；对剩余80个唯一标识稳定排序、Random(10028).shuffle后选前56。挑选不使用GT数量或模型结果。原v3训练132张、验证44张和测试44张均保留。','',
        f'微图目录：`{D/"micrographs"}`',f'清单与独立配置：`{M}`',f'三方法GT与输入链接：`{A}`',f'原始GT：`{G}`','',
        '300份原始CryoPPP CSV此前已存在，本次未重新下载标注。仅下载缺少的56张MRC，不下载其他微图或movie。测试推理、测试阈值扫描和重构均未启动；运行中的ANN训练未停止或修改。','',
        '| 新增微图 | GT数量 |','|---|---:|']
    lines += [f'| {n} | {len(gt[n])} |' for n in added]
    (R/'final_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('PREPARATION_COMPLETE',json.dumps(stats),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--select-only',action='store_true');args=ap.parse_args()
    s,added,protocol=select()
    if args.select_only:return
    if protocol['status']=='complete':
        print('ALREADY_COMPLETE',json.dumps(protocol['statistics']));return
    gt={};gtsha={}
    for name in s['test']+added:gt[name],gtsha[name]=getgt(name)
    # Count after selection was locked; GT cannot influence the selected members.
    csvout(M/'selection_locked_gt_counts.csv',[{'micrograph_name':n,'gt_count':len(gt[n]),'added':n in added} for n in s['test']+added])
    PART.mkdir(parents=True,exist_ok=True)
    need=[n for n in added if not (D/'micrographs'/n).exists()]
    fs=os.statvfs(D);free=fs.f_bavail*fs.f_frsize
    if free-len(need)*SIZE<5_000_000_000:raise RuntimeError('Insufficient safe free space for selected MRCs')
    print('DOWNLOAD_PREFLIGHT',len(need),'new files',len(need)*SIZE,'bytes; free',free,flush=True)
    record=R/'download_records.json';downloads={}
    if record.exists():downloads=json.loads(record.read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(download,n):n for n in added}
        for f in concurrent.futures.as_completed(futures):
            n=futures[f];result=f.result()
            # Preserve original transfer record when resuming a completed earlier download.
            if n not in downloads or not result['reused_existing']:downloads[n]=result
            dump(record,downloads);print('COMPLETE',len(downloads),'/56',n,flush=True)
    organize(s,added,protocol,gt,gtsha,downloads)

if __name__=='__main__':main()
