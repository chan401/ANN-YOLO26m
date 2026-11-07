#!/usr/bin/env python3
"""Unified EMPIAR-10028 micrograph-level evaluation. CPU/std-library only."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, sys, tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROTOCOL_ID = 'EMPIAR-10028-THREE-METHOD-EVAL-v3-pretrained-val'
METHODS = ('ANN-YOLO26', 'crYOLO', 'Topaz')
EXPECTED_COUNTS = {'train': (132, 11592), 'val': (44, 3824), 'test': (44, 3899)}
RADIUS_PX = 56.0  # 1/4 of the 224 px original-scale particle box diameter
ORIGINAL_GT_ROOT = Path('annotations_original/cryoppp_10028/10028/ground_truth/particle_coordinates')
STANDARD_COORD_ROOT = Path('annotations_standardized/split_v3_paper60_20_20/ann_yolo26/coords')
SPLIT_ROOT = Path('manifests/split_v3_paper60_20_20')

@dataclass(frozen=True)
class Point:
    x: float
    y: float
    score: float = 1.0
    idx: int = -1


def _read_names(path: Path) -> list[str]:
    out=[]
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        s=line.strip()
        if s and not s.startswith('#'):
            out.append(Path(s).name if Path(s).suffix else Path(s).name+'.mrc')
    if len(out)!=len(set(out)): raise ValueError(f'duplicate micrograph names in {path}')
    return out


def load_gt(dataset_root: Path, split: str) -> tuple[list[str], dict[str,list[Point]]]:
    manifest_dir=dataset_root/SPLIT_ROOT
    names_by_split={s:_read_names(manifest_dir/f'{s}.txt') for s in ('train','val','test')}
    expected_n,expected_gt=EXPECTED_COUNTS[split]
    if len(names_by_split[split])!=expected_n: raise ValueError(f'{split}: expected {expected_n} names, got {len(names_by_split[split])}')
    all_names=[n for values in names_by_split.values() for n in values]
    if len(all_names)!=220 or len(set(all_names))!=220: raise ValueError('frozen manifests must contain 220 unique micrographs')
    # Confirm exact split membership is also represented in split_manifest.csv.
    table={s:set() for s in ('train','val','test')}
    with (manifest_dir/'all_micrographs.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f): table[row['split']].add(row['micrograph_name'])
    for s in table:
        if table[s]!=set(names_by_split[s]): raise ValueError(f'{s}.txt and split_manifest.csv membership differ')
    gt={}; counts_from_manifest={}
    with (manifest_dir/'all_micrographs.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            if row['split']==split: counts_from_manifest[row['micrograph_name']]=int(row['gt_count'])
    for name in names_by_split[split]:
        path=dataset_root/ORIGINAL_GT_ROOT/f'{Path(name).stem}.csv'
        if not path.is_file(): raise FileNotFoundError(path)
        points=[]; seen=set()
        with path.open(encoding='utf-8-sig',newline='') as handle:
            reader=csv.DictReader(handle)
            required={'X-Coordinate','Y-Coordinate','Diameter'}
            if not reader.fieldnames or not required.issubset(reader.fieldnames): raise ValueError(f'{path}: missing CryoPPP positive-coordinate fields {sorted(required)}')
            for n,row in enumerate(reader,2):
                x=float(row['X-Coordinate']); y=float(row['Y-Coordinate']); diameter=float(row['Diameter'])
                if diameter!=224.0: raise ValueError(f'{path}:{n}: unexpected GT diameter {diameter}; expected 224px')
                if not (math.isfinite(x) and math.isfinite(y) and 0<=x<4096 and 0<=y<4096): raise ValueError(f'{path}:{n}: coordinate out of 4096 bounds: {x},{y}')
                if (x,y) in seen: raise ValueError(f'{path}:{n}: duplicate positive GT coordinate {x},{y}')
                seen.add((x,y)); points.append(Point(x,y,1.0,len(points)))
        if len(points)!=counts_from_manifest.get(name): raise ValueError(f'{path}: {len(points)} CryoPPP positives vs manifest count {counts_from_manifest.get(name)}')
        # Method-neutral coordinates are cross-checked, but evaluation uses the original CryoPPP CSV above.
        standardized=dataset_root/STANDARD_COORD_ROOT/split/f'{Path(name).stem}.coord'
        std=[]
        for n,line in enumerate(standardized.read_text(encoding='utf-8-sig').splitlines(),1):
            if line.strip():
                cols=line.replace(',',' ').split()
                if len(cols)<2: raise ValueError(f'{standardized}:{n}: expected x y')
                std.append((float(cols[0]),float(cols[1])))
        if [(p.x,p.y) for p in points]!=std: raise ValueError(f'{name}: standardized coordinates differ from CryoPPP original positive CSV')
        gt[name]=points
    total=sum(map(len,gt.values()))
    if total!=expected_gt: raise ValueError(f'{split}: expected {expected_gt} GT, found {total}')
    return names_by_split[split],gt


def read_predictions(path: Path, split: str) -> dict[str,dict[str,list[Point]]]:
    required={'method','split','micrograph_name','x_original_px','y_original_px','score'}
    result={m:defaultdict(list) for m in METHODS}
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if not reader.fieldnames or not required.issubset(reader.fieldnames): raise ValueError(f'predictions CSV requires columns: {sorted(required)}')
        seq=0
        for row in reader:
            if row['split']!=split: raise ValueError(f"wrong split {row['split']} in prediction CSV (requested {split})")
            m=row['method']
            if m not in METHODS: raise ValueError(f'unknown method {m!r}; expected {METHODS}')
            name=Path(row['micrograph_name']).name
            if not Path(name).suffix: name += '.mrc'
            x=float(row['x_original_px']); y=float(row['y_original_px']); score=float(row['score'])
            if not all(map(math.isfinite,(x,y,score))): raise ValueError(f'non-finite prediction row: {row}')
            if not (0<=x<4096 and 0<=y<4096): raise ValueError(f'prediction not in original 4096 coordinate system: {name} ({x},{y})')
            result[m][name].append(Point(x,y,score,seq)); seq+=1
    return result


def _edges(pred: Point, gt: list[Point], radius: float) -> list[int]:
    lim=radius*radius
    pairs=[]
    for j,g in enumerate(gt):
        d2=(pred.x-g.x)**2+(pred.y-g.y)**2
        if d2<=lim: pairs.append((d2,j))
    pairs.sort(key=lambda z:(z[0],z[1]))
    return [j for _,j in pairs]


def match_one_image(gt: list[Point], predictions: list[Point], radius: float=RADIUS_PX) -> tuple[dict[int,int],dict[int,float]]:
    """Deterministic maximum-cardinality bipartite match, per image, radius inclusive."""
    pred_order=sorted(range(len(predictions)),key=lambda i:(-predictions[i].score,predictions[i].idx,i))
    adj={i:_edges(predictions[i],gt,radius) for i in pred_order}
    gt_to_pred={}
    def augment(pi: int, seen: set[int]) -> bool:
        for gi in adj[pi]:
            if gi in seen: continue
            seen.add(gi)
            if gi not in gt_to_pred or augment(gt_to_pred[gi],seen):
                gt_to_pred[gi]=pi
                return True
        return False
    for pi in pred_order: augment(pi,set())
    pred_to_gt={pi:gi for gi,pi in gt_to_pred.items()}
    distances={pi:math.hypot(predictions[pi].x-gt[gi].x,predictions[pi].y-gt[gi].y) for pi,gi in pred_to_gt.items()}
    return pred_to_gt,distances


def metric_counts(tp:int,fp:int,fn:int)->dict[str,float|int]:
    precision=tp/(tp+fp) if tp+fp else 0.0
    recall=tp/(tp+fn) if tp+fn else 0.0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {'TP':tp,'FP':fp,'FN':fn,'Precision':precision,'Recall':recall,'F1':f1}


def evaluate_method(names:list[str],gt_by_name:dict[str,list[Point]],pred_by_name:dict[str,list[Point]],threshold:float):
    per_image=[]; matches=[]; sums=[0,0,0]
    for name in names:
        gt=gt_by_name[name]
        all_pred=pred_by_name.get(name,[])
        preds=[p for p in all_pred if p.score>=threshold]
        pair,dists=match_one_image(gt,preds)
        tp=len(pair); fp=len(preds)-tp; fn=len(gt)-tp
        if tp+fn!=len(gt): raise AssertionError('TP+FN invariant violated')
        sums[0]+=tp; sums[1]+=fp; sums[2]+=fn
        per_image.append({'micrograph_name':name,'GT':len(gt),'Predictions':len(preds),'TP':tp,'FP':fp,'FN':fn,**metric_counts(tp,fp,fn)})
        for pi,p in enumerate(preds):
            if pi in pair:
                gi=pair[pi]; g=gt[gi]
                matches.append({'micrograph_name':name,'record':'TP','prediction_index':p.idx,'gt_index':g.idx,'score':p.score,'distance_px':dists[pi],'pred_x_original_px':p.x,'pred_y_original_px':p.y,'gt_x_original_px':g.x,'gt_y_original_px':g.y})
            else:
                matches.append({'micrograph_name':name,'record':'FP','prediction_index':p.idx,'gt_index':'','score':p.score,'distance_px':'','pred_x_original_px':p.x,'pred_y_original_px':p.y,'gt_x_original_px':'','gt_y_original_px':''})
        matched_gt=set(pair.values())
        for gi,g in enumerate(gt):
            if gi not in matched_gt: matches.append({'micrograph_name':name,'record':'FN','prediction_index':'','gt_index':g.idx,'score':'','distance_px':'','pred_x_original_px':'','pred_y_original_px':'','gt_x_original_px':g.x,'gt_y_original_px':g.y})
    summary=metric_counts(*sums)
    summary['Predictions']=sums[0]+sums[1]
    summary['GT']=sums[0]+sums[2]
    return per_image,matches,summary


def threshold_sweep(names,gt,preds):
    scores=sorted({p.score for rows in preds.values() for p in rows},reverse=True)
    if scores: scores.insert(0,math.nextafter(scores[0],math.inf))
    else: scores=[1.0]
    rows=[]
    for threshold in scores:
        _,_,summary=evaluate_method(names,gt,preds,threshold)
        rows.append({'threshold':threshold,**summary})
    best=max(rows,key=lambda r:(r['F1'],r['Precision'],r['Recall'],r['threshold']))
    return rows,best


def _write_csv(path:Path,rows:list[dict],fields:list[str]|None=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if fields is None: fields=list(rows[0]) if rows else []
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)


def _audit(dataset_root:Path):
    report={}
    for s,(n,g) in EXPECTED_COUNTS.items():
        names,gt=load_gt(dataset_root,s)
        report[s]={'micrographs':len(names),'positive_gt':sum(map(len,gt.values())),'min_gt_per_micrograph':min(map(len,gt.values())),'max_gt_per_micrograph':max(map(len,gt.values()))}
    return report


def _sha(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_cli(args):
    split=args.split
    if split not in ('val','test'): raise ValueError('only val and test may be evaluated; training is for fitting only')
    if split=='val' and args.frozen_config: raise ValueError('do not supply frozen config for validation sweep')
    if split=='val' and not args.sweep: raise ValueError('validation evaluation must produce a threshold sweep (--sweep)')
    if split=='test' and args.sweep: raise ValueError('test threshold sweep is forbidden')
    if split=='test' and not args.frozen_config: raise ValueError('test requires --frozen-config produced after model/rule freeze')
    dataset=args.dataset_root.resolve()
    names,gt=load_gt(dataset,split)
    preds=read_predictions(args.predictions.resolve(),split)
    observed={m for m,d in preds.items() if sum(map(len,d.values()))>0}
    if observed!=set(METHODS): raise ValueError(f'prediction CSV must contain all 3 methods; found {sorted(observed)}')
    out=args.output_dir.resolve()
    if split=='val':
        if args.frozen_config: raise ValueError('do not supply frozen config for validation sweep')
        if not args.sweep: raise ValueError('validation evaluation must produce a threshold sweep (--sweep)')
        out.mkdir(parents=True,exist_ok=True)
        all_sweeps=[]; thresholds={}; summaries=[]
        for method in METHODS:
            sweep,best=threshold_sweep(names,gt,preds[method]); thresholds[method]=float(best['threshold'])
            for r in sweep: all_sweeps.append({'method':method,**r})
            per,matches,summary=evaluate_method(names,gt,preds[method],thresholds[method])
            summaries.append({'method':method,'selected_validation_threshold':thresholds[method],**summary})
            _write_csv(out/f'{method}_val_per_image.csv',per)
            _write_csv(out/f'{method}_val_matching_records.csv',matches)
        _write_csv(out/'validation_threshold_sweep.csv',all_sweeps)
        _write_csv(out/'validation_selected_metrics.csv',summaries)
        (out/'validation_thresholds.json').write_text(json.dumps({'protocol_id':PROTOCOL_ID,'split':'val','matching_radius_px':RADIUS_PX,'threshold_selection':'max micro-F1; ties: higher Precision, then Recall, then threshold','methods':thresholds},indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'status':'PASS','split':'val','thresholds':thresholds,'summaries':summaries,'output_dir':str(out)},indent=2))
        return
    if split!='test': raise ValueError('only val and test may be evaluated; training split is for fitting only')
    if args.sweep: raise ValueError('test threshold sweep is forbidden')
    if not args.frozen_config: raise ValueError('test requires --frozen-config produced after model/rule freeze')
    lock=dataset/'qc/three_method_evaluation/test_evaluation_started.json'
    if lock.exists() or out.exists(): raise FileExistsError('final test evaluation already started or output directory exists; refusing repeat/overwrite')
    cfg=json.loads(args.frozen_config.resolve().read_text(encoding='utf-8'))
    if cfg.get('protocol_id')!=PROTOCOL_ID or cfg.get('frozen') is not True or cfg.get('matching_radius_px')!=RADIUS_PX: raise ValueError('frozen config protocol/radius/frozen flag mismatch')
    if set(cfg.get('methods',{}))!=set(METHODS): raise ValueError('frozen config must have thresholds/checkpoint metadata for all 3 methods')
    for method in METHODS:
        item=cfg['methods'][method]
        if not isinstance(item.get('threshold'),(int,float)) or not math.isfinite(item['threshold']): raise ValueError(f'{method}: invalid frozen threshold')
        for k in ('checkpoint_sha256','config_sha256'):
            if not isinstance(item.get(k),str) or len(item[k])!=64: raise ValueError(f'{method}: {k} must be a 64-char SHA256')
    if out!=dataset/'qc/three_method_evaluation/test_final': raise ValueError('final test output must use fixed qc/three_method_evaluation/test_final directory')
    summaries=[]; all_per=[]; all_matches=[]
    for method in METHODS:
        threshold=float(cfg['methods'][method]['threshold'])
        per,matches,s=evaluate_method(names,gt,preds[method],threshold)
        if s['TP']+s['FN']!=7050: raise AssertionError(f'{method}: TP+FN != 7050')
        summaries.append({'Method':method,'TP':s['TP'],'FP':s['FP'],'FN':s['FN'],'Precision':s['Precision'],'Recall':s['Recall'],'F1':s['F1']})
        all_per.extend({'method':method,**r} for r in per)
        all_matches.extend({'method':method,**r} for r in matches)
    out.mkdir(parents=True,exist_ok=False)
    fields=['Method','TP','FP','FN','Precision','Recall','F1']
    _write_csv(out/'paper_table.csv',summaries,fields)
    _write_csv(out/'per_micrograph_metrics.csv',all_per)
    _write_csv(out/'matching_records.csv',all_matches)
    _write_csv(out/'frozen_thresholds_used.csv',[{'method':m,'threshold':cfg['methods'][m]['threshold'],'checkpoint_sha256':cfg['methods'][m]['checkpoint_sha256'],'config_sha256':cfg['methods'][m]['config_sha256']} for m in METHODS])
    # A single fixed sentinel prevents subsequent test sweeps/reruns via this evaluator.
    lock.parent.mkdir(parents=True,exist_ok=True)
    lock.write_text(json.dumps({'protocol_id':PROTOCOL_ID,'status':'completed','predictions_sha256':_sha(args.predictions.resolve()),'frozen_config_sha256':_sha(args.frozen_config.resolve()),'output_dir':str(out)},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'PASS','split':'test','summary':summaries,'output_dir':str(out)},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('audit'); a.add_argument('--dataset-root',type=Path,default=Path('/root/autodl-tmp/cryoem_datasets/EMPIAR-10028'))
    e=sub.add_parser('evaluate'); e.add_argument('--dataset-root',type=Path,default=Path('/root/autodl-tmp/cryoem_datasets/EMPIAR-10028')); e.add_argument('--split',choices=('train','val','test'),required=True); e.add_argument('--predictions',type=Path,required=True); e.add_argument('--output-dir',type=Path,required=True); e.add_argument('--sweep',action='store_true'); e.add_argument('--frozen-config',type=Path)
    args=p.parse_args()
    if args.command=='audit': print(json.dumps({'status':'PASS','protocol_id':PROTOCOL_ID,'splits':_audit(args.dataset_root.resolve())},indent=2))
    else: evaluate_cli(args)

if __name__=='__main__': main()


