import argparse,csv,math,sys
from common import *
from ultralytics import YOLO
sys.path.insert(0,str(DATA/'metadata/three_method_evaluation'))
from evaluate_empiar10028_v3_pretrained_val import Point,load_gt,evaluate_method
sys.path.insert(0,'/root/autodl-tmp/cryoem_baselines/experiments/empiar10028_pretrained_val_v3')
from evaluate_val import incremental_threshold_sweep

def write_csv(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['micrograph_name']);w.writeheader();w.writerows(rows)

def infer(weight,out,smoke=False):
    out.mkdir(parents=True,exist_ok=False)
    m=YOLO(str(weight));m.model.model[-1].max_det=2000
    ns=names('val')[:1] if smoke else names('val');preds={};allrows=[];info=[]
    for name in ns:
        image,meta=normalize_mrc(DATA/'micrographs'/name); candidates=[];tile_counts=[]
        for iy,y0 in enumerate(STARTS):
            for ix,x0 in enumerate(STARTS):
                tile=image[y0:y0+SIZE,x0:x0+SIZE]
                tensor=torch.from_numpy(np.repeat(tile[None,None],3,axis=1)).to('cuda')
                r=m.predict(tensor,imgsz=SIZE,device=0,conf=.00001,max_det=2000,verbose=False)[0]
                xywh=r.boxes.xywh.detach().cpu().numpy();scores=r.boxes.conf.cpu().numpy()
                tile_counts.append({'x0':x0,'y0':y0,'candidates':len(scores),'at_max_det':len(scores)>=2000})
                for (x,y,w,h),score in zip(xywh,scores):
                    xx,yy=float(x)+x0,float(y)+y0
                    if BOUNDS[ix]<=xx<BOUNDS[ix+1] and BOUNDS[iy]<=yy<BOUNDS[iy+1]:
                        candidates.append((xx,yy,float(score)))
        # Deterministic suppression only of nearly-identical centers, diameter/8=28 px.
        keep=[];grid={}
        for x,y,s in sorted(candidates,key=lambda v:(-v[2],v[0],v[1])):
            gx,gy=int(x//28),int(y//28)
            nearby=[v for xx in range(gx-1,gx+2) for yy in range(gy-1,gy+2) for v in grid.get((xx,yy),[])]
            if all((x-a)**2+(y-b)**2>28**2 for a,b in nearby):
                keep.append((x,y,s));grid.setdefault((gx,gy),[]).append((x,y))
        preds[name]=[Point(x,y,s,j) for j,(x,y,s) in enumerate(keep)]
        allrows += [dict(method='ANN-YOLO26',split='val',micrograph_name=name,x_original_px=x,y_original_px=y,score=s) for x,y,s in keep]
        info.append(dict(micrograph_name=name,before_center_merge=len(candidates),after_center_merge=len(keep),normalization=meta,tile_counts=tile_counts))
        print('INFERENCE',name,len(keep),flush=True)
    write_csv(out/'predictions.csv',allrows)
    write_json(out/'inference.json',{'checkpoint':str(weight),'checkpoint_sha256':sha(weight),'score_floor':.00001,'tile_max_det':2000,
        'ownership_regions':BOUNDS,'center_merge_radius':28,'tile_size':SIZE,'original_gt_coordinates':'tile center + crop offset; input image already in CryoPPP orientation',
        'micrographs':info,'test_accessed':False})
    if smoke:
        write_json(out/'smoke_inference.json',{'status':'PASS','images':1,'finite_in_bounds':all(0<=p.x<4096 and 0<=p.y<4096 and math.isfinite(p.score) for pp in preds.values() for p in pp),'prediction_count':len(allrows)})
        return
    ns,gt=load_gt(DATA,'val')
    sweep,best=incremental_threshold_sweep(ns,gt,preds,label='ANN-YOLO26')
    best_f1=dict(best)
    targets={'Precision':.7498,'Recall':.9610,'F1':.8424}
    feasible=[r for r in sweep if all(r[k]>v for k,v in targets.items())]
    if feasible:best=max(feasible,key=lambda r:(r['F1'],r['Precision'],r['Recall'],r['threshold']))
    per,matches,summary=evaluate_method(ns,gt,preds,best['threshold'])
    assert summary['TP']+summary['FN']==3824 and len(per)==44
    write_csv(out/'threshold_sweep.csv',sweep);write_csv(out/'per_micrograph_metrics.csv',per);write_csv(out/'matching_records.csv',matches)
    passed=all(summary[k]>v for k,v in targets.items())
    write_json(out/'validation_summary.json',{'status':'PASS','metrics':summary,'threshold':best['threshold'],'targets':targets,'all_targets_met':passed,
        'selection_rule':'max F1 among validation thresholds satisfying all targets if any; otherwise global max validation F1',
        'unconstrained_max_f1':best_f1,'feasible_threshold_count':len(feasible),
        'best_precision_at_required_recall':max((r['Precision'] for r in sweep if r['Recall']>.9610),default=0),
        'same_model_and_global_threshold':True,'match_radius_px':56.,'validation_gt':3824,'test_accessed':False})
    print('VALIDATION_RESULT',json.dumps({'metrics':summary,'threshold':best['threshold'],'all_targets_met':passed}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--smoke',action='store_true');a=p.parse_args();infer(a.weights,a.out,a.smoke)
