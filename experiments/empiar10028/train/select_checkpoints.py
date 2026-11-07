"""Select a particle-picking checkpoint using Validation only, after a run completes."""
import argparse,shutil,subprocess,sys
from common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='standard_m_p3p5');a=p.parse_args()
    run=EXP/'runs'/a.run;weights=run/'weights';candidates=[weights/'best.pt']
    if a.run=='standard_m_p3p5':
        candidates+=sorted([p for p in (EXP/'snapshots').glob('early_best_epoch*.pt')
            if (EXP/'validation'/('early_epoch'+p.stem.split('epoch')[-1])/'validation_summary.json').exists()],reverse=True)
    candidates.append(weights/'last.pt')
    candidates+=sorted((q for q in weights.glob('epoch*.pt') if int(q.stem[5:])>=10),key=lambda q:int(q.stem[5:]),reverse=True)
    results=[];seen=set()
    for ck in candidates:
        if ck==weights/'last.pt' and any(r['all_targets_met'] for r in results):break
        if not ck.is_file():continue
        h=sha(ck)
        if h in seen:continue
        seen.add(h)
        out=EXP/'validation'/(('early_epoch'+ck.stem.split('epoch')[-1]) if ck.name.startswith('early_best_epoch') else (a.run if ck.name=='best.pt' else a.run+'_'+ck.stem))
        summary=out/'validation_summary.json'
        if not summary.exists():
            with (EXP/(out.name+'_validation.log')).open('w') as log:
                r=subprocess.run([sys.executable,str(Path(__file__).with_name('infer_eval.py')),'--weights',str(ck),'--out',str(out)],stdout=log,stderr=subprocess.STDOUT)
            if r.returncode:raise RuntimeError(f'Validation failed for {ck}; see log')
        result=json.loads(summary.read_text());results.append({'checkpoint':str(ck),'sha256':h,'result_dir':str(out),**result})
        print('CHECKPOINT_RESULT',ck.name,json.dumps(result['metrics']),'targets',result['all_targets_met'],flush=True)
    if not results:raise RuntimeError('No completed checkpoints')
    feasible=[x for x in results if x['all_targets_met']]
    selected=max(feasible or results,key=lambda r:(r['metrics']['F1'],r['metrics']['Precision'],r['metrics']['Recall']))
    folder=EXP/'selections'/a.run;folder.mkdir(parents=True,exist_ok=True)
    shutil.copy2(selected['checkpoint'],folder/'best.pt')
    write_json(folder/'selection.json',{'run':a.run,'selection_rule':'largest Validation F1 among goal-feasible final best-AP and prevalidated milestones; fallback largest Validation F1',
        'selected':selected,'candidates':results,'test_accessed':False})
    print('SELECTED',json.dumps(selected),flush=True)

if __name__=='__main__':main()
