"""Source-only selection, then one frozen TRPV1 check with unchanged baselines."""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,os,shutil,subprocess,sys,time,zipfile
G=Path('/root/autodl-tmp/cryoem_generalization/v3_low_snr');V2=G.parent/'v2';V1=G.parent/'v1'
os.environ['ANN_GENERALIZATION_RUN_ROOT']=str(G)
sys.path.insert(0,str(V2/'code'))
from common import *
spec=importlib.util.spec_from_file_location('source_selector_v2',V2/'code/select_and_evaluate.py');sel=importlib.util.module_from_spec(spec);spec.loader.exec_module(sel)
PY='/root/autodl-tmp/cryoem_annyolo26/.venv/bin/python'

def infer(weight,split,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 with (out/'wrapper.log').open('a') as f:subprocess.run([PY,'-u',str(G/'code/infer_native.py'),'--weights',str(weight),'--split',split,'--out',str(out)],stdout=f,stderr=subprocess.STDOUT,check=True)
 return load_predictions(out/'predictions.csv')

def protect():
 frozen=json.loads((G/'final_frozen_config.json').read_text())
 assert sha(G/'final_frozen_config.json')==(G/'final_frozen_config.sha256').read_text().strip()
 for p,h in frozen['protected'].items():assert sha(p)==h,('Frozen input changed',p)
 return frozen

def freeze(selected,bc,baseline):
 protected={}
 for folder in [G/'code',G/'core',ULTRA/'ultralytics']:
  for p in folder.rglob('*'):
   if p.is_file() and p.suffix in ['.py','.yaml','.yml']:protected[str(p)]=sha(p)
 for p in [G/'protocol.json',G/'secondary_HCN1_protocol.json',G/'source_image_registry.json',G/'image_registry.json',G/'selection.json',G/'target_integrity.json',Path(selected['checkpoint']),bc.CRYO_WEIGHT,bc.TOP_WEIGHT,V1/'code/baseline_external.py',V2/'code/common.py',V2/'code/select_and_evaluate.py',V2/'code/prepare_data.py']:
  protected[str(p)]=sha(p)
 for p in [D/'metadata/three_method_evaluation/evaluate_empiar10028_v3_pretrained_val.py',D/'experiments/test_100_v1_three_methods/code/baseline_helpers.py',D/'experiments/test_100_v1_three_methods/topaz_runtime_compat/sitecustomize.py',D/'experiments/test_100_v1_three_methods/topaz_preprocess_repro/sitecustomize.py',bc.B/'cryolo/env-gpu-py38/bin/cryolo_predict_gpu.sh',bc.B/'topaz/.venv-gpu/bin/topaz',V2/'baseline_selection.json']:
  protected[str(p)]=sha(p)
 for p in [G/'v2_control_protocol.json',V2/'selections/best.pt',V2/'results/final_check/ANN_v2/predictions.csv',V2/'results/final_check/ANN_v2/done.json']:protected[str(p)]=sha(p)
 for p in [G/'development_gate.json',G/'augmentation_check.json',G/'preparation_report.json',G/'source_stress/v2_reference/fixed_threshold_metrics.json',G/'source_stress/v3_selected/fixed_threshold_metrics.json']:protected[str(p)]=sha(p)
 for r in records('val'):
  assert sha(r['GT_path'])==r['GT_sha256'];protected[r['GT_path']]=r['GT_sha256']
 for p in (G/'baseline_configs').glob('*.json'):protected[str(p)]=sha(p)
 for method in ['crYOLO','Topaz']:
  for name in ['predictions.csv','done.json']:
   p=V2/'results/baseline_final_check'/method/'10081'/name;protected[str(p)]=sha(p)
 for r in [r for r in json.loads((G/'image_registry.json').read_text()) if r['split'] in ['final_check','regression_check']]:
  assert sha(r['path'])==r['sha256'] and sha(r['GT_path'])==r['GT_sha256'];protected[r['path']]=r['sha256'];protected[r['GT_path']]=r['GT_sha256']
 dump(G/'final_frozen_config.json',dict(created_unix=time.time(),protected=protected,ANN=selected,baselines=baseline,
  HCN1_archive=str(G.parent/'archives/HCN1_v2_frozen'),HCN1_different_model_version=True,target='10059',new_images=20,
  one_global_ANN_threshold=True,target_threshold_scan=False,preprocessing_unchanged_from_v2=True,
  matching_radius_px={'10059':33,'10081':38.5},HCN1_secondary_regression_images=20,
  matching='maximum-cardinality one-to-one inclusive radius',GT_used_for_training=False))
 (G/'final_frozen_config.sha256').write_text(sha(G/'final_frozen_config.json')+'\n')

def package(report):
 out=G/'exports/TRPV1_v3_low_snr';out.mkdir(parents=True,exist_ok=True)
 for name in ['final_report.json','protocol.json','secondary_HCN1_protocol.json','v2_control_protocol.json','final_frozen_config.json','final_frozen_config.sha256','selection.json','target_integrity.json','augmentation_check.json','low_snr20_training_config.json']:
  shutil.copy2(G/name,out/name)
 shutil.copy2(V2/'final_report.json',out/'previous_v2_final_report.json')
 operational=out/'operational_records';operational.mkdir(exist_ok=True)
 for name in ['image_registry.json','source_image_registry.json','preparation_report.json','capacity_plan.json','capacity_cleanup_candidates.csv','capacity_deleted_files.csv','capacity_cleanup_result.json','transport_revision.json','transport_capacity_fix.json','development_gate.json']:
  p=G/name
  if p.exists():shutil.copy2(p,operational/name)
 shutil.copytree(G/'baseline_configs',out/'baseline_configs',dirs_exist_ok=True)
 for folder in ['reports','code','core']:
  shutil.copytree(G/folder,out/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
 for tag in ['v2_reference','v3_selected']:
  p=G/'source_stress'/tag/'fixed_threshold_metrics.json'
  if p.exists():shutil.copy2(p,out/('source_stress_'+tag+'.json'))
 for tag in ['best','last']:
  dest=out/'source_validation'/tag;dest.mkdir(parents=True,exist_ok=True)
  for name in ['done.json','metrics.json','threshold_grid.csv','inference.json']:
   p=G/'validation'/tag/name
   if p.exists():shutil.copy2(p,dest/name)
 shutil.copy2(report['ANN']['checkpoint'],out/'best.pt')
 shutil.copy2(G/'runs/low_snr20/results.csv',out/'training_results.csv')
 shutil.copy2(G/'runs/low_snr20/completion.json',out/'training_completion.json')
 training=json.loads((G/'low_snr20_training_config.json').read_text())
 assert training['source_code_sha256']['train_low_snr.py']==sha(G/'code/train_low_snr.py')
 dump(out/'training_code_provenance.json',dict(status='PASS',actual_training_wrapper_sha256=sha(G/'code/train_low_snr.py'),
  wrapper_matches_training_start_record=True,inference_core_sha256={p.name:sha(p) for p in (G/'core').glob('*.py')},
  note='Evaluation packaging code was extended before final freeze; actual training wrapper/core retained unchanged.'))
 for name in ['predict_particles.py','model_configuration.json','README_运行说明.md','ultralytics_runtime_source.zip','runtime_versions.json','server_environment_freeze.txt']:
  shutil.copy2(V2/'exports/generalization_v2'/name,out/name)
 cfg=json.loads((out/'model_configuration.json').read_text());cfg.update(threshold=report['ANN']['threshold'],global_threshold=report['ANN']['threshold'],checkpoint_sha256=report['ANN']['checkpoint_sha256']);dump(out/'model_configuration.json',cfg)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 table=[r for r in report['table'] if r['dataset']=='10059'];fig,ax=plt.subplots(figsize=(6,4),dpi=180)
 bars=ax.bar([r['method'] for r in table],[r['F1'] for r in table],color=['#1976D2','#F29E25','#55A868']);ax.set_ylim(0,1);ax.set_ylabel('F1');ax.set_title('TRPV1: 20 new independent micrographs')
 for bar,row in zip(bars,table):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.02,f"{row['F1']:.4f}",ha='center')
 fig.tight_layout();fig.savefig(out/'TRPV1_F1.png');fig.savefig(out/'TRPV1_F1.pdf');plt.close(fig)
 curve=list(csv.DictReader((out/'training_results.csv').open()));epochs=[int(float(r['epoch'])) for r in curve]
 fig,axes=plt.subplots(1,3,figsize=(13,3.8),dpi=180)
 for ax,kind in zip(axes[:2],['box','cls']):
  for split in ['train','val']:ax.plot(epochs,[float(r[split+'/'+kind+'_loss']) for r in curve],label=split)
  ax.set_title(kind+' loss');ax.set_xlabel('Epoch');ax.legend(frameon=False)
 for field,label in [('metrics/mAP50(B)','mAP50'),('metrics/mAP50-95(B)','mAP50-95')]:axes[2].plot(epochs,[float(r[field]) for r in curve],label=label)
 axes[2].set_title('Source tile validation proxy');axes[2].set_xlabel('Epoch');axes[2].set_ylim(0,1);axes[2].legend(frameon=False)
 fig.tight_layout();fig.savefig(out/'training_curves.png');fig.savefig(out/'training_curves.pdf');plt.close(fig)
 lines=['# TRPV1 源域低信噪比增强实验 v3','',
  'HCN1 v2 已另行冻结保存。本报告补充同一v3权重、同一阈值在原20张HCN1微图上的回归检查，不替换v2归档，也不把不同版本模型拼接为同一模型成绩。',
  '七源域训练276张，验证92张；HCN1/TRPV1均未参与监督训练或阈值选择。新增20张TRPV1通过固定种子100283选取，并排除所有v1/v2测试微图。',
  '唯一新增训练模块：65%概率在源域原始灰度图块上降低信号幅度并添加独立高斯噪声，然后重新生成LP4/LP2/raw通道。原生尺度、1536图块、标签几何与推理后处理保持不变。',
  '本轮针对已知TRPV1失败开展开发，因此是已知目标域的新微图独立检查，不能称为完全未见目标域的盲测。三方法GPU推理、相同GT、33px一对一匹配，阈值冻结后仅评价一次。',
  '另在同批新TRPV1上运行冻结v2模型（仍用原0.22阈值）作为版本对照；HCN1 v2对照复用已校验的原始预测。该对照不参与本轮模型或阈值选择。',
  '版本对照同时包含继续训练的影响，尚不能把全部变化单独归因于低信噪比增强；分离该因素需要另做同轮数、同初始化的无新增增强对照。',
  '',f"ANN统一阈值：{report['ANN']['threshold']}；新TRPV1微图GT：{report['GT']}。HCN1为旧20张回归检查，GT2328，不是新增盲测。",'',
  '|数据集/角色|方法|TP|FP|FN|Precision|Recall|F1|','|---|---|---:|---:|---:|---:|---:|---:|']
 for r in report['table']:lines.append(f"|{r['dataset']} / {r['role']}|{r['method']}|{r['TP']}|{r['FP']}|{r['FN']}|{r['Precision']:.4f}|{r['Recall']:.4f}|{r['F1']:.4f}|")
 lines+=['',f"TRPV1 ANN F1 是否高于两个基线：{report['ANN_F1_exceeds_both_baselines']}。同一v3模型在本轮两个检查上是否都领先：{report['same_v3_model_both_datasets_exceed_baselines']}。",'',
  '此前审计记录显示crYOLO通用模型在HCN1条目存在已知训练重叠；Topaz完整预训练数据重叠尚未排除。这是公开预训练权重的直接迁移比较，不是统一训练条件的架构比较。已保留此前失败结果，不在本批最终微图上重新调阈值。']
 lines+=['','## 同批微图上的冻结v2对照','','|数据集|TP|FP|FN|Precision|Recall|F1|','|---|---:|---:|---:|---:|---:|---:|']
 for r in report['v2_control_table']:lines.append(f"|{r['dataset']}|{r['TP']}|{r['FP']}|{r['FN']}|{r['Precision']:.4f}|{r['Recall']:.4f}|{r['F1']:.4f}|")
 (out/'实验报告.md').write_text('\n'.join(lines)+'\n')
 dump(out/'SHA256SUMS.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'})
 archive=G/'exports/TRPV1_v3_low_snr.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for p in out.rglob('*'):
   if p.is_file():z.write(p,str(Path(out.name)/p.relative_to(out)))
 dump(G/'export_manifest.json',dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive)))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--finalize',action='store_true');args=parser.parse_args()
 (G/'reports').mkdir(exist_ok=True)
 if (G/'final_report.json').exists():package(json.loads((G/'final_report.json').read_text()));return
 rs=records('val');assert len(rs)==92
 for r in rs:assert sha(r['GT_path'])==r['GT_sha256']
 if not (G/'selection.json').exists():
  candidates=[]
  for tag in ['best','last']:
   weight=G/'runs/low_snr20/weights'/(tag+'.pt');out=G/'validation'/tag
   pred=infer(weight,'val',out);best,sweep,metrics=sel.choose(rs,pred,sel.ANN_GRID,minimum_domain_F1=.55)
   csvout(out/'threshold_grid.csv',sweep);dump(out/'metrics.json',metrics)
   candidates.append(dict(tag=tag,checkpoint=str(weight),checkpoint_sha256=sha(weight),threshold=best['threshold'],validation_macro=metrics['macro'],validation_datasets=metrics['datasets']))
  eligible=[c for c in candidates if min(m['F1'] for m in c['validation_datasets'].values())>=.55]
  chosen=dict(max(eligible or candidates,key=lambda c:c['validation_macro']['F1']));chosen['candidates']=candidates
  dest=G/'selections/best.pt';dest.parent.mkdir(exist_ok=True);shutil.copy2(chosen['checkpoint'],dest);chosen['checkpoint']=str(dest);dump(G/'selection.json',chosen)
 chosen=json.loads((G/'selection.json').read_text());previous=json.loads((V2/'selected_model.json').read_text())
 gate=dict(ANN_source_macro_F1=chosen['validation_macro']['F1'],v2_source_macro_F1=previous['validation_macro']['F1'],ANN_min_domain_F1=min(m['F1'] for m in chosen['validation_datasets'].values()))
 baseline=json.loads((V2/'baseline_selection.json').read_text())
 # Source-only gate: no more than .01 macro loss; still beat references and preserve worst domain.
 gate['passed']=gate['ANN_source_macro_F1']>=gate['v2_source_macro_F1']-.01 and gate['ANN_min_domain_F1']>=.55 and all(gate['ANN_source_macro_F1']>m['validation_macro']['F1'] for m in baseline.values())
 dump(G/'development_gate.json',gate);print('SOURCE_SELECTION',json.dumps(chosen),flush=True);print('SOURCE_GATE',json.dumps(gate),flush=True)
 diagnostic=G/'source_stress/v3_selected/fixed_threshold_metrics.json'
 if not diagnostic.exists():
  with (G/'stress_selected.log').open('a') as log:subprocess.run([PY,'-u',str(G/'code/stress_diagnostic.py'),'--weights',chosen['checkpoint'],'--out',str(diagnostic.parent)],stdout=log,stderr=subprocess.STDOUT,check=True)
 if not gate['passed'] or not args.finalize:return
 if not (G/'target_integrity.json').exists():subprocess.run([PY,'-u',str(G/'code/prepare_targets.py')],check=True)
 import importlib
 sys.path.insert(0,str(V1/'code'));bc=importlib.import_module('baseline_external');bc.G=G
 bc.records=lambda split,datasets=None:[r for r in records(split) if datasets is None or r['dataset'] in datasets]
 (G/'baseline_configs').mkdir(exist_ok=True)
 for name in ['cryolo_10059.json','cryolo_10081.json','protocol.json']:shutil.copy2(V2/'baseline_configs'/name,G/'baseline_configs'/name)
 if not (G/'final_frozen_config.json').exists():freeze(chosen,bc,baseline)
 protect();test=records('final_check');assert len(test)==20
 predictions={'ANN_v3':infer(chosen['checkpoint'],'final_check',G/'results/ANN_v3')}
 import gc
 gc.collect();torch.cuda.empty_cache()
 for method in ['crYOLO','Topaz']:
  protect();p,_=bc.one('10059',method,'final_check');predictions[method]=p
 hcn=records('regression_check');assert len(hcn)==20 and {r['dataset'] for r in hcn}=={'10081'}
 predictions['ANN_v3'].update(infer(chosen['checkpoint'],'regression_check',G/'results/ANN_v3_HCN1_regression'))
 for method in ['crYOLO','Topaz']:
  out=V2/'results/baseline_final_check'/method/'10081';done=json.loads((out/'done.json').read_text())
  assert sha(out/'predictions.csv')==done['predictions_sha256']
  assert done['checkpoint_sha256']==sha(bc.CRYO_WEIGHT if method=='crYOLO' else bc.TOP_WEIGHT)
  predictions[method].update(load_predictions(out/'predictions.csv'))
 control_protocol=json.loads((G/'v2_control_protocol.json').read_text());assert sha(V2/'selections/best.pt')==control_protocol['checkpoint_sha256']
 control_predictions=infer(V2/'selections/best.pt','final_check',G/'results/ANN_v2_same_batch_control')
 olddone=json.loads((V2/'results/final_check/ANN_v2/done.json').read_text())
 assert sha(V2/'results/final_check/ANN_v2/predictions.csv')==olddone['predictions_sha256']
 oldpred=load_predictions(V2/'results/final_check/ANN_v2/predictions.csv');control_predictions.update({key(r):oldpred.get(key(r),[]) for r in hcn})
 protect();assert not (G/'reports/evaluation_lock.json').exists();dump(G/'reports/evaluation_lock.json',dict(status='started',started_unix=time.time()))
 table=[];allmetrics={}
 for method,pred in predictions.items():
  threshold=chosen['threshold'] if method=='ANN_v3' else baseline[method]['threshold'];metrics=evaluate(test+hcn,pred,threshold);allmetrics[method]=metrics
  for ds in ['10059','10081']:table.append(dict(method=method,dataset=ds,role='new_independent_micrographs' if ds=='10059' else 'previous_HCN1_regression',images=20,threshold=threshold,**metrics['datasets'][ds]))
  dump(G/'reports'/(method+'_metrics.json'),metrics);csvout(G/'reports'/(method+'_per_micrograph.csv'),metrics['per_image'])
  picks=[dict(method=method,dataset=r['dataset'],micrograph_name=r['micrograph_name'],x_original_px=p.x,y_original_px=p.y,x_mrc_array_px=p.x,y_mrc_array_px=r['height']-1-p.y,score=p.score) for r in test+hcn for p in pred.get(key(r),[]) if p.score>=threshold]
  csvout(G/'reports'/(method+'_coordinates.csv'),picks,['method','dataset','micrograph_name','x_original_px','y_original_px','x_mrc_array_px','y_mrc_array_px','score'])
 csvout(G/'reports/per_dataset_metrics.csv',table)
 control=evaluate(test+hcn,control_predictions,control_protocol['threshold'])
 control_table=[dict(method='ANN_v2_control',dataset=ds,images=20,threshold=control_protocol['threshold'],**m) for ds,m in control['datasets'].items()]
 csvout(G/'reports/v2_same_batch_control.csv',control_table);csvout(G/'reports/v2_control_per_micrograph.csv',control['per_image']);dump(G/'reports/v2_same_batch_control.json',control)
 rng=np.random.default_rng(100283);ci={}
 for ds in ['10059','10081']:
  per={m:[r for r in metric['per_image'] if r['dataset']==ds] for m,metric in allmetrics.items()}
  names=[[r['micrograph_name'] for r in rows] for rows in per.values()];assert all(n==names[0] for n in names)
  sample=rng.integers(0,20,size=(1000,20));boots={}
  for method,rows in per.items():
   counts=np.array([[r[k] for k in ['TP','FP','FN']] for r in rows]);s=counts[sample].sum(axis=1);den=2*s[:,0]+s[:,1]+s[:,2];boots[method]=np.divide(2*s[:,0],den,out=np.zeros(len(den)),where=den>0)
  ci[ds]={m:dict(F1_95CI=np.percentile(v,[2.5,97.5]).tolist()) for m,v in boots.items()}
  ci[ds]['paired_ANN_F1_difference_95CI']={m:np.percentile(boots['ANN_v3']-boots[m],[2.5,97.5]).tolist() for m in ['crYOLO','Topaz']}
 dump(G/'reports/bootstrap_micrograph_95CI.json',ci)
 report=dict(status='complete',ANN=chosen,table=table,images=20,GT=sum(r['GT_count'] for r in test),
  ANN_F1_exceeds_both_baselines=all(allmetrics['ANN_v3']['datasets']['10059']['F1']>allmetrics[m]['datasets']['10059']['F1'] for m in ['crYOLO','Topaz']),
  same_v3_model_both_datasets_exceed_baselines=all(allmetrics['ANN_v3']['datasets'][ds]['F1']>allmetrics[m]['datasets'][ds]['F1'] for ds in ['10059','10081'] for m in ['crYOLO','Topaz']),
  HCN1_result_retained_with_different_v2_model=True,HCN1_secondary_regression_same_v3_model=True,
  v2_control_table=control_table,bootstrap=ci,no_target_training=True,no_target_threshold_scan=True,matching_radius_px={'10059':33,'10081':38.5},frozen_config_sha256=sha(G/'final_frozen_config.json'))
 dump(G/'final_report.json',report);dump(G/'reports/evaluation_lock.json',dict(status='complete',report_sha256=sha(G/'final_report.json')))
 print('FINAL_RESULTS',json.dumps(report),flush=True);package(report)

if __name__=='__main__':main()
