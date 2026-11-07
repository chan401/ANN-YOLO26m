"""Development-only calibration, parameter freeze, one-time two-target comparison."""
import importlib,shutil,sys,time,zipfile,gc,subprocess,argparse
from common import *
from common import _edges
from infer import infer
ANN_GRID=[.005,.01,.02,.03]+[round(i/50,3) for i in range(2,50)]
CRYO_GRID=[.0001,.001,.005,.01]+[round(i/50,3) for i in range(1,50)]
TOP_GRID=[-8.,-6.,-4.,-3.,-2.,-1.,0.,.5,1.,1.5,2.,2.5,2.7722246646881104,3.,3.5,4.,5.,6.,8.]
PY='/root/autodl-tmp/cryoem_annyolo26/.venv/bin/python'

def forward(selected,rs,split,out):
 if selected['profile']=='canonical64_lowpass1':return infer(Path(selected['checkpoint']),rs,out)
 assert selected['profile']=='native_multiscale'
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 with (out/'wrapper.log').open('a') as f:subprocess.run([PY,'-u',str(G/'code/native_wrapper.py'),'infer','--weights',selected['checkpoint'],'--split',split,'--out',str(out)],stdout=f,stderr=subprocess.STDOUT,cwd=G/'code',check=True)
 return load_predictions(out/'predictions.csv')

def choose(rs,pred,grid,minimum_domain_F1=None):
 sys.setrecursionlimit(10000)
 domains=sorted({r['dataset'] for r in rs});totals={t:{d:[0,0,0] for d in domains} for t in grid}
 for r in rs:
  gt=points(r);pp=sorted(pred.get(key(r),[]),key=lambda p:(-p.score,p.idx));adj={};gt_to_pred={};n=tp=0
  def augment(pi,seen):
   for gi in adj[pi]:
    if gi in seen:continue
    seen.add(gi)
    if gi not in gt_to_pred or augment(gt_to_pred[gi],seen):gt_to_pred[gi]=pi;return True
   return False
  for t in sorted(grid,reverse=True):
   while n<len(pp) and pp[n].score>=t:
    adj[n]=_edges(pp[n],gt,r['diameter']/4.)
    if augment(n,set()):tp+=1
    n+=1
   c=totals[t][r['dataset']];c[0]+=tp;c[1]+=n-tp;c[2]+=len(gt)-tp
 sweep=[]
 for t in grid:
  mm={d:metric_counts(*v) for d,v in totals[t].items()}
  sweep.append(dict(threshold=t,macro_F1=float(np.mean([v['F1'] for v in mm.values()])),min_domain_F1=min(v['F1'] for v in mm.values()),**{f'{d}_{k}':v[k] for d,v in mm.items() for k in ['TP','FP','FN','Precision','Recall','F1']}))
 feasible=[r for r in sweep if minimum_domain_F1 is None or r['min_domain_F1']>=minimum_domain_F1]
 best=max(feasible or sweep,key=lambda r:(r['macro_F1'],r['min_domain_F1'],r['threshold']))
 actual=evaluate(rs,pred,best['threshold'])
 for d,m in actual['datasets'].items():assert [m[k] for k in ['TP','FP','FN']]==totals[best['threshold']][d]
 return best,sweep,actual

def baseline_engine():
 sys.path.insert(0,str(V1/'code'));bc=importlib.import_module('baseline_external');bc.G=G
 bc.records=lambda split,datasets=None:[r for r in records(split) if datasets is None or r['dataset'] in datasets]
 (G/'baseline_configs').mkdir(exist_ok=True)
 cfg=json.loads((bc.B/'cryolo/pretrained_baseline/config_lowpass_0p1_box224.json').read_text())
 for ds in DOMAINS+TARGETS:
  r=next(r for r in json.loads((G/'image_registry.json').read_text()) if r['dataset']==ds)
  current=json.loads(json.dumps(cfg));current['model']['anchors']=[r['diameter']]*2;dump(G/'baseline_configs'/('cryolo_'+ds+'.json'),current)
 dump(G/'baseline_configs/protocol.json',dict(v1_protocol_sha256=sha(V1/'baseline_configs/protocol.json'),preprocessing_and_GPU_engine='Unchanged v1 baseline_external.py',engine_sha256=sha(V1/'code/baseline_external.py'),new_domains=DOMAINS[-3:],target_domains=TARGETS,validation='same92 validation micrographs across seven source domains; no HCN1/TRPV1 calibration',pretraining_overlap='crYOLO 10081 known exposure; Topaz complete pretrained list unavailable'))
 return bc

def freeze(selected,baseline,bc):
 deployment=json.loads((G/'deployment_equivalence_v2.json').read_text());assert deployment['status']=='PASS'
 assert deployment['picker_sha256']==sha(G/'code/predict_particles.py')
 protected={str(p):sha(p) for p in (G/'code').glob('*.py')}
 for p in [G/'protocol.json',G/'image_registry.json',G/'training_data/tile_registry.json',G/'adapter_equivalence.json',G/'deployment_equivalence_v2.json',G/'development_selection_rule.json',G/'input_integrity.json',G/'selected_model.json',G/'baseline_selection.json',Path(selected['checkpoint']),bc.CRYO_WEIGHT,bc.TOP_WEIGHT,V1/'code/baseline_external.py',G/'baseline_configs/protocol.json']:
  protected[str(p)]=sha(p)
 for p in G.joinpath('baseline_configs').glob('cryolo_*.json'):protected[str(p)]=sha(p)
 for p in [D/'metadata/three_method_evaluation/evaluate_empiar10028_v3_pretrained_val.py',D/'experiments/test_100_v1_three_methods/code/baseline_helpers.py',D/'experiments/test_100_v1_three_methods/topaz_runtime_compat/sitecustomize.py',D/'experiments/test_100_v1_three_methods/topaz_preprocess_repro/sitecustomize.py',bc.B/'cryolo/env-gpu-py38/bin/cryolo_predict_gpu.sh',bc.B/'topaz/.venv-gpu/bin/topaz']:
  protected[str(p)]=sha(p)
 for r in records('val'):
  assert sha(r['GT_path'])==r['GT_sha256'];protected[r['GT_path']]=r['GT_sha256']
 for p in ULTRA.joinpath('ultralytics').rglob('*'):
  if p.is_file() and p.suffix in ['.py','.yaml','.yml']:protected[str(p)]=sha(p)
 target=[r for r in json.loads((G/'image_registry.json').read_text()) if r['split']=='final_check']
 for r in target:
  assert sha(r['path'])==r['sha256'] and sha(r['GT_path'])==r['GT_sha256']
  protected[r['path']]=r['sha256'];protected[r['GT_path']]=r['GT_sha256']
 for p in (G/'native7/code').glob('*.py'):protected[str(p)]=sha(p)
 native=selected['profile']=='native_multiscale'
 f=dict(frozen=True,created_unix=time.time(),ANN=selected,baselines=baseline,protected=protected,targets=TARGETS,final_check_images=len(target),profile=selected['profile'],tile_size=1536 if native else SIZE,particle_diameter_scaled=None if native else PARTICLE,lowpass_sigma=[2,4] if native else SIGMA,overlap=256,pad=128 if native else PAD,max_det=2000,score_floor=1e-5 if native else .001,merge_radius=28 if native else PARTICLE/8,merge_radius_units='native pixels' if native else 'canonical pixels',matching='same maximum-cardinality one-to-one, published native D/4',target_images_in_ANN_training=False,target_outcomes_previously_known_on_different_v1_micrographs=True,no_final_check_threshold_scan=True)
 dump(G/'final_frozen_config.json',f);(G/'final_frozen_config.sha256').write_text(sha(G/'final_frozen_config.json')+'\n')
 return f

def frozen():
 assert sha(G/'final_frozen_config.json')==(G/'final_frozen_config.sha256').read_text().strip()
 f=json.loads((G/'final_frozen_config.json').read_text())
 for p,h in f['protected'].items():assert sha(p)==h,('Frozen input changed',p)
 return f

def package(report):
 export=G/'exports/generalization_v2';export.mkdir(parents=True,exist_ok=True)
 for rel in ['final_report.json','final_frozen_config.json','final_frozen_config.sha256','protocol.json','preparation_report.json','cleanup_result.json','connectivity.json','adapter_equivalence.json','deployment_equivalence_v2.json','development_selection_rule.json','input_integrity.json','validation_gate.json','selected_model.json','baseline_selection.json','image_registry.json']:
  shutil.copy2(G/rel,export/Path(rel).name)
 for folder in ['reports','code']:
  dest=export/folder;dest.mkdir(exist_ok=True)
  for p in (G/folder).rglob('*'):
   if p.is_file() and p.suffix in ['.py','.json','.csv','.md','.png']:
    q=dest/p.relative_to(G/folder);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
 shutil.copy2(report['ANN']['checkpoint'],export/'best.pt')
 canonical_dir=export/'canonical_candidate';canonical_dir.mkdir(exist_ok=True)
 shutil.copy2(G/'runs/canonical64_lowpass1_7domains/results.csv',canonical_dir/'training_results.csv')
 shutil.copy2(G/'canonical64_lowpass1_7domains_config.json',canonical_dir/'training_config.json')
 shutil.copy2(G/'runs/canonical64_lowpass1_7domains/completion.json',canonical_dir/'completion.json')
 native_dir=export/'native_control';native_dir.mkdir(exist_ok=True)
 for src,name in [(G/'native7/native7_training_config.json','training_config.json'),(G/'native7/runs/native7/results.csv','training_results.csv'),(G/'native7/adapter_config.json','adapter_config.json')]:shutil.copy2(src,native_dir/name)
 for p in (G/'native7/code').glob('*.py'):shutil.copy2(p,native_dir/p.name)
 shutil.copy2(G/'native7/runs/native7/completion.json',native_dir/'completion.json')
 chosen_dir=native_dir if report['ANN']['profile']=='native_multiscale' else canonical_dir
 shutil.copy2(chosen_dir/'training_config.json',export/'training_config.json')
 shutil.copy2(chosen_dir/'training_results.csv',export/'training_results.csv')
 warmup=G/'source_scale_adaptation'
 for p,name in [(warmup/'canonical64_source4_config.json','source_scale_adaptation_config.json'),(warmup/'runs/canonical64_source4/results.csv','source_scale_adaptation_results.csv')]:
  shutil.copy2(p,export/name)
 shutil.copy2(warmup/'runs/canonical64_source4/weights/best.pt',export/'source_scale_initialization.pt')
 pipeline=json.loads((G/'final_frozen_config.json').read_text())
 dump(export/'model_configuration.json',{k:pipeline[k] for k in ['profile','tile_size','particle_diameter_scaled','lowpass_sigma','overlap','pad','max_det','score_floor','merge_radius','merge_radius_units']}|{'input_profile':report['ANN']['profile'],'threshold':report['ANN']['threshold'],'global_threshold':report['ANN']['threshold'],'checkpoint_sha256':report['ANN']['checkpoint_sha256']})
 shutil.copy2(G/'code/predict_particles.py',export/'predict_particles.py')
 for name in ['ultralytics_runtime_source.zip','runtime_versions.json','server_environment_freeze.txt']:shutil.copy2(V1/'exports/generalization_v1/model'/name,export/name)
 instructions='''# 独立颗粒挑选程序

需要 best.pt、model_configuration.json、predict_particles.py，以及匹配的 PyTorch/Ultralytics 运行环境。运行时不读取 GT、测试划分清单或 EMPIAR-10028 项目目录。匹配的 Ultralytics 源码和环境版本随包保存。

在已有服务器 Python 环境中运行：

    python predict_particles.py --config model_configuration.json --weights best.pt --input /path/to/micrographs --output new_picks

输入为单帧2D MRC，可传入目录或一个/多个文件。输出目录应为空。native_multiscale 模式保留原生像素尺度；canonical64_lowpass1 模式还需通过 --diameter 指定事先确定的原生颗粒直径，同一次调用使用相同粒径。

coordinates_mrc_array/*.txt 是0起始的MRC数组坐标：X=列，Y=行。predictions.csv 同时记录 MRC 数组坐标和 CryoPPP 坐标，关系为 Y_MRC=图像高度-1-Y_CryoPPP。

reports/*_retained_coordinates.csv 中 x_original_px/y_original_px 为本项目评估使用的 CryoPPP 原生像素坐标；另给出 x_mrc_array_px/y_mrc_array_px 以供图像叠加和导出。

模型与阈值由源域验证集选择。两个目标数据集使用同一个最终权重和统一阈值，不能在目标检查数据上重新扫描阈值。
'''
 (export/'README_运行说明.md').write_text(instructions,encoding='utf-8')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 names={'10081':'HCN1','10059':'TRPV1'};colors={'ANN_v2':'#1976D2','crYOLO':'#F29E25','Topaz':'#55A868'}
 fig,ax=plt.subplots(figsize=(7.6,4.5),dpi=180)
 for j,method in enumerate(['ANN_v2','crYOLO','Topaz']):
  values=[next(r['F1'] for r in report['table'] if r['dataset']==ds and r['method']==method) for ds in TARGETS]
  x=np.arange(2)+(j-1)*.24;ax.bar(x,values,width=.22,label=method,color=colors[method])
  for xx,v,ds in zip(x,values,TARGETS):
   ci=report['bootstrap'][ds][method]['F1_95CI'];ax.plot([xx,xx],ci,color='black',lw=1);ax.text(xx,ci[1]+.018,f'{v:.3f}',ha='center',fontsize=8)
 ax.set_xticks(np.arange(2),[names[d] for d in TARGETS]);ax.set_ylim(0,1.08);ax.set_ylabel('F1');ax.legend(loc='upper left',ncol=3,frameon=False);ax.spines[['right','top']].set_visible(False)
 ax.set_title('New micrograph check: 20 images per target dataset');fig.tight_layout();fig.savefig(export/'F1_comparison.png');fig.savefig(export/'F1_comparison.pdf');plt.close(fig)
 curve_rows=list(csv.DictReader((export/'training_results.csv').open()));epochs=np.array([int(float(r['epoch'])) for r in curve_rows])
 fig,axes=plt.subplots(1,3,figsize=(13.5,3.8),dpi=180)
 for ax,kind in zip(axes[:2],['box','cls']):
  for split in ['train','val']:ax.plot(epochs,[float(r[split+'/'+kind+'_loss']) for r in curve_rows],label=split)
  ax.set_title('Box loss' if kind=='box' else 'Classification loss');ax.set_xlabel('Epoch');ax.set_ylabel('Loss');ax.legend(frameon=False);ax.spines[['right','top']].set_visible(False)
 for field,label in [('metrics/mAP50(B)','mAP50'),('metrics/mAP50-95(B)','mAP50-95')]:axes[2].plot(epochs,[float(r[field]) for r in curve_rows],label=label)
 axes[2].set_title('Tile-level validation proxy');axes[2].set_ylim(0,1);axes[2].set_xlabel('Epoch');axes[2].legend(frameon=False);axes[2].spines[['right','top']].set_visible(False)
 fig.suptitle('Selected ANN candidate: '+report['ANN']['profile']);fig.tight_layout();fig.savefig(export/'training_curves.png');fig.savefig(export/'training_curves.pdf');plt.close(fig)
 input_description='统一到64px颗粒尺度、抗混叠缩小、sigma1低通、重采样后稳健归一化、1024图块、256重叠、96反射边界' if report['ANN']['profile']=='canonical64_lowpass1' else '保持原生像素尺度，原始灰度与sigma2/sigma4低通三通道，实际通道顺序LP4/LP2/raw，1536图块、256重叠、128反射边界；沿用v1训练与推理规则'
 a_epochs=len(list(csv.DictReader((G/'runs/canonical64_lowpass1_7domains/results.csv').open())))
 b_epochs=len(list(csv.DictReader((G/'native7/runs/native7/results.csv').open())))
 lines=['# ANN-YOLO26 泛化 v2','',f'目标：HCN1（EMPIAR-10081）与 TRPV1（EMPIAR-10059），各20张新增微图。','',
  '训练使用10028、10017、10077、10291、10289、10240、10760；HCN1和TRPV1不进入监督训练或阈值选择。每方法统一阈值由相同92张源域验证微图选择。',
  '此前已查看过两目标条目其他微图的结果，因此本轮是已知目标域上的新微图检查，不能称为完全未见目标域的盲测。','',
  '既往v1在MS2的F1=0.0586失败记录仍保留；本轮未评价v2在MS2的表现。结论只覆盖用户指定的HCN1和TRPV1，不能扩展为任意蛋白的泛化保证。','',
  f"最终选择的输入方案：{report['ANN']['profile']}，{input_description}。全程float32灰度，预测恢复原生坐标。候选A先用原四源域做20轮尺度适配，再以七源域训练{a_epochs}轮（原计划最多40轮，中期源域宏F1=0.7014后提前停止）；候选B保留v1原生多频率输入，在七源域新增{b_epochs}轮。两候选及其best/last由同一92张开发验证微图选择。不把不同验证集上的代理mAP混作同一指标。",'',
  '匹配：三方法相同GT、原生粒径四分之一一对一规则（HCN1 38.5px，TRPV1 33px）。完整失败保留为漏检。','',
  '|数据集|方法|TP|FP|FN|Precision|Recall|F1|','|---|---|---:|---:|---:|---:|---:|---:|']
 for row in report['table']:
  lines.append(f"|{row['dataset']}|{row['method']}|{row['TP']}|{row['FP']}|{row['FN']}|{row['Precision']:.4f}|{row['Recall']:.4f}|{row['F1']:.4f}|")
 lines+=['',f"两数据集F1均高于两个基线：{report['both_targets_F1_exceed_both_baselines']}",'',
  '比较限制：crYOLO公开通用模型在HCN1条目存在已知训练重叠；本轮未独立排除Topaz的预训练重叠。这里比较已发布通用权重的直接迁移表现，不是统一训练条件的架构比较。',
  '本轮只对最终检查图按冻结参数评价一次；不能根据结果反复选阈值或改变匹配半径。若后续根据这些结果优化，需要新的独立检查数据。']
 (export/'泛化报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 manifest={str(p.relative_to(export)):sha(p) for p in export.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'};dump(export/'SHA256SUMS.json',manifest)
 archive=G/'exports/ANN_YOLO26m_generalization_v2.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for p in export.rglob('*'):
   if p.is_file():z.write(p,str(Path('generalization_v2')/p.relative_to(export)))
 dump(G/'export_manifest.json',dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive)))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--finalize',action='store_true');args=parser.parse_args()
 reports=G/'reports';reports.mkdir(exist_ok=True)
 if (G/'final_report.json').exists():package(json.loads((G/'final_report.json').read_text()));return
 rs=records('val');assert len(rs)==92 and not {r['dataset'] for r in rs}&set(TARGETS)
 if not(G/'selected_model.json').exists():
  candidates=[];seen=set()
  sources=[(profile,tag,folder/(tag+'.pt')) for profile,folder in [('canonical64_lowpass1',G/'runs/canonical64_lowpass1_7domains/weights'),('native_multiscale',G/'native7/runs/native7/weights')] for tag in ['best','last']]
  for profile,tag,weight in sources:
   h=sha(weight)
   if h in seen:continue
   seen.add(h);out=G/'validation'/(profile+'_'+tag);pred=forward(dict(profile=profile,checkpoint=str(weight)),rs,'val',out);best,sweep,metrics=choose(rs,pred,ANN_GRID,minimum_domain_F1=.55)
   csvout(out/'threshold_grid.csv',sweep);dump(out/'metrics.json',metrics)
   candidates.append(dict(tag=tag,profile=profile,checkpoint=str(weight),checkpoint_sha256=h,threshold=best['threshold'],validation_macro=metrics['macro'],validation_datasets=metrics['datasets']))
   gc.collect();torch.cuda.empty_cache()
  eligible=[c for c in candidates if min(m['F1'] for m in c['validation_datasets'].values())>=.55]
  selected=dict(max(eligible or candidates,key=lambda c:c['validation_macro']['F1']));dest=G/'selections/best.pt';dest.parent.mkdir(exist_ok=True);shutil.copy2(selected['checkpoint'],dest);selected['checkpoint']=str(dest)
  selected['selection_rule']='prospective source gate min-domain F1>=0.55 where feasible, then maximum mean per-domain validation F1 across seven source accessions; same fixed grid, target GT excluded';selected['candidates']=candidates;dump(G/'selected_model.json',selected)
 selected=json.loads((G/'selected_model.json').read_text());print('V2_MODEL_SELECTED',json.dumps(selected),flush=True)
 gc.collect();torch.cuda.empty_cache()
 bc=baseline_engine()
 if not(G/'baseline_selection.json').exists():
  baseline={}
  for method,grid in [('crYOLO',CRYO_GRID),('Topaz',TOP_GRID)]:
   pred={};reused={}
   for ds in DOMAINS:
    if ds in ['10028','10017','10077','10291']:
     p=V1/'results/baseline_val'/method/ds/'predictions.csv';pred.update(load_predictions(p));reused[str(p)]=sha(p)
    else:
     preds,_=bc.one(ds,method,'val');pred.update(preds)
   best,sweep,metrics=choose(rs,pred,grid);csvout(reports/(method+'_validation_grid.csv'),sweep);dump(reports/(method+'_validation_metrics.json'),metrics)
   baseline[method]=dict(threshold=best['threshold'],validation_macro=metrics['macro'],validation_datasets=metrics['datasets'],same_validation_images=92,reused_prediction_sha256=reused)
  dump(G/'baseline_selection.json',baseline)
 baseline=json.loads((G/'baseline_selection.json').read_text())
 # Do not spend the independent final check on a candidate that already fails
 # the source-domain development reference. All conditions are target-free.
 gate=dict(rule='ANN macro validation F1 exceeds both pretrained references on the same92 source-validation images; worst source-domain F1 >= 0.55',
  ANN_macro_F1=selected['validation_macro']['F1'],baseline_macro_F1={m:d['validation_macro']['F1'] for m,d in baseline.items()},
  ANN_min_domain_F1=min(d['F1'] for d in selected['validation_datasets'].values()))
 gate['passed']=all(gate['ANN_macro_F1']>v for v in gate['baseline_macro_F1'].values()) and gate['ANN_min_domain_F1']>=.55
 dump(G/'validation_gate.json',gate)
 if not gate['passed']:
  print('DEVELOPMENT_GATE_NOT_PASSED_NO_FINAL_CHECK',json.dumps(gate),flush=True);return
 if not args.finalize:
  print('DEVELOPMENT_VALIDATION_COMPLETE_FINAL_CHECK_NOT_STARTED',json.dumps(gate),flush=True);return
 if not(G/'final_frozen_config.json').exists():freeze(selected,baseline,bc)
 f=frozen();test=records('final_check');assert len(test)==40
 # No target metrics are produced until all three fixed-parameter forward passes finish.
 predictions={'ANN_v2':forward(selected,test,'final_check',G/'results/final_check/ANN_v2')}
 gc.collect();torch.cuda.empty_cache()
 for method in ['crYOLO','Topaz']:
  predictions[method]={}
  for ds in TARGETS:
   frozen();p,_=bc.one(ds,method,'final_check');predictions[method].update(p)
 frozen();assert not(reports/'evaluation_lock.json').exists(),'Final check already evaluated'
 dump(reports/'evaluation_lock.json',dict(status='started',frozen_config_sha256=sha(G/'final_frozen_config.json'),started_unix=time.time()))
 table=[];allmetrics={}
 for method,pred in predictions.items():
  t=selected['threshold'] if method=='ANN_v2' else baseline[method]['threshold'];metrics=evaluate(test,pred,t);allmetrics[method]=metrics
  for ds,m in metrics['datasets'].items():table.append(dict(method=method,dataset=ds,images=sum(r['dataset']==ds for r in test),threshold=t,**m))
  csvout(reports/(method+'_per_micrograph.csv'),metrics['per_image']);dump(reports/(method+'_metrics.json'),metrics)
  picks=[]
  for r in test:
   picks.extend(dict(method=method,dataset=r['dataset'],micrograph_name=r['micrograph_name'],x_original_px=p.x,y_original_px=p.y,x_mrc_array_px=p.x,y_mrc_array_px=r['height']-1-p.y,score=p.score) for p in pred.get(key(r),[]) if p.score>=t)
  csvout(reports/(method+'_retained_coordinates.csv'),picks,['method','dataset','micrograph_name','x_original_px','y_original_px','x_mrc_array_px','y_mrc_array_px','score'])
 csvout(reports/'per_dataset_metrics.csv',table)
 rng=np.random.default_rng(100282);ci={}
 for ds in TARGETS:
  per={m:[r for r in metrics['per_image'] if r['dataset']==ds] for m,metrics in allmetrics.items()}
  names=[[r['micrograph_name'] for r in rr] for rr in per.values()];assert all(n==names[0] for n in names)
  samples=rng.integers(0,len(names[0]),size=(1000,len(names[0])))
  boots={}
  for method,rr in per.items():
   counts=np.array([[r[k] for k in ['TP','FP','FN']] for r in rr]);summed=counts[samples].sum(axis=1);den=2*summed[:,0]+summed[:,1]+summed[:,2];boots[method]=np.divide(2*summed[:,0],den,out=np.zeros(len(den)),where=den>0)
  ci[ds]={method:dict(F1_95CI=np.percentile(v,[2.5,97.5]).tolist()) for method,v in boots.items()}
  ci[ds]['paired_F1_difference_95CI']={method:np.percentile(boots['ANN_v2']-boots[method],[2.5,97.5]).tolist() for method in ['crYOLO','Topaz']}
 dump(reports/'bootstrap_micrograph_95CI.json',ci)
 passed=all(allmetrics['ANN_v2']['datasets'][ds]['F1']>allmetrics[m]['datasets'][ds]['F1'] for ds in TARGETS for m in ['crYOLO','Topaz'])
 report=dict(status='complete',ANN=selected,baselines=baseline,table=table,final_images=len(test),final_GT=sum(r['GT_count'] for r in test),both_targets_F1_exceed_both_baselines=passed,bootstrap=ci,targets_not_in_supervised_training=True,known_target_domain_check_not_pristine_domain_blind=True,matching_changed=False,frozen_config_sha256=sha(G/'final_frozen_config.json'))
 dump(G/'final_report.json',report);dump(reports/'evaluation_lock.json',dict(status='complete',report_sha256=sha(G/'final_report.json'),completed_unix=time.time()))
 print('V2_FINAL_RESULTS',json.dumps(report),flush=True);package(report)

if __name__=='__main__':main()
