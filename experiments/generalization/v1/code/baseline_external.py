"""GPU pretrained reference inference; source-validation thresholds; published diameter-based box/NMS sizes."""
import argparse,subprocess
from general_common import *
B=Path('/root/autodl-tmp/cryoem_baselines');OLD=D/'experiments/test_100_v1_three_methods'
sys.path.insert(0,str(OLD/'code'));import baseline_helpers as helper
CRYO_WEIGHT=B/'cryolo/weights/gmodel_phosnet_202005_N63_c17.h5'
TOP_WEIGHT=B/'topaz/topaz/pretrained/detector/resnet16_u64.sav'
THRESHOLDS={'crYOLO':.6545040607452393,'Topaz':2.7722246646881104}
FIELDS=['method','dataset','split','micrograph_name','image_key','x_original_px','y_original_px','score']

def run(cmd,log,extra=None):
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',NVIDIA_TF32_OVERRIDE='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 if extra:env.update(extra)
 with log.open('w') as h:
  h.write('COMMAND '+json.dumps(list(map(str,cmd)))+'\n');h.flush();subprocess.run(list(map(str,cmd)),stdout=h,stderr=subprocess.STDOUT,check=True,env=env,cwd=log.parent)
def input_path(r):
 p=Path(r['path'])
 with mrcfile.open(p,permissive=True) as m:
  a=np.array(m.data,copy=True)
  unsigned=r['dataset']=='10059' and int(m.header.mode)==0
  if unsigned:a=a.view(np.uint8)
  a=a.astype(np.float32)
 q1,q99=np.percentile(a,[1,99]);span=float(q99-q1);assert span>0
 low,high=float(q1-10*span),float(q99+10*span);mask=(a<low)|(a>high);count=int(mask.sum())
 if unsigned or count:
  dest=G/'datasets'/r['dataset']/'baseline_input_compat'/p.name;dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists():
   a=np.clip(a,low,high)
   with mrcfile.new(dest,overwrite=False) as m:m.set_data(a)
  dump(dest.with_suffix('.provenance.json'),{'original_path':str(p),'original_sha256':sha(p),'derived_sha256':sha(dest),'unsigned_mode0_format_cast':unsigned,
   'clipped_pixels':count,'bounds':[low,high],'rule':'P1-10*(P99-P1) to P99+10*(P99-P1); existing source-validation080 robust rule, applied uniformly prospectively; no GT/model outputs used'})
  return dest
 return p
def prepare_config():
 root=G/'baseline_configs';root.mkdir(exist_ok=True);p=protocol()
 for ds,v in {**p['datasets'],'10028':{'metadata':{'diameter':224}}}.items():
  cfg=json.loads((B/'cryolo/pretrained_baseline/config_lowpass_0p1_box224.json').read_text());cfg['model']['anchors']=[v['metadata']['diameter']]*2
  dump(root/('cryolo_'+ds+'.json'),cfg)
 dump(root/'protocol.json',{'crYOLO':{'weight':str(CRYO_WEIGHT),'sha256':sha(CRYO_WEIGHT),'threshold':THRESHOLDS['crYOLO'],'device':0,'prediction_batch_size':1,'score_floor':0,'IoU_NMS':.3,'anchors':'published native particle diameter','normalization':'STANDARD','lowpass':.1},
  'Topaz':{'weight':str(TOP_WEIGHT),'sha256':sha(TOP_WEIGHT),'threshold':THRESHOLDS['Topaz'],'model':'resnet16_u64','device':0,'preprocess':'same CPU GMM; scale8; seed10028','segment_patch_size':144,'score_floor':'-inf','NMS_radius':'max(1,round(published_D/16)) in downsampled pixels','extract_upscale':8},
  'coordinate_adapters':'unchanged validated helper functions except native image width/height values; x_CryoPPP=x_native; y_Topaz=H-1-y_native; crYOLO y=H-(boxY+h/2)',
  'operational_failure_policy':'nonfinite candidates cannot be matched; retain only finite scores; entirely nonfinite score maps ->zero valid predictions, all GT counted as FN; no repair or image dropping',
  'input_robustness':'Uniform prospective P1/P99 +/-10span pixel outlier guard based on old10028 validation080 rule; protects variance calculation from extreme finite values; source files unchanged; no blind data tuning',
  'threshold_selection':'probe initially uses source-validation thresholds; final comparison calibrates one global threshold per method on the same4 development-validation accessions before heldout freeze; no target-heldout calibration','pretraining_overlap':'not independently excluded; reference pretrained models may have seen evaluation accessions'})
def one(ds,method,split):
 rs=records(split,[ds]);out=G/'results'/('baseline_'+split)/method/ds;out.mkdir(parents=True,exist_ok=True)
 if (out/'done.json').exists():
  d=json.loads((out/'done.json').read_text());assert sha(out/'predictions.csv')==d['predictions_sha256'];return load_predictions(out/'predictions.csv'),rs
 paths=[input_path(r) for r in rs];per=out/'per_image';per.mkdir(exist_ok=True);failures=[]
 if method=='crYOLO':
  raw=out/'raw'
  run([B/'cryolo/env-gpu-py38/bin/cryolo_predict_gpu.sh','-c',G/'baseline_configs'/('cryolo_'+ds+'.json'),'-w',CRYO_WEIGHT,'-i',*paths,'-o',raw,'-t','0','-nc','1','-pbs','1','-g','0','--otf','--write_empty'],out/'GPU_predict.log')
  for r in rs:
   helper.IMAGE_WIDTH=r['width'];helper.IMAGE_HEIGHT=r['height'];cbox=raw/'CBOX'/(Path(r['micrograph_name']).stem+'.cbox')
   helper.cryolo_cbox_to_csv(cbox,per/(key(r)+'.csv'),r['micrograph_name'],allow_empty=cbox.exists() and cbox.stat().st_size==0)
 else:
  cli=B/'topaz/.venv-gpu/bin/topaz';pre=out/'preprocessed';scores=out/'scores';pre.mkdir(exist_ok=True);scores.mkdir(exist_ok=True)
  run([cli,'preprocess','--scale','8','--metadata','--device','-1','--num-workers','0','--num-threads','1','--destdir',pre,*paths],out/'preprocess.log',{'PYTHONPATH':str(OLD/'topaz_preprocess_repro')})
  run([cli,'segment','--model','resnet16_u64','--device','0','--num-threads','1','--patch-size','144','--destdir',scores,*[pre/p.name for p in paths]],out/'GPU_segment.log',{'PYTHONPATH':str(OLD/'topaz_runtime_compat')})
  radius=max(1,round(rs[0]['diameter']/16));native=out/'raw_candidates.tsv'
  run([cli,'extract','--model','none','--radius',str(radius),'--threshold=-inf','--up-scale','8','--device','-1','--num-threads','1','--num-workers','0','--format','csv','--output',native,*[scores/(p.stem+'.tiff') for p in paths]],out/'extract.log')
  with native.open(newline='') as h:
   first=h.readline();h.seek(0);reader=csv.DictReader(h,delimiter='\t' if '\t' in first else ',');fields=reader.fieldnames;groups={Path(r['micrograph_name']).stem:[] for r in rs}
   for row in reader:groups[Path(row['image_name']).stem].append(row)
  for r in rs:
   nn=Path(r['micrograph_name']).stem;allrows=groups[nn];valid=[row for row in allrows if all(math.isfinite(float(row[k])) for k in ['x_coord','y_coord','score'])]
   if len(valid)!=len(allrows):failures.append({'micrograph_name':r['micrograph_name'],'nonfinite_candidates':len(allrows)-len(valid),'valid_candidates':len(valid)})
   nt=per/(key(r)+'_native.tsv');csvout(nt,valid,fields);helper.IMAGE_WIDTH=r['width'];helper.IMAGE_HEIGHT=r['height'];helper.topaz_csv_to_canonical(nt,per/(key(r)+'.csv'),r['micrograph_name'])
 rows=[]
 for r in rs:
  with (per/(key(r)+'.csv')).open(newline='') as h:
   for row in csv.DictReader(h):rows.append({**{k:row[k] for k in ['method','micrograph_name','x_original_px','y_original_px','score']},'dataset':ds,'split':split,'image_key':key(r)})
 csvout(out/'predictions.csv',rows,FIELDS);dump(out/'done.json',{'status':'complete','method':method,'images':len(rs),'predictions_sha256':sha(out/'predictions.csv'),'checkpoint_sha256':sha(CRYO_WEIGHT if method=='crYOLO' else TOP_WEIGHT),'baseline_protocol_sha256':sha(G/'baseline_configs/protocol.json'),'GPU':0,'failure_records':failures,'probe_source_threshold_reference':THRESHOLDS[method]})
 return load_predictions(out/'predictions.csv'),rs
def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['probe','val','heldout'],default='probe');p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
 if a.prepare_only:prepare_config();return
 assert (G/'baseline_configs/protocol.json').exists();results=[]
 for method in ['crYOLO','Topaz']:
  for ds in (HELDOUT if a.split=='heldout' else ['10028']+DEVELOPMENT if a.split=='val' else DEVELOPMENT):
   pred,rs=one(ds,method,a.split)
   if a.split=='probe':
    m=evaluate(rs,pred,THRESHOLDS[method]);dump(G/'results/baseline_probe'/method/ds/'metrics.json',m);print('BASELINE_PROBE_METRICS',method,ds,json.dumps(m['datasets']),flush=True)
if __name__=='__main__':main()
