"""Archive HCN1 v2, create an isolated source-only low-SNR experiment."""
from pathlib import Path
import csv, hashlib, json, os, random, shutil, time, zipfile

ROOT=Path('/root/autodl-tmp/cryoem_generalization')
V2=ROOT/'v2';G=ROOT/'v3_low_snr';ARCH=ROOT/'archives/HCN1_v2_frozen'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def link(src,dst):
 dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
 if not dst.exists():os.link(src,dst)
 assert sha(src)==sha(dst)

def main():
 ARCH.mkdir(parents=True,exist_ok=True);G.mkdir(parents=True,exist_ok=True)
 final=json.loads((V2/'final_report.json').read_text())
 assert final['ANN']['checkpoint_sha256']=='92776d3c11fd1157a012fac6f3c38ba280c8e077b22a18c9c67176cd31703679'
 for name in ['best.pt','model_configuration.json','predict_particles.py','README_运行说明.md','ultralytics_runtime_source.zip','runtime_versions.json','server_environment_freeze.txt']:
  link(V2/'exports/generalization_v2'/name,ARCH/name)
 for name in ['final_frozen_config.json','final_frozen_config.sha256','selected_model.json','baseline_selection.json','protocol.json']:
  link(V2/name,ARCH/name)
 for method in ['ANN_v2','crYOLO','Topaz']:
  for suffix in ['per_micrograph.csv','retained_coordinates.csv']:
   src=V2/'reports'/(method+'_'+suffix);dst=ARCH/(method+'_'+suffix)
   with src.open() as f:
    reader=csv.DictReader(f);fields=reader.fieldnames;rows=[r for r in reader if r['dataset']=='10081']
   with dst.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 report=dict(dataset='HCN1',accession='10081',images=20,GT=2328,ANN_checkpoint_sha256=final['ANN']['checkpoint_sha256'],
  ANN_threshold=final['ANN']['threshold'],matching_radius_px=38.5,table=[r for r in final['table'] if r['dataset']=='10081'],
  source_report_sha256=sha(V2/'final_report.json'),source_archive_unchanged=True,
  scope='Known target domain; new micrographs, excluded from ANN supervised training; crYOLO pretrained accession overlap known.')
 dump(ARCH/'HCN1_result.json',report)
 dump(ARCH/'SHA256SUMS.json',{str(p.relative_to(ARCH)):sha(p) for p in ARCH.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'})
 archive=ARCH.with_suffix('.zip')
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for p in ARCH.rglob('*'):
   if p.is_file():z.write(p,str(Path(ARCH.name)/p.relative_to(ARCH)))
 dump(ROOT/'archives/HCN1_v2_archive_manifest.json',dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive)))
 # Reuse the exact 350 source images and the same tile selection, without raw-data copies.
 src=V2/'native7/training_data';dest=G/'training_data'
 if not dest.exists():shutil.copytree(src,dest,copy_function=os.link)
 for name in ['train.txt','val.txt','smoke_train.txt','smoke_val.txt','data.yaml','smoke.yaml']:
  p=dest/name;text=(src/name).read_text().replace(str(src),str(dest))
  # Unlink before writing: source files are hardlinked and must remain immutable.
  if p.exists():p.unlink()
  p.write_text(text)
 rows=json.loads((V2/'image_registry.json').read_text());source=[r for r in rows if r['split'] in ['train','val']]
 assert len(source)==368 and not {r['dataset'] for r in source}&{'10059','10081'}
 dump(G/'image_registry.json',source)
 old=list(csv.DictReader((ROOT/'v1/all_images.csv').open()))
 used={r['micrograph_name'] for r in rows+old if r['dataset']=='10059'}
 import re,urllib.parse
 ds=V2/'datasets/10059';available={urllib.parse.unquote(n) for n in re.findall(r'href="([^"]+)"',(ds/'index.html').read_text()) if n.lower().endswith('.mrc')}
 annotated={p.stem+'.mrc' for p in (ds/'annotations_original').glob('*.csv')}
 pool=sorted((available&annotated)-used);sorted_pool_sha256=hashlib.sha256('\n'.join(pool).encode()).hexdigest()
 random.Random(100283).shuffle(pool);assert len(pool)>=20
 files=[dict(dataset='10059',split='final_check',micrograph_name=n,diameter=132,url='https://ftp.ebi.ac.uk/empiar/world_availability/10059/data/micrographs/'+urllib.parse.quote(n)) for n in pool[:20]]
 protocol=dict(protocol_id='v3-source-low-SNR-TRPV1',created_unix=time.time(),source_datasets=sorted({r['dataset'] for r in source}),
  source_train_images=276,source_validation_images=92,source_registry_sha256=sha(G/'image_registry.json'),
  HCN1_frozen_archive=str(ARCH),HCN1_result_not_replaced=True,initial_checkpoint=str(V2/'selections/best.pt'),
  initial_checkpoint_sha256=final['ANN']['checkpoint_sha256'],only_new_training_component='source-only lower-SNR input augmentation before multiscale channels',
  augmentation=dict(probability=.65,signal_scale=[.35,.8],independent_gaussian_sigma=[.02,.10],units='float intensity [0,1]',channels_regenerated_after_augmentation=True),
  inference_profile_unchanged='native_multiscale LP4/LP2/raw; 1536 tiles, pad128, overlap256, merge28native,maxdet2000,floor1e-5',
  training=dict(epochs=20,lr=.00005,init='v2 selected checkpoint',balanced_tiles_per_domain=192),
  checkpoint_selection='best/last on same92 source-validation images; maximize 7-domain macro F1 with min-domain F1 >= .55 where feasible; one global threshold',
  target_GT_used_for_training_or_selection=False,target_domains=['10059','10081'],
  final_selection=dict(seed=100283,algorithm='stable filename sort; exclude ALL v1 and v2 target micrographs; random.Random(seed).shuffle; first20',candidate_count=len(pool),sorted_pool_sha256=sorted_pool_sha256),
  files=files,matching='unchanged maximum-cardinality one-to-one native D/4; TRPV1 radius33px for all methods',
  baseline_policy='reuse unchanged GPU pretrained weights, source92 calibrated thresholds crYOLO .14/Topaz -1.0; no target threshold tuning',
  scope='TRPV1-targeted development with known previous outcomes; new independent micrograph check, not a pristine domain-blind test; HCN1 v2 is a different frozen model version',
  final_check_policy='one final evaluation after model and global threshold freeze; no threshold scans or model selection using final20 outcomes')
 if (G/'protocol.json').exists():
  assert sha(G/'protocol.json')==(G/'protocol.sha256').read_text().strip()
 else:dump(G/'protocol.json',protocol);(G/'protocol.sha256').write_text(sha(G/'protocol.json')+'\n')
 # Copy v1 core, never change protected v1/v2 files.
 for name in ['general_common.py','train_general.py','infer_ann.py']:
  dst=G/'core'/name;dst.parent.mkdir(exist_ok=True)
  if not dst.exists():shutil.copy2(ROOT/'v1/code'/name,dst)
  assert sha(dst)==sha(ROOT/'v1/code'/name)
 dump(G/'initialization.json',dict(status='PASS',HCN1_archive_sha256=sha(archive),new_target_names=pool[:20],source_images=len(source),source_data_reused=True))
 print('INITIALIZED',G,'HCN_ARCHIVE',sha(archive),flush=True)

if __name__=='__main__':main()
