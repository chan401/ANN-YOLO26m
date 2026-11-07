"""Reuse the exact v1 native multi-frequency pipeline with seven source domains.

Only run root, source-domain registry, and initialization are adapted in memory.
The protected v1 files are copied and verified, never edited.
"""
import argparse,importlib,shutil,subprocess
import common as v2

def setup():
 n=v2.G/'native7';code=n/'code';code.mkdir(parents=True,exist_ok=True)
 core={}
 for name in ['general_common.py','prepare_training.py','train_general.py','infer_ann.py']:
  src=v2.V1/'code'/name;dst=code/name
  if not dst.exists():shutil.copy2(src,dst)
  assert v2.sha(src)==v2.sha(dst);core[name]=v2.sha(dst)
 adapter=code/'v2_native_wrapper.py'
 if not adapter.exists():shutil.copy2(v2.G/'code/native_wrapper.py',adapter)
 assert v2.sha(adapter)==v2.sha(v2.G/'code/native_wrapper.py'),'Adapter code changed after snapshot'
 p=n/'protocol.json'
 if not p.exists():shutil.copy2(v2.G/'protocol.json',p)
 assert v2.sha(p)==v2.sha(v2.G/'protocol.json')
 initial=v2.V1/'selections/final_general/best.pt'
 assert v2.sha(initial)=='0a3449ace204e2ad5a7cbb64264a540258f0b6f63b837b46bf6ff7b94ea42f6b'
 v2.sys.path.insert(0,str(code));ref=importlib.import_module('general_common')
 ref.G=n;ref.DEVELOPMENT=v2.DOMAINS[1:];ref.HELDOUT=v2.TARGETS
 ref.records=lambda split,datasets=None:[r for r in v2.records(split) if datasets is None or r['dataset'] in datasets]
 def protocol():
  d=v2.json.loads(p.read_text());d.update(base_checkpoint=str(initial),base_checkpoint_sha256=v2.sha(initial));return d
 ref.protocol=protocol
 v2.dump(n/'adapter_config.json',dict(root=str(n),source_domains=v2.DOMAINS,targets_excluded=v2.TARGETS,initial_checkpoint=str(initial),initial_sha256=v2.sha(initial),core_sha256=core,preprocessing_changed_from_v1=False,mode='native_multiscale',training_epochs=30,
  in_memory_adaptations=['G run root','DEVELOPMENT accession list','HELDOUT accession list','records from verified v2 source registry','protocol initialization fields']))
 return n

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','smoke','train','infer']);p.add_argument('--weights');p.add_argument('--split',choices=['val','final_check']);p.add_argument('--out');a=p.parse_args()
 n=setup()
 if a.stage=='prepare':
  v2.sys.argv=['prepare_training.py','--mode','native'];importlib.import_module('prepare_training').main()
 elif a.stage in ['smoke','train']:
  v2.sys.argv=['train_general.py','--mode','native_multiscale','--name','native7','--epochs','30','--lr','0.0001']+(['--smoke'] if a.stage=='smoke' else [])
  importlib.import_module('train_general').main()
 else:
  assert a.weights and a.split and a.out
  importlib.import_module('infer_ann').infer(v2.Path(a.weights),v2.records(a.split),'native_multiscale',v2.Path(a.out))

if __name__=='__main__':main()
