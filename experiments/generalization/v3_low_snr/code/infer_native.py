"""Exact v2 native inference, isolated root and source/final roles."""
from pathlib import Path
import argparse,json,sys
G=Path('/root/autodl-tmp/cryoem_generalization/v3_low_snr')
sys.path.insert(0,str(G/'core'))
import general_common as ref
ref.G=G;ref.DEVELOPMENT=['10017','10077','10291','10289','10240','10760'];ref.HELDOUT=['10059','10081']
def records(split,datasets=None):
 if split in ['final_check','regression_check']:assert (G/'final_frozen_config.json').exists()
 rows=json.loads((G/'image_registry.json').read_text());result=[r for r in rows if r['split']==split and (datasets is None or r['dataset'] in datasets)]
 assert result
 if split in ['train','val']:assert not {r['dataset'] for r in result}&set(ref.HELDOUT)
 return result
ref.records=records
def main():
 p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--split',choices=['val','final_check','regression_check'],required=True);p.add_argument('--out',required=True);a=p.parse_args()
 from infer_ann import infer
 infer(Path(a.weights),records(a.split),'native_multiscale',Path(a.out))
if __name__=='__main__':main()
