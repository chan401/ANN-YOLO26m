"""GPU Topaz source-validation cache while the ANN loader is CPU bound."""
from common import *
from select_and_evaluate import baseline_engine

def main():
 bc=baseline_engine()
 for ds in DOMAINS[-3:]:
  dump(G/'early_topaz_validation_status.json',dict(status='running',dataset=ds,target_images_used=False))
  bc.one(ds,'Topaz','val')
 dump(G/'early_topaz_validation_status.json',dict(status='complete',images=24,GPU=0,target_images_used=False))
 print('TOPAZ_NEW_SOURCE_VALIDATION_CACHED',flush=True)

if __name__=='__main__':main()
