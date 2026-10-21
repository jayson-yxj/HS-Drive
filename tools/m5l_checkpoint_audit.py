"""Verify matched initialization/order and selected-checkpoint SHA; no new training."""
import json
from pathlib import Path
import torch
from tools.m5l_common import write,asset
from simulator.recorder import file_hash

def main():
 results=json.loads(Path('reports/m5l_vehicle_training.json').read_text())['records'];records=[]
 for seed in [101,102]:
  b=next(r for r in results if r['mode']=='no_mobile'and r['seed']==seed);c=next(r for r in results if r['mode']=='mobile'and r['seed']==seed);assert b['order']['sha256']==c['order']['sha256'];assert b['parameters']==c['parameters']
  initb=torch.load(next(r['path']for r in b['history']if r['step']==0),weights_only=True)['adapter'];initc=torch.load(next(r['path']for r in c['history']if r['step']==0),weights_only=True)['adapter'];assert all(torch.equal(v,initc[k])for k,v in initb.items());assert not initb['delta.weight'].any()and not initb['delta.bias'].any()
  for r in [b,c]:assert file_hash(r['selected']['path'])==r['selected']['sha256']
  records.append(dict(seed=seed,initial_weights_exact=True,order_sha256=b['order']['sha256'],parameters=b['parameters'],steps_each=600,selected=[b['selected'],c['selected']],peak_mib=[b['peak_mib'],c['peak_mib']],elapsed_s=[b['elapsed_s'],c['elapsed_s']]))
 write('reports/m5l_checkpoint_audit.json',dict(kind='actual_checkpoint_tensor_and_SHA_verification',records=records,limits='Elapsedoptimization includesVal passes/cacheexcluded; trainingshortcachedfeatures, not endtoendbackbone cost. P4probe timing ranwithotherGPUdiagnostics, sharedload cost notisolatedlatencycertification. OriginalDepth/Mobile remainsfrozen.'))
if __name__=='__main__':main()
