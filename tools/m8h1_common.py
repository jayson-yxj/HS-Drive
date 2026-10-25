"""Append-only M8-H1 evidence; no simulation mutation interfaces."""
import json,subprocess,time
from pathlib import Path
from tools.m8a_common import asset,sha,write
ROOT=Path('/root/gpufree-data/hs-drive-experiments/m8h1-native-template-smoke')
HISTORY=Path('/root/gpufree-data/hs-drive-experiments/m8g-placement-feasibility/unified_history_v1.json')
BASE='5e4701c0cdb81792d696c37157cf23d544843fb0'
LOG=Path('reports/m8h1_logs')
def provenance(kind):
 paths=[*Path('configs/m8h1').glob('*.json'),*Path('tools').glob('m8h1_*.py'),*Path('datasets').glob('m8h1_*.py'),*Path('tests').glob('test_m8h1*.py')]
 return dict(kind=kind,base=BASE,execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources={str(p):sha(p) for p in sorted(paths)},new_model_forwards=0,training_steps=0,visual_control=False)
def run(name,cmd,timeout=1800,env=None):
 t=time.monotonic()
 with (LOG/(name+'.txt')).open('x') as f:
  try:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=timeout,env=env);code=r.returncode
  except subprocess.TimeoutExpired:code=124
 write(LOG/(name+'.json'),dict(**provenance('ActualServerProcess'),command=cmd,returncode=code,elapsed_s=time.monotonic()-t,log=asset(LOG/(name+'.txt'))))
 if code:raise RuntimeError('Failed process preserved: '+name)
 print(name,'completed',flush=True)
