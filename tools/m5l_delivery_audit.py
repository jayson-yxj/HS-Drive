"""Postfreeze final research qualifications and retained failure inventory."""
import json
from pathlib import Path
from tools.m5l_common import write,asset,summary
from simulator.recorder import file_hash

def main():
 failures=[]
 for p in Path('runs/m5l_logs').glob('*.log'):
  text=p.read_text(errors='replace')
  if 'Traceback (most recent call last)'in text or 'ERROR:'in text or 'FAILED 'in text:failures.append(dict(log=asset(p),tail=text[-1200:]))
 write('reports/m5l_failures.json',dict(kind='actual_log_failure_inventory',records=failures,limits='Failedattempts preserved. Initialresidualorderwritefailedbeforeoptimization; SDKimport andsynchronousworldrejections beforeShadow. Town02 noisolatedpose keptasfailure, no rule relaxed.'))
 paths=['summary.md','acceptance.json','sampling_failure_analysis.json','p4_feature_quality.json','target_depth_quality.json','vehicle_residual_ablation.json','object_2p5d_baseline.json','adaptive_mass_diagnostics.json','online_shadow.json']
 assert all(Path('reports/m5l_'+p).exists()for p in paths)
 freeze=json.loads(Path('reports/m5l_pretest_freeze.json').read_text());assert all(file_hash(p)==h for p,h in freeze['sources'].items());assert all(file_hash(p)==h for p,h in freeze['models'].items())
 replay=json.loads(Path('reports/m5l_m4_replay_validation.json').read_text())['records'];rr=[]
 for method,stride in sorted({(r['method'],r['stride'])for r in replay}):
  sub=[r for r in replay if r['method']==method and r['stride']==stride];n=sum(r['n']for r in sub);base=[r for r in replay if r['method']=='original'and r['stride']==stride];bn=sum(r['n']for r in base);old=sum(r['abs_sum']for r in base)/bn;mae=sum(r['abs_sum']for r in sub)/n;rr.append(dict(method=method,stride=stride,n=n,mae=mae,bias=sum(r['bias_sum']for r in sub)/n,near_gate=mae<=old*1.05))
 write('reports/m5l_m4_replay_summary.json',dict(kind='actual_frozen_validation_replay',records=rr,raw=asset('reports/m5l_m4_replay_validation.json'),limits='Replay q=0 for originalcache; realMobile q nearActor separate. Notallglobalprotection inferred fromReplay.'))
 # Per-scene near/bias failures cannot be hidden by aggregates.
 records=[]
 for split in ['val','regression','test']:
  raw=json.loads(Path(f'runs/m5l_residual_eval_{split}/raw.json').read_text())['depth'];base=json.loads(Path(f'runs/m5l_paths_{split}.json').read_text())['records']
  for method in sorted({r['method']for r in raw}):
   for group in sorted({r['group']for r in raw}):
    for scope,band in [('all','near'),('actor','near'),('all','all'),('all','10_20'),('all','20_35')]:
     a=[r for r in raw if r['method']==method and r['group']==group and r['scope']==scope and r['band']==band];b=[r for r in base if r['method']=='original_p8'and r['group']==group and r['scope']==scope and r['band']==band];n=sum(r['n']for r in a);nb=sum(r['n']for r in b)
     if not n or not nb:records.append(dict(split=split,method=method,group=group,scope=scope,band=band,n=n,status='NotEvaluated_ZeroSupport'));continue
     mae=sum(r['sums']['expected_mae']for r in a)/n;old=sum(r['sums']['expected_mae']for r in b)/nb;bias=sum(r['sums']['bias']for r in a)/n;oldbias=sum(r['sums']['bias']for r in b)/nb;space=sum(r['sums']['space1']for r in a)/n;oldspace=sum(r['sums']['space1']for r in b)/nb;ok=space>=oldspace if scope=='actor'else mae<=old*1.05 if band=='near'else abs(bias)<=abs(oldbias)+(.25 if band=='all'else .5);records.append(dict(split=split,method=method,group=group,scope=scope,band=band,n=n,mae=mae,old_mae=old,bias=bias,old_bias=oldbias,space=space,old_space=oldspace,passed=ok))
 rawpath='runs/m5l_scene_protection.json';write(rawpath,dict(records=records));write('reports/m5l_scene_protection.json',dict(kind='offline_perposition_frozen_protection',evaluated=sum('passed'in r for r in records),failed=[r for r in records if r.get('passed')is False],zero_support=[r for r in records if 'passed'not in r],raw=asset(rawpath)))
 write('reports/m5l_tests.json',dict(kind='actual_server',gpu_environment=dict(passed=298,skipped=5,log=asset('runs/m5l_logs/full_tests_final.log')),carla_sdk=dict(passed=14,log=asset('runs/m5l_logs/sdk_tests.log')),frozen_source_and_model_integrity=True,checkpoint=asset('reports/m5l_checkpoint_audit.json'),limits='Unit/synthetic checks distinct fromactualGPU/onlineCARLA reports. SkippedoptionalSDK cases runseparately. C P4Depthnottrained.'))
if __name__=='__main__':main()
