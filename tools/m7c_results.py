"""Independent post-run summaries, attribution and qualification limits."""
from collections import Counter
from pathlib import Path
import json
import numpy as np
from tools.m7c_common import *


def stats(v):return {k:float(np.quantile(v,q)) for k,q in [('p50',.5),('p95',.95),('p99',.99),('max',1)]} | {'count':len(v),'over200ms':sum(x>200 for x in v)} if v else None


def main():
 prov=provenance();eq=read('reports/m7c_geometry_cache_equivalence_prepared.json');files={s:f'reports/m7c_{s}_{suffix}.json' for s,suffix in [('development','v2'),('validation','frozen'),('fresh_final','frozen')]};data={s:read(p) for s,p in files.items()}
 latency=dict(equivalence_pair={k:stats([r['timing_ms'][k] for r in eq['records']]) for k in ['baseline_core','optimized_cold','optimized_warm','stream']},new_scenarios={s:{k:stats([r['timing_ms'][k] for r in d['records'] if 'timing_ms' in r]) for k in ['baseline_core','with_slow_core','slow_extra','hypotheses_and_road_support','research_total']} for s,d in data.items()})
 write('reports/m7c_planning_latency.json',dict(kind='ActualServerOfflinePairedMeasuredComponents',provenance=prov,results=latency,profile=asset('reports/m7c_execution_logs/baseline_profile.txt'),preparation=dict(map_read_ms=eq['map_setup_ms'],fixtures=eq['fixture_setup']),qualification='NoGo_OverallOfflineBudget_NotOnlineCertification',cold_definition='Per-candidate empty cache, union/buffer/query/body included',warm_definition='Exact same candidate immediately repeated; favorable replay bound, not a moving ego',stream_definition='Same frozen scenario sequence, shared bounded geometry+query cache; dynamic results and age never cached',research_total='New-scenario wall elapsed includes fixture, hypotheses, all candidate checks, independent reference audit; excludes map setup and JSON disk serialization',core='Generation+hypotheses+support+cached candidate geometry+dynamic+selection, excludes independent references/serialization; not full IPC or CARLA tick cycle',limits=['No shared GPU stress','No live control','Some diagnostic profile and early development ran concurrently; final all-history paired sweep run sequentially internally','No repeat seeds or timing confidence claimed']))
 summary={};false={}
 for split,d in data.items():
  valid=[r for r in d['records'] if r['fixture']['status']=='FixtureValid'];dynamic=[r for r in valid if r['spec']['kind'] in ['lead','crossing']];gains=[];losses=[];acc=[];selected_conf=[];base_conf=[];progress=[];margin=[]
  for r in valid:
   for e in r['evaluations']:
    if e['accepted'] and e['reference']['collisions']:acc.append(dict(scene=r['scene_id'],candidate=e['name'],reference=e['reference']))
   picked=next((e for e in r['evaluations'] if e['name']==r['selected']),None);old=next((e for e in r['evaluations'] if e['name']==r['baseline_selected']),None)
   if picked and picked['reference']['collisions']:selected_conf.append(r['scene_id'])
   if old and old['reference']['collisions']:base_conf.append(r['scene_id'])
   if r['selected']=='E_slow_follow' and r['baseline_selected'] in [None,'D_stop']:gains.append(r['scene_id'])
   if r['selected'] is None and r['baseline_selected'] is not None:losses.append(r['scene_id'])
   if picked:
    met=dict(picked['geometry']['metrics']);progress.append(dict(scene=r['scene_id'],selected=r['selected'],progress_m=met.get('route_progress_m'),reference_min_gap=picked['reference']['min_OBB_gap_m'],stop_reserve=picked['stop_reserve']))
  summary[split]=dict(attempted=len(d['records']),valid=len(valid),invalid=[dict(scene=r['scene_id'],errors=r['fixture']['errors']) for r in d['records'] if r not in valid],road_errors=[r for r in d['roads'] if r['status']!='MapReady'],dynamic_valid=len(dynamic),dynamic_selected=sum(r['selected'] is not None for r in dynamic),baseline_choices=dict(Counter(str(r['baseline_selected']) for r in valid)),new_choices=dict(Counter(str(r['selected']) for r in valid)),continuing_gain=gains,selection_losses=losses,accepted_candidates=sum(e['accepted'] for r in valid for e in r['evaluations']),unknown_scenes=[r['scene_id'] for r in valid if r['prediction_unknown']],progress=progress)
  false[split]=dict(accepted_conflicts=acc,selected_conflicts=selected_conf,baseline_selected_conflicts=base_conf)
 write('reports/m7c_candidate_coverage.json',dict(kind='OfflineFrozenBaselineVersusSingleSlowFamily',provenance=prov,per_split=summary,raw={s:asset(p) for s,p in files.items()},C_stop='NotRun_no_new_admission',metrics='Continue means selected non-stop trajectory over6s, not route completion; rejection is finite-library failure not global impossibility',VehicleExecutedSuccessfully='NotRun'))
 write('reports/m7c_dynamic_false_accepts.json',dict(kind='IndependentSampledEvaluatorOnlyFuture',provenance=prov,per_split=false,historical_seven=asset('reports/m7b_historical_seven_v2.json'),equivalence_preserves_historical_risk=eq['all_equal'],limits='Same synthetic OpenDRIVE/OBB; separate Shapely0.025s method, not independent physical Mesh truth'))
 # Attribute every historical stop/reject, without labeling a finite-library failure as necessary in all plans.
 attrs=[];oldlimits=read('reports/m7b_coverage_limits.json')['per_split']['fresh_final']['realized_future_feasible_but_robust_rejected'];referenceonly={r['scene_id'] for r in oldlimits}
 slowdev={r['scene_id']:r for r in read('reports/m7c_slow_follow_feasibility.json')['records']}
 for name in ['development_v2','validation_frozen','fresh_final_frozen','historical_regression_frozen']:
  for r in read('reports/m7b_'+name+'.json')['records']:
   if r['fixture']['status']!='FixtureValid':attrs.append(dict(scene_id=r['scene_id'],categories=['FixtureInvalid_route_support'],necessary_stop_proven=False));continue
   if r['decisions']['B_hypotheses'] not in [None,'D_stop']:continue
   categories=[];ev=[e for e in r['candidate_evaluations'] if not e['name'].startswith('E_stop')];moving=[e for e in ev if e['name']!='D_stop']
   if r['spec'].get('fault'):categories.append('InputFaultOrTimeInvalid')
   if r['prediction_unknown']:categories.append('PredictionSupportInsufficient')
   if any(any('road' in reason or 'lane_change' in reason for reason in e['geometry']['reasons']) for e in moving):categories.append('MovingCandidateRoadOrLaneRestricted')
   if any(e['dynamic']['collisions'] for e in moving):categories.append('MovingCandidateConflictUnderReasonableHypotheses')
   if any(e['dynamic']['stop_margin_failures'] for e in ev if e['name']=='D_stop'):categories.append('Stop3mReserveNotCertified')
   if r['scene_id'] in referenceonly:categories.append('SafeForOneReference_NotRobust')
   if r['scene_id'] in slowdev and slowdev[r['scene_id']]['continue_gain']:categories.append('DevelopmentProvenCandidateLibraryGap')
   elif r['decisions']['B_hypotheses']=='D_stop':categories.append('StopOnlyInCurrentLibrary_NotProofStopNecessary')
   attrs.append(dict(scene_id=r['scene_id'],selected=r['decisions']['B_hypotheses'],categories=categories,necessary_stop_proven=False,FeasibleToExecute='NotValidated'))
 write('reports/m7c_stop_coverage_attribution.json',dict(kind='HistoricalPerSceneCausalLimits',provenance=prov,records=attrs,count_by_category=dict(Counter(c for r in attrs for c in r['categories'])),definitions=dict(SafeForOneReference='One evaluator script only',RobustWithinHypotheses='No collision across enumerated finite assumptions',FeasibleToExecute='Requires actual actuation; NotValidated'),cannot_infer='Conflicting alternative futures are not simultaneous physical actors; rejected alternatives do not prove no possible forward plan'))
 fresh=data['fresh_final'];prior={2,3,4,18,19,20,65,67,76}
 write('reports/m7c_fresh_final_roads.json',dict(kind='FrozenNewRoadIDsSameTown03',provenance=prov,freeze=read('reports/m7c_execution_logs/rule_freeze.json'),roads=fresh['roads'],new_ids=[r['road'] for r in fresh['roads'] if r['road'] not in prior],selection=asset('reports/m7c_execution_logs/map_only_inventory.json'),opened_once=True,limitations='Only2newroads, shared23templates, sameTown03; no cross-map or prior-project universal novelty claim',invalid=summary['fresh_final']['invalid']))
 allfalse=any(v['accepted_conflicts'] or v['selected_conflicts'] for v in false.values());freshok=summary['fresh_final']['valid']==summary['fresh_final']['attempted'] and summary['fresh_final']['attempted']>0
 write('reports/m7c_acceptance.json',dict(kind='IndependentEngineeringResearchStates',provenance=prov,status='Partial',decisions=dict(A_cache='Passed_ExactHistoryEquivalence_PartialLatency',B_continuation='Partial_UniqueForwardOnly_AmbiguityRemains',C_attribution='Completed_FiniteLibraryNotNecessityProof',D_slow='NoGo' if allfalse else 'LimitedResearchGain' if summary['fresh_final']['continuing_gain'] else 'NoGo_NoStableFreshGain',E_Pilot='NotApproved_FinalFixtureOrCoverageRisks' if not freshok or allfalse or not summary['fresh_final']['continuing_gain'] else 'MayProposeNonRealtimeOracleOnly'),fresh_fixture_all_pass=freshok,geometry_equivalent=eq['all_equal'],latency_gate='NoGo_ColdAndStreamOrExpandedCore',raw_final=summary['fresh_final'],reference_conflicts=false,unchanged=dict(vehicle_execution='NotRun',normal_stopping='ControlNoGo',strict20Hz='RuntimePartial',RGB='CandidateNoGo',visual_driving='NotValidated',Bench2Drive='PreparationOnly',PETR='NotRun',Occupancy='NotPassed'),new_training_steps=0,new_vehicle_runs=0,new_GPU_inference=0))
 print(json.dumps({k:{n:v[n] for n in ['attempted','valid','dynamic_valid','dynamic_selected','baseline_choices','new_choices','continuing_gain']} for k,v in summary.items()},indent=2));print('false',false)
if __name__=='__main__':main()
