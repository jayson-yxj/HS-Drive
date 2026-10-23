"""Separate fixture, dynamic, coverage and latency decisions; no gate promotion."""
from pathlib import Path
from collections import Counter
import subprocess,json
import numpy as np
from tools.m7b_common import *


def quant(values):
    return {k:float(np.quantile(values,q)) for k,q in [('p50',.5),('p95',.95),('p99',.99),('max',1)]} if values else None


def summarize(d):
    records=d['records'];valid=[r for r in records if r['fixture']['status']=='FixtureValid'];dynamic=[r for r in valid if r['spec']['kind'] in ['lead','crossing']]
    result=dict(attempted=len(records),fixture_valid=len(valid),invalid=[dict(scene_id=r['scene_id'],reasons=r['fixture']['errors']) for r in records if r not in valid],valid_dynamic_scenes=len(dynamic))
    for method in ['A_CV','B_hypotheses']:
        acc=[];selected=[];unknown=[];witness_scenes=0;witness_missed=0;dynamic_success=0;kinfail=0;fault_reject=0;allreject=0
        accepted_key=lambda e:e['nominal']['planning_feasible'] if method=='A_CV' else e['robust_accepted']
        for r in valid:
            es=r['candidate_evaluations'];base=[e for e in es if not e['name'].startswith('E_stop')]
            for e in base:
                if accepted_key(e):acc.append((r,e))
                if e['name']=='D_stop' and any(reason in ['initial_state_mismatch','position_velocity_inconsistent','acceleration_limit','internal_jerk_limit','curvature_limit','heading_path_inconsistent','stop_not_reached_in_horizon'] for reason in e['geometry']['reasons']):kinfail+=1
            chosen=next((e for e in base if e['name']==r['decisions'][method]),None)
            if chosen:selected.append((r,chosen))
            else:allreject+=1
            if r.get('prediction_unknown'):unknown.append(r['scene_id'])
            if r['spec'].get('fault') and chosen is None:fault_reject+=1
            # A robust finite-library witness is a sufficient existence certificate,
            # never a claim that every physically feasible solution was enumerated.
            exists=any(e['robust_accepted'] for e in es)
            if exists:
                witness_scenes+=1
                if chosen is None:witness_missed+=1
            if r in dynamic and chosen is not None and not chosen['reference']['collisions']:dynamic_success+=1
        false=[dict(scene_id=r['scene_id'],candidate=e['name'],reference=e['reference']) for r,e in acc if e['reference']['collisions']]
        selected_false=[dict(scene_id=r['scene_id'],candidate=e['name']) for r,e in selected if e['reference']['collisions']]
        result[method]=dict(generated_candidates=sum(len([e for e in r['candidate_evaluations'] if not e['name'].startswith('E_stop')]) for r in valid),geometry_legal_candidates=sum(e['geometry']['planning_feasible'] for r in valid for e in r['candidate_evaluations'] if not e['name'].startswith('E_stop')),accepted_candidates=len(acc),accepted_reference_conflicts=false,selected_scenes=len(selected),selected_reference_conflicts=selected_false,all_reject_scenes=allreject,known_finite_library_witness_scenes=witness_scenes,known_witness_all_rejected=witness_missed,dynamic_selected_collision_free=dynamic_success,dynamic_selection_fraction=dynamic_success/len(dynamic) if dynamic else None,D_stop_kinematic_failures=kinfail,invalid_input_correct_rejects=fault_reject,prediction_out_of_support_scenes=unknown,correct_no_solution='Not globally certified; finite library all-reject and fault counts reported separately')
    return result


def main():
    files={'development':'reports/m7b_development_v2.json','validation':'reports/m7b_validation_frozen.json','fresh_final':'reports/m7b_fresh_final_frozen.json','historical_regression':'reports/m7b_historical_regression_frozen.json'}
    data={k:read(p) for k,p in files.items()};summaries={k:summarize(d) for k,d in data.items()};prov=provenance();allrows=[r for d in data.values() for r in d['records']]
    fixtures=[dict(scene_id=r['scene_id'],**r['fixture']) for r in allrows]
    write('reports/m7b_fixture_consistency.json',dict(kind='OfflineCARLAMap_SyntheticActorGeometry',provenance=prov,records=fixtures,per_split={k:{n:s[n] for n in ['attempted','fixture_valid','invalid']} for k,s in summaries.items()},old_results_untouched=True,reference='OpenDRIVE lane polygon+CARLA waypoint, planar OBB; not render Mesh truth'))
    write('reports/m7b_future_motion_validity.json',dict(kind='OfflineAnalyticReferenceAndFiniteDifferenceAudit',provenance=prov,records=[dict(scene_id=r['scene_id'],checks=r['fixture']['future_checks']) for r in allrows],definition='t0 position/speed, first difference<=0.12m/s, acceleration[-3,1.5], heading, full road OBB samples; crossing explicit corridor exception',no_future_input=True,limits='Analytic bounded acceleration references; sampled body-road checks. Not measured actors, behavior probabilities or exact physics.'))
    write('reports/m7b_dynamic_hypotheses.json',dict(kind='OfflineFiniteMotionHypotheses',provenance=prov,protocol=asset('configs/m7b/protocol.json'),hypothesis_files={k:asset(p) for k,p in files.items()},definition='Road arc constant speed, braking[-3,-1.5,-0.5], acceleration[0.75,1.5], onset[0,1,2], speed[0,8],6s. Static geometry held; stopped vehicles may accelerate.',unknown='Missing route, mismatch, initial speed beyond support or any route-end hypothesis means PredictionOutOfSupport and no robust acceptance',safety_scope='Finite set only; arbitrary intermediate accelerations, intent changes, off-route movement, mesh contacts not certified',historical_seven=asset('reports/m7b_historical_seven_v2.json'),C_gate=asset('reports/m7b_C_gate.json')))
    write('reports/m7b_robust_candidate_evaluation.json',dict(kind='OfflinePairedA_B_NoCActivation',provenance=prov,per_split=summaries,records={k:asset(p) for k,p in files.items()},VehicleExecutedSuccessfully='NotRun',PlanningFeasible='Conditional current scene and finite assumptions; no actuation authorization'))
    write('reports/m7b_dynamic_false_accepts.json',dict(kind='IndependentDensePolygonFutureAudit',provenance=prov,per_split={k:{m:s[m]['accepted_reference_conflicts'] for m in ['A_CV','B_hypotheses']} for k,s in summaries.items()},selected={k:{m:s[m]['selected_reference_conflicts'] for m in ['A_CV','B_hypotheses']} for k,s in summaries.items()},historical=read('reports/m7b_historical_seven_v2.json'),method='0.025s Shapely polygon overlap vs separate evaluator future; not the SAT method; same synthetic geometry therefore not independent physical truth',invalid_fixture_excluded=True))
    write('reports/m7b_candidate_coverage.json',dict(kind='OfflineCandidateLibraryCoverage',provenance=prov,per_split=summaries,C='NotRun: development did not reveal legacy all-reject with a legal stop witness',witness='Cosine speed jerk-bounded stopsT2.5..5.5 used only as independent finite existence witnesses; not selected or actuation',no_solution='General completeness not established. NoFeasibleCandidate is not proof no physical solution exists.',success='Selected AND reference collision-free; stops counted separately in per-scene decisions, not route completion'))
    old={3,18,67,19,65,20};fresh=data['fresh_final']['roads'];roadinfo=[]
    for f in fresh:
        p=np.array(f.get('route',[]));dist={}
        for split in ['development','validation']:
            for other in data[split]['roads']:
                q=np.array(other.get('route',[]))
                if len(p) and len(q):dist[str(other['road'])]=float(np.linalg.norm(p[:,None,:2]-q[None,:,:2],axis=-1).min())
        roadinfo.append(dict(road=f['road'],not_M7A_road=f['road'] not in old,minimum_sampled_route_distance_m=dist,geometry=f))
    write('reports/m7b_fresh_final_roads.json',dict(kind='FrozenBeforeHeldOutEvaluation',provenance=prov,freeze_commit='a18f754',map_only_inventory=asset('reports/m7b_execution_logs/map_only_road_inventory.json'),roads=roadinfo,untuned_after_open=True,scope='New M7A/M7B Road IDs, same Town03 and shared scenario templates; not new map or globally never-used CARLA areas',historical_roads_only_regression=[65,67,18],final_validity=summaries['fresh_final']['invalid']))
    latency={}
    for split,d in data.items():
        totals={x:[] for x in ['A_CV','B_hypotheses','independent_reference','fixture','generation','hypothesis_generation','selection','body_road_static','dynamic_SAT']}
        for r in d['records']:
            q=r['timing_ms'];totals['fixture'].append(q['fixture'])
            if 'candidates' not in q:continue
            for k in ['generation','hypothesis_generation','selection']:totals[k].append(q[k])
            base=[c for c in q['candidates'] if not c['witness']]
            geom=sum(c['body_road_static_ms'] for c in base);dyn=sum(c['multi_hypothesis_collision_ms'] for c in base)
            totals['body_road_static'].append(geom);totals['dynamic_SAT'].append(dyn)
            totals['A_CV'].append(q['generation']+q['selection']+sum(c['legacy_CV_ms'] for c in base))
            totals['B_hypotheses'].append(q['generation']+q['hypothesis_generation']+q['selection']+geom+dyn)
            totals['independent_reference'].append(sum(c['independent_reference_ms'] for c in q['candidates']))
        latency[split]=dict(map_read_ms=d['map_read_ms'],map_setup_ms=[r.get('map_setup_ms') for r in d['roads']],stage_ms={k:quant(v) for k,v in totals.items()},B_exceeds200ms=sum(v>200 for v in totals['B_hypotheses']),B_requests=len(totals['B_hypotheses']))
    write('reports/m7b_planning_latency.json',dict(kind='ActualServerOfflineMeasuredTiming',provenance=prov,per_split=latency,profile=asset('reports/m7b_execution_logs/function_profile.txt'),interpretation='Offline component sum excluding witnesses and independent GT audit, not actual end-to-end online scheduling. Includes full original six-second checks.',optimization='Batch SAT equivalent500random cases + unchanged development decisions; no checks removed',budget_ms=200,online5Hz='NotApproved_tail_exceeds_budget',strict20Hz='RuntimePartial',new_live_trials=0))
    base=read('reports/m7b_execution_logs/base_freeze.json');changed=[p for p,h in base['tracked_files'].items() if sha(p)!=h]
    assert not changed,changed
    final=summaries['fresh_final'];seven=read('reports/m7b_historical_seven_v2.json');A=final['fixture_valid']==final['attempted'] and bool(final['attempted']);B=seven['detected']==7 and not final['B_hypotheses']['accepted_reference_conflicts'];C=final['B_hypotheses']['dynamic_selection_fraction']>=.5 and final['B_hypotheses']['known_witness_all_rejected']==0
    write('reports/m7b_acceptance.json',dict(kind='IndependentResearchDecisions',provenance=prov,status='Partial',decisions=dict(A_fixture='Passed_LimitedFreshRoads' if A else 'Partial_FixtureSupport',B_dynamic='Passed_WithinFiniteReferences' if B else 'NoGo',C_candidates='Passed_LimitedCandidateCoverage' if C else 'Partial_Coverage',D_latency='NoGo_Online200ms_NotDemonstrated',E_OraclePilot='ProposalOnly_OfflineOrNonRealtime_NotAuthorizedControl' if A and B and C else 'NoGo_A_B_C_not_all_passed'),gate_values=dict(A=A,B=B,C=C),historical_seven_detected=seven['detected'],fresh_final=final,unchanged=dict(low_speed='ControlNoGo',strict20Hz='RuntimePartial',visual='CandidateNoGo',visual_driving='NotValidated',Bench2Drive='PreparationOnly'),preservation=dict(base_files=len(base['tracked_files']),changed=changed),new_vehicle_runs=0,new_GPU_inference=0,training_steps=0,C_generator='NotRun_conditional_gate_not_met',tests='reports/m7b_execution_logs/full_tests.xml',not_run=['live5Hz','strict20Hz','vehicle_actuation','new_visual_inference','Bench2Drive','PETR']))
    print(json.dumps({k:{n:s[n] for n in ['attempted','fixture_valid','valid_dynamic_scenes']} for k,s in summaries.items()},indent=2))
    print('decisions',A,B,C)
if __name__=='__main__':main()
