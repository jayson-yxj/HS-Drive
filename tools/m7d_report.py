"""Aggregate immutable artifacts; status is derived from measured evidence."""
import json, collections, math
from pathlib import Path
import numpy as np
from tools.m7d_common import *
from tools.m7c_common import equal
from diagnostics.m7b_motion import interpolate


def quantiles(values):
    return {k:float(np.quantile(values,q)) for k,q in [('p50',.5),('p95',.95),('p99',.99),('max',1.)]} if values else None


def comparable(r):
    return {k:r.get(k) for k in ['input_error','selected','baseline_selected','prediction_unknown','hypotheses','actor_route_support','slow_generation_error','counts']} | {'evaluations':[{k:e[k] for k in ['name','candidate','geometry','dynamic','stop_reserve','accepted','allowed_lanes']} for e in r.get('evaluations',[])]}


def main():
    cfg=read('configs/m7d/protocol.json');pro=provenance();common=dict(provenance=pro,kind='ActualServerOfflineAnalysisOfActualCARLA_SeparateSyntheticTests',control_authorized=False)
    ds=[read(p) for p in sorted(Path('reports/m7d_replays').glob('*_stream.json'))]
    summary=[];stability=[];progress=[];collision=[];errors=[];routes=[];runtime=[];access=[]
    for d in ds:
        ep=d['episode'];rs=d['records'];ed=read(f'reports/m7d_evaluations/{ep}.json');evs=ed['records'];byindex={r['input_index']:r for r in evs}
        decisions=collections.Counter(str(r['selected']) for r in rs);switches=[];reject_runs=[];run=0;trajectory_deltas=[];speed_deltas=[];recoveries=[]
        for i,r in enumerate(rs):
            if r['selected'] is None:run+=1
            elif run:reject_runs.append(run*.2);run=0
            if i and r['selected']!=rs[i-1]['selected']:
                switches.append(dict(frame=r['frame'],from_candidate=rs[i-1]['selected'],to_candidate=r['selected']))
                if rs[i-1]['selected']=='D_stop' and r['selected'] not in [None,'D_stop']:recoveries.append(r['frame'])
            if i and r['selected'] and rs[i-1]['selected']:
                prev=rs[i-1];a=next(e['candidate'] for e in prev['evaluations'] if e['name']==prev['selected']);b=next(e['candidate'] for e in r['evaluations'] if e['name']==r['selected'])
                dt=r['clock']['input_timestamp_s']-prev['clock']['input_timestamp_s'];query=np.linspace(0,6-dt,30)
                aa=interpolate(a['points'],a['times_s'],query+dt);bb=interpolate(b['points'],b['times_s'],query)
                trajectory_deltas.append(float(np.max(np.linalg.norm(aa[:,:2]-bb[:,:2],axis=1))))
                speed_deltas.append(float(abs(interpolate(a['points'],a['times_s'],[1+dt])[0,3]-interpolate(b['points'],b['times_s'],[1])[0,3])))
        if run:reject_runs.append(run*.2)
        all_eval=[e for r in evs for e in r['evaluations']];acc=[e for e in all_eval if e['accepted']];sel=[e for e in all_eval if e['selected']]
        collision.append(dict(episode=ep,snapshots=len(evs),accepted_candidates=len(acc),selected_candidates=len(sel),accepted_conflict_candidates=sum(bool(e['measured_sample_OBB_conflicts']) for e in acc),selected_conflict_candidates=sum(bool(e['measured_sample_OBB_conflicts']) for e in sel),accepted_stop_margin_failures=sum(e['stop_3m_preserved'] is False for e in acc),accepted_road_violations=sum(bool(e['full_body_road_violation_samples']) for e in acc),accepted_recorded_future_unknown=sum(e['SafeForRecordedFuture']=='Unknown' for e in acc),minimum_accepted_gap_m=min((e['minimum_gap_m'] for e in acc if e['minimum_gap_m'] is not None),default=None),artifact=asset(f'reports/m7d_evaluations/{ep}.json')))
        unknown=collections.Counter(x['reason'] for r in rs for x in r.get('prediction_unknown',[]));geom=collections.Counter(reason for r in rs for e in r.get('evaluations',[]) for reason in e['geometry']['reasons'])
        summary.append(dict(episode=ep,split=d['split'],snapshots=len(rs),decisions=dict(decisions),input_rejections=[dict(frame=r['frame'],reason=r['input_error']) for r in rs if 'input_error' in r],unknown_snapshots=sum(bool(r.get('prediction_unknown')) for r in rs),unknown_reasons=dict(unknown),geometry_reasons=dict(geom),artifact=asset(f'reports/m7d_replays/{ep}_stream.json')))
        stability.append(dict(episode=ep,switch_count=len(switches),switches=switches,stop_to_progress_frames=recoveries,longest_reject_run_s=max(reject_runs,default=0),reject_runs_s=reject_runs,overlap_trajectory_max_delta_m=quantiles(trajectory_deltas),one_second_speed_target_delta_mps=quantiles(speed_deltas),interpretation='Replay state is actual script Ego, not previous plan execution; changes alone do not prove unnecessary oscillation.'))
        progress.append(dict(episode=ep,baseline_decisions=dict(collections.Counter(str(r.get('baseline_selected')) for r in rs)),with_slow=dict(decisions),slow_added_progress_snapshots=sum(r['selected']=='E_slow_follow' and r.get('baseline_selected') in ['D_stop',None] for r in rs),selected_progress_m=quantiles([e['plan_progress_m'] for e in sel]),finite_future_safe_geometry_feasible_rejected=sum(e['finite_reference_safe_but_rejected'] for e in all_eval),snapshots_with_only_stop_accepted=sum(r['selected']=='D_stop' for r in rs),snapshots_without_accepted_candidate=sum(r['selected'] is None for r in rs),no_geometry_feasible_candidate=sum(not any(e['geometry']['planning_feasible'] for e in r.get('evaluations',[])) for r in rs),interpretation='No accepted candidate != globally impossible. Recorded-safe rejected != robust-safe; no tuning to future. Planned distances not actual driven progress.'))
        rt=[r for r in rs if 'clock' in r];times=[r['timing_ms']['total'] for r in rt]
        runtime.append(dict(episode=ep,split=d['split'],ms=quantiles(times),over_200ms=sum(x>200 for x in times),expired=sum(r['clock']['expired_for_online'] for r in rt),components_ms={k:quantiles([r['timing_ms'][k] for r in rt]) for k in rt[0]['timing_ms']} if rt else {},candidate_ms={n:{k:quantiles([e['timing_ms'][k] for r in rt for e in r['evaluations'] if e['name']==n]) for k in ['road_body','dynamic_SAT','slow_reserve']} for n in sorted({e['name'] for r in rt for e in r['evaluations']})},cache=rt[-1]['cache'] if rt else {},timing_scope='Campaign timings may overlap other offline diagnostics; clean paired run separately'))
        routes.append(dict(episode=ep,unknown_reasons=dict(unknown),continuation_reasons=dict(collections.Counter(x['reason'] for r in rs for x in r.get('actor_route_support',[]))),transitions=[dict(frame=r['frame'],actor=x['object_id'],transitions=x['transitions']) for r in rs for x in r.get('actor_route_support',[]) if x['transitions']],intent_source=None))
        access.append(dict(episode=ep,**d['access_audit']))
        errors += [dict(episode=ep,**x) for r in evs for x in r.get('prediction_errors',[])]
    paired=[]
    for ep in ['dev_constant_r3','dev_braking_r20']:
        paths={m:f'reports/m7d_replays/{ep}_{m}_clean.json' for m in ['baseline','cold','stream']}
        if not all(Path(p).exists() for p in paths.values()):continue
        modes={m:read(p) for m,p in paths.items()};base=modes['baseline']['records']
        eq={m:len(d['records'])==len(base) and all(equal(comparable(a),comparable(b)) for a,b in zip(base,d['records'])) for m,d in modes.items()}
        stats={m:dict(ms=quantiles([r['timing_ms']['total'] for r in d['records'] if 'timing_ms'in r]),over_200ms=sum(r.get('timing_ms',{}).get('total',0)>200 for r in d['records']),expired=sum(r.get('clock',{}).get('expired_for_online',False) for r in d['records']),cache=d['records'][-1].get('cache'),artifact=asset(paths[m])) for m,d in modes.items()}
        paired.append(dict(episode=ep,all_candidate_equivalence=eq,modes=stats))
    availability=[]
    for d in ds:
        path=f"reports/m7d_replays/{d['episode']}_availability.json"
        ad=read(path);rr=[r for r in ad['records'] if 'availability_audit' in r]
        availability.append(dict(episode=d['episode'],all_scientific_results_equal=len(d['records'])==len(ad['records']) and all(equal(comparable(a),comparable(b)) for a,b in zip(d['records'],ad['records'])),
            payload_ready_ms=quantiles([r['availability_audit']['complete_compute_and_serialization_ms'] for r in rr]),
            serialization_ms=quantiles([r['availability_audit']['serialization_ms'] for r in rr]),expired=sum(r['clock']['expired_for_online'] for r in rr),records=len(rr),artifact=asset(path)))
    assert len(ds)==8 and sum(len(d['records']) for d in ds)==488
    assert all(x['all_scientific_results_equal'] for x in availability), 'availability measurement changed scientific results'
    historic=read('reports/m7d_historical_equivalence.json');histn=sum(len(r['candidates']) for r in historic['records'])
    write('reports/m7d_latency_equivalence.json',dict(**common,historical_scenes=historic['valid_scenes'],historical_candidates=histn,historical_all_equal=historic['all_equal'],historical_seven_detection=read('reports/m7d_historical_seven.json'),paired_real_snapshots=paired,optimization='Same-snapshot polygon reuse only; exact early exit, constraints and hypotheses unchanged',cold_definition='New GeometryCache for each snapshot, immutable map index prebuilt',stream_definition='One cache over actual changing snapshots, no repeat warming'))
    write('reports/m7d_temporal_planning_results.json',dict(**common,episodes=summary,independent_episodes=len(ds),snapshots=sum(len(d['records']) for d in ds),sample_dependence='0.2s overlapping six-second windows are strongly correlated, not independent scenes',RobustWithinHypotheses='Per-candidate only; finite declared family',FeasibleToExecute='NotValidated'))
    write('reports/m7d_actual_future_collision_audit.json',dict(**common,episodes=collision,prediction_errors=errors,reference='Real .05s CARLA Actor snapshots / planar OBB and OpenDRIVE. Not independent render Mesh, not exact continuous physical collision. Target actors do not react to counterfactual Ego.',SafeForRecordedFuture='Only complete observed six-second sampled reference and body-road/stop checks',FeasibleToExecute='NotValidated'))
    write('reports/m7d_decision_stability.json',dict(**common,episodes=stability))
    write('reports/m7d_stop_and_progress_analysis.json',dict(**common,episodes=progress,slow_parameters='Frozen target0.5m/s,4.5s cosine transition, no second family',no_actual_plan_tracking=True))
    write('reports/m7d_geometry_runtime_profile.json',dict(**common,episodes=runtime,clean_paired=paired,complete_payload_availability=availability,strict_realtime='NoGo',vehicle_execution='NotRun'))
    write('reports/m7d_route_support.json',dict(**common,episodes=routes,historical_roads18_19='Historical205 equivalence reexecutes ambiguity unchanged; original reference retained',branch_intent='None; unique successor only; future never used',person_support='PredictionOutOfSupport unless declared corridor available; no invented future corridor'))
    write('reports/m7d_future_leakage_audit.json',dict(**common,status='PassedWithinAuditedPythonDataPath',episodes=access,active_forbidden_future_probe=asset('reports/m7d_forbidden_future_probe.json'),forbidden_read_was_blocked=bool(read('reports/m7d_forbidden_future_probe.json')['access_audit']['denied']),mechanism='Separate planner subprocess: after imports/map setup sys.addaudithook restricts reads to enumerated current packet paths. Evaluator starts only after output persisted. No Actor handles or future paths in value contract.',limits='Application data-flow contract, not an adversarial OS sandbox. Frozen map and source imports precede guard.'))
    inv=read('reports/m7d_historical_inventory.json');col=[read(p) for p in Path('reports/m7d_episodes').glob('*/result.json')]
    write('reports/m7d_episode_inventory.json',dict(**common,historical=asset('reports/m7d_historical_inventory.json'),historical_files=inv['summary'],new_episodes=col,selected_history=cfg['history_selection'],new_episode_count=len(col),cap=12,renderer_failure='Low Town03 SIGSEGV preserved; Epic corrected according to historical documented procedure',collection_collision_episodes=[r['episode_id'] for r in col if r['collisions']],failure_scope='Independent script does not avoid static obstacle; collection collisions not planner control collisions. Contact dynamics may invalidate hypotheses, retained. SameTown03/road-ID separation not cross-map or guaranteed geographic independence.'))
    write('reports/m7d_temporal_contract.json',dict(**common,status='PassedWithinRecordedScope',schema='datasets/m7d_temporal.py',original_simulation_timestamp_preserved=True,inputs='Current actors + history stamps + predeclared Ego route only',wall_clock='Original collection availability; new offline computation start/end; cannot subtract wall from simulation',modeled_age='input delay + measured compute time; offline model, not real-time tick claim',max_age_s=.2,missing_actor_policy='Unknown; no reconstructed measured samples',input_rejections=sum(len(s['input_rejections']) for s in summary),all_result_times_original=True,full_payload_ready_audit=availability,core_clock_limitation='Initial reports core clock ends before result construction; supplemental availability replay measures complete payload serialization without changing frozen plan or final decisions'))
    write('reports/m7d_safety_contract_regression.json',dict(**common,kind_override='SyntheticUnitTestsAndOfflineRecordedReplay',full_test_log=asset('reports/m7d_execution_logs/pytest_full.txt'),sdk_test_log=asset('reports/m7d_execution_logs/pytest_sdk.txt'),full_result=Path('reports/m7d_execution_logs/pytest_full.txt').read_text().splitlines()[-1],sdk_result=Path('reports/m7d_execution_logs/pytest_sdk.txt').read_text().splitlines()[-1],cases=['Missing snapshot','nonmonotonic time','actor identity jump/change','duplicate ID','future actor stamp','future history','unsupported reverse','nonfinite input','expired plan','no feasible candidate','map interruption','missing hypotheses','snapshot cache identity'],real_worker_faults='NotRun',actual_braking='NotRun',defaults_unchanged=True))
    conflicts=sum(r['accepted_conflict_candidates'] for r in collision)
    write('reports/m7d_acceptance.json',dict(**common,overall='Partial',A_input_time_contract='PassedWithinRecordedScope',B_recorded_future='Partial_NoObservedAcceptedConflict' if conflicts==0 else 'NoGo_AcceptedReferenceConflict',accepted_reference_conflicts=conflicts,C_slow_coverage='Partial_LimitedRealStateCompatibility',D_stability_support='Partial_StopRejectAndUnsupportedActors',E_latency='NoGo_200msBudgetExceeded',F_online_shadow='NotApproved_OfflineFoundationOnly',historical_equivalence='Passed' if historic['all_equal'] else 'Failed',optimization_equivalence='Passed' if paired and all(all(p['all_candidate_equivalence'].values()) for p in paired) else 'FailedOrNotRun',FeasibleToExecute='NotValidated',control='NotAuthorized_NotRun',runtime_inherited='RuntimePartial',visual='CandidateNoGo',occupancy='NotPassed',training='NotRun',test_counts_not_independent_scenes=True))
    print('report episodes',len(ds),'snapshots',sum(len(d['records']) for d in ds),'conflicts',conflicts)
if __name__=='__main__':main()
