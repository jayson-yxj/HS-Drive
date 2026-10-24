"""Derive stage decisions from preserved paired and future audit artifacts."""
from pathlib import Path
import collections
import numpy as np
from tools.m7e_common import *
from tools.m7c_common import equal
from diagnostics.m7b_motion import interpolate


def quantile(v):
    return {k:float(np.quantile(v,q)) for k,q in [('p50',.5),('p95',.95),('p99',.99),('max',1.)]} if v else None


def scientific(r,old_only=False):
    return dict(selected=r.get('baseline_selected') if old_only else r.get('selected'),hypotheses=r.get('hypotheses'),prediction_unknown=r.get('prediction_unknown'),
        evaluations=[{k:e[k] for k in ['name','candidate','geometry','dynamic','stop_reserve','accepted','allowed_lanes']} for e in r['evaluations'] if not(old_only and e['name'].startswith('F_'))])


def stability(rs,key='selected'):
    switches=[];speed=[];space=[];reject_runs=[];run=0;recover=[]
    for i,r in enumerate(rs):
        chosen=r.get(key)
        if chosen is None:run+=1
        elif run:reject_runs.append(run*.2);run=0
        if not i:continue
        prev=rs[i-1];before=prev.get(key)
        if before!=chosen:
            switches.append(dict(frame=r['frame'],before=before,after=chosen))
            if before in [None,'D_stop'] and chosen not in [None,'D_stop']:recover.append(r['frame'])
        if before and chosen:
            a=next(e['candidate'] for e in prev['evaluations'] if e['name']==before);b=next(e['candidate'] for e in r['evaluations'] if e['name']==chosen)
            dt=r['delivery']['original_timestamp_s']-prev['delivery']['original_timestamp_s'];q=np.linspace(0,6-dt,30)
            aa=interpolate(a['points'],a['times_s'],q+dt);bb=interpolate(b['points'],b['times_s'],q)
            space.append(float(np.linalg.norm(aa[:,:2]-bb[:,:2],axis=1).max()));speed.append(float(abs(interpolate(a['points'],a['times_s'],[1+dt])[0,3]-interpolate(b['points'],b['times_s'],[1])[0,3])))
    if run:reject_runs.append(run*.2)
    return dict(switch_count=len(switches),switches=switches,recovery_frames=recover,longest_reject_s=max(reject_runs,default=0),speed_target_delta_mps=quantile(speed),overlap_trajectory_max_delta_m=quantile(space),note='Actual script state each step; these are not executed plan transitions or jerk at vehicle.')


def main():
    common=dict(provenance=provenance(),kind='OfflineActualCARLARecords_SeparateSyntheticTests',control_authorized=False)
    ds=[read(p) for p in sorted(Path('reports/m7e_replays').glob('*_B.json'))];ablation=[];trajectory=[];future=[];stabilities=[];latency=[];allbase=True
    for d in ds:
        ep=d['episode'];rs=d['records'];isnew=ep.startswith('m7e_');ev=read(f'reports/m7e_evaluations/{ep}.json')['records'];fs=[e for r in rs for e in r['evaluations'] if e['name']=='F_state_anchored_slow']
        fe=[e for r in ev for e in r['evaluations'] if e['name']=='F_state_anchored_slow'];acc=[e for e in fe if e['accepted']];allacc=[e for r in ev for e in r['evaluations'] if e['accepted']]
        if not isnew:
            old=read(f'reports/m7d_replays/{ep}_stream.json')['records'];matches=[]
            for a,b in zip(rs,old):
                aa=scientific(a,True);bb=scientific(b);matches.append(equal(aa,bb))
            baseeq=len(old)==len(rs) and all(matches);allbase &= baseeq
        else:baseeq='SeparatePairedSameSnapshotAEmbedded'
        added=[r for r in rs if r['selected']=='F_state_anchored_slow' and r['baseline_selected'] in [None,'D_stop']]
        ablation.append(dict(episode=ep,group='new_final' if isnew else 'M7D_development' if ep in ['dev_constant_r3','dev_braking_r20'] else 'historical_regression',snapshots=len(rs),A=dict(collections.Counter(str(r.get('baseline_selected')) for r in rs)),B=dict(collections.Counter(str(r['selected']) for r in rs)),added_continue=len(added),added_frames=[r['frame'] for r in added],old_candidates_unchanged=baseeq,
            F_generated=len(fs),F_geometry_pass=sum(e['geometry']['planning_feasible'] for e in fs),F_dynamic_pass=sum(e['dynamic']['robust'] for e in fs),F_reserve_pass=sum(e['stop_reserve']['passed'] for e in fs),F_accepted=sum(e['accepted'] for e in fs),
            F_final_connection_status=dict(collections.Counter('GeometryKinematicsPassed' if e['geometry']['planning_feasible'] else 'InitialStateConnectionInfeasible:'+','.join(e['geometry']['reasons']) for e in fs)),F_infeasible=dict(collections.Counter(r['f_connection_error'] for r in rs if r.get('f_connection_error'))),F_geometry_reasons=dict(collections.Counter(k for e in fs for k in e['geometry']['reasons'])),unknown_snapshots=sum(bool(r.get('prediction_unknown')) for r in rs),artifact=asset(f'reports/m7e_replays/{ep}_B.json')))
        anchors=[];first_integral=[];first_curvature=[]
        for r in rs:
            for e in r['evaluations']:
                if e['name']!='F_state_anchored_slow':continue
                p=np.array(e['candidate']['points']);actual=np.array(r['request']['scene']['ego_xy_yaw_speed']);ds0=np.linalg.norm(p[1,:2]-p[0,:2]);anchors.append(float(abs(p[0]-actual).max()));first_integral.append(float(abs(ds0-(p[0,3]+p[1,3])*.05)));first_curvature.append(float(abs(np.arctan2(np.sin(p[1,2]-p[0,2]),np.cos(p[1,2]-p[0,2])))/max(ds0,1e-6)))
        trajectory.append(dict(episode=ep,generated=len(fs),max_t0_state_error=max(anchors,default=None),first_integral_error_m=quantile(first_integral),first_curvature_inv_m=quantile(first_curvature),F_progress_m=quantile([e['plan_progress_m'] for e in fe if e['accepted']]),initial_state_contract_pass=sum(x<=1e-12 for x in anchors),initial_acceleration_and_yaw_rate='NotObserved_assumed_zero_planned_acceleration_and_curvature',FeasibleToExecute='NotValidated'))
        future.append(dict(episode=ep,group='new_final' if isnew else 'historical_or_development',accepted_total=len(allacc),accepted_F=len(acc),selected_F=sum(e['selected'] for e in acc),accepted_conflict_candidates=sum(bool(e['measured_sample_OBB_conflicts']) for e in allacc),accepted_F_conflict_candidates=sum(bool(e['measured_sample_OBB_conflicts']) for e in acc),accepted_F_road_violations=sum(bool(e['full_body_road_violation_samples']) for e in acc),accepted_F_recorded_reserve_failures=sum(not e['recorded_slow_reserve_passed'] for e in acc),accepted_F_min_gap_m=min((e['minimum_gap_m'] for e in acc if e['minimum_gap_m'] is not None),default=None),accepted_F_min_reserve_m=min((e['recorded_slow_reserve_min_m'] for e in acc if e['recorded_slow_reserve_min_m'] is not None),default=None),artifact=asset(f'reports/m7e_evaluations/{ep}.json')))
        stabilities.append(dict(episode=ep,A=stability(rs,'baseline_selected'),B=stability(rs)))
        latency.append(dict(episode=ep,scope='Full campaign, may share CPU with diagnostics; clean paired timings separately',full_ms=quantile([r['delivery']['full_ms'] for r in rs]),expired=sum(r['delivery']['expired'] for r in rs),snapshots=len(rs),valid_fraction=sum(not r['delivery']['expired'] for r in rs)/len(rs),map_setup_ms=d['map_setup_ms'],components_ms={k:quantile([r['timing_ms'][k] for r in rs if 'timing_ms'in r]) for k in rs[0].get('timing_ms',{})},serialization_ms=quantile([r['delivery']['serialization_ms'] for r in rs]),candidate_ms={n:{k:quantile([e['timing_ms'][k] for r in rs for e in r['evaluations'] if e['name']==n]) for k in ['road_body','dynamic_SAT','slow_reserve']} for n in sorted({e['name'] for r in rs for e in r['evaluations']})},region_cache=rs[-1].get('region_cache'),body_cache=rs[-1].get('cache')))
    clean=[]
    for ep in ['dev_constant_r3','dev_braking_r20']:
        modes={a:read(f'reports/m7e_replays/{ep}_{a}_clean.json') for a in ['A_uncached','A','B']};eq=all(equal(scientific(a),scientific(b)) for a,b in zip(modes['A_uncached']['records'],modes['A']['records']))
        clean.append(dict(episode=ep,static_cache_scientific_equal=eq,modes={arm:dict(full_ms=quantile([r['delivery']['full_ms'] for r in d['records']]),expired=sum(r['delivery']['expired'] for r in d['records']),valid_fraction=sum(not r['delivery']['expired'] for r in d['records'])/len(d['records']),map_setup_ms=d['map_setup_ms'],region_cache=d['records'][-1].get('region_cache'),body_cache=d['records'][-1].get('cache'),artifact=asset(f'reports/m7e_replays/{ep}_{arm}_clean.json')) for arm,d in modes.items()}))
    write('reports/m7e_candidate_ablation.json',dict(**common,episodes=ablation,old_candidates_all_equal=allbase,new_F_not_equivalent_by_design=True,development_gate=read('reports/m7e_development_gate.json')))
    write('reports/m7e_state_anchored_trajectory.json',dict(**common,episodes=trajectory,algorithm='Single quintic XY Hermite with actual-state tangent and arc length speed; all original checks retained',parameters=asset('configs/m7e/protocol.json'),unsupported_negative_speed='F rejected, no clipping to pretend measured zero',FeasibleToExecute='NotValidated'))
    write('reports/m7e_actual_future_collision_audit.json',dict(**common,episodes=future,RobustWithinHypotheses='Per candidate, frozen finite motion family',SafeForRecordedFuture='Actual .05s samples, planar OBB only; candidate interpolation not mesh truth',FeasibleToExecute='NotValidated',limits='Scripted actors do not react to hypothetical Ego; overlapping windows not independent samples. No control execution.'))
    write('reports/m7e_temporal_stability.json',dict(**common,episodes=stabilities))
    write('reports/m7e_full_payload_latency.json',dict(**common,episodes=latency,clean_paired=clean,OnlinePlanner='OnlinePlannerNoGo',result_ready_definition='Complete scientific result serialized bytes readable in memory; original stamp retained; envelope/disk persistence afterward. No IPC/control claim',map_setup_excluded_but_reported=True,budget_ms=200,validity_s=.2))
    geo=read('reports/m7e_geometry_equivalence.json');seven=read('reports/m7e_historical_seven.json');new=[a for a in ablation if a['group']=='new_final'];newadd=sum(a['added_continue'] for a in new)
    write('reports/m7e_acceptance.json',dict(**common,overall='Partial_OfflineOnly',A_initial_state='PassedWithinDeclaredSupport',B_initial_rejection='ImprovedWithOtherConstraintsRetained',C_independent_progress='Partial_ImprovedFiniteRecordedCoverage' if newadd else 'NoGo_NoIndependentGain',new_final_added_continue=newadd,D_frozen_risks='Passed' if geo['all_equal'] and seven['detected']==7 and allbase else 'Failed',E_static_cache='Equivalent_NoConsistentMaterialSpeedup' if any(x['modes']['A']['full_ms']['p95']>=.98*x['modes']['A_uncached']['full_ms']['p95'] for x in clean) else 'Equivalent_MeasuredSpeedup',F_latency='OnlinePlannerNoGo',online_shadow='NotApproved',vehicle_execution='NotValidated',visual_control='NotRun_Forbidden',default_modules_modified=False,training='NotRun',automatic_M7F='NotAuthorized_NoUnboundedExpansion',hist205=geo['scenes'],hist1662=geo['candidates'],accepted_reference_conflicts=sum(f['accepted_conflict_candidates'] for f in future)))
    assert len(ds)==11 and allbase
    assert all(x['static_cache_scientific_equal'] for x in clean)
    print('episodes',len(ds),'new added',newadd,'all old equal',allbase)
if __name__=='__main__':main()
