"""Unfiltered measurement metrics; phase boundaries retain transition derivatives."""
import argparse,json
from pathlib import Path
import numpy as np
from tools.m47_common import dynamics_metrics
from tools.m6e_common import write,asset,provenance
from diagnostics.m6a_oracle_metrics import quantiles


def inspect(path):
    d=json.loads(path.read_text());rows=[json.loads(s) for s in (path.parent/'trace.jsonl').read_text().splitlines()];phases={}
    for phase in dict.fromkeys(q['phase'] for q in rows):
        inds=[i for i,q in enumerate(rows) if q['phase']==phase];r=[rows[i] for i in inds];metric_rows=rows[max(0,inds[0]-2):inds[-1]+1]
        t=np.array([q['measurement']['t'] for q in r]);v=np.array([q['signed']['v_forward_mps'] for q in r]);norm=np.array([q['signed']['speed_norm_mps'] for q in r]);p=np.array([q['measurement']['position'][:2] for q in r])*[1,-1]
        yaw=r[0]['signed']['yaw_rad'];progress=(p-p[0])@np.array([np.cos(yaw),np.sin(yaw)])
        raw=dynamics_metrics([q['measurement'] for q in metric_rows]) if len(metric_rows)>=8 else None
        forward=np.flatnonzero(v>.1);stopped=np.flatnonzero(norm<=.02)
        if phase in ['normal_stop','final_stop','diagnostic_action','warm_stop']:stopped=np.array([i for i in stopped if i>=20 and np.all(norm[i-20:i+1]<=.02)],int)
        refs=[(q['signed']['v_forward_mps']-q['reference_speed'])**2 for q in r if q['reference_speed'] is not None]
        phases[phase]=dict(ticks=len(r),start_t=float(t[0]),initial_speed=float(v[0]),final_speed=float(v[-1]),max_speed=float(norm.max()),
            start_delay_s=None if not len(forward) else float(t[forward[0]]-t[0]),
            stop_duration_s=None if not len(stopped) else float(t[stopped[0]]-t[0]),
            stopped_travel_m=None if not len(stopped) else float(progress[stopped[0]]),total_travel_m=float(progress[-1]),
            net_backward_excursion_m=float(max(0,-progress.min())),backward_frames=int((v<-.03).sum()),
            max_displacement_m=float(np.linalg.norm(p-p[0],axis=1).max()),
            final_stable_1s=len(r)>=21 and bool(np.all(norm[-21:]<=.08)),
            hold_passed=len(r)>=21 and bool(np.all(norm[-21:]<=.02)) and float(np.linalg.norm(p-p[0],axis=1).max())<=.05,
            reference_speed_RMSE_mps=None if not refs else float(np.sqrt(np.mean(refs))),
            actual_gear=sorted({q['measurement']['applied']['gear'] for q in r}),
            raw=raw,comfort_passed=bool(raw and raw['max_decel_raw']<=1.8 and raw['jerk_peak_raw']<=5 and raw['jerk_p95_moving']<=2.5),
            applied_brake_range=[min(q['measurement']['applied']['brake'] for q in r),max(q['measurement']['applied']['brake'] for q in r)],
            applied_throttle_range=[min(q['measurement']['applied']['throttle'] for q in r),max(q['measurement']['applied']['throttle'] for q in r)])
    first_risk=next((q for q in rows if q['guard']['risk']),None);first_em=next((q for q in rows if q['protection_applied']),None)
    gaps=[q['obstacle_obb_gap_m'] for q in rows if q['obstacle_obb_gap_m'] is not None]
    return dict(name=path.parent.name,config=d['config'],result=asset(path),trace=d['trace'],physics=d['physics'],phases=phases,raw_all=d['raw'],complete=d['complete'],
        normal_comfort_passed=all(p['comfort_passed'] for k,p in phases.items() if k!='initialize'),
        initial=d['initial'],collisions=d['collisions'],boundary_ticks=d['online_boundary_ticks'],final_body_road=d['final_body_road'],lane_events=d['lane_events'],
        rollback_protection=dict(risk=first_risk is not None,emergency=first_em is not None,risk_to_apply_s=None if first_em is None or first_risk is None else first_em['measurement']['t']-first_risk['measurement']['t'],
            observation_to_apply_ms=None if first_em is None else (first_em['timeline']['apply_control']-first_em['timeline']['snapshot_end'])*1000,
            confirmed_net_m=None if first_em is None else first_em['guard']['net_backward_m'],
            final_stopped=all(q['signed']['speed_norm_mps']<=.02 for q in rows[-21:])),
        obstacle=dict(min_gap_m=min(gaps) if gaps else None,final_gap_m=gaps[-1] if gaps else None,final_gap3_passed=bool(gaps and gaps[-1]>=3)),
        runtime={k:d[k] for k in ['control_interval_ms','tick_ms','mean_hz','deadline_miss','skipped_slots','worker','expired_normal_plan']},
        stages_ms={name:quantiles([(q['timeline'][b]-q['timeline'][a])*1000 for q in rows]) for name,a,b in [('snapshot','wake_up','snapshot_end'),('guard_geometry_road','snapshot_end','safety_end'),('command','safety_end','controller_end')]},
        min_grade=min(q['road']['grade'] for q in rows),max_grade=max(q['road']['grade'] for q in rows))


def main():
    p=argparse.ArgumentParser();p.add_argument('batch');a=p.parse_args();plan=json.loads(Path(f'reports/m6e_{a.batch}_plan.json').read_text())
    records=[inspect(Path('reports/m6e_trials')/task['name']/'result.json') for task in plan['tasks']]
    write(f'reports/m6e_{a.batch}_analysis.json',dict(**provenance('offline_actual_CARLA_raw_phase_metrics'),trials=records,
        limits='Raw derivatives retain previous two samples at phase changes. No filtered acceptance; initialization archived. A/B command schedules differ from C trajectory feedback; not a single-factor physical ablation.'))
    for r in records:
        print(r['name'],'comfort',r['normal_comfort_passed'],'phases',[(k,round(p['raw']['jerk_peak_raw'],2) if p['raw'] else None,round(p['raw']['max_decel_raw'],2) if p['raw'] else None,p['start_delay_s']) for k,p in r['phases'].items() if k in ['cold_run','normal_stop','warm_run','diagnostic_action']],flush=True)
if __name__=='__main__':main()
