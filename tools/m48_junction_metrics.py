"""Separate real CARLA latency, route geometry and scheduling acceptance."""
import json
from pathlib import Path
import numpy as np
from simulator.recorder import atomic_json,file_hash
from tools.m47_common import dynamics_metrics

def read(p):return json.loads(Path(p).read_text())
def distribution(x):
    return dict(p50_ms=float(np.quantile(x,.5)),p95_ms=float(np.quantile(x,.95)),max_ms=float(np.max(x)),n=len(x)) if len(x) else None

def main():
    output=Path('reports/m48_junction_validation.json')
    if output.exists():raise FileExistsError(output)
    records=[]
    for path in ['reports/m48_junction_pilot_v3.json','reports/m48_junction_short.json','reports/m48_multirate_verified.json']:
        source=read(path)
        for result in source['results']:
            assert file_hash(result['log'])==result['log_sha256']
            rows=[json.loads(x) for x in Path(result['log']).read_text().splitlines()];planner=np.array([r['planner_ms'] for r in rows]);safety=np.array([r['safety_ms'] for r in rows]);total=np.array([r['latency_ms'] for r in rows]);active=np.array([r.get('planning_tick',True) for r in rows]);age=[r.get('plan_age_s',0) for r in rows];multi='multirate' in path
            records.append(dict(source=path,scenario=result['scenario'],route_status=result['status'],completion=result['route_completion'],collision_count=result['collision_count'],lane_invasions=result['lane_invasions'],safety_events=result['safety_forced_braking_events'],safety_ticks=result['safety_forced_braking_ticks'],
                planner=distribution(planner[active]),safety=distribution(safety[active]),planner_safety=distribution((planner+safety)[active]),controller_only_ticks=distribution(total[~active]),all_tick_total=distribution(total),
                plan_generation_period_s=.2 if multi else .05,max_plan_age_s=max(age),
                planning_budget_p95_pass=bool(np.quantile((planner+safety)[active],.95)<=(200 if multi else 50)),
                all_tick_50ms_deadline_pass=bool((total<=50).all()),controller_wallclock_decoupled=False,
                actual_dynamics=dynamics_metrics([r['measurement'] for r in rows]),log_sha256=result['log_sha256'],runtime_sources=source['source_hashes_at_start']))
    atomic_json(output,dict(kind='offline_metrics_of_actual_carla_closedloop',records=records,source_sha256=file_hash(__file__),
        selected_release='No replacement of default; M48 full-horizon route fix is a research candidate.',
        states=dict(route_geometry_recovery='Passed on three fixed-green Town03 routes; not traffic-rule generalization',latency_20hz_full_horizon='Failed',multirate_interface='Passed interpolation/expiry and simulated scheduling tests',multirate_wallclock_realtime='Partial: serial computation blocks controller on planning ticks',safety_closedloop='Partial: comfort and hard realtime not passed'),
        short_horizon_scope='2s horizon was performance diagnostic only, not accepted as replacement for 6s obstacle anticipation.',
        limitations=['Exact-map road corners are discrete, not polygon-swept road containment.','No asynchronous planner worker; nominal simulated frequencies do not prove wall-clock 20Hz.','Resampled tracking reference is checked against generation state for kinematic limits, not actual state drift. Current footprint and immediate obstacle guard remain every tick.','Fixed green lights are route regression, not traffic-rule validation.']))
if __name__=='__main__':main()
