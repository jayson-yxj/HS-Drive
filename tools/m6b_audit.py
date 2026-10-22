"""No model changes: original stopping geometry and adjacent clock relationships."""
import json
from pathlib import Path
import numpy as np
from simulator.motion_oracle import route_projection
from tools.m6b_common import write, provenance, asset


def main():
    stops=[]
    for folder in ['m6a_low_speed_clean_artifacts','m6a_oracle_remaining_artifacts']:
        root=Path('reports')/folder
        route=np.array(json.loads((root/'low_speed_stop_route.json').read_text())['route_world_hs'])
        rows=[json.loads(s) for s in (root/'low_speed_stop.jsonl').read_text().splitlines()]
        last=rows[-1];xy=np.array(last['measurement']['position'][:2])*[1,-1]
        s,arc,_,lat=route_projection(route,xy);target_s=arc[-1]-1
        target=[float(np.interp(target_s,arc,route[:,i])) for i in (0,1)]
        first_brake=next((r for r in rows[10:] if r['command']['brake']>0 and r['reason'] is None),None)
        stops.append(dict(source='Privileged',trace=asset(root/'low_speed_stop.jsonl'),route_length_m=float(arc[-1]),
            navigation_end_xy=route[-1,:2].tolist(),implicit_stop_target_xy=target,actual_stop_xy=xy.tolist(),
            fixed_route_margin_m=1.0,arc_distance_to_navigation_end_m=float(arc[-1]-s),
            residual_to_implicit_target_m=float(target_s-s),lateral_error_m=lat,
            euclidean_target_error_m=float(np.linalg.norm(xy-target)),
            fixed_margin_fraction_of_endpoint_error=float(1/(arc[-1]-s)),
            first_normal_brake=first_brake,
            conclusion='fixed 1m explains part of endpoint shortfall; residual anticipatory envelope/HOLD and nonlinear dynamics remain; no causal percentage for controller vs physics'))
    root=Path('reports/m6a_oracle_artifacts');rows=[json.loads(s) for s in (root/'straight.jsonl').read_text().splitlines()]
    interval=np.diff([r['control_apply_wall'] for r in rows]);i=int(np.argmax(interval))+1
    prev,cur=rows[i-1],rows[i]
    timing=dict(previous_frame=prev['frame'],current_frame=cur['frame'],
        apply_interval_ms=float(interval[i-1]*1000),previous_world_tick_ms=prev['profile']['world_tick_ms'],
        previous_apply_wall=prev['control_apply_wall'],previous_tick_end_wall=prev['tick_end_wall'],
        current_wake_wall=cur['wall_begin'],current_apply_wall=cur['control_apply_wall'],
        between_tick_return_and_next_wake_ms=(cur['wall_begin']-prev['tick_end_wall'])*1000,
        current_preapply_ms=(cur['control_apply_wall']-cur['wall_begin'])*1000,
        source=asset(root/'straight.jsonl'),
        conclusion='Previous tick blocking dominates this interval; client log cannot separate UE rendering/physics/internal scheduling. Historical deadline reset exists; new runner uses fixed slot grid with explicit skipped slots.')
    write('reports/m6b_goal_stop_geometry.json',dict(**provenance('offline_geometry_and_clock_audit_of_actual_CARLA'),stops=stops,timing=timing,
        semantic_contract='route point=ego actor origin; footprint requires body beyond point; obstacle target=rear bumper clearance minus 3m, not navigation endpoint',
        controller_review=['HOLD at remaining<=.25 and speed<1.5 can invoke low-speed brake discontinuity.',
            'Alternate horizon<.8m, end speed<.08 and ego<.3 can latch early while target remains farther.',
            'remaining at controller is aged via speed*age, not reprojected actual current pose.',
            'low-speed effort max(0,effort) before rate limiting can retain negative previous effort.',
            'coast cap sqrt(.4*(remaining-.5)) already encodes a second .5m anticipatory offset.']))
    print([(r['arc_distance_to_navigation_end_m'],r['residual_to_implicit_target_m']) for r in stops],timing['apply_interval_ms'])

if __name__=='__main__':main()
