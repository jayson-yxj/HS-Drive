"""Read-only original snapshots and frozen candidate failure attribution."""
from pathlib import Path
import numpy as np
import carla
from shapely.geometry import Polygon
from diagnostics.m7d_scene import RoadIndex
from diagnostics.m7e_trajectory import initial_state
from datasets.m7d_temporal import from_packet
from diagnostics.m7a_planning import center,swept_polygon
from simulator.oracle_driver import footprint
from simulator.motion_oracle import wrap
from tools.m7e_common import *


def main():
    mp='reports/m7d_execution_logs/Town03.xodr';roads=RoadIndex(carla.Map('Town03',Path(mp).read_text()));rows=[]
    for path in sorted(Path('reports/m7d_replays').glob('*_stream.json')):
        d=read(path)
        for r in d['records']:
            p=read(f"reports/m7d_inputs/{d['episode']}/{r['input_index']:05d}.json");inp=from_packet(p);req=inp.request();state=initial_state(req)
            state['v_forward_mps']=inp.signed_speed_mps;state['request_speed_clipping_mps']=req.scene.ego_xy_yaw_speed[3]-inp.signed_speed_mps
            keys,_,_=roads.ego_support(inp);body=Polygon(footprint(*center(req.scene.ego_xy_yaw_speed,req.scene.body_center_offset),inp.ego_xy_yaw[2],*inp.body_half_extent))
            state['body_road_validity']=roads.region(keys).covers(body);es=[]
            for e in r.get('evaluations',[]):
                c=e['candidate'];points=np.array(c['points']);first=points[:2];dt=c['times_s'][1]-c['times_s'][0];ds=float(np.linalg.norm(first[1,:2]-first[0,:2]));curv=abs(float(wrap(first[1,2]-first[0,2])))/max(ds,1e-6)
                region=roads.region({tuple(k) for k in e['allowed_lanes']}).buffer(-.03);sweep=swept_polygon(*first,np.array(inp.body_half_extent)+[.2,.15],inp.body_offset)
                pos=float(np.linalg.norm(points[0,:2]-inp.ego_xy_yaw[:2]));heading=abs(float(wrap(points[0,2]-inp.ego_xy_yaw[2])));integ=abs(ds-(first[:,3].sum())*dt/2)
                flags=dict(initial_position_mismatch=pos>1e-6,initial_heading_mismatch=heading>1e-6,initial_speed_mismatch=abs(points[0,3]-inp.signed_speed_mps)>1e-6,
                    first_segment_integral_error=integ>.025,first_segment_curvature=curv>.10,first_segment_body_road=not region.covers(sweep),
                    dynamic_collision=bool(e['dynamic']['collisions']),dynamic_stop_margin=bool(e['dynamic']['stop_margin_failures']),motion_support_insufficient=bool(e['dynamic']['unknown']),
                    actor_corridor_unknown=any(u['reason'] in ['ActorCategoryWithoutDeclaredMotionCorridor','missing_motion_observation'] for u in e['dynamic']['unknown']),
                    slow_stop_reserve_failed=e['stop_reserve'] is not None and not e['stop_reserve']['passed'])
                flags={k:bool(v) for k,v in flags.items()}
                es.append(dict(name=e['name'],accepted=e['accepted'],flags=flags,initial_position_error_m=pos,initial_heading_error_rad=heading,
                    first_segment_integral_error_m=integ,first_segment_curvature_inv_m=curv,reasons=e['geometry']['reasons'],
                    initial_contract_only_blocker=bool(e['geometry']['reasons']) and set(e['geometry']['reasons'])<= {'initial_state_mismatch'} and e['dynamic']['robust'] and not flags['slow_stop_reserve_failed']))
            rows.append(dict(episode=d['episode'],frame=r['frame'],index=r['input_index'],state=state,evaluations=es))
    import collections
    totals={}
    for name in sorted({e['name'] for r in rows for e in r['evaluations']}):
        es=[e for r in rows for e in r['evaluations'] if e['name']==name]
        totals[name]=dict(candidates=len(es),accepted=sum(e['accepted'] for e in es),initial_contract_only_blocker=sum(e['initial_contract_only_blocker'] for e in es),flags=dict(collections.Counter(k for e in es for k,v in e['flags'].items() if v)),full_geometry_reasons=dict(collections.Counter(k for e in es for k in e['reasons'])))
    write('reports/m7e_initial_state_failure_attribution.json',dict(kind='OfflineFrozenM7DMeasuredStateAudit',provenance=provenance(),records=rows,summary=totals,limits='Overlapping failure categories, not mutually exclusive. Removing initial mismatch alone is not a safe acceptance rule. Original request clipped tiny negative speed; F must reject any negative measured speed.'))
    print(totals)
if __name__=='__main__':main()
