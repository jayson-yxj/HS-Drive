"""Actual SDK connectivity audit with explicit historical reference support."""
import numpy as np
import carla
from pathlib import Path
from shapely.geometry import Polygon
from shapely.ops import unary_union
from tools.m7c_common import *
from tools.m7b_fixtures import map_fixture
from tools.m7a_scenarios import pose,lane_polygon
from diagnostics.m7c_routes import extend_existing,continue_waypoints
from diagnostics.m7b_motion import longitudinal,build_hypotheses,ObservedMotion
from simulator.oracle_driver import footprint
from tools.m7c_equivalence import request_from


def main():
    cfg=read('configs/m7b/protocol.json');m=carla.Map('Town03',Path(cfg['planning_map']).read_text());records=[];support=[]
    for road,file in [(19,'validation_frozen'),(18,'historical_regression_frozen')]:
        f=map_fixture(m,road,'history');path,conn,join=extend_existing(m,f['route'],f['z'])
        records.append(dict(road=road,original_length=f['arc'].length,extended_length=path.length,join_distance=join,status=conn.status,stop_reason=conn.stop_reason,transitions=conn.transitions,nodes=conn.nodes,original_prefix_exact=np.array_equal(path.points[:len(f['route'])],f['route'])))
        old=read('reports/m7b_'+file+'.json')
        for r in old['records']:
            if r['road']!=road or r['spec']['kind']!='lead':continue
            req=request_from(r);spec=r['spec'];obs=[ObservedMotion('lead',path,spec['d'])];hs,u=build_hypotheses(req.scene,obs,cfg)
            t=np.arange(0,6.0001,.025);ds,v=longitudinal(t,spec['v'],spec['a'],spec['onset']);ref_supported=spec['d']+ds.max()<=path.length
            # Original reference parameters untouched; report only support, not replacement result.
            support.append(dict(scene_id=r['scene_id'],old_fixture=r['fixture']['status'],old_unknown=r.get('prediction_unknown'),new_prediction_unknown=u,hypotheses=len(hs),same_reference_parameters=spec,reference_now_supported=bool(ref_supported),continuation_reason=conn.stop_reason,classification='SupportedByUniqueSuccessors' if ref_supported and not u else 'AmbiguousSuccessorsUnknown' if conn.stop_reason=='AmbiguousSuccessors' else 'StillOutOfSupport'))
    # Independent topology examples, map-only: not new final planning scenes.
    examples=[]
    ws=m.generate_waypoints(1.)
    for road in [0,1,6,19,18,60]:
        q=[w for w in ws if w.road_id==road and not w.is_junction and w.lane_type==carla.LaneType.Driving]
        lane=min(w.lane_id for w in q);q=sorted([w for w in q if w.lane_id==lane],key=lambda w:w.s,reverse=lane>0)
        conn=continue_waypoints(q[-5],required_length=40,max_extra=60)
        examples.append(dict(road=road,status=conn.status,reason=conn.stop_reason,length=conn.length,transitions=conn.transitions,nodes=conn.nodes))
    write('reports/m7c_connected_route_support.json',dict(kind='ActualOfflineCARLA0915SDKTopology',provenance=provenance(),historical_routes=records,topology_examples=examples,scope='Forward unique legal successors only; multi-branch without explicit intent remains Unknown; no tangent continuation',full_body_future='Separate sampled geometry audit required; connectivity alone is insufficient'))
    write('reports/m7c_prediction_support_audit.json',dict(kind='HistoricalSupportOnly_NoReopenedTest',provenance=provenance(),records=support,reference_redefined=False,decision_changed=False,limits='This is support diagnosis, not rerun historical acceptance; original M7B failures retained.'))
    print([(r['road'],r['extended_length'],r['stop_reason']) for r in records]);print([(s['scene_id'],s['classification']) for s in support])
if __name__=='__main__':main()
