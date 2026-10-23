"""Historical seven risk audit and offline function-level profile."""
import cProfile,pstats,io
from dataclasses import replace
import numpy as np
from shapely.geometry import Polygon
from datasets.m7a_contracts import *
from diagnostics.m7b_motion import *
from diagnostics.m7a_planning import evaluate
from tools.m7b_common import *


def main():
    cfg=read('configs/m7b/protocol.json');failed=read('reports/m7a_fixture_validity.json')['valid_future_failures'];out=[]
    for item in failed:
        scene_id=item['scene_id'];split=scene_id.split('_road')[0];old=read('reports/m7a_planner_'+('development_v2' if split=='development' else split)+'.json')
        r=next(x for x in old['records'] if x['scene_id']==scene_id);sr=r['request']['scene']
        scene=OracleSceneState(Stamp(**sr['stamp']),tuple(sr['ego_xy_yaw_speed']),tuple(Box(**b) for b in sr['boxes']),tuple(sr['body_half_extent']),tuple(sr['body_center_offset']))
        cd=next(x for x in r['candidates'] if x['name']==item['candidate']);c=CandidateTrajectory(**{**cd,'stamp':Stamp(**cd['stamp'])})
        hs=[];unknown=[]
        for b in scene.boxes:
            # Preserve the historical straight-tangent snapshot support explicitly.
            # These old actors were not placed on the curved map. This historical
            # regression uses their declared tangent corridor, not fresh road claims.
            direction=np.array([np.cos(b.yaw),np.sin(b.yaw)])
            path=ArcRoute([(*b.xy,b.yaw),(*(np.array(b.xy)+direction*100),b.yaw)])
            h,u=build_hypotheses(replace(scene,boxes=(b,)),[ObservedMotion(b.object_id,path,0.,'historical_declared_tangent')],cfg);hs+=h;unknown+=u
        result=evaluate_dynamic_hypotheses(c,scene,hs,unknown)
        out.append(dict(scene_id=scene_id,candidate=c.name,risk_detected=bool(result['collisions']),result=result,scope='Historical seven only; old declared tangent corridor, not new road placement validity'))
    write('reports/m7b_historical_seven_v2.json',dict(kind='OfflineFrozenM7AReplay',provenance=provenance(),records=out,detected=sum(r['risk_detected'] for r in out),total=len(out)))
    r=next(x for x in read('reports/m7b_development_v1.json')['records'] if x['scene_id']=='development_road3_braking_0')
    sr=r['request']['scene'];scene=OracleSceneState(Stamp(**sr['stamp']),tuple(sr['ego_xy_yaw_speed']),tuple(Box(**b) for b in sr['boxes']),tuple(sr['body_half_extent']),tuple(sr['body_center_offset']))
    req=PlanningRequest(scene,tuple(map(tuple,r['request']['route'])),r['scene_id'],tuple(r['request']['goal_xy_yaw']))
    cd=r['candidate_evaluations'][0]['candidate'];c=CandidateTrajectory(**{**cd,'stamp':Stamp(**cd['stamp'])})
    obs=[ObservedMotion(x['object_id'],ArcRoute(x['route']),x['s0'],x['mode']) for x in r['observed_motion']]
    hs,u=build_hypotheses(scene,obs,cfg)
    import carla
    from tools.m7b_fixtures import map_fixture
    m=carla.Map('Town03',Path(cfg['planning_map']).read_text());f=map_fixture(m,3,'development');road=f['polygons'][f['lane']].buffer(-.03)
    def exact(x,y):
        w=m.get_waypoint(carla.Location(x=float(x),y=-float(y),z=f['z']),project_to_road=False,lane_type=carla.LaneType.Driving)
        return w is not None and w.road_id==3 and w.lane_id==f['lane']
    pr=cProfile.Profile();pr.enable()
    for _ in range(3):
        evaluate(req,c,road,100.,exact_road=exact);evaluate_dynamic_hypotheses(c,scene,hs,u)
    pr.disable();stream=io.StringIO();pstats.Stats(pr,stream=stream).sort_stats('cumulative').print_stats(45)
    Path('reports/m7b_execution_logs/function_profile.txt').write_text(stream.getvalue())
    print('historical risks',sum(x['risk_detected'] for x in out),'/',len(out))
if __name__=='__main__':main()
