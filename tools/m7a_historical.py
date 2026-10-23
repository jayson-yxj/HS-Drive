"""Preserve all M6A/B executions and re-evaluate bounded historical snapshots."""
import json,time,math
from pathlib import Path
from dataclasses import asdict
from collections import Counter
import numpy as np
import carla
from shapely.ops import unary_union
from datasets.m7a_contracts import *
from diagnostics.m7a_planning import evaluate,generate
from simulator.motion_oracle import MotionPlanner,OracleState
from simulator.m47_motion import safety_check
from tools.m7a_scenarios import lane_polygon
from tools.m7a_common import read,write,asset,provenance


def quant(v):
    return None if not len(v) else dict(n=len(v),**{k:float(np.quantile(v,q)) for k,q in [('p50',.5),('p95',.95),('p99',.99),('max',1.)]})


def main():
    cmap=carla.Map('Town03',Path('reports/m6b_execution_logs/Town03.xodr').read_text())
    execution=[];intervals=[];ticks=[];workers=[];snapshots=[]
    for path in sorted(Path('reports/m6b_trials').glob('*/result.json')):
        result=read(path);root=path.parent;trace=[json.loads(s) for s in (root/'trace.jsonl').read_text().splitlines()]
        timeline=[r['timeline'] for r in trace];ii=np.diff([t['apply_control'] for t in timeline])*1000
        tt=[(t['tick_end']-t['tick_start'])*1000 for t in timeline]
        plans=read(root/'plans.json')
        ww=[(p['output_wall']-p['start_wall'])*1000 for p in plans]
        intervals.extend(ii);ticks.extend(tt);workers.extend(ww)
        positions=np.array([r['measurement']['position'] for r in trace])
        execution.append(dict(name=root.name,evidence='HistoricalActualCARLA0915_Privileged',result=asset(path),trace=asset(root/'trace.jsonl'),
          acquisition_commit=result.get('git_head'),physics=asset(root/'physics.json'),
          planning_feasible='NotRetrospectivelyInferredFromVehicleSuccess',
          vehicle_executed_successfully=bool(result.get('goal_passed',False)),
          execution_classification='ActuationFailure_Comfort' if not result.get('comfort_pass',False) else 'LimitedHistoricalPass',
          goal_passed=result.get('goal_passed'),comfort_pass=result.get('comfort_pass'),completion=result.get('completion'),
          measured_distance_m=float(np.linalg.norm(np.diff(positions,axis=0),axis=1).sum()),
          collision_count=len(result.get('collisions',[])),illegal_lane_events=len(result.get('illegal_lane_events',[])),
          curb_contact='Unknown_no_mesh_reference',goal_error_m=result.get('final_goal_error_m'),
          control_interval_ms=quant(ii),world_tick_ms=quant(tt),worker_ms=quant(ww),
          deadline_miss=result['deadline_miss'],skipped_slots=result['skipped_wall_slots'],
          expired_normal_control=result['expired_normal_control'],degradation=result.get('degradation'),
          plan_rejections=result.get('planner_rejected'),dynamics=result.get('dynamics')))
    # Exact frozen historical map/actor/state snapshots, not synthetic fixtures.
    names=['final_straight_s101','final_mild_curve_s101','final_junction_turn_s101',
      'final_bypass_p126_s101','final_bypass_p170_s102','final_blocked_p126_s101_fixture_retry',
      'final_blocked_p170_s102','final_lead_s101']
    allwps=cmap.generate_waypoints(.5)
    body=read('reports/m6b_body_reference.json')
    for name in names:
        root=Path('reports/m6b_trials')/name
        if not root.exists():
            snapshots.append(dict(name=name,status='NotAvailable_missing_historical_trial'));continue
        trace=[json.loads(s) for s in (root/'trace.jsonl').read_text().splitlines()]
        route=np.array(read(root/'route.json')['selected'])
        for fraction in [.15,.5]:
            index=int((len(trace)-1)*fraction);row=trace[index];m=row['measurement']
            state=OracleState(m['position'][0],-m['position'][1],-math.radians(m['rotation'][2]),m['speed'],row['sim_time'])
            st=Stamp(row['frame'],state.timestamp_s,state.timestamp_s+.2,WORLD)
            objects=tuple(Box(str(o['actor_id']),tuple(o['xy']),o['yaw'],o['half_length'],o['half_width'],tuple(o['velocity'])) for o in row['obstacles'])
            scene=OracleSceneState(st,(state.x,state.y,state.yaw,state.speed_mps),objects,tuple(body['extent'][:2]),(body['location'][0],-body['location'][1]))
            req=PlanningRequest(scene,tuple(map(tuple,route)),name+':'+str(row['frame']),tuple(route[-1]))
            localwps=[]
            ids=set()
            for x,y,yaw in route:
                w=cmap.get_waypoint(carla.Location(x=float(x),y=float(-y),z=m['position'][2]),project_to_road=False,lane_type=carla.LaneType.Driving)
                if w:ids.add((w.road_id,w.lane_id))
            polys=[]
            for rid,lid in ids:
                ws=sorted([w for w in allwps if w.road_id==rid and w.lane_id==lid],key=lambda w:w.s)
                if len(ws)>1:polys.append(lane_polygon(ws))
            region=unary_union(polys).buffer(-.03)
            start=time.perf_counter()
            tr,meta=MotionPlanner().plan(route,state,row['obstacles'],max_speed=read(root/'config.json')['speed'])
            c,s=np.cos(state.yaw),np.sin(state.yaw)
            xy=tr.points[:,:2]@np.array([[c,s],[-s,c]])+[state.x,state.y]
            p=np.vstack((scene.ego_xy_yaw_speed,np.c_[xy,tr.points[:,2]+state.yaw,tr.points[:,3]]))
            candidate=CandidateTrajectory(st,'HistoricalFrozenMotionPlanner',tuple(map(tuple,p)),tuple(np.arange(len(p))*.1),tuple(map(tuple,route)))
            old=safety_check(tr,state,row['obstacles'],road_predicate=lambda x,y:cmap.get_waypoint(carla.Location(x=float(x),y=-float(y),z=m['position'][2]),project_to_road=False,lane_type=carla.LaneType.Driving) is not None)
            future={}
            for obj in objects:
                future[obj.object_id]=[(r['sim_time']-state.timestamp_s,tuple(o['xy'])) for r in trace[index:] if 0<=r['sim_time']-state.timestamp_s<=6 for o in r['obstacles'] if str(o['actor_id'])==obj.object_id]
            ev=evaluate(req,candidate,region,state.timestamp_s,future_reference=future)
            snapshots.append(dict(name=name,index=index,source='Privileged',frame=st.frame_id,request=asdict(req),
               source_trace=asset(root/'trace.jsonl'),source_route=asset(root/'route.json'),old_safety_reason=old,
               evaluation=asdict(ev),elapsed_ms=(time.perf_counter()-start)*1000,
               comparison='FrozenPlanner cold replay; not exact historical worker state; stronger M7A offline evaluator not replacement Safety',
               reference_future='Recorded future only in evaluator; missing later actor samples not invented'))
    a=read('reports/m6a_oracle_planner_regression.json')
    write('reports/m7a_historical_regression.json',dict(kind='OfflineReplayOfActualM6A_B_CARLA',provenance=provenance(),
      M6A=dict(report=asset('reports/m6a_oracle_planner_regression.json'),trials=a['trials'],count=a['total_trials']),
      M6B=dict(trials=execution,count=len(execution),initialization_failures=read('reports/m6b_acceptance.json')['trial_counts']['initialization_failure_logs']),
      frozen_snapshot_replays=snapshots,new_CARLA_trials=0,
      runtime=dict(control_interval_ms=quant(intervals),world_tick_ms=quant(ticks),worker_ms=quant(workers),
       deadline_miss=sum(r['deadline_miss'] for r in execution),skipped_slots=sum(r['skipped_slots'] for r in execution),
       expired_normal_control=sum(r['expired_normal_control'] for r in execution)),
      limits=['No new paired vehicle execution','All historical failure and initialization records included','A/B stop thresholds differ; no pooled success fraction','No claim of actuator improvement']))
    print('M6A',a['total_trials'],'M6B',len(execution),'snapshot replays',len(snapshots))

if __name__=='__main__':main()
