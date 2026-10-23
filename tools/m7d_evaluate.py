"""Future-only evaluator executed AFTER planner results exist; no feedback path."""
import argparse, json, time
from pathlib import Path
import numpy as np
import carla
from shapely.geometry import Polygon
from diagnostics.m7a_planning import center
from diagnostics.m7b_motion import interpolate
from diagnostics.m7d_scene import RoadIndex
from simulator.oracle_driver import footprint
from tools.m7d_common import *


def evaluate_record(record,future,roads):
    if not record.get('evaluations'):return dict(frame=record['frame'],input_index=record['input_index'],status='NoCandidate',evaluations=[])
    req=record['request'];scene=req['scene'];origin=scene['stamp']['timestamp_s'];fs=future['samples'];ts=np.array([f['sim_time']-origin for f in fs])
    complete=future['complete_six_seconds'] and len(fs)==121 and abs(ts[0])<1e-6 and abs(ts[-1]-6)<1e-5 and all(fs[i+1]['frame']==fs[i]['frame']+1 and abs(ts[i+1]-ts[i]-.05)<1e-5 for i in range(120))
    identities=[{str(a['actor_id']) for a in f['actors']} for f in fs]
    initial={b['object_id'] for b in scene['boxes']};missing=any(not initial<=ids for ids in identities)
    # New actors are included in evaluation even if absent at planning time.
    actorpolys=[[Polygon(footprint(*a['xy'],a['yaw'],a['half_length'],a['half_width'])) for a in f['actors']] for f in fs]
    es=[]
    for e in record['evaluations']:
        c=e['candidate'];ep=interpolate(c['points'],c['times_s'],ts);conflicts=[];gaps=[];roadfail=[]
        allowed={tuple(k) for k in e['allowed_lanes']};region=roads.region(allowed);body=[]
        for i,(p,f,ops) in enumerate(zip(ep,fs,actorpolys)):
            polygon=Polygon(footprint(*center(p,scene['body_center_offset']),p[2],*scene['body_half_extent']));body.append(polygon)
            if not region.covers(polygon):roadfail.append(i)
            for a,op in zip(f['actors'],ops):
                gap=polygon.distance(op);gaps.append(gap)
                if polygon.intersects(op):conflicts.append(dict(frame=f['frame'],time_s=float(ts[i]),actor_id=a['actor_id'],category=a.get('category','Unknown')))
        minimum=min(gaps) if gaps else None
        stopok=minimum is None or minimum>=3.-1e-8 or e['name']!='D_stop'
        safe=complete and not missing and not conflicts and not roadfail and stopok
        es.append(dict(name=e['name'],accepted=e['accepted'],selected=e['name']==record['selected'],
            RobustWithinHypotheses=e['dynamic']['robust'],SafeForRecordedFuture=safe if complete and not missing else 'Unknown',FeasibleToExecute='NotValidated',
            measured_sample_OBB_conflicts=conflicts,minimum_gap_m=minimum,stop_3m_preserved=stopok if e['name']=='D_stop' else 'NotApplicable',full_body_road_violation_samples=roadfail,
            finite_reference_safe_but_rejected=bool(safe and not e['accepted'] and e['geometry']['planning_feasible']),
            plan_progress_m=float(np.linalg.norm(ep[-1,:2]-ep[0,:2])),geometry_feasible=e['geometry']['planning_feasible']))
    errors=[]
    for horizon in [1.,3.,6.]:
        index=int(round(horizon/.05));f=fs[index]
        for a in f['actors']:
            hs=[h for h in record['hypotheses'] if h['object_id']==str(a['actor_id'])]
            vals=[(h['name'],float(np.linalg.norm(interpolate(h['poses'],h['times'],[horizon])[0,:2]-a['xy']))) for h in hs]
            cv=next((v for n,v in vals if n=='constant_speed_on_route'),None)
            errors.append(dict(actor_id=a['actor_id'],category=a.get('category','Unknown'),horizon_s=horizon,best_finite_hypothesis_error_m=min(v for _,v in vals) if vals else None,constant_speed_error_m=cv,hypotheses=len(vals)))
    return dict(frame=record['frame'],input_index=record['input_index'],complete_measured_future=complete,missing_original_actor=missing,
        new_actor_ids=sorted(set.union(*identities)-initial),evaluations=es,prediction_errors=errors,
        reference='ActualCARLA0.05sActorSnapshot_planarOBB_notMesh_notExactContinuousCollision')


def run(episode):
    resultpath=f'reports/m7d_replays/{episode}_stream.json';d=read(resultpath);m=read(f'reports/m7d_inputs/{episode}/manifest.json')['map'];roads=RoadIndex(carla.Map('Town03',Path(m['path']).read_text()));out=[]
    for r in d['records']:
        f=Path('reports/m7d_evaluator')/episode/f"{r['input_index']:05d}.json"
        out.append(evaluate_record(r,read(f),roads))
    write(f'reports/m7d_evaluations/{episode}.json',dict(kind='OfflineIndependentFutureReadOfMeasuredCARLA',provenance=provenance(),planner_output=asset(resultpath),records=out,
        limits='No future feedback to Planner. Sampled .05s OBB, not physical mesh or exact continuous truth. Ego candidate is counterfactual; actual target actors followed collection script and do not react to candidate Ego.'))
    print(episode,len(out),'accepted conflicts',sum(bool(e['measured_sample_OBB_conflicts']) for r in out for e in r['evaluations'] if e['accepted']))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',required=True);a=p.parse_args();run(a.episode)
