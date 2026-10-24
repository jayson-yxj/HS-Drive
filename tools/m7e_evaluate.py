"""Evaluator-only actual future reference, separated from frozen planner process."""
import argparse
from pathlib import Path
import numpy as np
import carla
from tools.m7d_evaluate import evaluate_record
from diagnostics.m7d_scene import RoadIndex
from diagnostics.m7b_motion import interpolate
from diagnostics.m7a_planning import center
from shapely.geometry import Polygon
from simulator.oracle_driver import footprint
from tools.m7e_common import *


def run(episode,input_root='reports/m7d_inputs',future_root='reports/m7d_evaluator'):
    path=f'reports/m7e_replays/{episode}_B.json';d=read(path);m=read(Path(input_root)/episode/'manifest.json')['map'];roads=RoadIndex(carla.Map('Town03',Path(m['path']).read_text()));out=[]
    for r in d['records']:
        f=read(Path(future_root)/episode/f"{r['input_index']:05d}.json");res=evaluate_record(r,f,roads)
        # F/E inherits conservative slow reserve; also audit on actual observed futures.
        for e in res['evaluations']:
            if e['name'] not in ['E_slow_follow','F_state_anchored_slow']:continue
            c=next(x['candidate'] for x in r['evaluations'] if x['name']==e['name']);s=r['request']['scene'];ts=[x['sim_time']-f['origin_timestamp'] for x in f['samples']];poses=interpolate(c['points'],c['times_s'],ts);margins=[]
            for p,row in zip(poses,f['samples']):
                ego=Polygon(footprint(*center(p,s['body_center_offset']),p[2],*s['body_half_extent']))
                for a in row['actors']:
                    ob=Polygon(footprint(*a['xy'],a['yaw'],a['half_length'],a['half_width']))
                    margins.append(ego.distance(ob)-(3+p[3]**2/3.6+p[3]*1.8/2.5))
            e['recorded_slow_reserve_min_m']=min(margins) if margins else None
            e['recorded_slow_reserve_passed']=bool(not margins or min(margins)>=-1e-8)
        out.append(res)
    write(f'reports/m7e_evaluations/{episode}.json',dict(kind='ActualRecordedCARLAFuture_OfflineOBBAudit',provenance=provenance(),planner_output=asset(path),records=out,
        limits='Measured .05s OBB samples, candidate interpolated; not exact continuous Mesh collision. Counterfactual Ego not executed, target actors do not react to candidate.'))
    print(episode,'accepted',sum(e['accepted'] for r in out for e in r['evaluations']),'conflicts',sum(e['accepted'] and bool(e['measured_sample_OBB_conflicts']) for r in out for e in r['evaluations']))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',required=True);p.add_argument('--input-root',default='reports/m7d_inputs');p.add_argument('--future-root',default='reports/m7d_evaluator');a=p.parse_args();run(a.episode,a.input_root,a.future_root)
