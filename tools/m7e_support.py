"""Actual collection support and geographic separation, no result tuning."""
import json
from pathlib import Path
import numpy as np
from tools.m7e_common import *
from datasets.m7d_temporal import from_packet
from diagnostics.m7e_trajectory import initial_state


def main():
    cfg=read('configs/m7e/collection.json');episodes=[];pairs=[]
    old={ep:np.array([json.loads(s)['ego']['xy'] for s in Path(f'reports/m7d_episodes/{ep}/trace.jsonl').read_text().splitlines()]) for ep in ['dev_constant_r3','dev_braking_r20']}
    for spec in cfg['episodes']:
        ep=spec['id'];root=Path('reports/m7e_episodes')/ep;rs=[json.loads(s) for s in (root/'trace.jsonl').read_text().splitlines()];xy=np.array([r['ego']['xy'] for r in rs]);signed=np.array([r['ego']['signed_speed'] for r in rs])
        man=read(f'reports/m7e_inputs/{ep}/manifest.json');state=[initial_state(from_packet(read(x['input']['path'])).request()) for x in man['records']]
        yaw=np.unwrap([r['ego']['yaw'] for r in rs]);v=np.array([np.linalg.norm(r['actors'][0]['velocity']) for r in rs]);a=np.diff(v)/.05
        episodes.append(dict(episode=ep,road=spec['road'],kind=spec['kind'],actual_rows=len(rs),snapshots=len(man['records']),duration=rs[-1]['sim_time']-rs[0]['sim_time'],actual_script_ego_distance_m=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),frame_gaps=sum(rs[i+1]['frame']!=rs[i]['frame']+1 for i in range(len(rs)-1)),
            max_abs_lateral_error_m=max(abs(s['lateral_error_m']) for s in state),max_abs_heading_error_rad=max(abs(s['heading_error_rad']) for s in state),minimum_signed_speed=float(signed.min()),yaw_span_rad=float(np.ptp(yaw)),lead_speed_minmax=[float(v.min()),float(v.max())],lead_acceleration_minmax=[float(a.min()),float(a.max())],actor_categories=sorted({a['category'] for r in rs for a in r['actors']}),collisions=read(root/'result.json')['collisions'],trace=asset(root/'trace.jsonl'),physics=asset(root/'physics.json'),snapshot_states=state))
        for dev,d in old.items():pairs.append(dict(final=ep,development=dev,minimum_ego_path_distance_m=float(np.linalg.norm(xy[:,None]-d[None],axis=-1).min())))
    freeze=read('reports/m7e_execution_logs/pre_final_freeze.json')['provenance']['sources'];protected=[p for p in freeze if p in ['diagnostics/m7e_trajectory.py','diagnostics/m7e_region.py','tools/m7e_replay.py','tools/m7e_collect.py','tools/m7e_prepare.py','configs/m7e/protocol.json','configs/m7e/collection.json']];checks={p:sha(p)==freeze[p] for p in protected}
    write('reports/m7e_fresh_episode_support.json',dict(kind='ActualCARLA0915IndependentScriptCollection',provenance=provenance(),new_episodes=episodes,historical_M7D_episodes=8,historical_M7D_snapshots=488,new_snapshot_count=sum(e['snapshots'] for e in episodes),pairs=pairs,freeze_checks=checks,all_frozen_unchanged=all(checks.values()),limits='SameTown03; roads seen in prior synthetic regression but new actual episodes and no M7E development use. Independent road IDs not independent cities. No F planner execution. Counterfactual actors do not react to plans. Actual acceleration can exceed finite assumed family.'))
    print('new',len(episodes),'freeze',checks,'min distance',min(p['minimum_ego_path_distance_m'] for p in pairs))
if __name__=='__main__':main()
