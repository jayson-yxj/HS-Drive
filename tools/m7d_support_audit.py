"""Measured geographic/physical support and frozen-source checks, no retuning."""
import json
from pathlib import Path
import numpy as np
from tools.m7d_common import *


def main():
    cfg=read('configs/m7d/protocol.json');episodes={}
    for spec in cfg['episodes']:
        ep=spec['id'];root=Path('reports/m7d_episodes')/ep;rs=[json.loads(s) for s in (root/'trace.jsonl').read_text().splitlines()]
        xy=np.array([r['ego']['xy'] for r in rs]);s=np.array([r['ego']['signed_speed'] for r in rs]);v=np.array([np.linalg.norm(r['actors'][0]['velocity']) for r in rs]);a=np.diff(v)/.05
        episodes[ep]=dict(split=spec['split'],road=spec['road'],rows=len(rs),actual_ego_distance_m=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),ego_signed_speed_minmax=[float(s.min()),float(s.max())],lead_speed_minmax=[float(v.min()),float(v.max())],lead_measured_acceleration_minmax=[float(a.min()),float(a.max())],lead_acceleration_outside_declared_family_samples=int(np.sum((a< -3)|(a>1.5))),physics=asset(root/'physics.json'),xy=xy)
    pairs=[]
    for final,f in episodes.items():
        if f['split']!='fresh_final':continue
        for dev,d in episodes.items():
            if d['split']!='development':continue
            dist=float(np.linalg.norm(f['xy'][:,None,:]-d['xy'][None,:,:],axis=-1).min())
            pairs.append(dict(final=final,development=dev,minimum_recorded_ego_distance_m=dist))
    for d in episodes.values():d.pop('xy')
    frozen=read('reports/m7d_execution_logs/pre_final_freeze.json')['provenance']['sources'];keys=[p for p in frozen if p.startswith(('diagnostics/','datasets/')) or p in ['tools/m7d_replay.py','tools/m7d_collect.py','tools/m7d_prepare.py','configs/m7d/protocol.json']]
    checked={p:sha(p)==frozen[p] for p in keys}
    write('reports/m7d_support_audit.json',dict(kind='ActualRecordedCARLAAndSourceSHAAnalysis',provenance=provenance(),episodes=episodes,geographic_pairs=pairs,frozen_planning_collection_protocol_unchanged=checked,all_frozen_unchanged=all(checked.values()),interpretation='Measured script accelerations may leave finite assumed family. No inference of robustness outside family. SameTown03, overlapping temporal windows and shared script templates; no independent-scene confidence interval.'))
    print(pairs,checked)
if __name__=='__main__':main()
