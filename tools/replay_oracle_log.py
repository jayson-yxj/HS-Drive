"""Recorded-state controller replay and segment cross-track audit, without CARLA."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulator.oracle_driver import OracleState,Trajectory,PurePursuitPID
from simulator.recorder import atomic_json,file_hash


def cross_track(points,route):
    start=route[:-1,:2];delta=np.diff(route[:,:2],axis=0)
    offset=points[:,None,:]-start[None]
    t=np.clip((offset*delta).sum(-1)/np.maximum((delta*delta).sum(-1),1e-10),0,1)
    return np.linalg.norm(offset-t[:,:,None]*delta,axis=-1).min(-1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifacts',default='reports/m45_oracle_driving_artifacts');p.add_argument('--output',default='reports/m45_oracle_replay.json');p.add_argument('--figure',default='reports/m45_oracle_driving_paths.png');a=p.parse_args()
    if Path(a.output).exists() or Path(a.figure).exists():raise FileExistsError('New paths required')
    results={};files=sorted(Path(a.artifacts).glob('*_controls.jsonl'));fig,axes=plt.subplots(3,3,figsize=(12,11))
    for index,path in enumerate(files):
        name=path.stem.removesuffix('_controls');rows=[json.loads(line) for line in path.read_text().splitlines()];route=np.array(json.loads((path.parent/(name+'_route.json')).read_text())['route_world_hs'])
        controller=PurePursuitPID();error=0.
        for r in rows:
            state=OracleState(r['x'],r['y'],r['yaw'],r['speed_mps'],r['timestamp_s'])
            tr=None if r['trajectory'] is None else Trajectory(np.array(r['trajectory'],float),r['trajectory_timestamp_s'])
            actual=controller.command(tr,state,r['safety_reason'])
            error=max(error,max(abs(actual[k]-r['command'][k]) for k in ('steer','throttle','brake')))
            if actual['reason']!=r['command']['reason']:raise AssertionError('Replay reason changed')
        xy=np.array([[r['x'],r['y']] for r in rows]);ct=cross_track(xy,route)
        results[name]={'ticks':len(rows),'controller_replay_max_abs_error':error,'segment_cross_track_mean_m':float(ct.mean()),'segment_cross_track_max_m':float(ct.max()),'source_sha256':file_hash(path)}
        ax=axes.flat[index];ax.plot(route[:,0],route[:,1],label='Route');ax.plot(xy[:,0],xy[:,1],label='Actual');ax.set_title(name);ax.set_aspect('equal');ax.set_xlabel('World HS X (m)');ax.set_ylabel('World HS Y (m)')
    axes.flat[0].legend();fig.tight_layout();fig.savefig(a.figure);plt.close(fig)
    atomic_json(a.output,dict(schema_version='hsd-m45-oracle-replay-v1',passed=all(r['controller_replay_max_abs_error']<1e-12 for r in results.values()),results=results,scope='Exact controller replay at recorded states, not physics re-simulation; live rerun uses --replay-routes. Original lateral_error fields are nearest-waypoint Euclidean distance; use segment_cross_track fields here for lateral tracking.'))


if __name__=='__main__':main()
