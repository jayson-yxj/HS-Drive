"""Locate exact failing footprint corners in saved M47 junction; no driving."""
import json,math
from pathlib import Path
import numpy as np
import carla
from simulator.m47_motion import safety_check
from simulator.motion_oracle import OracleState,Trajectory
from simulator.recorder import atomic_json,file_hash


def main():
    output=Path('reports/m48_road_diagnosis_actual_size.json')
    if output.exists():raise FileExistsError(output)
    c=carla.Client('127.0.0.1',2100);c.set_timeout(30)
    # Offline map parsing, independent of running world's map/tick owner.
    source=Path('reports/m47_routes_verified_artifacts/m47_junction_turn.jsonl')
    rows=[json.loads(x) for x in source.read_text().splitlines()]
    cmap=carla.Map('Town03',Path('/root/gpufree-data/simulators/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps/OpenDrive/Town03.xodr').read_text())
    samples=[]
    selected=next(i for i,r in enumerate(rows) if r['safety_reason']=='road_boundary')
    for i in [max(0,selected-10),selected,min(selected+50,len(rows)-1),len(rows)-1]:
        r=rows[i];bad=[]
        def check(x,y):
            wp=cmap.get_waypoint(carla.Location(x=x,y=-y,z=r['measurement']['position'][2]),project_to_road=False,lane_type=carla.LaneType.Driving)
            if wp is None:
                nearest=cmap.get_waypoint(carla.Location(x=x,y=-y,z=r['measurement']['position'][2]),project_to_road=True,lane_type=carla.LaneType.Any)
                bad.append(dict(x=x,y=y,nearest=[nearest.transform.location.x,-nearest.transform.location.y],lane_width=nearest.lane_width,lane_type=str(nearest.lane_type),junction=nearest.is_junction,road_id=nearest.road_id,lane_id=nearest.lane_id))
            return wp is not None
        state=OracleState(r['x'],r['y'],r['yaw'],r['v'],r['t'])
        reason=safety_check(Trajectory(np.array(r['trajectory']),r['t'],r['trajectory_dt']),state,[],half_length=2.395889759,half_width=1.081725,road_predicate=check)
        samples.append(dict(index=i,x=r['x'],y=r['y'],yaw=r['yaw'],v=r['v'],reason=reason,bad=bad,trajectory=r['trajectory']))
    atomic_json(output,dict(kind='offline_saved_states_exact_opendrive_queries',samples=samples,log_sha256=file_hash(source),source_sha256=file_hash(__file__)))
    print([(r['index'],r['v'],r['bad']) for r in samples])
if __name__=='__main__':main()
