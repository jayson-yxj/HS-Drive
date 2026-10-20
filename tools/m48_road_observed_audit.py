"""Exact full-map current-footprint audit of actual saved trajectories."""
import json
from pathlib import Path
import carla,numpy as np
from simulator.oracle_driver import footprint
from simulator.recorder import atomic_json,file_hash

def main():
    out=Path('reports/m48_road_observed_audit.json')
    if out.exists():raise FileExistsError(out)
    source=Path('/root/gpufree-data/simulators/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps/OpenDrive/Town03.xodr');cmap=carla.Map('Town03',source.read_text());records=[]
    for name in ['reports/m47_routes_verified.json','reports/m48_junction_pilot_v3.json','reports/m48_junction_short.json']:
        for r in json.loads(Path(name).read_text())['results']:
            if r['version']=='m46':continue
            rows=[json.loads(s) for s in Path(r['log']).read_text().splitlines()];count=0;violations=[]
            for row in rows:
                corners=footprint(row['x'],row['y'],row['yaw'],2.395889759+.2,1.081725+.15)
                bad=[p.tolist() for p in corners if cmap.get_waypoint(carla.Location(x=float(p[0]),y=-float(p[1]),z=row['measurement']['position'][2]),project_to_road=False,lane_type=carla.LaneType.Driving) is None]
                count+=bool(bad)
                if bad and len(violations)<3:violations.append(dict(t=row['t'],points=bad))
            records.append(dict(report=name,scenario=r['scenario'],frames=len(rows),current_footprint_boundary_frames=count,examples=violations,log_sha256=file_hash(r['log'])))
    atomic_json(out,dict(kind='offline_actual_closedloop_footprint_against_full_map',records=records,map_sha256=file_hash(source),source_sha256=file_hash(__file__),scope='Four footprint corners with unchanged safety margins; not full polygon or continuous road sweep certification.'))
if __name__=='__main__':main()
