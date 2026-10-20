"""Multi-map OpenDRIVE line certificate audit against full CARLA lookup."""
import json,time
from pathlib import Path
import numpy as np,carla
from simulator.m48_road import CertifiedRoad
from simulator.recorder import atomic_json,file_hash


def main():
    dest=Path('reports/m48_road_audit_v2.json')
    if dest.exists():raise FileExistsError(dest)
    root=Path('/root/gpufree-data/simulators/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps/OpenDrive');rng=np.random.default_rng(48);records=[]
    for town in ['Town01','Town02','Town03','Town04']:
        path=root/(town+'.xodr');cmap=carla.Map(town,path.read_text());road=CertifiedRoad(cmap);points=[];kinds=[]
        ways=cmap.generate_waypoints(4)
        for index in rng.choice(len(ways),min(500,len(ways)),replace=False):
            w=ways[index];tf=w.transform;left=np.array([-tf.get_forward_vector().y,-tf.get_forward_vector().x]);xy=np.array([tf.location.x,-tf.location.y])
            for off in [-w.lane_width,-w.lane_width/2-.05,-w.lane_width/2+.05,0,w.lane_width/2-.05,w.lane_width/2+.05,w.lane_width]:
                points.append(xy+off*left);kinds.append('junction' if w.is_junction else 'lane_boundary')
        for origin,d,left,length,half in road.rectangles[:300]:
            for a in [0.1,length/2,length-.1]:
                for b in [-half-.02,-half+.02,0,half-.02,half+.02]:points.append(origin+a*d+b*left);kinds.append('certificate_boundary')
        points=np.array(points);fast=road.certified(points);start=time.perf_counter();exact=np.array([road.exact(*p) for p in points]);ms=(time.perf_counter()-start)*1000
        records.append(dict(map=town,points=len(points),rectangles=len(road.rectangles),certified=int(fast.sum()),false_accept_against_api=int((fast&~exact).sum()),false_reject_fast=int((~fast&exact).sum()),init_ms=road.init_ms,exact_total_ms=ms,map_sha256=file_hash(path),by_region={k:dict(n=kinds.count(k),false_accept=int((fast&~exact&np.array([x==k for x in kinds])).sum())) for k in set(kinds)},counterexamples=points[fast&~exact][:20].tolist()))
        print(town,records[-1]['false_accept_against_api'],flush=True)
    atomic_json(dest,dict(kind='offline_geometry_actual_carla_map_api',records=records,source_sha256=file_hash('simulator/m48_road.py'),audit_sha256=file_hash(__file__),certificate_enabled_in_closedloop=False,limitations=['Exact-map API is road semantics, not physical drivable mesh.','Overlapping roads/elevations and routing legality require extra treatment; candidates remain disabled.','No certificate for curves or junctions; exact fallback, including holes.']))
if __name__=='__main__':main()
