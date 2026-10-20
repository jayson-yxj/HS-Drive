"""Audit a fast map corridor candidate before any closed-loop adoption."""
import json,time,argparse
from pathlib import Path
import numpy as np
from simulator.m47_road import RouteRoad
from simulator.oracle_driver import footprint
from simulator.recorder import atomic_json,file_hash


def main():
    import carla
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",default="reports/m47_road_profile.json");a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError(a.output)
    c=carla.Client("127.0.0.1",2100);c.set_timeout(30);w=c.get_world();cmap=w.get_map()
    if not cmap.name.endswith("Town03"):raise ValueError("Wrong map")
    root=Path("reports/m46_routes_verified_artifacts")
    route=np.array(json.loads((root/"m46_junction_turn_route.json").read_text())["route"])
    road=RouteRoad(cmap,route)
    rows=[json.loads(x) for x in (root/"m46_junction_turn.jsonl").read_text().splitlines()]
    timings=[];false_accept=[];total=0
    for row in rows[::10]:
        p=np.asarray(row["trajectory"]);cs,sn=np.cos(row["yaw"]),np.sin(row["yaw"])
        xy=p[:,:2]@np.array([[cs,sn],[-sn,cs]])+[row["x"],row["y"]]
        points=np.concatenate([footprint(*q,row["yaw"]+r[2],2.59589,1.231725) for q,r in zip(xy,p)])
        start=time.perf_counter();fast=road.batch(points);ms=(time.perf_counter()-start)*1000
        # Exact map checks are kept separate from timing, including accepted interior.
        exact=np.array([road.exact(x,y) for x,y in points])
        bad=fast&~exact;false_accept.extend(points[bad].tolist());total+=len(points)
        timings.append(ms)
    atomic_json(a.output,dict(kind="offline_recorded_trajectory_live_map_candidate",startup_ms=road.init_ms,
        p95_batch_ms=float(np.quantile(timings,.95)),total_points=total,false_accept_count=len(false_accept),
        false_accept_examples=false_accept[:30],adopted=False,
        status="CandidateOnly" if not false_accept else "Rejected",
        reason="Finite route sample audit cannot certify corridor over junctions, lane direction and disjoint road polygons.",
        source_sha256=file_hash("simulator/m47_road.py")))


if __name__=="__main__":main()
