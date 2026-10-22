"""Geometry-only allocation of map-isolated location groups and randomized layouts."""
import argparse, json, math
from pathlib import Path
import numpy as np
from tools.m5o_common import write, provenance


def layouts(seed):
    rng=np.random.default_rng(seed)
    vehicles=['vehicle.tesla.model3','vehicle.mercedes.sprinter','vehicle.audi.tt','vehicle.lincoln.mkz_2020']
    person='walker.pedestrian.0001';cone='static.prop.trafficcone01'
    result=[[]]
    # Balanced bearings; distances/yaws/types vary independently by location.
    for i in range(18):
        theta=math.radians([0,55,110,180,-110,-55][i%6]+rng.uniform(-12,12))
        dist=[7,14,25][i//6]+rng.uniform(-1.5,1.5)
        kind=vehicles[int(rng.integers(len(vehicles)))]
        result.append([[float(dist*math.cos(theta)),float(dist*math.sin(theta)),kind,0.05,float(rng.uniform(-160,160))]])
    for i in range(6):
        theta=math.radians([0,55,110,180,-110,-55][i]);dist=float(rng.uniform(5,17))
        result.append([[dist*math.cos(theta),dist*math.sin(theta),person if i%2==0 else cone,.05,0]])
    # Multi-actor / partial occlusion; shared pattern families disclosed, not independent scenes.
    for yaw in [0,90,180,-90]:
        theta=math.radians(yaw);near=float(rng.uniform(8,12));far=near+float(rng.uniform(4,8))
        result.append([[near*math.cos(theta),near*math.sin(theta),vehicles[0],.05,float(rng.uniform(-30,30))],
                       [far*math.cos(theta)-.8*math.sin(theta),far*math.sin(theta)+.8*math.cos(theta),vehicles[1],.05,0],
                       [(near+3)*math.cos(theta)+2*math.sin(theta),(near+3)*math.sin(theta)-2*math.cos(theta),person,.05,0]])
    result.extend([[[3,3,vehicles[2],.05,70]],[]])
    return result


def main():
    import carla
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['dev','sealed'],required=True);a=p.parse_args()
    if a.phase=='sealed':assert Path('reports/m5o_pretest_freeze.json').exists()
    client=carla.Client('127.0.0.1',2100);client.set_timeout(120)
    plan=[('train','Town01',6),('train','Town03',7),('train','Town04',7),('val','Town05',6),('calibration','Town02',6)] if a.phase=='dev' else [('sealed','Town10HD',6)]
    old=[]
    for path in Path('configs').glob('*.json'):
        try:d=json.loads(path.read_text())
        except ValueError:continue
        if isinstance(d,dict):old.extend(g for g in d.get('groups',[]) if isinstance(g,dict) and 'pose' in g and 'map' in g)
    groups=[];audit=[]
    for split,town,n in plan:
        world=client.load_world(town);mp=world.get_map()
        avoid=[np.array(g['pose'][:2]) for g in old if g['map'].split('/')[-1]==town]
        points=[]
        for wp in mp.generate_waypoints(8):
            if wp.lane_type!=carla.LaneType.Driving:continue
            t=wp.transform;xy=np.array([t.location.x,t.location.y])
            prior=min([np.linalg.norm(xy-v) for v in avoid] or [1000])
            if prior<45:continue
            pitch=math.degrees(math.asin(math.sin(math.radians(t.rotation.pitch))))
            points.append((wp,xy,float(prior),pitch))
        chosen=[]
        for i in range(n):
            viable=[r for r in points if min([np.linalg.norm(r[1]-v[1]) for v in chosen] or [1000])>=90]
            if not viable:break
            # Alternate slope/junction/straight preferences without rendered selection.
            def score(r):
                dist=min([np.linalg.norm(r[1]-v[1]) for v in chosen] or [150])
                preference=(abs(r[3])*10 if i%3==0 else 35*int(r[0].is_junction) if i%3==1 else 15*int(not r[0].is_junction))
                return preference+min(dist,300)+r[2]*.05
            r=max(viable,key=score);chosen.append(r);wp,xy,prior,pitch=r;t=wp.transform
            number=len(groups);seed=(53000 if a.phase=="sealed" else 51000)+number
            g=dict(group=f'{town}_m5o_{split}_{i:02d}',split=split,map=town,
                   pose=[t.location.x,t.location.y,t.location.z+.4,t.rotation.pitch,t.rotation.yaw,t.rotation.roll],
                   road_id=wp.road_id,lane_id=wp.lane_id,road_s=wp.s,is_junction=wp.is_junction,
                   min_prior_m=prior,actual_pitch_deg=pitch,geometry='slope' if abs(pitch)>1 else 'junction' if wp.is_junction else 'flat_road',
                   weather=['ClearNoon','CloudyNoon','ClearSunset'][i%3],layout_seed=seed,configurations=layouts(seed))
            groups.append(g);print(g['group'],g['geometry'],flush=True)
        audit.append(dict(split=split,map=town,requested=n,allocated=len(chosen)))
    write(f'configs/m5o_capture_{a.phase}.json',dict(groups=groups,allocation=audit,source=provenance(),min_intergroup_m=90,min_historical_pose_m=45,split_isolation='disjoint maps; layout families shared but continuous placement/yaw/type seeded per group',configurations=[]))
    assert all(x['requested']==x['allocated'] for x in audit),audit

if __name__=='__main__':main()
