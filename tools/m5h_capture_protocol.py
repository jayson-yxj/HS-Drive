"""Allocate unseen positions using map geometry, without rendered test observations."""
import json
from pathlib import Path
import carla,numpy as np
from simulator.recorder import atomic_json,file_hash

def main():
    client=carla.Client('localhost',2100);client.set_timeout(120);old=[]
    for f in ['m5a_protocol','m5b_protocol','m5c_protocol','m5d_protocol','m5e_protocol','m5f_capture_protocol','m5g_capture_protocol']:
        old+=json.loads(Path('configs',f+'.json').read_text())['groups']
    groups=[]
    plan=[('Town04','train'),('Town05','train')]+[(m,'test')for m in ['Town01','Town02','Town03','Town04','Town05','Town10HD']]
    for j,(town,split) in enumerate(plan):
        world=client.load_world(town);m=world.get_map();avoid=[g['pose'][:2]for g in old+groups if g['map']==town];choices=[]
        for i,p in enumerate(m.get_spawn_points()):
            distance=min([float(np.linalg.norm(np.array([p.location.x,p.location.y])-a))for a in avoid]or[1000])
            if distance<45:continue
            w=m.get_waypoint(p.location);ahead=w.next(15)
            if not ahead:continue
            target=p.transform(carla.Location(x=15))
            if m.get_waypoint(target,project_to_road=False,lane_type=carla.LaneType.Driving) is None:continue
            category='junction'if w.is_junction or any(x.is_junction for x in ahead)else'curve'if abs((ahead[0].transform.rotation.yaw-p.rotation.yaw+180)%360-180)>8 else'straight'
            choices.append((distance,i,p,category))
        if not choices:raise RuntimeError('No independent pose '+town)
        dist,i,p,category=max(choices,key=lambda x:x[0]);g=dict(group=f'{town}_h{split}',map=town,split=split,spawn_index=i,pose=[p.location.x,p.location.y,p.location.z,p.rotation.pitch,p.rotation.yaw,p.rotation.roll],geometry=category,min_previous_pose_distance_m=dist,weather=['ClearNoon','WetCloudySunset','ClearSunset','CloudyNoon'][j%4]);groups.append(g);print(g,flush=True)
    car='vehicle.tesla.model3';ped='walker.pedestrian.0001';cone='static.prop.trafficcone01'
    train=[[],[[-8,8,car,.05]],[[8,8,car,.05]],[[-12,-8,car,.05]],[[12,0,car,.05]],[[8,6,ped,.85]],[[-7,6,cone,.05]],[]]
    test=[[],[[10,0,car,.05]],[[25,0,car,.05]],[[32,0,car,.05]],[[10,10,car,.05]],[[-10,8,car,.05]],[[-15,0,car,.05]],[[15,1,ped,.85]],[[15,-1,cone,.05]],[[12,0,'vehicle.mercedes.sprinter',.05],[28,0,car,.05]],[[-10,-8,car,.05]],[]]
    for split,layout in [('train',train),('test',test)]:
        dest=Path(f'configs/m5h_capture_{split}.json')
        if dest.exists():raise FileExistsError(dest)
        atomic_json(dest,dict(schema='hsd-m5h-capture-v1',groups=[g for g in groups if g['split']==split],configurations=layout,source_sha256=file_hash(__file__),role='Geometryallocationonly; no test observations beforefreeze'))
if __name__=='__main__':main()
