"""Geometry-only allocation, heldoutRGB remains unrecorded until scientific freeze."""
import json
from pathlib import Path
import numpy as np,carla
from tools.m5j_common import write
from simulator.recorder import file_hash

def main():
    c=carla.Client('localhost',2100);c.set_timeout(120);old=[]
    for p in Path('configs').glob('*.json'):
        if p.name.startswith('m5j'):continue
        try:x=json.loads(p.read_text())
        except:continue
        if isinstance(x,dict):old += [g for g in x.get('groups',[])if 'pose'in g and 'map'in g]
    man=json.loads(Path('runs/m4_expanded_joint/manifest.json').read_text());ix=json.loads(Path(man['config']['index'],'index.json').read_text());groups=[]
    for j,town in enumerate(['Town01','Town03','Town04','Town05']):
        w=c.load_world(town);mp=w.get_map();avoid=[g['pose'][:2]for g in old+groups if g['map'].split('/')[-1]==town]
        for scene in ix['scenes']:
            if scene['map'].split('/')[-1]==town:avoid+=(np.array(scene['positions_xy'])*[1,-1]).tolist()
        choices=[]
        for i,p in enumerate(mp.get_spawn_points()):
            xy=np.array([p.location.x,p.location.y]);dist=min([np.linalg.norm(xy-v)for v in avoid]or[1000])
            if dist<45:continue
            wp=mp.get_waypoint(p.location);ahead=wp.next(15)
            if not ahead:continue
            nearjunction=wp.is_junction or any(v.is_junction for v in wp.next(8))
            choices.append((int(nearjunction),dist,i,p))
        if not choices:raise RuntimeError('No isolatedpose '+town)
        junction,dist,i,p=max(choices,key=lambda x:(x[0],x[1]));g=dict(group=f'{town}_jtest{j}',map=town,split='test',spawn_index=i,pose=[p.location.x,p.location.y,p.location.z,p.rotation.pitch,p.rotation.yaw,p.rotation.roll],min_prior_distance_m=float(dist),geometry='junction_near8m'if junction else'road',weather=['ClearNoon','CloudyNoon','ClearSunset','WetCloudySunset'][j]);groups.append(g);print(g,flush=True)
    car='vehicle.tesla.model3';van='vehicle.mercedes.sprinter';ped='walker.pedestrian.0001';cone='static.prop.trafficcone01'
    layouts=[[],[[8,0,car,.05]],[[8,-7,car,.05]],[[8,7,van,.05]],[[-8,-7,car,.05]],[[-8,7,van,.05]],[[-12,0,car,.05]],[[8,-6,ped,.85]],[[8,6,ped,.85]],[[-8,-6,ped,.85]],[[-8,6,ped,.85]],[[14,-9,cone,.05]],[[14,9,cone,.05]],[[-14,-8,cone,.05]],[[-14,8,cone,.05]],[[25,0,van,.05]],[[32,0,van,.05]],[[22,-10,van,.05]],[[25,1,ped,.85]],[]]
    write('configs/m5j_capture_test.json',dict(groups=groups,configurations=layouts,source_sha256=file_hash(__file__),coordinate='CARLA yright; avoidHSrecords converted',allocation='geometryonly prioritized nearjunction, no testRGB inspected'))
if __name__=='__main__':main()
