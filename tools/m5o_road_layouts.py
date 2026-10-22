"""Pre-render road-supported target placement; no RGB/model selection."""
import argparse,json,math
from pathlib import Path
import numpy as np
from tools.m5o_common import write,provenance


def main():
    import carla
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['dev','sealed'],required=True);a=p.parse_args()
    source=f'configs/m5o_capture_{a.phase}_v2.json' if a.phase=='dev' else 'configs/m5o_capture_sealed.json'
    data=json.loads(Path(source).read_text());c=carla.Client('localhost',2100);c.set_timeout(120);current=None
    for g in data['groups']:
        if current!=g['map']:world=c.load_world(g['map']);mp=world.get_map();wps=mp.generate_waypoints(2);current=g['map']
        xyz=g['pose'][:3];yaw=math.radians(g['pose'][4]);R=np.array([[math.cos(yaw),math.sin(yaw)],[-math.sin(yaw),math.cos(yaw)]])
        candidates=[]
        for wp in wps:
            pos=wp.transform.location;local=R@(np.array([pos.x,pos.y])-xyz[:2]);dist=np.linalg.norm(local)
            if not 5<dist<38 or abs(pos.z-xyz[2])>4:continue
            angle=math.atan2(local[1],local[0]);sector=int((angle+math.pi)/(math.pi/3))%6
            candidates.append((wp,local,sector,int(dist//10)))
        rng=np.random.default_rng(g['layout_seed']);rng.shuffle(candidates);selected=[]
        # Unique local actor positions >2m, balanced sector/distance bins where road exists.
        for _ in range(5):
            for sector in range(6):
                options=[r for r in candidates if r[2]==sector and all(np.linalg.norm(r[1]-v[1])>2 for v in selected)]
                if options:
                    count={b:sum(v[3]==b for v in selected) for b in range(4)}
                    selected.append(min(options,key=lambda r:count[r[3]]))
        if len(selected)<20:
            for r in candidates:
                if all(np.linalg.norm(r[1]-v[1])>2 for v in selected):selected.append(r)
                if len(selected)>=30:break
        cars=['vehicle.tesla.model3','vehicle.audi.tt','vehicle.lincoln.mkz_2020','vehicle.mercedes.sprinter']
        configs=[[]]
        for i,(wp,local,sector,band) in enumerate(selected):
            kind=cars[i%4] if i%6<4 else 'walker.pedestrian.0001' if i%6==4 else 'static.prop.trafficcone01'
            rotation=(wp.transform.rotation.yaw-g['pose'][4]+float(rng.uniform(-25,25))+180)%360-180
            configs.append([[float(local[0]),float(local[1]),kind,.05,rotation]])
        # Four extra multi-actor pairs, same road geometry, depth separated; not duplicates.
        for i in range(min(4,len(selected)//2)):
            x=selected[i][1];y=selected[-i-1][1]
            if np.linalg.norm(x-y)>7:configs.append([[float(x[0]),float(x[1]),cars[0],.05,0],[float(y[0]),float(y[1]),cars[3],.05,45]])
        configs.append([]);g['configurations']=configs;g['road_supported_positions']=len(selected);g['road_sectors']=sorted({r[2] for r in selected})
        print(g['group'],len(configs),g['road_sectors'],flush=True)
    write(f'configs/m5o_capture_{a.phase}_v3.json',dict(data,source=provenance(),placement='map road-supported unique position grid; same layout families; physics settling and collision-ground audit still mandatory'))
if __name__=='__main__':main()
