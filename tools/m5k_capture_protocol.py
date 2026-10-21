"""Geometry-only disjoint allocation and target-distance configurations."""
import json,math
from pathlib import Path
import numpy as np,carla
from tools.m5k_common import write
from simulator.recorder import file_hash

def main():
 c=carla.Client('localhost',2100);c.set_timeout(120);old=[]
 for p in Path('configs').glob('*.json'):
  try:x=json.loads(p.read_text())
  except:continue
  if isinstance(x,dict):old += [g for g in x.get('groups',[])if 'pose'in g and 'map'in g]
 man=json.loads(Path('runs/m4_expanded_joint/manifest.json').read_text());ix=json.loads(Path(man['config']['index'],'index.json').read_text());groups=[]
 for ti,town in enumerate(['Town01','Town03','Town04','Town05']):
  w=c.load_world(town);mp=w.get_map();avoid=[g['pose'][:2]for g in old if g['map'].split('/')[-1]==town]
  for scene in ix['scenes']:
   if scene['map'].split('/')[-1]==town:avoid+=(np.array(scene['positions_xy'])*[1,-1]).tolist()
  for split in ['train','val','test']:
   choices=[]
   for i,p in enumerate(mp.get_spawn_points()):
    xy=np.array([p.location.x,p.location.y]);dist=min([np.linalg.norm(xy-v)for v in avoid]or[1000])
    if dist<45:continue
    wp=mp.get_waypoint(p.location)
    if not wp.next(30):continue
    junction=wp.is_junction or any(v.is_junction for v in wp.next(12));choices.append((dist,int(junction),i,p))
   if not choices:raise RuntimeError('No isolatedpose '+town+' '+split)
   dist,junction,i,p=max(choices,key=lambda x:(x[1],x[0]));g=dict(group=f'{town}_k{split}{ti}',map=town,split=split,spawn_index=i,pose=[p.location.x,p.location.y,p.location.z,p.rotation.pitch,p.rotation.yaw,p.rotation.roll],min_prior_distance_m=float(dist),geometry='junction_near12m'if junction else'road',weather=['ClearNoon','CloudyNoon','ClearSunset','WetCloudySunset'][ti]);groups.append(g);avoid.append(g['pose'][:2]);print(g,flush=True)
 car='vehicle.tesla.model3';van='vehicle.mercedes.sprinter';ped='walker.pedestrian.0001';cone='static.prop.trafficcone01'
 def at(r,a,kind,z):return [round(r*math.cos(math.radians(a)),3),round(r*math.sin(math.radians(a)),3),kind,z]
 layouts=[[]]+[[at(25,a,van,.05)]for a in [0,-60,60,-120,120,180]]+[[at(r,a,car,.05)]for r,a in [(17,0),(18,60),(32,0),(32,-60)]]+[[at(23,a,ped,.85)]for a in [0,-60,60,-120,120,180]]+[[at(17,a,ped,.85)]for a in [-60,60]]+[[at(r,a,cone,.05)]for r,a in [(7,0),(10,60),(14,-60),(20,0),(10,180)]]+[[at(15,0,car,.05),at(25,0,ped,.85)],[]]
 for splitset,name in [(['train','val'],'dev'),(['test'],'test')]:write(f'configs/m5k_capture_{name}.json',dict(groups=[g for g in groups if g['split']in splitset],configurations=layouts,source_sha256=file_hash(__file__),allocation='geometry only, all old config poses and M4 replay avoided>=45m. TestRGB uncaptured until freeze. Nominal placements can be occluded or outsideBEV; actual eligibility measured.',coordinate='CARLA yright'))
 write('reports/m5k_position_allocation.json',dict(prior_groups=len(old),groups=groups,configurations=len(layouts),kind='actual_CARLA_map_geometry_no_test_images',source_sha256=file_hash(__file__)))
if __name__=='__main__':main()
