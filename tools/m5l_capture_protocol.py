"""New sealed geometry-only allocation after all312 historicaldiagnostics; no RGB selection."""
import json,math
from pathlib import Path
import numpy as np,carla
from tools.m5l_common import write
from simulator.recorder import file_hash

def main():
 c=carla.Client('localhost',2100);c.set_timeout(120);old=[]
 for p in Path('configs').glob('*.json'):
  try:d=json.loads(p.read_text())
  except (ValueError,OSError):continue
  if isinstance(d,dict):old +=[g for g in d.get('groups',[])if 'pose'in g and 'map'in g]
 man=json.loads(Path('runs/m4_expanded_joint/manifest.json').read_text());ix=json.loads(Path(man['config']['index'],'index.json').read_text());groups=[]
 for i,town in enumerate(['Town01','Town03','Town04','Town05']):
  w=c.load_world(town);mp=w.get_map();avoid=[r['pose'][:2]for r in old if r['map'].split('/')[-1]==town]
  for scene in ix['scenes']:
   if scene['map'].split('/')[-1]==town:avoid+=(np.array(scene['positions_xy'])*[1,-1]).tolist()
  choices=[]
  for wp in mp.generate_waypoints(8):
   p=wp.transform;dist=min([np.linalg.norm(np.array([p.location.x,p.location.y])-xy)for xy in avoid]or[1000]);slope=abs(p.rotation.pitch)
   if dist>=45 and wp.next(30):choices.append((slope,dist,p,wp.is_junction))
  assert choices,town
  slope,dist,p,junction=max(choices,key=lambda v:(v[0]if i in [1,2]else-v[0],v[1]));groups.append(dict(group=f'{town}_ltest{i}',map=town,split='test',pose=[p.location.x,p.location.y,p.location.z+.4,p.rotation.pitch,p.rotation.yaw,p.rotation.roll],min_prior_m=float(dist),map_pitch_deg=float(p.rotation.pitch),geometry='slope'if slope>1 else'flat_road',weather=['ClearNoon','CloudyNoon','ClearSunset','WetCloudySunset'][i]));print(groups[-1],flush=True)
 car='vehicle.tesla.model3';van='vehicle.mercedes.sprinter';ped='walker.pedestrian.0001';cone='static.prop.trafficcone01'
 def at(r,angle,kind,z):return [r*math.cos(math.radians(angle)),r*math.sin(math.radians(angle)),kind,z]
 layouts=[[]]+[[at(25,a,van,.05)]for a in [0,-60,60,-120,120,180]]+[[at(r,a,ped,.85)]for r,a in [(21,3),(23,-57),(25,63),(23,-117),(21,123),(25,177)]]+[[at(r,a,cone,.05)]for r,a in [(8,3),(13,-57),(20,63)]]+[[at(14,0,car,.05),at(24,0,ped,.85)],[at(15,-60,car,.05),at(26,-60,van,.05)],[]]
 write('configs/m5l_capture_test.json',dict(groups=groups,configurations=layouts,source_sha256=file_hash(__file__),limits='Geometry-only slope/flat selection; fixednewazimuths not chosenbytestRGB. Actorsplaced relativeegoplane legacycapture mayfloat/intersect onvaryingterrain; muststratify actualbottomGT in evaluation, no Meshgroundingclaim. No newTrainVal dueP4failure.'))
if __name__=='__main__':main()
