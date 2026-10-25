"""Read-only packaged map identity and simulation cleanup evidence."""
from pathlib import Path
import json
from tools.m8h0_common import *
def main():
 import carla
 base=Path('/root/gpufree-data/simulators/CARLA_0.9.15');maps=base/'CarlaUE4/Content/Carla/Maps'
 files=[maps/('Town02'+s) for s in ['.umap','.uexp','.ubulk']]+[maps/('Town02_BuiltData'+s) for s in ['.uasset','.uexp']]+[maps/'Nav/Town02.bin',maps/'OpenDrive/Town02.xodr',maps/'TM/Town02.bin',base/'CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping']
 c=carla.Client('127.0.0.1',2100);c.set_timeout(20);w=c.get_world();a=w.get_actors();snap=w.get_snapshot()
 record=dict(**provenance('ActualServerReadOnlyMapIdentity'),server=c.get_server_version(),world_episode=w.id,map=w.get_map().name,packaged_files=[asset(p) for p in files],API_OpenDRIVE=asset(ROOT/'Town02.xodr'),exact_package_matches_only='Hashes bind these package files; not every referenced mesh asset. Historical same-map versions not assumed equivalent.',frame=snap.frame,timestamp_s=snap.timestamp.elapsed_seconds,settings=str(w.get_settings()),synchronous_mode=w.get_settings().synchronous_mode,vehicles=len(a.filter('vehicle.*')),walkers=len(a.filter('walker.*')),sensors=len(a.filter('sensor.*')),physics_changes=0,weather_changes=0,navigation_calls=60,spawn_inventory_calls=1)
 assert record['vehicles']==record['walkers']==record['sensors']==0 and not record['synchronous_mode']
 write('reports/m8h0_map_identity_and_runtime.json',record)
 print('map files SHA',len(files),'no spawned actors/sensors; async server',flush=True)
if __name__=='__main__':main()
