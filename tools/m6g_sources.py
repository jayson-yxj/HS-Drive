"""Finite official-source audit; no UE download, compilation or simulator calls."""
import base64
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request
import urllib.error


def api(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'HS-Drive-M6G-source-audit'}),timeout=25) as r:return json.load(r)


def main():
    root=Path('reports/m6g_sources');root.mkdir(exist_ok=True);records=[]
    repos={'carla-simulator/carla':('0.9.16',[
      'LibCarla/source/carla/client/Vehicle.cpp','LibCarla/source/carla/client/detail/Client.cpp',
      'Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Server/CarlaServer.cpp',
      'Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Actor/CarlaActor.cpp',
      'Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Vehicle/CarlaWheeledVehicle.cpp',
      'Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Vehicle/MovementComponents/DefaultMovementComponent.cpp',
      'Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Vehicle/VehicleTelemetryData.h',
      'Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Vehicle/WheeledVehicleMovementComponentNW.cpp',
      'LibCarla/source/carla/rpc/VehicleTelemetryData.h','Docs/build_linux.md']),
      'NVIDIAGameWorks/PhysX-3.4':('master',[
      'PhysX_3.4/Source/PhysXVehicle/src/PxVehicleUpdate.cpp',
      'PhysX_3.4/Source/PhysXVehicle/src/PxVehicleTireFriction.cpp',
      'PhysX_3.4/Source/PhysXVehicle/src/PxVehicleComponents.cpp',
      'PhysX_3.4/Source/PhysXVehicle/src/VehicleUtilControl.cpp',
      'PhysX_3.4/Include/vehicle/PxVehicleComponents.h',
      'PhysX_3.4/Include/vehicle/PxVehicleUpdate.h','README.md'])}
    tasks=[]
    for repo,(ref,paths) in repos.items():
        info=api(f'https://api.github.com/repos/{repo}/commits/{ref}');commit=info['sha']
        (root/(repo.split('/')[-1]+'_commit.json')).write_text(json.dumps(info,indent=2))
        tasks.extend((repo,commit,p) for p in paths)
    def get(task):
        repo,commit,path=task;url=f'https://api.github.com/repos/{repo}/contents/{path}?ref={commit}'
        record=dict(repo=repo,commit=commit,path=path,url=f'https://github.com/{repo}/blob/{commit}/{path}')
        try:
            data=api(url);raw=base64.b64decode(data['content']);name=repo.split('/')[-1]+'_'+path.replace('/','__')+'.txt'
            with (root/name).open('xb') as f:f.write(raw)
            record.update(local_path=str(root/name),sha256=hashlib.sha256(raw).hexdigest(),git_blob=data['sha'],bytes=len(raw),status='Retrieved')
        except Exception as exc:record.update(status='Unavailable',error=repr(exc))
        return record
    records=list(concurrent.futures.ThreadPoolExecutor(4).map(get,tasks))
    with (root/'manifest.json').open('x') as f:json.dump(records,f,indent=2)
    print(json.dumps([{k:v for k,v in r.items() if k in ['path','status','error','bytes']} for r in records],indent=2))

if __name__=='__main__':main()
