"""Limited official source retrieval with TLS verification; no accounts or model changes."""
import urllib.request
from pathlib import Path
from tools.m8e_common import *
root=ROOT/'official';root.mkdir(exist_ok=True)
files=['Game/Tagger.cpp','Game/TaggedComponent.cpp','Sensor/InstanceSegmentationCamera.cpp']
records=[]
for name in files:
    url='https://raw.githubusercontent.com/carla-simulator/carla/0.9.15/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/'+name
    path=root/Path(name).name
    try:
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url,timeout=25) as response:
            body=response.read();status=response.status
        with path.open('xb') as stream:stream.write(body)
        records.append(dict(url=url,status=status,asset=asset(path)))
    except Exception as e:records.append(dict(url=url,error=str(e)))
write('reports/m8e_official_identity_sources.json',dict(**provenance('OfficialCARLA0915SourceAudit'),
    downloads=records,interpretation='Tagger encodes UE Actor.GetUniqueID low16bits: G=low8, B=high8. Python Actor.id is a different namespace; EnvironmentObject ID is not assumed equal. Historical decoder G*256+B is a bijective byte-swapped code, retained as opaque key; raw BGRA permits canonical UE low16 recovery. ID collisions and component grouping possible.',
    links=['https://carla.readthedocs.io/en/0.9.15/python_api/','https://carla.readthedocs.io/en/0.9.15/ref_sensors/']))
