"""Sensor frame IDs, timestamps and calibration identities from actual raw logs."""
import json
from pathlib import Path
from tools.m5k_common import write,asset
from simulator.recorder import file_hash

def main():
 records=[]
 for phase in ['dev','test']:
  root=Path('/root/gpufree-data/hs-drive-data/raw')/('m5k_'+phase)
  for directory in sorted(root.iterdir()):
   if not directory.is_dir():continue
   man=json.loads((directory/'manifest.json').read_text());cal=json.loads((directory/'calibration.json').read_text());assert file_hash(directory/'calibration.json')==man['calibration_sha256'];errors=[];timestamps=[];poses=[];sensorcount=0;ids=man['frames']+list(man['references'].values())
   for fid in ids:
    frame=json.loads((directory/'frames'/f'{fid:08d}.json').read_text());timestamps.append(frame['timestamp_s']);poses.append(frame['T_world_from_ego'])
    for camera in cal['cameras']:
     r=frame['cameras'][camera]
     for kind in ['rgb','depth','instance']:
      sensorcount+=1
      if r[kind+'_frame_id']!=frame['frame_id']or abs(r[kind+'_timestamp_s']-frame['timestamp_s'])>1e-5:errors.append([fid,camera,kind])
   assert not errors,errors
   import numpy as np
   delta=float(np.max(np.abs(np.array(poses)-poses[0])));assert delta<1e-8
   records.append(dict(group=man['group'],split=man['split'],recorded_configurations=len(man['frames']),ego_removed_references=len(man['references']),sensor_observations=sensorcount,sync_errors=errors,pose_max_delta=delta,calibration=asset(directory/'calibration.json'),manifest=asset(directory/'manifest.json')))
 write('reports/m5k_sync_audit.json',dict(kind='actual_CARLA_recorded_per_sensor_metadata_validation',records=records,all_passed=True,source_sha256=file_hash(__file__),limits='Same frame timestamp/calibration checked; semanticRGB materialappearance mismatch remainspossible. No independent Mesh claim.'))
if __name__=='__main__':main()
