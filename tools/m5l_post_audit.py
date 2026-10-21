"""Postfreeze physical-reference limitations and result freshness verification."""
import json
from pathlib import Path
import numpy as np
from tools.m5l_common import write,asset,summary
from simulator.recorder import file_hash

def main():
 shadow=json.loads(Path('runs/m5l_logs/shadow_raw.json').read_text());accepted=[r for r in shadow['outputs']if r['status']=='accepted'];assert all(r['end_to_end_ms']<=250 for r in accepted);assert len({r['frame']for r in accepted})==len(accepted);by=[]
 for split in ['train','val','regression','test']:
  raw=json.loads(Path(f'runs/m5l_objects_{split}.json').read_text());records=[]
  for seed in [101,102]:
   rr=[r for r in raw['actors']if r['seed']==seed];valid=[r for r in rr if r['valid']];records.append(dict(seed=seed,visible_bottom_z=summary([r['visible_bottom_z']for r in rr]),plane_contact_supported_views=sum(r['contact_matches_plane']for r in rr),plane_contact_supported_valid=sum(r['contact_matches_plane']for r in valid),flat_pose_views=sum(r['flat_pose']for r in rr),slope_pose_views=sum(not r['flat_pose']for r in rr),conditional_valid=len(valid),trusted_positions=0))
  by.append(dict(split=split,records=records))
 (write if not Path('reports/m5l_reference_and_freshness_audit.json').exists() else lambda *args:None)('reports/m5l_reference_and_freshness_audit.json',dict(kind='postfreeze_actual_record_diagnostic',shadow_accepted_fresh=True,accepted=len(accepted),accepted_latency_ms=summary([r['end_to_end_ms']for r in accepted]),all_output_latency_ms=summary([r['end_to_end_ms']for r in shadow['outputs']if 'end_to_end_ms'in r]),objects=by,limits='Conditional position estimates are not trustedpositions: binaryforeground cannotconfirmcontact. Visiblebottom isaftererosion andmaynot beactualgroundcontact. Fixedz plane notfit toGT. Freshness measuredatworkeroutput; no planner consumesoutput. Startupinference outlier retained. TickshadowP95 grows butbelowpredeclaredstop; no strict20Hzclaim.'))
 sync=[]
 root=Path('/root/gpufree-data/hs-drive-data/raw/m5l_test')
 for directory in sorted(root.iterdir()):
  if not directory.is_dir():continue
  man=json.loads((directory/'manifest.json').read_text());assert file_hash(directory/'calibration.json')==man['calibration_sha256'];ids=man['frames']+list(man['references'].values());errors=[]
  for fid in ids:
   frame=json.loads((directory/'frames'/f'{fid:08d}.json').read_text())
   for cam,r in frame['cameras'].items():
    for k in ['rgb','depth','instance']:
     if r[k+'_frame_id']!=fid or abs(r[k+'_timestamp_s']-frame['timestamp_s'])>1e-5:errors.append([fid,cam,k])
  assert not errors;sync.append(dict(group=man['group'],frames=len(man['frames']),references=len(man['references']),sensor_observations=len(ids)*18,sync_errors=errors,manifest=asset(directory/'manifest.json')))
 write('reports/m5l_sync_audit.json',dict(kind='actual_CARLA_sensor_metadata',records=sync,all_passed=True))
if __name__=='__main__':main()
