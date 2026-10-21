"""Per-sensor synchronization and checksums for completed M5-I campaigns."""
import argparse,json
from pathlib import Path
from simulator.recorder import atomic_json,file_hash

def main():
    p=argparse.ArgumentParser();p.add_argument('--test',action='store_true');a=p.parse_args();dest=Path('reports/m5i_capture_integrity'+('_test'if a.test else'')+'.json')
    if dest.exists():raise FileExistsError(dest)
    records=[]
    for campaign in (['m5i_sealed']if a.test else['m5i_added_train','m5i_added_val']):
        for scene in sorted(Path('/root/gpufree-data/hs-drive-data/raw',campaign).iterdir()):
            if not scene.is_dir():continue
            man=json.loads((scene/'manifest.json').read_text());assert man['calibration_sha256']==file_hash(scene/'calibration.json');count=0;maxdt=0
            for p in (scene/'frames').glob('*.json'):
                f=json.loads(p.read_text());assert len(f['cameras'])==6
                for cam,d in f['cameras'].items():
                    for key in ['rgb','depth','instance']:
                        assert d[key+'_frame_id']==f['frame_id'];dt=abs(d[key+'_timestamp_s']-f['timestamp_s']);assert dt<1e-6;maxdt=max(maxdt,dt);assert file_hash(scene/d[key])==d[key+'_sha256'];count+=1
            records.append(dict(scene=scene.name,sensor_records=count,frames=count//18,max_timestamp_difference_s=maxdt))
    atomic_json(dest,dict(kind='actual_recorded_CARLA_synchronization_checksum',records=records,passed=True,source_sha256=file_hash(__file__)))
if __name__=='__main__':main()
