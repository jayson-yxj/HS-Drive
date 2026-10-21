"""Final read-only per-sensor timestamps/calibration/file SHA audit."""
import json
from pathlib import Path
from simulator.recorder import atomic_json,file_hash

def main():
    dest=Path('reports/m5h_capture_integrity.json')
    if dest.exists():raise FileExistsError(dest)
    records=[]
    for campaign in ['m5h_added','m5h_sealed']:
        for scene in sorted(Path('/root/gpufree-data/hs-drive-data/raw',campaign).iterdir()):
            if not scene.is_dir():continue
            m=json.loads((scene/'manifest.json').read_text());cal=scene/'calibration.json';assert m['calibration_sha256']==file_hash(cal);n=channels=0;maxdt=0
            for p in (scene/'frames').glob('*.json'):
                f=json.loads(p.read_text());assert len(f['cameras'])==6
                for cam,d in f['cameras'].items():
                    for kind in ['rgb','depth','instance']:
                        assert d[kind+'_frame_id']==f['frame_id'];dt=abs(d[kind+'_timestamp_s']-f['timestamp_s']);assert dt<1e-6;maxdt=max(maxdt,dt);assert file_hash(scene/d[kind])==d[kind+'_sha256'];channels+=1
                n+=1
            records.append(dict(campaign=campaign,scene=scene.name,frames_including_ego_removed=n,sensor_records_verified=channels,max_timestamp_difference_s=maxdt,calibration_sha256=file_hash(cal)))
    atomic_json(dest,dict(kind='actual_recorded_CARLA_sensor_frame_timestamp_and_file_SHA_audit',records=records,passed=True,source_sha256=file_hash(__file__),limits='Recordedrenderalignment only; notindependentphysicaltruth. Initialadhocwalk assumed alldirectory andstoppedatrootmanifest; no datachanged.'))
if __name__=='__main__':main()
