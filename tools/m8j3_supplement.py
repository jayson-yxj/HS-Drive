"""Additional numeric checks on local annotations; no training or inference."""
import json
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
from tools.m8j3_validate_clip import RAW, PRIVATE, read_annotation, publish_json, CAMERAS, VIEW
from tools.m8j3_download_clip import now


def main():
    anomalies, field_counts, depth = [], Counter(), Counter()
    centers = []
    for f in sorted((RAW/'anno').glob('*.json.gz')):
        a = read_annotation(f); ordinal=f.name.split('.')[0]
        for obj in a['bounding_boxes']:
            for key, shape in [('center',(3,)),('extent',(3,)),('location',(3,)),('rotation',(3,))]:
                if key not in obj:
                    field_counts['missing_'+key] += 1
                    continue
                x=np.asarray(obj[key], dtype=float)
                if x.shape != shape or not np.isfinite(x).all():
                    anomalies.append(dict(ordinal=ordinal,kind='BadBoxNumeric',field=key))
            if 'extent' in obj and np.any(np.asarray(obj['extent'])<=0):
                field_counts['nonpositive_extent_'+obj['class']]+=1
            if obj['class']=='ego_vehicle':
                centers.append(obj['location'])
        for cam in CAMERAS:
            s=a['sensors'][cam]
            for key, shape in [('intrinsic',(3,3)),('cam2ego',(4,4)),('world2cam',(4,4))]:
                x=np.asarray(s[key],float)
                if x.shape!=shape or not np.isfinite(x).all():
                    anomalies.append(dict(ordinal=ordinal,kind='BadCameraNumeric',camera=cam,field=key))
            d=cv2.imread(str(RAW/'camera'/('depth_'+VIEW[cam])/(ordinal+'.png')),-1)
            depth['pixels']+=d.size;depth['value_255']+=int(np.count_nonzero(d==255))
            depth['value_zero']+=int(np.count_nonzero(d==0))
    # Spatial displacement only; no assumed time, velocity, acceleration or duration.
    centers=np.asarray(centers)
    result=dict(checked_at_utc=now(),kind='ActualPayloadOfflineNumericChecks',
        numeric_anomalies=anomalies,box_field_counts=dict(field_counts),
        depth_counts=dict(depth),depth_value_255_fraction=depth['value_255']/depth['pixels'],
        ego_recorded_polyline_distance_m=float(np.linalg.norm(np.diff(centers,axis=0),axis=1).sum()),
        duration_s=None,derived_velocity=None,raw_sensor_synchronization='Unverified',
        depth_255_meaning='Possible saturation from source float-to-uint8 PNG; not a reliable 255m surface',
        source_exporter_matches_payload_encoding='Consistent, not proof of exact historical capture executable',
        model_forwards=0,training_steps=0)
    publish_json(PRIVATE/'validation/numeric_supplement.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
