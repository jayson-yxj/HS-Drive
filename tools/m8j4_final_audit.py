"""Append-only CPU evidence audit; never opens a model or starts CARLA."""
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from datasets.m8j3_rgb_clip import CAMERAS, VIEW
from tools.m8j3_validate_clip import RAW, read_annotation
from tools.m8j4_sidecar import OUT, save


def common_region(obj):
    """A comparable positive only, never an exhaustive valid-negative region."""
    if not obj['positive_supervision_candidate']:
        return False
    center = np.asarray(obj['center_hs_m'])
    if not (np.all(center > [-51.2, -51.2, -5]) and np.all(center < [51.2, 51.2, 3])):
        return False
    return any(v['status'] == 'Valid' and v['projection']['bbox_xyxy'][3] * .88 > 280
               and v['projection']['bbox_xyxy'][1] * .88 < 792
               for v in obj['views'].values())


def main():
    os.umask(0o077)
    anomalies = []
    positive_classes = Counter()
    common_classes = Counter()
    common_frames = set()
    camera_support = Counter()
    visibility = Counter()
    max_dim_error = 0.
    for path in sorted((OUT / 'reviewed_sidecar').glob('*.json')):
        frame = json.loads(path.read_text())
        for obj in frame['objects']:
            if obj['geometry_status'] == 'Valid':
                max_dim_error = max(max_dim_error, obj['dimension_residual_m'])
            if obj['positive_supervision_candidate']:
                positive_classes[obj['model_class']] += 1
            if common_region(obj):
                common_classes[obj['model_class']] += 1
                common_frames.add(frame['ordinal'])
            for cam, view in obj['views'].items():
                if view['status'] == 'Valid':
                    camera_support[cam] += 1
                    visibility[view.get('occlusion_state', 'NotAvailable')] += 1
            if obj.get('reason') != 'HiddenBelowGround_NoTrainingTarget':
                continue
            annotation = read_annotation(RAW / 'anno' / (frame['ordinal'] + '.json.gz'))
            original = next(o for o in annotation['bounding_boxes'] if str(o.get('id')) == obj['actor_id'])
            counts = {}
            for cam in CAMERAS:
                ins = cv2.imread(str(RAW / 'camera' / ('instance_' + VIEW[cam]) / (frame['ordinal'] + '.png')), -1)
                sem = cv2.imread(str(RAW / 'camera' / ('semantic_' + VIEW[cam]) / (frame['ordinal'] + '.png')), -1)
                if ins is None or sem is None:
                    raise ValueError('Missing anomaly sensor evidence')
                counts[cam] = dict(instance_semantic12_pixels=int(np.count_nonzero(ins[:, :, 2] == 12)),
                                   semantic12_pixels=int(np.count_nonzero(sem == 12)),
                                   projected_status=obj['views'][cam]['projection']['status'])
            anomalies.append(dict(ordinal=frame['ordinal'], actor_id=obj['actor_id'],
                world_center=original['center'], world_z_bounds=[min(p[2] for p in original['world_cord']), max(p[2] for p in original['world_cord'])],
                hs_center=obj['center_hs_m'], camera_evidence=counts, label_status='Invalid',
                raw_record_retained=True, positive_supervision=False, background_negative=False,
                interpretation='Out-of-domain hidden/placeholder annotation is plausible; physical placement cause not proven',
                human_certification='Pending'))
    assert len(anomalies) == 23
    save(OUT / 'anomaly_detailed_audit.json', anomalies)
    anomaly_summary = dict(evidence_type='ActualPrivatePayload_CPUAudit', records=len(anomalies),
        ordinals=[a['ordinal'] for a in anomalies], distinct_actor_ids=len({a['actor_id'] for a in anomalies}),
        hs_center_z_range_m=[min(a['hs_center'][2] for a in anomalies), max(a['hs_center'][2] for a in anomalies)],
        world_center_z_range_m=[min(a['world_center'][2] for a in anomalies), max(a['world_center'][2] for a in anomalies)],
        instance_pedestrian_pixels_all_six_views=sum(v['instance_semantic12_pixels'] for a in anomalies for v in a['camera_evidence'].values()),
        semantic_pedestrian_pixels_all_six_views=sum(v['semantic12_pixels'] for a in anomalies for v in a['camera_evidence'].values()),
        projected_status=dict(Counter(v['projected_status'] for a in anomalies for v in a['camera_evidence'].values())),
        action='ExcludeAll23FromPositiveSupervision_RetainRecords_NoBackgroundNegatives',
        causal_explanation='NotProven_NoPhysicalStateOrRenderingEngineTrace',
        human_certification='Pending', private_record_sha256=hashlib.sha256((OUT / 'anomaly_detailed_audit.json').read_bytes()).hexdigest())
    save(Path('reports/m8j4_anomaly_audit.json'), anomaly_summary)
    summary = dict(evidence_type='ActualPrivatePayload_CPUAudit', positive_candidate_classes=dict(positive_classes),
        model_range_and_crop_positive_classes=dict(common_classes), model_range_and_crop_positive_frames=len(common_frames),
        view_support=dict(camera_support), visibility_proxy=dict(visibility), max_dimension_residual_m=max_dim_error,
        human_certified_training_targets=0, independent_scenes=1,
        coverage_caveat='214 correlated ordinal groups; candidates are actor-frame records, not unique independent actors',
        negative_supervision_valid=False, occupancy_free_valid=False)
    save(OUT / 'support_audit.json', summary)
    print(json.dumps(dict(anomalies=anomaly_summary, support=summary), indent=2))


if __name__ == '__main__':
    main()
