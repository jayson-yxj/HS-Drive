"""CPU-only private payload audit and RGB-only export; no model/CARLA calls."""
import gzip
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np
from datasets.m8j3_rgb_clip import CAMERAS, VIEW, load_rgb_packet
from geometry.reference import CARLA_EGO_TO_HS_EGO as FLIP
from geometry.reference import OPTICAL_TO_CARLA_CAM as OPT
from tools.m8j3_download_clip import PRIVATE, NAME, REV, now

CLIP = NAME[:-7]
RAW = PRIVATE / 'extracted' / CLIP


def read_annotation(p):
    with gzip.open(p, 'rb') as f:
        b = f.read(8 * 1024**2 + 1)
    if len(b) > 8 * 1024**2:
        raise ValueError('Annotation exceeds budget')
    return json.loads(b)


def special_keys(v, prefix=''):
    found = set()
    if isinstance(v, dict):
        for k, item in v.items():
            path = prefix + '.' + k
            if any(s in k.lower() for s in ('timestamp', 'frame', 'time', 'version', 'episode')):
                found.add(path)
            found |= special_keys(item, path)
    elif isinstance(v, list):
        for item in v:
            found |= special_keys(item, prefix + '[]')
    return found


def publish_json(p, data):
    with p.open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def export_packet(a, ordinal, output):
    rows = []
    for cam in CAMERAS:
        sensor = a['sensors'][cam]
        rel = f'{ordinal}/{cam}.jpg'
        src = RAW / 'camera' / ('rgb_' + VIEW[cam]) / f'{ordinal}.jpg'
        dst = output / rel
        dst.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(src, dst)
        rows.append(dict(name=cam, path=rel, sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),
            width=sensor['image_size_x'], height=sensor['image_size_y'],
            K=sensor['intrinsic'],
            T_hs_ego_from_optical=(FLIP @ np.asarray(sensor['cam2ego']) @ OPT).tolist()))
    packet = dict(schema='m8j3-rgb-only-v1', source='Bench2Drive_HF_LocalPersonalResearch',
        revision=REV, clip=CLIP, ordinal=ordinal, source_frame=None, timestamp_s=None,
        temporal_valid=False, camera_order=list(CAMERAS), cameras=rows)
    publish_json(output / f'{ordinal}.json', packet)
    return load_rgb_packet(output, f'{ordinal}.json')


def mosaic(a, ordinal, out):
    panels = []
    for cam in CAMERAS:
        sensor = a['sensors'][cam]
        rgb = cv2.imread(str(RAW / 'camera' / ('rgb_' + VIEW[cam]) / f'{ordinal}.jpg'))
        w2c, k = np.asarray(sensor['world2cam']), np.asarray(sensor['intrinsic'])
        for obj in a['bounding_boxes']:
            if obj['class'] not in ('vehicle', 'walker') or 'world_cord' not in obj:
                continue
            pts = np.asarray(obj['world_cord'])
            p = (np.linalg.inv(OPT) @ w2c @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
            if (p[:, 2] <= .1).any():
                continue
            uv = (k @ p.T).T
            uv = uv[:, :2] / uv[:, 2:]
            low, high = uv.min(0), uv.max(0)
            if high[0] < 0 or high[1] < 0 or low[0] >= 1600 or low[1] >= 900:
                continue
            low = np.maximum(low, [0, 0]).astype(int)
            high = np.minimum(high, [1599, 899]).astype(int)
            cv2.rectangle(rgb, tuple(low), tuple(high), (0, 210, 255), 2)
            cv2.putText(rgb, f"{obj['class']} {obj['id']} projected GT", tuple(low),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 210, 255), 1)
        rgb = cv2.resize(rgb, (640, 360))
        cv2.putText(rgb, cam, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
        panels.append(rgb)
    image = np.concatenate([np.concatenate(panels[:3], 1), np.concatenate(panels[3:], 1)], 0)
    cv2.imwrite(str(out / f'{ordinal}_six_camera_projected_boxes.jpg'), image)


def main():
    os.umask(0o077)
    result_dir = PRIVATE / 'validation'
    result_dir.mkdir(exist_ok=False)
    rgb_only = PRIVATE / 'rgb_only'
    rgb_only.mkdir(exist_ok=False)
    annos = sorted((RAW / 'anno').glob('*.json.gz'))
    ordinals = [p.name.split('.')[0] for p in annos]
    selected = [ordinals[0], ordinals[len(ordinals)//2], ordinals[-1]]
    report = dict(kind='ActualDownloadedPayload_CPUOfflineAudit', checked_at_utc=now(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        clip=CLIP, frames=len(annos), ordinal_first=ordinals[0], ordinal_last=ordinals[-1],
        ordinals_contiguous=[int(x) for x in ordinals] == list(range(len(ordinals))),
        temporal_valid=False, source_frame=None, timestamp_s=None)
    groups = {}
    for folder in sorted(p for p in RAW.rglob('*') if p.is_dir()):
        files = sorted(p for p in folder.iterdir() if p.is_file())
        if files:
            ids = [p.name.split('.')[0] for p in files]
            groups[str(folder.relative_to(RAW))] = dict(count=len(files), ordinal_set_matches=ids == ordinals)
    report['groups'] = groups
    stats = defaultdict(lambda: dict(count=0, shape_dtype=Counter(), min=1e9, max=-1e9))
    bad_images, time_keys, annotation_keys = [], set(), set()
    classes, unique_ids, flagged, center_errors, pose_errors, k_errors = Counter(), defaultdict(set), [], [], [], []
    max_rigid_error = 0.
    semantic_matches, semantic_total, detected_ids = 0, 0, set()
    calibration_reference = {}
    calibration_drift = 0.
    all_id_view_records = []
    for f in annos:
        ordinal = f.name.split('.')[0]
        a = read_annotation(f)
        annotation_keys |= set(a)
        time_keys |= special_keys(a)
        ego = [b for b in a['bounding_boxes'] if b['class'] == 'ego_vehicle']
        if len(ego) != 1:
            raise ValueError('Exactly one ego reference required')
        w2e = np.asarray(ego[0]['world2ego'])
        actor_ids = set()
        for b in a['bounding_boxes']:
            classes[b['class']] += 1; unique_ids[b['class']].add(str(b['id']))
            actor_ids.add(int(b['id']))
            if 'world_cord' in b:
                center_errors.append(float(np.linalg.norm(np.mean(b['world_cord'], axis=0) - b['center'])))
            if b['class'] in ('vehicle', 'walker'):
                z = float((w2e @ np.r_[b['center'], 1])[2])
                if z < -20:
                    flagged.append(dict(ordinal=ordinal, actor_id=str(b['id']), kind=b['class'],
                                        reason='BelowEgoMinus20m_DiagnosticNotAutoLabel', ego_z_m=z))
        for cam in CAMERAS:
            s = a['sensors'][cam]; k = np.asarray(s['intrinsic']); c2e = np.asarray(s['cam2ego']); w2c = np.asarray(s['world2cam'])
            expected_f = s['image_size_x'] / (2 * np.tan(np.deg2rad(s['fov']) / 2))
            expected_k = np.array([[expected_f, 0, s['image_size_x']/2], [0, expected_f, s['image_size_y']/2], [0, 0, 1]])
            k_errors.append(float(np.max(np.abs(k-expected_k))))
            pose_errors.append(float(np.max(np.abs(c2e @ w2c - w2e))))
            max_rigid_error = max(max_rigid_error, float(np.max(np.abs(c2e[:3,:3].T @ c2e[:3,:3]-np.eye(3)))))
            for key, value in [('K', k), ('cam2ego', c2e)]:
                refkey = cam + key
                calibration_reference.setdefault(refkey, value)
                calibration_drift = max(calibration_drift, float(np.max(np.abs(value - calibration_reference[refkey]))))
            loaded = {}
            for mode in ('rgb', 'depth', 'instance', 'semantic'):
                rel = f'camera/{mode}_{VIEW[cam]}/{ordinal}.' + ('jpg' if mode == 'rgb' else 'png')
                im = cv2.imread(str(RAW / rel), cv2.IMREAD_UNCHANGED)
                if im is None:
                    bad_images.append(rel); continue
                loaded[mode] = im
                st = stats[mode]; st['count'] += 1
                st['shape_dtype'][str(im.shape)+' '+str(im.dtype)] += 1
                st['min'] = min(st['min'], int(im.min())); st['max'] = max(st['max'], int(im.max()))
            if 'instance' in loaded and 'semantic' in loaded:
                inst, sem = loaded['instance'], loaded['semantic']
                semantic_matches += int(np.count_nonzero(inst[:, :, 2] == sem))
                semantic_total += sem.size
                # Diagnostic only: nominal CARLA BGRA instance decoding; no GT association granted.
                ids = np.unique(inst[:, :, 1].astype(np.uint32) + 256*inst[:, :, 0].astype(np.uint32))
                detected_ids.update(int(x) for x in ids)
                all_id_view_records.append(dict(ordinal=ordinal, camera=cam,
                    nominal_pixel_ids=len(ids), actor_id_numeric_intersections=len(set(ids.tolist()) & actor_ids)))
        if ordinal in selected:
            loaded_packet = export_packet(a, ordinal, rgb_only)
            assert loaded_packet['images'].shape == (6, 900, 1600, 3)
            mosaic(a, ordinal, result_dir)
        if int(ordinal) % 40 == 0:
            print('validated ordinal', ordinal, flush=True)
    report.update(annotation_keys=sorted(annotation_keys), timestamp_frame_version_keys_found=sorted(time_keys),
        sensor_stats={k:{**v,'shape_dtype':dict(v['shape_dtype'])} for k,v in stats.items()},
        image_decode_failures=bad_images, object_records_by_class=dict(classes),
        unique_ids_by_class={k:len(v) for k,v in unique_ids.items()}, below_ground_diagnostic_records=len(flagged),
        center_corner_mean_error_max_m=max(center_errors), camera_ego_matrix_residual_max=max(pose_errors),
        intrinsic_fov_residual_max=max(k_errors), cam2ego_rigid_residual_max=max_rigid_error,
        static_calibration_drift_max=calibration_drift,
        instance_semantic_channel_agreement=semantic_matches/semantic_total,
        nominal_instance_ids_count=len(detected_ids), instance_actor_association='NotValidated_DoNotJoinByRawID',
        selected_RGB_packets=selected, RGB_packet_shape=[6,900,1600,3],
        inference_calls=0, training_steps=0, CARLA_connections=0,
        untouched_modalities=['top_down RGB (excluded inference)', 'LiDAR LAZ', 'Radar H5', 'expert assessment NPZ'],
        depth_status='ObservedUint8SingleChannel_QuantizedSaturated_NotRaw24Bit_NoHSDepthGTApproval',
        synchronization_status='MatchedOrdinals_OriginalPerSensorFrameTimestampUnavailable',
        training_status='TrainingReadinessNoGo')
    publish_json(result_dir / 'payload_validation.json', report)
    publish_json(result_dir / 'flagged_actor_records.json', flagged)
    publish_json(result_dir / 'instance_id_diagnostic.json', all_id_view_records)
    manifest = []
    for p in sorted(RAW.rglob('*')):
        if p.is_file():
            manifest.append(dict(path=str(p.relative_to(RAW)), bytes=p.stat().st_size,
                                 sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    publish_json(result_dir / 'private_payload_manifest.json', manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
