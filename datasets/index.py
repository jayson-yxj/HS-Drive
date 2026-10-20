"""Immutable normalized index, constructed only from fully verified recordings.

Input records contain no supervision references. Targets live in separate files.
The raw recorder schema remains unchanged; this is a versioned derived artifact.
"""
import json
import re
import shutil
from pathlib import Path
import numpy as np
import yaml
from datasets.carla_dataset import CarlaDataset
from datasets.schema import CAMERA_IDS, SCHEMA_VERSION, calibration_hash
from datasets.split import assign_splits, verify_spatial_separation, SPLITS
from simulator.recorder import atomic_json, file_hash
from tools.validate_recording import validate
from tools.provenance import source_hashes

def validate_camera_geometry(calibration):
    for name in CAMERA_IDS:
        camera=calibration['cameras'][name]
        k=np.asarray(camera['K']);t=np.asarray(camera['T_hs_ego_from_opt'])
        if k.shape!=(3,3) or not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0 or not np.allclose(k[2],[0,0,1]):
            raise ValueError('Invalid intrinsic matrix')
        if (t.shape!=(4,4) or not np.isfinite(t).all() or not np.allclose(t[3],[0,0,0,1]) or
                not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-4) or
                not np.isclose(np.linalg.det(t[:3,:3]),1,atol=1e-4)):
            raise ValueError('Invalid optical-to-ego extrinsic matrix')


INDEX_SCHEMA = 'hsd-index-v1'
INPUT_SCHEMA = 'hsd-input-record-v1'
TARGET_SCHEMA = 'hsd-target-record-v1'


def read_json(path):
    return json.loads(Path(path).read_text())


def safe_path(root, relative):
    root = Path(root).resolve()
    rel = Path(relative)
    path = (root / rel).resolve()
    if rel.is_absolute() or '..' in rel.parts or not path.is_relative_to(root):
        raise ValueError(f'Path escapes dataset: {relative}')
    return path


def checked_json(root, reference):
    path = safe_path(root, reference['path'])
    if file_hash(path) != reference['sha256']:
        raise ValueError(f'Checksum mismatch: {path}')
    return read_json(path)


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError('scene_id must contain only letters, numbers, underscore, hyphen')
    return value


def build_index(registry_path, output):
    registry_path, output = Path(registry_path).resolve(), Path(output).resolve()
    registry = yaml.safe_load(registry_path.read_text())
    if registry.get('schema_version') != 'hsd-registry-v1':
        raise ValueError('Unknown registry schema')
    if output.exists():
        raise FileExistsError(output)
    raw_root = (registry_path.parent / registry['raw_root']).resolve()
    evidence_path = (registry_path.parent / registry['depth_evidence']).resolve()
    depth_evidence = read_json(evidence_path)
    if depth_evidence.get('convention') != 'optical_z' or not depth_evidence.get('status','').startswith('verified_'):
        raise ValueError('Verified optical_z evidence is required')
    scenes, source_paths = [], set()
    ids = set()
    for entry in registry['scenes']:
        sid = identifier(entry['scene_id'])
        scene_path = safe_path(raw_root, entry['path'])
        if sid in ids or scene_path in source_paths:
            raise ValueError('Duplicate scene ID or source recording')
        ids.add(sid); source_paths.add(scene_path)
        manifest = read_json(scene_path / 'manifest.json')
        if manifest['status'] != 'complete':
            raise ValueError(f'Incomplete recording: {sid}')
        scenes.append({**entry, 'map': manifest['map'].split('/')[-1]})
    policy = registry['split_policy']
    assignments = assign_splits(scenes, seed=policy['seed'], val_fraction=policy['val_fraction'],
                                test_maps=policy['test_maps'])
    stage = output.with_name(output.name + '.building')
    stage.mkdir(parents=True, exist_ok=False)
    rows, summaries, seen_content = [], [], {}
    try:
        for entry in sorted(scenes, key=lambda s: s['scene_id']):
            sid, split = entry['scene_id'], assignments[entry['scene_id']]
            scene_path = safe_path(raw_root, entry['path'])
            ds = CarlaDataset(scene_path)
            manifest = ds.manifest
            drivable_manifest = None
            if entry.get('drivable_labels'):
                from datasets.drivable import SCHEMA as DRIVABLE_SCHEMA, validate_arrays
                label_root = safe_path(raw_root,entry['drivable_labels'])
                drivable_manifest = read_json(label_root/'manifest.json')
                if (drivable_manifest['schema_version']!=DRIVABLE_SCHEMA or drivable_manifest['status']!='complete' or
                        drivable_manifest['raw_manifest_sha256']!=file_hash(scene_path/'manifest.json') or
                        drivable_manifest['bev_config_sha256']!=file_hash(registry_path.parent/registry['drivable_bev_config'])):
                    raise ValueError('Drivable provenance mismatch')
                if set(drivable_manifest['frames'])!={str(read_json(f)['frame_id']) for f in ds.frames}:
                    raise ValueError('Drivable frame set mismatch')
            seed = manifest.get('seed', manifest.get('config', {}).get('seed'))
            if not isinstance(seed, int) or isinstance(seed, bool) or not manifest.get('weather'):
                raise ValueError(f'Missing seed/weather provenance: {sid}')
            if manifest['schema_version'] != SCHEMA_VERSION or manifest['carla_version'] != '0.9.15':
                raise ValueError('Unsupported recording schema/CARLA version')
            if len(ds) != manifest['frames_committed'] or not len(ds):
                raise ValueError('Commit count mismatch/empty recording')
            if set(ds.calibration['cameras']) != set(CAMERA_IDS):
                raise ValueError('Need six cameras')
            validate_camera_geometry(ds.calibration)
            profile = manifest['camera_config'].get('imaging_profile_id', 'geometry-baseline-v1')
            if profile not in registry['allowed_profiles']:
                raise ValueError(f'Unapproved imaging profile: {profile}')
            positions, normalized, initial_pose, first_time, raw_id = [], [], None, None, None
            for frame_path in ds.frames:
                sample = read_json(frame_path)
                if raw_id is None:
                    raw_id = sample['scene_id']
                if sample['scene_id'] != raw_id:
                    raise ValueError('Raw scene ID changed within a recording')
                if sample['calibration_id'] != manifest['calibration_id']:
                    raise ValueError('Frame calibration mismatch')
                refs = [sample['cameras'][c][k] for c in CAMERA_IDS for k in ('rgb','depth')]
                refs += list(sample['labels'].values())
                if set(sample['labels'])-{'objects','traffic_rules'}:raise ValueError('Unknown raw label field')
                if set(refs) != set(sample['sha256']) or len(refs) != len(set(refs)):
                    raise ValueError('Missing, aliased, or extra checksum references')
                # Check path types before any reader follows paths from a raw record.
                for c in CAMERA_IDS:
                    for kind in ('rgb','depth'):
                        rel = sample['cameras'][c][kind]
                        if Path(rel).parts[:2] != (kind,c):
                            raise ValueError('Camera modality path mismatch')
                if Path(sample['labels']['objects']).parts[:2] != ('labels','objects'):
                    raise ValueError('Object label path mismatch')
                if 'traffic_rules' in sample['labels']:
                    if Path(sample['labels']['traffic_rules']).parts[:2]!=('labels','traffic_rules'):raise ValueError('Traffic label path mismatch')
                for rel in refs:
                    safe_path(scene_path, rel)
                ego = sample['ego']
                pose = np.asarray(ego['T_world_from_ego'], dtype=np.float64)
                if pose.shape != (4,4) or not np.isfinite(pose).all() or not np.allclose(pose[3],[0,0,0,1]):
                    raise ValueError('Invalid ego pose')
                if not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-4):
                    raise ValueError('Non-rigid ego pose')
                if initial_pose is None:
                    initial_pose, first_time = pose, sample['timestamp_s']
                positions.append(pose[:2,3].tolist())
                rotation = pose[:3,:3].T
                state = np.r_[ego['speed_mps'], rotation @ ego['velocity_world_mps'],
                              rotation @ ego['angular_velocity_world_radps']]
                if state.shape != (7,) or not np.isfinite(state).all():
                    raise ValueError('Invalid ego state')
                rgb_hashes = tuple(sample['sha256'][sample['cameras'][c]['rgb']] for c in CAMERA_IDS)
                if rgb_hashes in seen_content and seen_content[rgb_hashes] != split:
                    raise ValueError('Identical six-camera frame content crosses splits')
                seen_content[rgb_hashes] = split
                cameras, depths = {}, {}
                for c in CAMERA_IDS:
                    cal = ds.calibration['cameras'][c]
                    cameras[c] = {k: cal[k] for k in ('K','width','height','T_hs_ego_from_opt')}
                    for kind, target in [('rgb', cameras[c]), ('depth', depths)]:
                        rel = sample['cameras'][c][kind]
                        reference = {'path': (Path(entry['path'])/rel).as_posix(), 'sha256': sample['sha256'][rel]}
                        if kind == 'rgb':
                            target['rgb'] = reference
                        else:
                            target[c] = reference
                frame = sample['frame_id']
                input_record = {'schema_version': INPUT_SCHEMA, 'scene_id': sid, 'frame_id': frame,
                    'calibration_id': manifest['calibration_id'], 'cameras': cameras,
                    'ego_pose': (np.linalg.inv(initial_pose)@pose).tolist(), 'ego_state': state.tolist(),
                    'timestamp_s': float(sample['timestamp_s']-first_time), 'route_command': -1}
                target_record = {'schema_version': TARGET_SCHEMA, 'scene_id': sid, 'frame_id': frame,
                    'calibration_id': manifest['calibration_id'], 'depth': depths,
                    'objects': {'path': (Path(entry['path'])/sample['labels']['objects']).as_posix(),
                                'sha256': sample['sha256'][sample['labels']['objects']]},
                    'depth_convention': 'optical_z'}
                if 'traffic_rules' in sample['labels']:
                    from simulator.traffic_rules import validate_traffic
                    rel=sample['labels']['traffic_rules'];validate_traffic(read_json(scene_path/rel),frame,sample['timestamp_s'])
                    target_record['traffic_rules']={'path':(Path(entry['path'])/rel).as_posix(),'sha256':sample['sha256'][rel]}
                if drivable_manifest is not None:
                    ref=drivable_manifest['frames'][str(frame)]
                    path=safe_path(label_root,ref['path'])
                    if ref['raw_frame_sha256']!=file_hash(frame_path) or ref['sha256']!=file_hash(path):
                        raise ValueError('Drivable frame identity/checksum mismatch')
                    with np.load(path,allow_pickle=False) as data:
                        validate_arrays(data['drivable_static'],data['drivable_valid'],data['drivable_rule_valid'])
                    target_record['drivable']={'path':(Path(entry['drivable_labels'])/ref['path']).as_posix(),
                        'sha256':ref['sha256'],'schema_version':drivable_manifest['schema_version'],
                        'bev_config_sha256':drivable_manifest['bev_config_sha256'],
                        'label_manifest_sha256':file_hash(label_root/'manifest.json')}
                normalized.append((sample, frame_path, input_record, target_record))
            integrity = validate(scene_path)  # full hashes, dimensions, timestamps, same-frame sensor poses
            if not integrity['recording_integrity_checks_passed']:
                raise ValueError(f'Recording integrity failed: {sid}')
            manifest_hash = file_hash(scene_path/'manifest.json')
            for sample, frame_path, inp, target in normalized:
                key = f"{sid}/{sample['frame_id']:08d}"
                ip, tp = f'inputs/{key}.json', f'targets/{key}.json'
                atomic_json(stage/ip,inp); atomic_json(stage/tp,target)
                rows.append({'sample_id':key,'scene_id':sid,'split':split,'frame_id':sample['frame_id'],
                    'timestamp_s':inp['timestamp_s'], 'input':{'path':ip,'sha256':file_hash(stage/ip)},
                    'target':{'path':tp,'sha256':file_hash(stage/tp)},
                    'raw_frame_sha256':file_hash(frame_path)})
            summaries.append({'scene_id':sid,'raw_scene_id':raw_id,'path':entry['path'],'group_id':entry['group_id'],
                'split':split,'map':entry['map'],'weather':manifest['weather'],'seed':seed,'frames':len(ds),
                'period_s':manifest['camera_config']['sensor_tick_s'], 'calibration_id':manifest['calibration_id'],
                'imaging_profile_id':profile,'manifest_sha256':manifest_hash,
                'calibration_sha256':file_hash(scene_path/'calibration.json'),
                'positions_xy':positions,'integrity':integrity,
                'ego_source':'CARLA snapshot, ideal ego odometry proxy; not validated real sensor odometry',
                'route_source':'unavailable; -1 unknown, never inferred from future trajectory'})
            print(f'Indexed {sid}: {len(ds)} frames -> {split}',flush=True)
        separation = verify_spatial_separation(summaries, policy['min_cross_split_distance_m'])
        index = {'schema_version':INDEX_SCHEMA,'dataset_version':registry['dataset_version'],
            'raw_root':str(raw_root),'camera_order':list(CAMERA_IDS),'depth_convention':'optical_z',
            'split_policy':policy,'scenes':summaries,'samples':rows,'spatial_separation':separation,
            'registry_sha256':file_hash(registry_path),
            'drivable_bev_config_sha256':file_hash(registry_path.parent/registry['drivable_bev_config']) if registry.get('drivable_bev_config') else None,
            'depth_evidence':{'sha256':file_hash(evidence_path),'report':depth_evidence},
            'source_sha256':source_hashes(), 'scope':registry['scope']}
        atomic_json(stage/'index.json',index)
        for split in SPLITS:
            atomic_json(stage/'splits'/f'{split}.json',{'schema_version':'hsd-split-v1',
                'index_sha256':file_hash(stage/'index.json'),
                'scene_ids':[s['scene_id'] for s in summaries if s['split']==split],
                'sample_ids':[s['sample_id'] for s in rows if s['split']==split]})
        atomic_json(stage/'manifest.json',{'schema_version':INDEX_SCHEMA,'status':'complete',
            'index_sha256':file_hash(stage/'index.json'), 'samples':len(rows),
            'splits':{split:sum(s['split']==split for s in rows) for split in SPLITS}})
        stage.rename(output)
        return index
    except BaseException:
        shutil.rmtree(stage)
        raise
