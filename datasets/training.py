"""CPU streaming dataset and reproducible PyTorch DataLoader for M3/M4.

Inference reads only normalized input records and RGB files. Target I/O is an
explicit separate path. No CARLA import or simulator server is needed.
"""
from dataclasses import dataclass
from pathlib import Path
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets.index import (INDEX_SCHEMA, INPUT_SCHEMA, TARGET_SCHEMA, read_json,
                            safe_path, checked_json)
from datasets.inputs import LearnedInputs
from datasets.labels import depth_bin_edges, sample_depth
from datasets.schema import CAMERA_IDS
from geometry.cameras import resize_rgb_and_intrinsics
from simulator.recorder import file_hash


@dataclass(frozen=True)
class DataItem:
    inputs: LearnedInputs
    targets: dict | None
    metadata: dict | list

    def pin_memory(self):
        targets=None if self.targets is None else {
            k:v.pin_memory() if isinstance(v,torch.Tensor) else v for k,v in self.targets.items()}
        return DataItem(self.inputs.pin_memory(),targets,self.metadata)


def _read_rgb(root, reference, expected_shape):
    path=safe_path(root,reference['path'])
    if Path(reference['path']).suffix != '.png' or 'rgb' not in Path(reference['path']).parts:
        raise ValueError('Input image must reference RGB PNG')
    if file_hash(path)!=reference['sha256']:
        raise ValueError(f'RGB checksum mismatch: {path}')
    bgr=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if bgr is None or bgr.shape!=(*expected_shape,3):
        raise ValueError('Unreadable/misshaped RGB')
    return bgr[...,::-1].copy()


class HSDriveDataset(Dataset):
    def __init__(self,index_dir,split,*,supervision=True,load_objects=False,sequence_length=1,
                 sequence_stride=1,output_wh=(640,384),feature_stride=16,raw_root=None,road_overlay=None):
        if split not in ('train','val','test'):
            raise ValueError('Unknown split')
        if sequence_length<1 or sequence_stride<1:
            raise ValueError('Sequence limits must be positive')
        if feature_stride<1 or min(output_wh)<1 or any(x%feature_stride for x in output_wh):
            raise ValueError('Output dimensions must be divisible by feature stride')
        if load_objects and not supervision:
            raise ValueError('Inference cannot request actor labels')
        self.root=Path(index_dir).resolve()
        manifest=read_json(self.root/'manifest.json')
        if manifest['schema_version']!=INDEX_SCHEMA or manifest['status']!='complete':
            raise ValueError('Index is not committed')
        index_path=self.root/'index.json'
        if file_hash(index_path)!=manifest['index_sha256']:
            raise ValueError('Index checksum mismatch')
        self.index=read_json(index_path)
        if self.index['schema_version']!=INDEX_SCHEMA or self.index['camera_order']!=list(CAMERA_IDS):
            raise ValueError('Unsupported index or camera order')
        split_file=read_json(self.root/'splits'/f'{split}.json')
        rows=[r for r in self.index['samples'] if r['split']==split]
        if (split_file['index_sha256']!=manifest['index_sha256'] or
                split_file['sample_ids']!=[r['sample_id'] for r in rows]):
            raise ValueError('Split/index disagreement')
        scenes={s['scene_id']:s for s in self.index['scenes']}
        if split_file['scene_ids'] != [s['scene_id'] for s in self.index['scenes'] if s['split']==split]:
            raise ValueError('Split scene disagreement')
        self.raw_root=Path(raw_root or self.index['raw_root']).resolve()
        self.road_overlay=None;self.road_manifest=None
        if supervision and road_overlay:
            from datasets.road_supervision import SCHEMA as ROAD_SCHEMA
            self.road_overlay=Path(road_overlay).resolve();self.road_manifest=read_json(self.road_overlay/'manifest.json')
            if (self.road_manifest['schema_version']!=ROAD_SCHEMA or self.road_manifest['status']!='complete' or
                    self.road_manifest['index_sha256']!=manifest['index_sha256'] or
                    self.road_manifest['bev_config_sha256']!=self.index['drivable_bev_config_sha256']):
                raise ValueError('Road overlay provenance mismatch')
            if set(self.road_manifest['samples'])!={r['sample_id'] for r in self.index['samples']}:raise ValueError('Road overlay sample set mismatch')
        self.supervision,self.load_objects=supervision,load_objects
        self.output_wh,self.feature_stride=tuple(output_wh),feature_stride
        self.sequence_length=sequence_length
        self.windows=[]
        for sid in split_file['scene_ids']:
            sr=[r for r in rows if r['scene_id']==sid]
            if scenes[sid]['split']!=split:
                raise ValueError('Scene crosses split')
            period=scenes[sid]['period_s']
            for end in range((sequence_length-1)*sequence_stride,len(sr)):
                full=sr[end-(sequence_length-1)*sequence_stride:end+1]
                # No padding across scene starts, no bridging gaps or future frames.
                if len(full)>1 and not np.allclose(np.diff([r['timestamp_s'] for r in full]),period,atol=1e-5,rtol=0):
                    continue
                self.windows.append(full[::sequence_stride])
        self.scenes=scenes
        self.bin_edges=depth_bin_edges()

    def __len__(self):return len(self.windows)

    def _input(self,row):
        record=checked_json(self.root,row['input'])
        expected={'schema_version','scene_id','frame_id','calibration_id','cameras','ego_pose',
                  'ego_state','timestamp_s','route_command'}
        if set(record)!=expected or record['schema_version']!=INPUT_SCHEMA:
            raise ValueError('Input record schema mismatch; supervision fields forbidden')
        if (record['scene_id'],record['frame_id'])!=(row['scene_id'],row['frame_id']):
            raise ValueError('Input identity mismatch')
        if not np.isclose(record['timestamp_s'],row['timestamp_s'],atol=1e-8,rtol=0):
            raise ValueError('Input timestamp/index mismatch')
        if record['calibration_id']!=self.scenes[row['scene_id']]['calibration_id']:
            raise ValueError('Input calibration mismatch')
        if set(record['cameras'])!=set(CAMERA_IDS):
            raise ValueError('Input requires all six cameras')
        images,ks,ts=[],[],[]
        for name in CAMERA_IDS:
            c=record['cameras'][name]
            if set(c)!={'K','width','height','T_hs_ego_from_opt','rgb'}:
                raise ValueError('Unexpected camera fields')
            im=_read_rgb(self.raw_root,c['rgb'],(c['height'],c['width']))
            im,k=resize_rgb_and_intrinsics(im,c['K'],self.output_wh)
            images.append(im.transpose(2,0,1).astype(np.float32)/255)
            ks.append(k);ts.append(c['T_hs_ego_from_opt'])
        tensors={'images':torch.from_numpy(np.stack(images)),
            'intrinsics':torch.tensor(np.asarray(ks),dtype=torch.float32),
            'cam_to_ego':torch.tensor(np.asarray(ts),dtype=torch.float32),
            'ego_pose':torch.tensor(record['ego_pose'],dtype=torch.float64),
            'ego_state':torch.tensor(record['ego_state'],dtype=torch.float32),
            'timestamps_s':torch.tensor(record['timestamp_s'],dtype=torch.float64),
            'route_command':torch.tensor(record['route_command'],dtype=torch.int64),
            'valid_cameras':torch.ones(6,dtype=torch.bool)}
        return LearnedInputs(tensors),record

    def _target(self,row,record):
        target=checked_json(self.root,row['target'])
        if target['schema_version']!=TARGET_SCHEMA or target['depth_convention']!='optical_z':
            raise ValueError('Unknown supervision schema/depth convention')
        if (target['scene_id'],target['frame_id'],target['calibration_id']) != (
                record['scene_id'],record['frame_id'],record['calibration_id']):
            raise ValueError('Supervision identity mismatch')
        if set(target['depth'])!=set(CAMERA_IDS):
            raise ValueError('Supervision requires six depth maps')
        samples=[]
        for name in CAMERA_IDS:
            ref=target['depth'][name];path=safe_path(self.raw_root,ref['path'])
            if file_hash(path)!=ref['sha256']:
                raise ValueError(f'Depth checksum mismatch: {path}')
            with np.load(path,allow_pickle=False) as data:depth=data['depth_m']
            c=record['cameras'][name]
            if depth.shape!=(c['height'],c['width']):
                raise ValueError('Depth dimensions differ from paired RGB')
            samples.append(sample_depth(depth,output_wh=self.output_wh,stride=self.feature_stride,edges=self.bin_edges))
        result={k:torch.from_numpy(np.stack([s[k] for s in samples])) for k in samples[0]}
        # Optional labels have an explicit all-unknown fallback for legacy scenes.
        label=np.full((160,120),255,dtype=np.uint8);valid=np.zeros_like(label,dtype=bool);rules=np.zeros_like(valid)
        if 'drivable' in target:
            from datasets.drivable import SCHEMA,validate_arrays
            ref=target['drivable'];path=safe_path(self.raw_root,ref['path'])
            if (ref['schema_version']!=SCHEMA or file_hash(path)!=ref['sha256'] or
                    ref['bev_config_sha256']!=self.index.get('drivable_bev_config_sha256')):
                raise ValueError('Drivable schema/checksum mismatch')
            with np.load(path,allow_pickle=False) as data:
                label=data['drivable_static'];valid=data['drivable_valid'];rules=data['drivable_rule_valid']
            validate_arrays(label,valid,rules)
        direction=np.zeros((2,160,120),np.float32);direction_valid=np.zeros_like(valid);junction=np.zeros_like(valid)
        if self.road_manifest is not None:
            from datasets.road_supervision import validate_road_arrays
            ref=self.road_manifest['samples'][row['sample_id']];path=safe_path(self.road_overlay,ref['path'])
            if file_hash(path)!=ref['sha256'] or ref['raw_frame_sha256']!=row['raw_frame_sha256']:raise ValueError('Road overlay sample identity mismatch')
            with np.load(path,allow_pickle=False) as data:
                validate_road_arrays(data);label=data['drivable_static'];valid=data['drivable_valid'];rules=data['drivable_rule_valid']
                direction=data['lane_direction'];direction_valid=data['lane_direction_valid'];junction=data['junction_support']
        result.update(lane_direction=torch.from_numpy(direction),lane_direction_valid=torch.from_numpy(direction_valid),junction_support=torch.from_numpy(junction))
        result.update(drivable_static=torch.from_numpy(label.astype(np.float32)),
                      drivable_valid=torch.from_numpy(valid),drivable_rule_valid=torch.from_numpy(rules))
        stop=-1;stop_valid=False;signal=-1
        if 'traffic_rules' in target:
            from simulator.traffic_rules import validate_traffic
            traffic=checked_json(self.raw_root,target['traffic_rules'])
            # Absolute timestamps belong to raw supervision; identity is pinned by index/raw hashes.
            validate_traffic(traffic,row['frame_id'],traffic['timestamp_s'])
            stop=traffic['ego_stop_required'];stop_valid=traffic['ego_stop_valid'];signal=traffic['ego_signal_state']
        result.update(signal_stop_required=torch.tensor(stop,dtype=torch.int64),signal_stop_valid=torch.tensor(stop_valid),signal_state=torch.tensor(signal,dtype=torch.int64))
        if self.load_objects:
            result['objects']=checked_json(self.raw_root,target['objects'])
        return result

    def __getitem__(self,index):
        window=self.windows[index];items=[]
        for row in window:
            inputs,record=self._input(row)
            targets=self._target(row,record) if self.supervision else None
            items.append((inputs,targets))
        last=window[-1];scene=self.scenes[last['scene_id']]
        metadata={'schema_version':'hsd-training-item-v1','dataset_version':self.index['dataset_version'],
            'scene_id':last['scene_id'],'sample_id':last['sample_id'],'frame_ids':[r['frame_id'] for r in window],
            'calibration_id':scene['calibration_id'],'camera_order':list(CAMERA_IDS),
            'coordinate_frame':'hs_ego; ego_pose=episode_origin_from_ego',
            'units':{'depth':'m','linear_velocity':'m/s','angular_velocity':'rad/s','time':'s since scene start'},
            'ego_source':scene['ego_source'],'route_source':scene['route_source'],
            'depth_convention':'optical_z','depth_sampling':'nearest native pixel at feature-cell center; 3x3 edge guard 0.5m',
            'depth_bin_edges_m':self.bin_edges.tolist(),'image_normalization':'RGB float32 /255; no mean/std normalization'}
        if self.sequence_length==1:
            return DataItem(items[0][0],items[0][1],metadata)
        inputs=LearnedInputs({k:torch.stack([x[0][k] for x in items]) for k in items[0][0]})
        targets=None if not self.supervision else {
            k:torch.stack([x[1][k] for x in items]) if isinstance(items[0][1][k],torch.Tensor)
            else [x[1][k] for x in items] for k in items[0][1]}
        return DataItem(inputs,targets,metadata)


def collate_items(items):
    if not items or len({x.targets is None for x in items})!=1:
        raise ValueError('Cannot mix training and inference items')
    inputs=LearnedInputs({k:torch.stack([x.inputs[k] for x in items]) for k in items[0].inputs})
    targets=None if items[0].targets is None else {
        k:torch.stack([x.targets[k] for x in items]) if isinstance(items[0].targets[k],torch.Tensor)
        else [x.targets[k] for x in items] for k in items[0].targets}
    return DataItem(inputs,targets,[x.metadata for x in items])


def seed_worker(_):
    seed=torch.initial_seed()%2**32
    random.seed(seed);np.random.seed(seed);cv2.setNumThreads(0)


def make_dataloader(dataset,*,batch_size=1,workers=0,seed=7,shuffle=False,pin_memory=False):
    if workers<0 or batch_size<1:
        raise ValueError('Invalid DataLoader size')
    generator=torch.Generator().manual_seed(seed)
    return DataLoader(dataset,batch_size=batch_size,num_workers=workers,shuffle=shuffle,
        collate_fn=collate_items,worker_init_fn=seed_worker,generator=generator,
        pin_memory=pin_memory,**({'prefetch_factor':1,'multiprocessing_context':'spawn'} if workers else {}))


def loader_from_config(index_dir,split,config_path,*,supervision=None):
    """Construct a loader from the versioned data config; val/test never shuffle."""
    import yaml
    config=yaml.safe_load(Path(config_path).read_text())
    expected={'schema_version','output_wh','feature_stride','sequence_length','sequence_stride',
              'batch_size','workers','seed','shuffle','pin_memory','supervision','load_objects'}
    if set(config)!=expected or config['schema_version']!='hsd-loader-v1':
        raise ValueError('Unknown/missing loader configuration fields')
    for key in ('shuffle','pin_memory','supervision','load_objects'):
        if not isinstance(config[key],bool):
            raise TypeError(f'Expected boolean loader option: {key}')
    for key in ('feature_stride','sequence_length','sequence_stride','batch_size','workers','seed'):
        if isinstance(config[key],bool) or not isinstance(config[key],int):
            raise TypeError(f'Expected integer loader option: {key}')
    supervised=config['supervision'] if supervision is None else supervision
    dataset=HSDriveDataset(index_dir,split,supervision=supervised,
        load_objects=config['load_objects'] if supervised else False,
        **{k:config[k] for k in ('output_wh','feature_stride','sequence_length','sequence_stride')})
    return make_dataloader(dataset,shuffle=split=='train' and config['shuffle'],
        **{k:config[k] for k in ('batch_size','workers','seed','pin_memory')})
