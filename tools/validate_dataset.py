"""M3 full replay, model-boundary audit and CPU DataLoader integration gate."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import torch
from datasets.training import HSDriveDataset, make_dataloader
from datasets.inputs import INPUT_FIELDS
from datasets.index import read_json
from simulator.recorder import atomic_json, file_hash


def tensor_digest(inputs):
    h=hashlib.sha256()
    for key in sorted(inputs):
        h.update(key.encode());h.update(inputs[key].numpy().tobytes())
    return h.hexdigest()


def preview(item,path):
    panels=[]
    for i,name in enumerate(item.metadata['camera_order']):
        rgb=(item.inputs['images'][i].numpy().transpose(1,2,0)*255).round().astype(np.uint8)
        panel=rgb[...,::-1].copy()
        uv=item.targets['depth_uv'][i].numpy()
        valid=item.targets['depth_valid'][i].numpy()
        depth=item.targets['depth_m'][i].numpy()
        colors=cv2.applyColorMap((np.clip(depth/80,0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO)
        for point,color in zip(uv[valid],colors[valid]):
            cv2.circle(panel,tuple(np.rint(point).astype(int)),2,tuple(map(int,color)),-1)
        cv2.rectangle(panel,(0,0),(640,30),(0,0,0),-1)
        cv2.putText(panel,f'{name} | depth target rays (privileged)',(8,21),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        panels.append(panel)
    cv2.imwrite(str(path),np.concatenate([np.concatenate(panels[j:j+3],axis=1) for j in (0,3)]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',required=True);p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=2)
    args=p.parse_args();start=time.monotonic();out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(7);torch.set_num_threads(4);cv2.setNumThreads(0)
    index=read_json(Path(args.index)/'index.json')
    split_results={};checks={};replay_count=0
    for split in ('train','val','test'):
        ds=HSDriveDataset(args.index,split,load_objects=True)
        valid,total,objects=0,0,0;digest=hashlib.sha256()
        for batch in make_dataloader(ds,workers=args.workers):
            assert set(batch.inputs)==INPUT_FIELDS
            replay_count+=1;digest.update(tensor_digest(batch.inputs).encode())
            valid+=int(batch.targets['depth_valid'].sum());total+=batch.targets['depth_valid'].numel()
            objects+=sum(len(x) for x in batch.targets['objects'])
        split_results[split]={'samples':len(ds),'valid_depth_fraction':valid/total,'actor_records':objects,
                              'input_digest_ordered':digest.hexdigest()}
        checks[split+'_full_replay']=len(ds)>0 and valid>0
        preview(ds[0],out/f'{split}_depth_targets.png')
        print(f'Replayed {split}: {len(ds)} frames',flush=True)
    train=HSDriveDataset(args.index,'train')
    inference=HSDriveDataset(args.index,'train',supervision=False)
    selected=sorted(set((0,len(train)//2,len(train)-1)))
    checks['repeatable_reads']=all(tensor_digest(train[i].inputs)==tensor_digest(train[i].inputs) for i in selected)
    expected={i:tensor_digest(train[i].inputs) for i in selected}
    with patch('numpy.load',side_effect=AssertionError('Inference opened depth')), \
         patch.object(HSDriveDataset,'_target',side_effect=AssertionError('Inference requested supervision')):
        checks['rgb_only_path']=all(inference[i].targets is None and tensor_digest(inference[i].inputs)==expected[i] for i in selected)
    try:inference[0].inputs['actor_gt']
    except KeyError:checks['forbidden_access_rejected']=True
    else:checks['forbidden_access_rejected']=False
    # Test ordered shuffled subset across zero and spawned workers.
    subset=torch.utils.data.Subset(inference,selected)
    def order(workers):
        return [(b.metadata[0]['sample_id'],tensor_digest(b.inputs))
                for b in make_dataloader(subset,workers=workers,seed=17,shuffle=True)]
    checks['worker_seed_reproducibility']=order(0)==order(args.workers)
    temporal=HSDriveDataset(args.index,'train',sequence_length=3,supervision=False)
    sequence=next(iter(make_dataloader(temporal,workers=0)))
    checks['temporal_shape']=list(sequence.inputs['images'].shape)==[1,3,6,3,384,640]
    checks['causal_scene_windows']=all(len({r['scene_id'] for r in w})==1 and
        all(b['frame_id']>a['frame_id'] for a,b in zip(w,w[1:])) for w in temporal.windows)
    # Small interface probe, NOT the BEV network or a quality/training benchmark.
    batch=next(iter(make_dataloader(train,workers=0)))
    probe=torch.nn.Conv2d(3,48,1)
    pooled=torch.nn.functional.avg_pool2d(batch.inputs['images'].flatten(0,1),16)
    logits=probe(pooled)
    labels=batch.targets['depth_bin'].flatten(0,1)
    loss=torch.nn.functional.cross_entropy(logits,labels,ignore_index=-100)
    loss.backward()
    checks['supervision_loss_backward']=bool(torch.isfinite(loss) and probe.weight.grad is not None and torch.isfinite(probe.weight.grad).all())
    train_maps={s['map'] for s in index['scenes'] if s['split']!='test'}
    test_maps={s['map'] for s in index['scenes'] if s['split']=='test'}
    checks['held_out_map']=bool(test_maps) and not train_maps&test_maps
    groups={}
    for scene in index['scenes']:groups.setdefault(scene['group_id'],set()).add(scene['split'])
    checks['group_disjoint']=all(len(v)==1 for v in groups.values())
    checks['spatial_separation']=all(x['min_distance_m']>=index['split_policy']['min_cross_split_distance_m'] for x in index['spatial_separation'])
    report={'schema_version':'hsd-m3-gate-v1','checks':checks,'passed':all(checks.values()),
        'index':str(Path(args.index).resolve()),'index_sha256':file_hash(Path(args.index)/'index.json'),
        'dataset_version':index['dataset_version'],'splits':split_results,'frames_replayed':replay_count,
        'temporal_windows_train':len(temporal),'probe_loss':float(loss.detach()),
        'environment':{'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available(),
                       'workers':args.workers,'threads':torch.get_num_threads()},
        'elapsed_s':time.monotonic()-start,'spatial_separation':index['spatial_separation'],
        'scope':'Engineering gate on 5 small scenes; no learned BEV model or perception accuracy claim',
        'limitations':['ego state uses ideal simulator odometry proxy','route command unavailable (-1)',
            'only depth and actor-box supervision available; no drivable/occupancy/future labels',
            'train/val same map; test Town10HD_Opt is held out','depth edge mask and sampling need M4 ablation',
            'CPU DataLoader validation does not validate CUDA training environment']}
    atomic_json(out/'gate.json',report)
    print(json.dumps(report,indent=2))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
