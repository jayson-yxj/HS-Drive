"""CPU-only official checkpoint/template identity audit in isolated network namespace."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
import mmcv

def main():
    cfg=json.loads(Path('configs/m8j12_protocol.json').read_text());rt=Path(cfg['runtime_root']);assets=rt/'artifacts'
    path=assets/cfg['checkpoint'];payload=torch.load(path,map_location='cpu',weights_only=False)
    assert isinstance(payload,dict) and 'state_dict' in payload
    state=payload['state_dict'];meta=payload.get('meta',{})
    if 'config' in meta:(Path(cfg['private_root'])/'checkpoint_config.py').write_text(meta['config'])
    tensors={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in state.items()}
    arrays={}
    for p in assets.glob('*.np*'):
        d=np.load(p,allow_pickle=False)
        v={k:d[k] for k in d.files} if isinstance(d,np.lib.npyio.NpzFile) else {'array':d}
        arrays[p.name]={k:dict(shape=list(x.shape),dtype=str(x.dtype),finite=bool(np.isfinite(x).all()),min=float(x.min()),max=float(x.max())) for k,x in v.items()}
    comparisons={}
    names={'motion_anchor':'kmeans_motion_6.npy','lat_anchor':'path_1m_pts_15_1024_b2d_new_ego.npy','lon_anchor':'vel_seq_K256_t30.npy','traj_anchor':'trajectory_1024_256_b2d.npz','traj_anchor_mask':'trajectory_1024_256_b2d.npz'}
    for tail,name in names.items():
        key='head.motion_plan_head.'+tail
        expected=state[key].numpy();a=np.load(assets/name,allow_pickle=False)
        if isinstance(a,np.lib.npyio.NpzFile):a=a['trajectory_mask' if tail.endswith('mask') else 'trajectory']
        a=a.astype(expected.dtype)
        same=a.shape==expected.shape
        comparisons[key]=dict(shape_match=same,checkpoint_shape=list(expected.shape),file_shape=list(a.shape),equal=bool(np.array_equal(a,expected)) if same else False,max_abs=float(abs(a-expected).max()) if same else None)
    c=mmcv.Config.fromfile(str(rt/'official-source'/cfg['config']))
    embedded=mmcv.Config.fromfile(str(Path(cfg['private_root'])/'checkpoint_config.py')) if 'config' in meta else None
    structure={}
    for k in ('model','class_names','plan_config','time_points','input_shape','test_pipeline','data_aug_conf'):
        x=c.get(k);y=embedded.get(k) if embedded else None
        structure[k]=dict(exact_equal=x==y,embedded_present=y is not None)
    r=dict(status='Audited_NoModelForward',checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),checkpoint_size=path.stat().st_size,keys=len(state),parameter_elements=sum(x.numel() for x in state.values()),meta_keys=list(meta),tensors=tensors,arrays=arrays,anchor_comparisons=comparisons,embedded_config_comparison=structure)
    Path('reports/m8j12_sources/checkpoint_integrity.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({'arrays':arrays,'comparisons':comparisons,'config':structure},indent=2))
if __name__=='__main__':main()
