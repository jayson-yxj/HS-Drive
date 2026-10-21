"""Separate physically associated Actor rays from broader semantic foreground metrics."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from tools.m5i_data import rows,ROOT,TEST
from tools.m5f_common import inputs,pixel_label
from tools.m5a_learning import encoder_load
from tools.m5g_depth_data import supervision,conditioned
from models.m5d_stride8 import cache_p8
from simulator.recorder import atomic_json,file_hash
from training.runtime import prepare_cuda

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['val','test'],default='val');a=p.parse_args();out=Path(f'reports/m5i_associated_actor_depth_{a.split}.json')
    if out.exists():raise FileExistsError(out)
    if a.split=='test':
        from tools.m5i_freeze import verify
        verify()
    prepare_cuda();torch.set_num_threads(2);net=encoder_load();base={k:v.clone()for k,v in net.depth_head.state_dict().items()};models=[('original',0,base)];study=json.loads(Path('reports/m5h_depth_preservation.json').read_text())['records']
    models += [('replay',r['seed'],torch.load(r['selected']['checkpoint'],weights_only=True)['depth'])for r in study if r['name']=='replay'];records=[]
    for row in rows(TEST if a.split=='test'else ROOT):
        if row['split']!=a.split:continue
        x=inputs(row);d=pixel_label(row);net.depth_head.load_state_dict(base);c=cache_p8(net,x);f=conditioned(net,c);dep,v,g,_=supervision(net,c,d);associated=torch.from_numpy((d['actor'][:,4::8,4::8]>0)&d['target'][:,4::8,4::8])[None].cuda();v=v&associated
        for name,seed,weights in models:
            net.depth_head.load_state_dict(weights)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):pr=net.depth_head(f.flatten(0,1)).reshape(1,6,48,48,80).float().softmax(2)
            exp=(pr*net.splat.depth_centers[None,None,:,None,None]).sum(2);delta=exp-dep;sp=(pr*g['same_layer']*(g['xy']<=1)).sum(2)
            for cidx,cam in enumerate(row['camera_names']):
                for lo,hi in [(1.5,10),(10,20),(20,35)]:
                    m=v[0,cidx]&(dep[0,cidx]>=lo)&(dep[0,cidx]<hi);n=int(m.sum());inside=m&g['inside'][0,cidx];ni=int(inside.sum());records.append(dict(name=name,seed=seed,group=row['group'],sample_id=row['sample_id'],camera=cam,distance=f'{lo}_{hi}',n=n,mae=float(delta[0,cidx][m].abs().mean())if n else None,bias=float(delta[0,cidx][m].mean())if n else None,inside_n=ni,space_1m_mass=float(sp[0,cidx][inside].mean())if ni else None))
    aggregate=[]
    for name,seed,_ in models:
        for dist in ['1.5_10','10_20','20_35']:
            rr=[r for r in records if r['name']==name and r['seed']==seed and r['distance']==dist];n=sum(r['n']for r in rr);ni=sum(r['inside_n']for r in rr);aggregate.append(dict(name=name,seed=seed,distance=dist,n=n,mae=sum((r['mae']or 0)*r['n']for r in rr)/n if n else None,bias=sum((r['bias']or 0)*r['n']for r in rr)/n if n else None,inside_n=ni,space_1m_mass=sum((r['space_1m_mass']or 0)*r['inside_n']for r in rr)/ni if ni else None))
    atomic_json(out,dict(kind='actual_GPU_readonly_associatedActor_subset',records=records,aggregate=aggregate,source_sha256=file_hash(__file__),limits='Uses rendererinstance/approximateActor association; not independentMesh. Broader actor scope in m5i_depth_* means semanticforeground; this report restricts associatedActor. No depthtraining/reselection.'))
if __name__=='__main__':main()
