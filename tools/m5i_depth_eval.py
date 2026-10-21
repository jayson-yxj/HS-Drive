"""Fixed-raster probability geometry metrics, Oracle2D isolates depth error."""
import argparse,json
from pathlib import Path
import torch,numpy as np
from models.m5d_stride8 import cache_p8
from tools.m5g_depth_data import supervision,conditioned
from tools.m5a_learning import encoder_load
from tools.m5f_common import inputs,pixel_label
from tools.m5f_factorization import reference
from models.m5f_lift import lift
from models.view_transform import unproject,feature_uv
from datasets.labels import depth_bin_edges
from diagnostics.m5c_localization import matching
from diagnostics.m5f_metrics import histogram,metrics
from simulator.recorder import atomic_json,file_hash
from training.runtime import prepare_cuda
from tools.m5i_data import ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default=ROOT);p.add_argument('--split',default='val');p.add_argument('--final',action='store_true');a=p.parse_args();dest=Path(f'reports/m5i_depth_{a.split}{"_final" if a.final else ""}.json')
    if dest.exists():raise FileExistsError(dest)
    if a.split=='test':
        from tools.m5i_freeze import verify
        verify()
    prepare_cuda();torch.set_num_threads(2);net=encoder_load();edges=torch.tensor(depth_bin_edges(),device='cuda',dtype=torch.float32);study=json.loads(Path('reports/m5h_depth_preservation.json').read_text());models=[dict(name='original',seed=0,weights={k:v.detach().clone()for k,v in net.depth_head.state_dict().items()})]
    for r in study['records']:
        if r['name']!='replay':continue
        cp=f"runs/m5h_depth/m5h_{r['name']}_{r['seed']}_step1200.pt" if a.final else r['selected']['checkpoint']
        models.append(dict(name=r['name'],seed=r['seed'],weights=torch.load(cp,weights_only=True)['depth'],checkpoint_sha256=file_hash(cp)))
    rows=[r for r in json.loads(Path(a.root,'index.json').read_text())['records']if r['split']==a.split];records=[];bev=[];aggregate={};camera_records=[];support=[]
    for row in rows:
        x=inputs(row);d=pixel_label(row);c=cache_p8(net,x);f=conditioned(net,c);dep,v,g,sp=supervision(net,c,d);target=torch.from_numpy(d['target'][:,4::8,4::8][None]).cuda();truth,known,actors=reference(d,x,row,row['camera_names']);gtid=torch.bucketize(dep.contiguous(),edges,right=True)-1
        for ci,cname in enumerate(row['camera_names']):
            mm=v[0,ci]&target[0,ci]&(dep[0,ci]>=20)&(dep[0,ci]<35);cls=d['classes'][ci,4::8,4::8];actor=d['actor'][ci,4::8,4::8];mask=mm.cpu().numpy();support.append(dict(sample_id=row['sample_id'],group=row['group'],camera=cname,n=int(mm.sum()),inside_n=int((mm&g['inside'][0,ci]).sum()),classes={str(k):int((mask&(cls==k)).sum())for k in [1,2,3]},actor_ids=np.unique(actor[mask]).tolist(),uv_yx=np.argwhere(mask).tolist()))
        for model in models:
            net.depth_head.load_state_dict(model['weights'])
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):log=net.depth_head(f.flatten(0,1)).reshape(1,6,48,48,80);pr=log.float().softmax(2)
            centers=net.splat.depth_centers;exp=(pr*centers[None,None,:,None,None]).sum(2);mode=centers[pr.argmax(2)];delta=g['points']-g['gt'][:,:,None];fullerr=(pr*delta.norm(dim=-1)).sum(2);uv=feature_uv(48,80,(384,640),'cuda').round();expoint=unproject(uv.expand(1,6,48,80,2),exp,c['K'],c['T']);expectederr=(expoint-g['gt']).norm(dim=-1);binids=torch.arange(48,device='cuda')[None,None,:,None,None]
            values=dict(expected_mae=(exp-dep).abs(),signed_bias=exp-dep,mode_mae=(mode-dep).abs(),gt_bin_mass=(pr*(binids==gtid[:,:,None])).sum(2),gt_bin_pm1_mass=(pr*((binids-gtid[:,:,None]).abs()<=1)).sum(2),correct_layer_mass=(pr*g['same_layer']).sum(2),distribution_expected_space_error=fullerr,expected_depth_point_error=expectederr)
            values['outside_bev_probability']=(pr*~g['bin_inside']).sum(2)
            values['near_range_probability']=(pr*(centers[None,None,:,None,None]<10)).sum(2)
            for tol in [.5,1,2]:values[f'same_layer_{tol}m_mass']=(pr*g['same_layer']*(g['xy']<=tol)).sum(2)
            for scope in ['all','actor','background']:
                for lo,hi in [(1.5,80),(1.5,10),(10,20),(20,35),(35,60),(60,80)]:
                    m=v&(dep>=lo)&(dep<hi)
                    if scope=='actor':m &= target
                    if scope=='background':m &= ~target
                    n=int(m.sum());inside=int((m&g['inside']).sum());r=dict(name=model['name'],seed=model['seed'],group=row['group'],sample_id=row['sample_id'],scope=scope,distance=f'{lo}_{hi}',n=n,bev_inside_n=inside)
                    # Layer/XY mass meaningful only where GT insideBEV; depth/3D allvalid.
                    for key,val in values.items():
                        mask=m&g['inside']if 'mass'in key and 'bin'not in key else m;nn=int(mask.sum());r[key]=float(val[mask].mean())if nn else None;r[key+'_n']=nn
                    records.append(r)
                    for ci,cname in enumerate(row['camera_names']):
                        cm=m[0,ci];nn=int(cm.sum());cn=dict(name=model['name'],seed=model['seed'],group=row['group'],weather=row['weather'],sample_id=row['sample_id'],camera=cname,scope=scope,distance=r['distance'],n=nn)
                        for vk in ['expected_mae','signed_bias','gt_bin_mass','correct_layer_mass','same_layer_1m_mass','outside_bev_probability']:
                            vm=cm&g['inside'][0,ci]if 'mass'in vk and 'bin'not in vk else cm;cnt=int(vm.sum());cn[vk]=float(values[vk][0,ci][vm].mean())if cnt else None;cn[vk+'_n']=cnt
                        camera_records.append(cn)
                    ak=(model['name'],model['seed'],scope,r['distance']);acc=aggregate.setdefault(ak,dict(n=0,bev_inside_n=0,values={}));acc['n']+=n;acc['bev_inside_n']+=inside
                    for key in values:
                        pair=acc['values'].setdefault(key,[0.,0]);pair[0]+=(r[key]or 0)*r[key+'_n'];pair[1]+=r[key+'_n']
            with torch.no_grad():out=lift(net.splat,target.float(),pr,c['K'],c['T'],c['image_hw'],valid=v)
            score=out['event_score'][0].cpu().numpy();pred=(score>=int(.2*256)/256)&known;bev.append(dict(name=model['name'],seed=model['seed'],group=row['group'],configuration=row['configuration'],metrics=metrics(histogram(score,truth,known),.2),matching=matching(pred,truth&known),actor=[dict(type_id=ac['type_id'],distance_m=ac['distance_m'],n=int(ac['mask'].sum()),matching=matching(pred,ac['mask']))for ac in actors],threshold=.2))
        # restore original before next RGBcache (features independent but probs copied forcertainty)
        net.depth_head.load_state_dict(models[0]['weights']);print(row['sample_id'],flush=True)
    agg=[dict(name=k[0],seed=k[1],scope=k[2],distance=k[3],n=v['n'],bev_inside_n=v['bev_inside_n'],metrics={key:dict(n=pair[1],mean=pair[0]/pair[1]if pair[1]else None)for key,pair in v['values'].items()})for k,v in aggregate.items()]
    atomic_json(dest,dict(kind='actual_GPU_fixed_RGB_visible_depth_distribution_and_Oracle2D_projection',records=records,camera_records=camera_records,far_support=support,aggregate=agg,bev=bev,checkpoint_mode='final1200_diagnostic_not_selected' if a.final else 'validation_neargate_selected',checkpoint_sha256={m['name']+str(m['seed']):m.get('checkpoint_sha256','originalM4') for m in models},source_sha256=file_hash(__file__),limits='SameMask/UV/models. OpticalZbins1.5-80 unchanged. Layer/spatialmass GTinsideBEV denominator; depthMAE allvalid. Full3D expectederr includesoutofgrid probability. Selfrenderer reference not independentMesh. NoGTformalforward.'))

if __name__=='__main__':main()
