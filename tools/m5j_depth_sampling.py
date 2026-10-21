"""Read-only depth distributions and Oracle sampling phase controls; no S4 model."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from tools.m5j_common import rows,write
from tools.m5f_common import inputs,pixel_label
from tools.m5a_learning import encoder_load
from models.m5d_stride8 import cache_p8
from tools.m5g_depth_data import supervision,conditioned
from tools.m5f_factorization import reference
from models.m5f_lift import lift
from diagnostics.m5c_localization import matching
from datasets.labels import depth_bin_edges
from simulator.recorder import file_hash
from training.runtime import prepare_cuda

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',default='val');a=p.parse_args()
    if a.split=='test':
        from tools.m5j_freeze import verify
        verify()
    prepare_cuda();torch.set_num_threads(2);net=encoder_load();base={k:v.clone()for k,v in net.depth_head.state_dict().items()};models=[('original',0,base)];prior=json.loads(Path('reports/m5h_depth_preservation.json').read_text())['records'];models += [('replay',r['seed'],torch.load(r['selected']['checkpoint'],weights_only=True)['depth'])for r in prior if r['name']=='replay'];records=[];sampling=[];support=[];edges=torch.tensor(depth_bin_edges(),device='cuda');cacheidx={r['sample_id']:r for r in json.loads(Path(f'runs/m5j_{a.split}/index.json').read_text())['records']}
    for row in rows(a.split):
        x=inputs(row);d=pixel_label(row);net.depth_head.load_state_dict(base);c=cache_p8(net,x);f=conditioned(net,c);dep,v,g,_=supervision(net,c,d);actor=torch.from_numpy((d['actor'][:,4::8,4::8]>0)&d['target'][:,4::8,4::8])[None].cuda();truth,known,_=reference(d,x,row,row['camera_names']);gtbin=torch.bucketize(dep.contiguous(),edges,right=True)-1;binids=torch.arange(48,device='cuda')[None,None,:,None,None];z=np.load(cacheidx[row['sample_id']]['cache']);oraclefg=torch.from_numpy(d['target'][:,4::8,4::8][None]).float().cuda()
        for ci,cam in enumerate(row['camera_names']):
            m=(v&actor&(dep>=20)&(dep<35))[0,ci].cpu().numpy();classes=d['classes'][ci,4::8,4::8];support.append(dict(group=row['group'],sample_id=row['sample_id'],camera=cam,n=int(m.sum()),types={str(k):int((m&(classes==k)).sum())for k in [1,2,3]},uv=np.argwhere(m).tolist()))
        for name,seed,weight in models:
            net.depth_head.load_state_dict(weight)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):pr=net.depth_head(f.flatten(0,1)).reshape(1,6,48,48,80).float().softmax(2)
            exp=(pr*net.splat.depth_centers[None,None,:,None,None]).sum(2);fields=dict(mae=(exp-dep).abs(),bias=exp-dep,gt_bin_mass=(pr*(binids==gtbin[:,:,None])).sum(2),correct_layer=(pr*g['same_layer']).sum(2),space1m=(pr*g['same_layer']*(g['xy']<=1)).sum(2),distribution3d=(pr*(g['points']-g['gt'][:,:,None]).norm(dim=-1)).sum(2))
            for scope in ['all','associated_actor']:
                for lo,hi in [(1.5,80),(1.5,10),(10,20),(20,35),(35,80)]:
                    m=v&(dep>=lo)&(dep<hi)
                    if scope=='associated_actor':m &=actor
                    for ci,cam in enumerate(row['camera_names']):
                        cm=m[0,ci];out=dict(name=name,seed=seed,group=row['group'],sample_id=row['sample_id'],camera=cam,scope=scope,distance=f'{lo}_{hi}',n=int(cm.sum()))
                        for k,value in fields.items():
                            vm=cm&g['inside'][0,ci]if k in ['correct_layer','space1m']else cm;nn=int(vm.sum());out[k]=float(value[0,ci][vm].mean())if nn else None;out[k+'_n']=nn
                        records.append(out)
            out=lift(net.splat,oraclefg,pr,c['K'],c['T'],c['image_hw'],valid=v);sampling.append(dict(group=row['group'],sample_id=row['sample_id'],name=name,seed=seed,mode='Oracle2D_LearnedDepth_P8_strict',matching=matching((out['event_score'][0].cpu().numpy()>=.2)&known,truth&known)))
        for stride,phase in [(4,2),(4,0),(8,4),(8,0),(16,8),(16,0)]:
            vv,uu=np.mgrid[phase:384:stride,phase:640:stride];uv=torch.from_numpy(np.stack([uu,vv],-1)).cuda().float();dp=torch.from_numpy(d['depth'][:,vv,uu][None]).cuda();fg=torch.from_numpy(d['target'][:,vv,uu][None]).float().cuda();valid=torch.from_numpy(d['valid'][:,vv,uu][None]).cuda();h,w=vv.shape;dummy=torch.ones(1,6,1,h,w,device='cuda');out=lift(net.splat,fg,dummy,c['K'],c['T'],(384,640),depth=dp,valid=valid,uv=uv);score=out['event_score'][0].cpu().numpy();sampling.append(dict(group=row['group'],sample_id=row['sample_id'],mode='Oracle_sampling',stride=stride,phase=phase,matching=matching((score>=.2)&known,truth&known)))
            for seed in [101,102]:
                q=torch.from_numpy(z[f'{seed}_pixel'][:,vv,uu][None].astype(np.float32)).cuda();out=lift(net.splat,fg*(q>=.9),dummy,c['K'],c['T'],(384,640),depth=dp,valid=valid,uv=uv);sampling.append(dict(group=row['group'],sample_id=row['sample_id'],seed=seed,mode='MobileROI_Oracle_sampling',stride=stride,phase=phase,matching=matching((out['event_score'][0].cpu().numpy()>=.2)&known,truth&known),target_rays=int((valid&(fg>0)).sum()),retained_rays=int((valid&(fg>0)&(q>=.9)).sum())))
        print(row['sample_id'],flush=True)
    write(f'reports/m5j_depth_shadow_{a.split}.json',dict(kind='actual_GPU_readonly_depth_and_offline_Oracle_sampling',records=records,sampling=sampling,far_support=support,source_sha256=file_hash(__file__),limits='S4/phase usesGTDepthonly, no productionS4Depth; MobileROI isRGBonly. Alltargetreference includesROI misses. AssociatedActor3x3strictdepthmask; notindependentMesh orstatisticallyindependentrays. No newtraining.'))
if __name__=='__main__':main()
