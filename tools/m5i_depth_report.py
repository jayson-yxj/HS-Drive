"""Fixed qualification gates; no fit or calibration on Test."""
import json
from pathlib import Path
import numpy as np
from simulator.recorder import atomic_json,file_hash

def summarize(report):
    aggregate=report['aggregate'];models=sorted({(r['name'],r['seed'])for r in aggregate});out=[]
    def find(name,seed,scope,dist):return next(r for r in aggregate if r['name']==name and r['seed']==seed and r['scope']==scope and r['distance']==dist)
    def measure(model,scope,dist,key):return find(*model,scope,dist)['metrics'][key]['mean']
    for name,seed in models:
        near=[]
        for group in sorted({r['group']for r in report['records']}):
            vals=[]
            for n,s in [('original',0),(name,seed)]:
                rs=[r for r in report['records']if r['group']==group and r['name']==n and r['seed']==s and r['scope']=='all'and r['distance']=='1.5_10'];count=sum(r['n']for r in rs);vals.append(sum((r['expected_mae']or 0)*r['n']for r in rs)/count if count else None)
            near.append(dict(group=group,original=vals[0],candidate=vals[1],passed=vals[1]<=1.05*vals[0]))
        bias=[]
        for dist,margin in [('1.5_80',.25),('10_20',.5),('20_35',.5)]:
            a=measure(('original',0),'all',dist,'signed_bias');b=measure((name,seed),'all',dist,'signed_bias');bias.append(dict(distance=dist,original=a,candidate=b,absolute_margin=margin,passed=abs(b)<=abs(a)+margin))
        space=[]
        for dist in ['10_20','20_35']:
            a=measure(('original',0),'actor',dist,'same_layer_1m_mass');b=measure((name,seed),'actor',dist,'same_layer_1m_mass');space.append(dict(distance=dist,original=a,candidate=b,passed=a is not None and b is not None and b>=a))
        bevs=[]
        for n,s in [('original',0),(name,seed)]:
            rr=[r for r in report['bev']if r['name']==n and r['seed']==s];count=sum(r['matching']['tolerances']['1.0']['gt_count']for r in rr);hit=sum(r['matching']['tolerances']['1.0']['matched_gt']for r in rr);bevs.append(hit/count if count else None)
        far=report['far_support'];groups=sorted({r['group']for r in far if r['n']});types=[k for k in ['1','2','3']if sum(r['classes'][k]for r in far)>0];count=sum(r['n']for r in far);fresh=[r for r in far if '_itest' in r['group'] or '_ival' in r['group']];freshgroups=sorted({r['group']for r in fresh if r['n']});freshn=sum(r['n']for r in fresh);freshtypes=[k for k in ['1','2','3']if sum(r['classes'][k]for r in fresh)>0];support=dict(rays=count,bev_inside_rays=sum(r['inside_n']for r in far),groups=groups,types=types,actor_views=sum(bool(r['actor_ids'])for r in far),fresh_rays=freshn,fresh_groups=freshgroups,fresh_types=freshtypes,passed=count>=100 and len(groups)>=4 and len(types)>=2 and (not any('_itest' in r['group']for r in far)or(freshn>=100 and len(freshgroups)>=4 and len(freshtypes)>=2)))
        ok=all(r['passed']for r in near+bias+space)and support['passed']and bevs[1]>=bevs[0]-.01
        out.append(dict(name=name,seed=seed,near_by_scene=near,bias_checks=bias,space_checks=space,oracle_2d_recall=dict(original=bevs[0],candidate=bevs[1],passed=bevs[1]>=bevs[0]-.01),far_support=support,qualified_on_this_split=ok))
    return out

def main():
    dest=Path('reports/m5i_depth_qualification.json')
    if dest.exists():raise FileExistsError(dest)
    val=json.loads(Path('reports/m5i_depth_val.json').read_text());test=json.loads(Path('reports/m5i_depth_test.json').read_text());v=summarize(val);t=summarize(test);orig=json.loads(Path('reports/m5i_depth_original_task.json').read_text());original_checks=[]
    for split,decisions in [('val',v),('test',t)]:
        support=json.loads(Path(f'reports/m5i_depth_support_{split}.json').read_text())
        for decision in decisions:
            decision['associated_actor_support']={k:support[k]for k in ['semantic_rays','associated_actor_rays','groups','classes','passed']}
            decision['qualified_on_this_split'] &= support['passed']
    for seed in [101,102]:
        for group in sorted({r['sample_id'].split('/')[0]for r in orig['records']}):
            nums=[]
            for name,s in [('original',0),('replay',seed)]:
                rr=[r for r in orig['records']if r['name']==name and r['seed']==s and r['sample_id'].startswith(group+'/')];n=sum(r['n']for r in rr);nums.append(sum(r['mae']*r['n']for r in rr)/n)
            original_checks.append(dict(seed=seed,group=group,original=nums[0],candidate=nums[1],passed=nums[1]<=nums[0]*1.05))
    decisions=[dict(seed=seed,qualified=all(next(r for r in rr if r['name']=='replay'and r['seed']==seed)['qualified_on_this_split']for rr in [v,t])and all(r['passed']for r in original_checks if r['seed']==seed))for seed in [101,102]]
    atomic_json(dest,dict(kind='offline_qualification_actual_GPU_readonly_candidates',validation=v,test=t,original_S16_checks=original_checks,decisions=decisions,default_changed=False,new_depth_training=False,calibration=False,source_sha256=file_hash(__file__),limits='Rays areunique sample/camera/UV support but correlated surfacepixels; notstatistically independent. Near/Bias/farspace/support separatechecks. No testfitting.'))
    # Stratification retains global, scene, camera, weather, range for bias attribution.
    reports={}
    for split,data in [('val',val),('test',test)]:
        rows=data['camera_records'];ag={}
        for r in rows:
            for category in ['all','camera:'+r['camera'],'scene:'+r['group'],'weather:'+str(r['weather'])]:
                key=(r['name'],r['seed'],r['scope'],r['distance'],category);a=ag.setdefault(key,dict(n=0,error=0.,bias=0.));a['n']+=r['n'];a['error']+=(r['expected_mae']or 0)*r['n'];a['bias']+=(r['signed_bias']or 0)*r['n']
        reports[split]=[dict(name=k[0],seed=k[1],scope=k[2],distance=k[3],category=k[4],n=a['n'],mae=a['error']/a['n']if a['n']else None,bias=a['bias']/a['n']if a['n']else None)for k,a in ag.items()]
    atomic_json('reports/m5i_depth_bias.json',dict(kind='offline_attribution_actual_GPU_fixed_depth_predictions',splits=reports,source_sha256=file_hash(__file__),calibration='Not executed; no globaloffset, defaultHeadunchanged',limits='Scene/camera/weather confounded unlesspaired; observationalstratification notcausalproof.'))
if __name__=='__main__':main()
