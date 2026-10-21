"""Descriptive scene/camera/class comparisons of already frozen models; no reselection."""
import json
from pathlib import Path
import numpy as np
from simulator.recorder import atomic_json,file_hash

def read(n):return json.loads(Path('reports',n+'.json').read_text())
def main():
    out=Path('reports/m5i_generalization_breakdown.json')
    if out.exists():raise FileExistsError(out)
    records=read('m5i_pixel_test')['records'];comparisons=[];sizes=[]
    for seed in [101,102]:
        base=next(r for r in records if r['name']=='gt'and r['seed']==seed)
        for name in ['output_kd','mobile_gt']:
            r=next(r for r in records if r['name']==name and r['seed']==seed)
            for category in ['by_scene','strata']:
                for key,m in r['result'][category].items():
                    if category=='strata'and not key.startswith('camera:'):continue
                    b=base['result'][category][key];comparisons.append(dict(seed=seed,name=name,category=category,key=key,positive=m['positive'],precision_delta=m['precision']-b['precision'],recall_delta=(m['recall']-b['recall']) if m['recall'] is not None and b['recall'] is not None else None,ap_delta=(m['histogram_ap']or 0)-(b['histogram_ap']or 0)))
    for r in records:
        for lo,hi in [(3,32),(33,512),(513,10**9)]:
            rr=[a for a in r['result']['actors']if lo<=a['pixels']<=hi];n=len(rr);sizes.append(dict(name=r['name'],seed=r['seed'],pixel_area=[lo,hi],actor_views=n,detected=sum(a['detected']for a in rr),recall=sum(a['detected']for a in rr)/n if n else None))
    frozen=read('m5i_pretest_freeze');ss=read('m5i_shadow_comparison')['records'];spatial=[]
    for seed in [101,102]:
        for mode in sorted({r['mode']for r in ss}):
            rr=[r for r in ss if r['seed']==seed and r['mode']==mode];entry=dict(seed=seed,mode=mode)
            for direction in ['pred_to_gt','gt_to_pred']:
                ds=[r['voxel3d'][direction]for r in rr];n=sum(r['n']-r['no_candidate']for r in ds);entry[direction]=dict(n=n,no_candidate=sum(r['no_candidate']for r in ds),mean_m=sum((r['mean_m']or 0)*(r['n']-r['no_candidate'])for r in ds)/n if n else None,unmatched_1m=sum(r['unmatched_1m']for r in ds))
            spatial.append(entry)
    atomic_json(out,dict(kind='offline_descriptive_frozen_test_comparisons',comparisons=comparisons,small_object_actor_views=sizes,coarse_spatial_errors=spatial,source_sha256=file_hash(__file__),limits='Six correlated location groups; signs/counts not significance tests. Differentselectedthresholds fixedbyVal; AP thresholdfree. Area usesGTvisiblepixels evaluationonly. No model changes basedonTest.'))
    lines=['# M5-I 补充分层诊断','', '以下为封存后描述统计，不用于重选模型/阈值。六个位置组不足以代表广泛驾驶域；重复配置和像素不是独立场景。','', '|候选|seed|未见位置AP提高数|位置P/R同时提高数|六摄P/R同时提高数|','|---|---:|---:|---:|---:|']
    for seed in [101,102]:
        for name in ['output_kd','mobile_gt']:
            rr=[r for r in comparisons if r['seed']==seed and r['name']==name];sc=[r for r in rr if r['category']=='by_scene'];ca=[r for r in rr if r['category']=='strata'and r['key'].endswith('/all')];lines.append(f"|{name}|{seed}|{sum(r['ap_delta']>0 for r in sc)}/{len(sc)}|{sum(r['precision_delta']>0 and (r['recall_delta']or 0)>0 for r in sc)}/{len(sc)}|{sum(r['precision_delta']>0 and (r['recall_delta']or 0)>0 for r in ca)}/{len(ca)}|")
    lines+=['', '|模型|seed|小目标3–32像素Actor-view|检出Recall|','|---|---:|---:|---:|']
    for r in sizes:
        if r['pixel_area'][0]!=3:continue
        lines.append(f"|{r['name']}|{r['seed']}|{r['actor_views']}|{r['recall']}|")
    lines+=['', '实际三维误差以四层粗体素中心为参考，两个方向与无候选数量均单列在JSON，不能称精确物理表面/Actor定位。Teacher不支持cone，只能在vehicle/person同Mask上比较。','', 'Bias的相机/天气归因是相关性：本数据没有独立控制背景与天气，不把某个天气桶的变化解释为天气的因果影响。远场支持还需通过具体Actor关联，单纯语义目标像素不能充数。']
    Path('reports/m5i_findings.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
