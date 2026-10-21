"""Independent Go/No-go research decisions from frozen actual experiments."""
import json,subprocess
from pathlib import Path
import numpy as np
from tools.m5j_common import write
from tools.m5j_metrics import aggregate
from simulator.recorder import file_hash

def read(n):return json.loads(Path('reports',n+'.json').read_text())
def main():
    from tools.m5j_freeze import verify
    freeze=verify();val=read('m5j_aggregation_val');test=read('m5j_aggregation_test');sel=read('m5j_selection');details=read('m5j_spatial_details_test');pop=read('m5j_attribution_population_test');lat=read('m5j_latency');pix=read('m5j_camera_quality_test');depth=read('m5j_depth_shadow_test');migration=read('m5j_mobile_migration');dataset=read('m5j_dataset_quality');tests=read('m5j_test_results');paired=[];work=[];scenes=[]
    for seed in [101,102]:
        raw=next(r for r in test['aggregate']if r['seed']==seed and r['method']=='raw'and r['threshold']==.2)
        for method in ['raw','gate','regularized']:
            for label in ['selected','recall_matched','precision_matched']:
                t=sel['workpoints'][f'{seed}_{method}'][label];r=next(r for r in test['aggregate']if r['seed']==seed and r['method']==method and r['threshold']==t);work.append(dict(seed=seed,method=method,workpoint=label,threshold=t,precision=r['precision'],recall1=r['recall1'],actor_recall=r['actor_recall'],actor_detection=r['actor_detection'],fpr=r['fpr'],precision_gain=r['precision']-raw['precision'],recall1_delta=r['recall1']-raw['recall1'],passed=r['precision']>=raw['precision']+.05 and r['recall1']>=raw['recall1']-.02 and r['actor_detection']>=raw['actor_detection']-.02))
            t=sel['workpoints'][f'{seed}_{method}']['selected']
            for group in sorted({r['group']for r in test['records']}):
                rr=[r for r in test['records']if r['seed']==seed and r['method']==method and r['threshold']==t and r['group']==group];scenes.append(dict(seed=seed,method=method,group=group,**aggregate(rr)))
            rr=[r for r in details['pairs']if r['seed']==seed and r['method']==method];n=sum(r['n']for r in rr);paired.append(dict(seed=seed,method=method,n=n,removed_fpr=sum(r['removed_fp']for r in rr)/n if n else None))
    write('reports/m5j_spatial_precision_recall.json',dict(kind='offline_frozen_Valmatched_Testworkpoints',workpoints=work,scenes=scenes,removed=paired,curve_source='m5j_aggregation_test.json',limits='Matching doneVal; Testrecallneednotmatch duegeneralization. TestPRcurves descriptive no posthoc thresholdselection.'))
    attribution=[]
    for seed in [101,102]:
        for domain in ['target','background','unknown']:
            rr=[r for r in pop['records']if r['seed']==seed and r['domain']==domain and r['area']=='all'];mass=sum(r['mass']for r in rr);sources={k:sum(r['source_mass'][k]for r in rr)for k in rr[0]['source_mass']};attribution.append(dict(seed=seed,domain=domain,high_cells=sum(r['high_cells']for r in rr),mass=mass,source_fraction={k:v/mass if mass else None for k,v in sources.items()},lowq_mass_fraction=sum(r['low_q_mass']for r in rr)/mass if mass else None,subthreshold_ray_accumulation_cells=sum(r['D_all_individual_rays_subthreshold']for r in rr),ray_bin_count=sum(r['ray_bin_count']for r in rr),distinct_ray_cells=sum(r['distinct_ray_cells']for r in rr)))
    write('reports/m5j_evidence_attribution.json',dict(kind='offline_allhighcell_source_summary',records=attribution,trace='m5j_evidence_attribution_test.json',population='m5j_attribution_population_test.json',limits=pop['limits']))
    pp=[]
    for seed in [101,102]:
        mobile=next(r for r in pix['records']if r['name']=='mobile_gt'and r['seed']==seed);old=next(r for r in pix['records']if r['name']=='legacy'and r['seed']==seed);m=mobile['metrics'];b=old['metrics'];pp.append(dict(seed=seed,mobile=m,legacy=b,scene_PR_both=sum(mobile['by_scene'][g]['precision']>old['by_scene'][g]['precision']and mobile['by_scene'][g]['recall']>old['by_scene'][g]['recall']for g in mobile['by_scene']),precision_gain=m['precision']-b['precision'],recall_gain=m['recall']-b['recall'],sizes=mobile['sizes']))
    write('reports/m5j_camera_quality.json',dict(kind='frozen_newTest_vs_historical_regression',paired=pp,new_test='m5j_camera_quality_test.json',historical='m5j_camera_quality_historical.json',limits='Legacy M5G has different historical training protocol, architecture comparison not singlevariable; reportfixedexistingcandidates.'))
    sampling=[]
    for mode in ['Oracle_sampling','MobileROI_Oracle_sampling']:
        for stride,phase in [(4,2),(4,0),(8,4),(8,0),(16,8),(16,0)]:
            for seed in ([None]if mode=='Oracle_sampling'else[101,102]):
                rr=[r for r in depth['sampling']if r['mode']==mode and r['stride']==stride and r['phase']==phase and r.get('seed')==seed];n=sum(r['matching']['tolerances']['1.0']['gt_count']for r in rr);hit=sum(r['matching']['tolerances']['1.0']['matched_gt']for r in rr);sampling.append(dict(mode=mode,stride=stride,phase=phase,seed=seed,gt_cells=n,recall1=hit/n if n else None))
    agg=[]
    for name,seed in [('original',0),('replay',101),('replay',102)]:
        for scope in ['all','associated_actor']:
            for dist in ['1.5_80','1.5_10','10_20','20_35','35_80']:
                rr=[r for r in depth['records']if r['name']==name and r['seed']==seed and r['scope']==scope and r['distance']==dist];out=dict(name=name,seed=seed,scope=scope,distance=dist,n=sum(r['n']for r in rr))
                for k in ['mae','bias','gt_bin_mass','correct_layer','space1m','distribution3d']:
                    n=sum(r[k+'_n']for r in rr);out[k]=sum((r[k]or 0)*r[k+'_n']for r in rr)/n if n else None;out[k+'_n']=n
                agg.append(out)
    write('reports/m5j_depth_shadow.json',dict(kind='offline_frozen_readonly_depth_sampling_summary',aggregate=agg,sampling=sampling,far_rays=sum(r['n']for r in depth['far_support']),far_groups=sorted({r['group']for r in depth['far_support']if r['n']}),far_types={str(k):sum(r['types'][str(k)]for r in depth['far_support'])for k in [1,2,3]},new_depth_training=False,default_changed=False,details='m5j_depth_shadow_test.json'))
    write('reports/m5j_aggregation_ablation.json',dict(kind='controlled_fixed_pixel_depth_geometry_ablation',selection=sel,test=work,fixed_threshold=[r for r in test['aggregate']if r['threshold']==.2],oracle='m5j_spatial_details_test.json',calibration_executed=False,limits='Threepredeclared formulas only, Dconditionfailed no learning. Correctmasked references identical.'))
    visualgo=all(r['precision_gain']>0 and r['recall_gain']>0 for r in pp);aggo=sel['selected']!='raw'and all(r['passed']for r in work if r['method']==sel['selected']and r['workpoint']=='selected');budget=all(r['p95']<=50 for r in lat['records']if r['visual']=='mobile_gt');gates=dict(engineering='Passed'if migration['passed']and dataset['passed']and tests['passed']else'Partial',mobile_migration='Go_OptInResearchOnly'if visualgo else'NoGo',visual_absolute_generalization='Partial',aggregation='GoResearchOnly'if aggo else'NoGo_PredeclaredGainNotMet',depth_upgrade='NoGo_NotRequalified',BEV_spatial_quality='Partial',complete_offline_gpu_budget='Passed'if budget else'Partial',runtime='PartialInherited_NoRuntimeChanges',oracle_driving_safety='NotRevalidated',pure_visual_driving='NotValidated',occupancy_pilot='NoGo_ThisRound')
    write('reports/m5j_acceptance.json',dict(stage='M5-J',status='Partial',gates=gates,Go_NoGo=dict(mobile_optin=visualgo,aggregation_upgrade=aggo,default_encoder_change=False,occupancy_pilot=False,online_visual_control=False),source_sha256=file_hash(__file__),freeze_commit=freeze['commit'],delivery_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    fmt=lambda v:'n/a'if v is None else f'{v*100:.2f}%'
    lines=['# M5-J — Mobile Visual Migration & BEV Evidence Calibration','', '**结论：研究有明确Go/No-go，整体质量仍Partial。** Mobile为显式独立双RGB分支；Depth与Context仍使用原ResNet，未交叉喂入Mobile特征。旧默认、48LID、HeightSplat、四层BEV、Controller/Physics全部保留。','', '## 决策','',f'- Mobile独立研究迁移：{gates["mobile_migration"]}；完整双分支GPU预算：{gates["complete_offline_gpu_budget"]}。',f'- 新聚合升级：{gates["aggregation"]}。固定公式A/B/C未通过两seed预声明+5pp Precision、最多2pp同层1m Recall下降与最多2pp Actor检出下降。D条件未满足，未训练校准层；无视觉或Depth训练。','- 正式BEV质量、Occupancy Pilot、Runtime及纯视觉安全门禁没有自动关闭。本轮不启动联合训练、Temporal、Sparse Query或视觉控制。','', '## 新位置二维对照','', '|seed|Mobile P/R|旧候选P/R|P/R同增位置|','|---:|---|---|---:|']
    for r in pp:lines.append(f"|{r['seed']}|{fmt(r['mobile']['precision'])}/{fmt(r['mobile']['recall'])}|{fmt(r['legacy']['precision'])}/{fmt(r['legacy']['recall'])}|{r['scene_PR_both']}/4|")
    lines+=['', '仅4个独立位置，车型/配置有共享，不能按像素或重复帧数量宣称广泛泛化；历史Test仅诊断未训练。小目标支持见逐Actor报告。','', '## BEV工作点（Val选择，Test不调阈值）','', '|seed|方法|阈值|精确表面P|同层1m R|Actor检出|背景FPR|','|---:|---|---:|---:|---:|---:|---:|']
    for r in work:
        if r['workpoint']=='selected':lines.append(f"|{r['seed']}|{r['method']}|{r['threshold']}|{fmt(r['precision'])}|{fmt(r['recall1'])}|{fmt(r['actor_detection'])}|{fmt(r['fpr'])}|")
    lines+=['', '固定0.2、Val锁定、Recall匹配和Precision匹配分别保存。Test曲线只是诊断，不按Test重选。正则化保证score≤raw且分母total+1，不会放大小质量；固定0.2的FPR下降必须同时看Recall损失。','', '## 高分格来源','', '|seed|区域|高分格|非目标源质量A|目标错位质量B|未知源E|低q质量比例|','|---:|---|---:|---:|---:|---:|---:|']
    for r in attribution:lines.append(f"|{r['seed']}|{r['domain']}|{r['high_cells']}|{fmt(r['source_fraction']['A_wrong2D'])}|{fmt(r['source_fraction']['B_target_depth_othercell'])}|{fmt(r['source_fraction']['E_source_unknown'])}|{fmt(r['lowq_mass_fraction'])}|")
    lines+=['', 'A/B/E为射线源贡献描述，C概率扩散与D累积可重叠，不是独立因果百分比。来源GT不可用时为E；Unknown高分不叫物理FP。ray_bin_count与distinct camera/pixel射线数严格区分；不同射线仍相关。','', '## 采样与Depth','', '|Oracle采样|phase|同层1m R|','|---|---:|---:|']
    for r in sampling:
        if r['mode']=='Oracle_sampling':lines.append(f"|S{r['stride']}|{r['phase']}|{fmt(r['recall1'])}|")
    lines+=['',f"新Test严格20–35m关联Actor射线：{sum(r['n']for r in depth['far_support'])}；详细类型/位置在m5j_depth_shadow.json。MAE、Bias、GTBin、正确层和同层1m概率分开评估，Replay仍只读，不因MAE单独改善升级。S4仅Oracle深度采样诊断，没有生产S4Depth网络；ROI结果分母包含漏掉的目标。",'', '## 完整推理成本','', '|视觉分支|seed|P50 ms|P95 ms|峰值MiB|','|---|---:|---:|---:|---:|']
    for r in lat['records']:lines.append(f"|{r['visual']}|{r['seed']}|{r['p50']:.2f}|{r['p95']:.2f}|{r['peak_mib']:.1f}|")
    lines+=['', '含输入检查、Pixel、原Depth、Lift和精确支持计数；CPU读图/解码/SHA、序列化、反序列化和H2D另测。不包含CARLA控制周期，不称20Hz在线闭环。','', '## 交付与限制','', '所有模型/配置/数据及源码SHA、原始日志、失败恢复记录保留。标签来源仍为CARLA可靠可见首表面，非独立Mesh。空间事件分数不是物理Occupancy或安全Free。', '', '复现见[实施说明](../docs/M5J_IMPLEMENTATION.md)，决策见[ADR](../docs/M5J_ARCHITECTURE_DECISION.md)。[PR曲线](m5j_figures/m5j_pr.png)和[错位射线图](m5j_figures/m5j_trace_B_target_depth_misprojected.png)展示实际结果。']
    Path('reports/m5j_summary.md').write_text('\n'.join(lines)+'\n')
    Path('docs/M5J_ARCHITECTURE_DECISION.md').write_text('# M5-J Architecture Decision\n\n'+json.dumps(gates,ensure_ascii=False,indent=2)+'\n\nGo：仅在相对视觉增益及接口通过时，保留Mobile双分支为显式离线研究Candidate。No-go：不替换默认Encoder，不升级聚合，不升级Replay，不启动Occupancy联合Pilot。\n\nA–C的阈值匹配收益不足，停止继续增加公式/校准复杂度。下一步将空间研究集中到可靠Actor支持、Depth概率落点与真实采样密度；S4Oracle收益不能自动外推到学习S4网络。未来有限训练必须另立协议并使用新封存组。\n\n来源分解见报告：二维错误、GT目标错位、未知射线与累积同时存在。当前干预只能评估固定模型条件下的效果，不声称数学识别唯一因果。Runtime和驾驶安全保持独立开放。\n')
if __name__=='__main__':main()
