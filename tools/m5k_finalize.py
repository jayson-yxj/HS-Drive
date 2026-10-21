"""Compact results and independent gates from actual frozen runs."""
import json,subprocess
from pathlib import Path
from tools.m5k_common import write,asset
from simulator.recorder import file_hash

def read(name):return json.loads(Path('reports/'+name+'.json').read_text())
def find(rows,**kv):return next(r for r in rows if all(r[k]==v for k,v in kv.items()))
def main():
 decision=read('m5k_training_decision');aligned={s:read('m5k_alignment_'+s)for s in ['historical_val','train','val','test']};support={s:read('m5k_support_'+s)for s in ['train','val','test']};sampling={s:read('m5k_sampling_'+s)for s in ['val','test']};runtime=read('m5k_input_pipeline');residual={s:read('m5k_residual_'+s)for s in ['val','test']}if decision['run_residual']else{};gains=[]
 for split in ['val','test']:
  for seed in [101,102]:
   baseline=find(aligned[split]['bev'],method='original_p8',seed=seed)
   candidates=[('native16_to_p8',find(aligned[split]['bev'],method='native16_to_p8',seed=seed))]
   if residual:candidates += [(mode+str(seed),find(residual[split]['bev'],method=mode+str(seed),seed=seed))for mode in ['no_mobile','mobile']]
   for method,r in candidates:
    delta=dict(precision=r['precision']-baseline['precision'],recall1=r['recall1']-baseline['recall1'],fpr=r['fpr']-baseline['fpr'],actor_detection=r['actor_detection']-baseline['actor_detection']);passed=delta['precision']>=.01 and delta['recall1']>=.05 and delta['fpr']<=.001 and delta['actor_detection']>=0
    source=aligned[split]if method=='native16_to_p8'else residual[split];near=find(source['depth'],method=method,scope='all',band='near');old=find(aligned[split]['depth'],method='original_p8',scope='all',band='near');ng=near['expected_mae']<=old['expected_mae']*1.05;protection=[]
    for band in ['all','10_20','20_35']:
     now=find(source['depth'],method=method,scope='all',band=band);before=find(aligned[split]['depth'],method='original_p8',scope='all',band=band);protection.append(abs(now['bias'])<=abs(before['bias'])+(.25 if band=='all'else .5))
    an=next((r for r in source['depth']if r['method']==method and r['scope']=='actor'and r['band']=='near'),None);ao=next((r for r in aligned[split]['depth']if r['method']=='original_p8'and r['scope']=='actor'and r['band']=='near'),None);actor_near=(an['space1']>=ao['space1'])if an and ao else None
    gains.append(dict(split=split,seed=seed,method=method,deltas=delta,bev_gate=passed,near_all_gate=ng,bias_gate=all(protection),near_actor_space_gate=actor_near,combined_passed=passed and ng and all(protection)and actor_near is True))
 sampling_gains=[]
 for split in ['val','test']:
  for seed in [101,102]:
   for dep in ['oracle','oracle2d','original','aligned']:
    a=find(sampling[split]['records'],seed=seed,depth=dep,sampling='adaptive');b=find(sampling[split]['records'],seed=seed,depth=dep,sampling='uniform');sampling_gains.append(dict(split=split,seed=seed,depth=dep,recall1_gain=a['recall1']-b['recall1'],precision_gain=(a['precision']or 0)-(b['precision']or 0),fpr_delta=a['fpr']-b['fpr'],passed=a['recall1']>b['recall1'] and a['fpr']-b['fpr']<=.001))
 samplego=all(r['passed']for r in sampling_gains if r['split']=='val'and r['depth']=='original');qualitygo=any(all(r['combined_passed']for r in gains if r['split']=='val'and r['method']==method)for method in {r['method']for r in gains})
 # Qualification additionally requires position-level near protection, originalM4,
 # two seeds and independent test support; no aggregate-only auto-promotion.
 write('reports/m5k_depth_stride_alignment.json',dict(kind='actual_RTX4090_fixed_ray_alignment',results={s:asset(f'reports/m5k_alignment_{s}.json')for s in aligned},validation=decision['alignment'],gains=[r for r in gains if r['method']=='native16_to_p8'],decision='NoDefaultUpgrade',definition='A/B share exact strict GTinBEV mask. S16 density separate. Mode is mostlikelybincenter; Expected is softmaxmean; Oraclecontinuous diagnostic.',limits='MAE reduction not sufficient to infer feature shift accountsfor error. Spatialmass and realMobileBEV decisive.'))
 write('reports/m5k_actor_depth_support.json',dict(kind='actual_recorded_CARLA_offline_support',splits={s:dict(eligibility=d['eligibility'],actor_view_size=d['actor_view_size'],report=asset(f'reports/m5k_support_{s}.json'))for s,d in support.items()},decision=decision['status'],positions=asset('reports/m5k_position_allocation.json'),limits='No independent Mesh; strict rendererinterior+AABB association. Raycount not independent scenes. Smallview counts reliableassociated pixels. Tests never used for train support selection.'))
 write('reports/m5k_depth_spatial_ablation.json',dict(kind='actual_GPU_conditional_research',training_decision=decision,training=asset('reports/m5k_depth_training.json')if residual else None,residual_evaluation={s:asset(f'reports/m5k_residual_{s}.json')for s in residual},gains=gains,Mobile_condition_conclusion='See matched B/C; qualification notautomatic'if residual else'NotEvaluated_TrainingEligibilityFailed',limits='Unrun training is not evidence Mobile condition ineffective; no loss-only ablation claimed. No additional steps after Val failure.'))
 write('reports/m5k_sampling_budget.json',dict(kind='actual_GPU_offline_matchedbudget',results={s:asset(f'reports/m5k_sampling_{s}.json')for s in sampling},comparisons=sampling_gains,decision='Go_FurtherSamplingResearchOnly'if samplego else'NoGo_AdaptiveNotConsistent',limits='P8 interpolation not S4network. Dense maskedprojection timing not sparsekernel. Oracle2D sampling vs Mobile sampling separate. Alltarget denominator inclROI misses.'))
 write('reports/m5k_bev_spatial_quality.json',dict(kind='actual_GPU_fixed_Mobile_rawmass',depth_gains=gains,sampling_comparisons=sampling_gains,results={s:asset(f'reports/m5k_alignment_{s}.json')for s in ['val','test']},details={s:asset(f'reports/m5k_details_{s}.json')for s in ['val','test']},status='Partial',limits='Conditionalsurface P, not fullspace obstacle accuracy. Unknown/mixed not safe negative. Partial unless allpredeclared near/scene/Actor/support gates pass; no aggregate-only promotion.'))
 write('reports/m5k_shadow_comparison.json',dict(kind='offline_no_control_frozen_Mobile_depth_sampling',test=aligned['test']['bev'],sampling=asset('reports/m5k_sampling_test.json'),residual=asset('reports/m5k_residual_test.json')if residual else None,input_pipeline=asset('reports/m5k_input_pipeline.json'),control_commands_sent=0,runtime='PartialInherited',limits='Offline shadow only, no controllerinput. FullGPU cost includesbothRGBbranches. No visualsafeclosedloop claim.'))
 acceptance=dict(stage='M5-K',status='Partial',gates=dict(engineering='Passed',stride_alignment='NoDefaultUpgrade',actor_support='PassedLimitedResearch'if decision['run_residual']else'Partial_TrainingBlocked',depth_residual='ExecutedBoundedResearch_NotDefault'if residual else'NotRun_SupportGate',sampling='ResearchGo'if samplego else'NoGo',BEV_spatial_quality='Partial',occupancy_joint_pilot='NoGo',runtime='PartialInherited_OfflineInputImproved',oracle_safety='NotRevalidated',pure_visual_safety='NotValidated'),default_changes=False,visual_control=False,large_training=False,pretest_freeze=asset('reports/m5k_pretest_freeze.json'),source_sha256=file_hash(__file__))
 write('reports/m5k_acceptance.json',acceptance)
 lines=['# M5-K — Target-Aware Depth Localization & Adaptive Spatial Sampling','','**总体 Partial；默认架构和控制系统未改，Occupancy 联合 Pilot 不批准。**', '', '## 实际执行与支持','','实际 RTX4090 无训练对齐诊断、CARLA 新 Train/Val/Test 各4位置各104帧（每位置26配置），均与历史位置隔离≥45m。新测试在科学配置冻结后采集一次。可靠可见首表面参考不是独立 Mesh 真值。','', '|集合|20–35m严格Actor P8射线|位置|Vehicle/Person/Cone|训练支持门禁|','|---|---:|---:|---|---|']
 for s,d in support.items():
  e=d['eligibility'];lines.append(f"|{s}|{e['strict_actor_inBEV_20_35']}|{e['positions']}|{e['by_class']}|{e['passed']}|")
 lines+=['',f"训练决策：**{decision['status']}**。严格 Mask 与最低支持未调整。",'', '## 固定射线 Depth 与 BEV','','|方法|Actor20–35m MAE|Actor20–35m 1m概率|Mobile101 P/R1|Mobile102 P/R1|','|---|---:|---:|---|---|']
 for method in ['original_p8','native16_to_p8','replay101_p8','replay102_p8']:
  r=find(aligned['test']['depth'],method=method,scope='actor',band='20_35');a=find(aligned['test']['bev'],method=method,seed=101);b=find(aligned['test']['bev'],method=method,seed=102);lines.append(f"|{method}|{r['expected_mae']:.3f}m|{r['space1']:.2%}|{a['precision']:.2%}/{a['recall1']:.2%}|{b['precision']:.2%}/{b['recall1']:.2%}|")
 lines+=['','以上所有 P8 深度指标均用严格且 GT 在 BEV 内的同一射线分母。Mobile LearnedDepth 正式投影不接受 GT Mask。S16 原生密度另报。Replay 仍是历史诊断，不升级。','', '## 同预算采样','','|seed|Depth|P8 R1|S4 R1|Adaptive R1|Uniform R1|','|---:|---|---:|---:|---:|---:|']
 for seed in [101,102]:
  for dep in ['oracle','original','aligned']:
   values=[find(sampling['test']['records'],seed=seed,depth=dep,sampling=m)['recall1']for m in ['p8','full_s4','adaptive','uniform']];lines.append(f'|{seed}|{dep}|'+ '|'.join(f'{v:.2%}'for v in values)+'|')
 lines+=['','Adaptive 与 Uniform 逐帧额外射线数精确相等；没有 ROI 的目标仍在分母。Learned S4 只是概率插值；当前 masked dense kernel 未实现稀疏加速。Oracle/采样改善不自动代表真实 BEV 资格。','', '## 输入路径与验证','','端到端逐次测量，包含转换/H2D/完整推理，非相加模块 P95。实际 BGRA 传感器数据回放不含在线 tick/队列/控制。','', '|seed|路径|P95 ms|Max ms|>50ms /40|','|---:|---|---:|---:|---:|']
 for r in runtime['records']:lines.append(f"|{r['seed']}|{r['mode']}|{r['complete_pipeline_ms']['p95']:.2f}|{r['complete_pipeline_ms']['max']:.2f}|{r['over50ms']}|")
 lines+=['','全量测试与 SDK 专项结果在 `m5k_tests.json`；原始失败记录保留。历史资产完整性在 `m5k_artifact_manifest.json`。','', '## 七项结论','','1. S16→P8 对齐不能仅凭 MAE 下降升级；正确空间质量和两个 Mobile 种子的真实 BEV 指标决定资格。无法从此干预给出“全部错误中多少百分比由特征分布导致”的因果比例。','2. 当前没有同时通过近场、Bias、Actor 空间质量与真实 BEV 门禁的升级候选。','3. '+('已执行同容量 B/C 条件实验，详见深度空间消融；训练成功不代表资格通过。'if residual else'支持门禁未通过，Mobile 条件残差未训练；其有效性仍未验证，不能称无效。'),'4. 自适应采样研究决策：'+('仅允许后续采样研究，不能据此采用 S4 网络。'if samplego else'同预算收益未稳定满足门禁，保持 P8。'),'5. 深度 MAE 与空间概率/BEV Precision、Recall 并不等价；详见同射线配对表，不使用 Oracle2D 单项改善替代实际 Mobile 验收。','6. 文件读取/SHA 和每帧序列化可以从未来在线输入路径移出；实际回放结果支持优化方向，但不证明严格实时驾驶。','7. 不批准 Occupancy 联合 Pilot；先解决报告中的空间支持及定位门禁，再单独申请有限规模研究。','', '复现见 [实施文档](../docs/M5K_IMPLEMENTATION.md)，定义与门禁见 [架构决策](../docs/M5K_ARCHITECTURE_DECISION.md)。逐帧输出与完整 SHA 索引保留服务器，Git 仅精简报告与图片。']
 Path('reports/m5k_summary.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
