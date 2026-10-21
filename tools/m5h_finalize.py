"""Aggregate frozen actual experiments without changing decisions or thresholds."""
import json,subprocess
from pathlib import Path
import numpy as np
from simulator.recorder import atomic_json,file_hash
from tools.m5h_freeze import verify

def read(name):return json.loads(Path('reports',name+'.json').read_text())
def weighted_pairs(result):
    ps=result['pairs'];n=sum(p['removed_n']for p in ps);return dict(removed_n=n,removed_fp=sum(p['removed_fp']for p in ps),removed_fpr=sum(p['removed_fp']for p in ps)/n if n else None)
def ratio(a,b):return a/b if b else None

def main():
    dest=Path('reports/m5h_acceptance.json')
    if dest.exists():raise FileExistsError(dest)
    frozen=verify();pix=read('m5h_pixel_test');teacher=read('m5h_teacher_test');depth=read('m5h_depth_test');shadow=read('m5h_shadow_comparison');train=read('m5h_distillation_ablation');dt=read('m5h_depth_preservation');lat=read('m5h_inference_latency');data=read('m5h_dataset_quality');coverage=read('m5h_training_coverage');roi=read('m5h_roi_diagnostics');ps=[]
    for r in pix['records']+teacher['records']:
        q=r['result'];ps.append(dict(name=r['name'],seed=r['seed'],threshold=q['threshold'],metrics=q['metrics'],removed=weighted_pairs(q),fp_per_million=q['false_foreground_per_million_valid_pixels'],actor_miss_rate=q['actor_miss_rate'],observable_actor_views=q['observable_actor_views'],false_components=sum(x['false_components']for x in q['false_detection_components']),by_scene=q['by_scene'],empty_by_scene=q['empty_by_scene'],low_fpr_locked=q['low_fpr_locked_evaluation'],supported_metrics=r['supported']['metrics'],supported_removed=weighted_pairs(r['supported']),class_metrics={k:v['metrics']for k,v in r.get('classes',{}).items()}))
    atomic_json('reports/m5h_pixel_quality.json',dict(kind='offline_summary_actual_frozen_test_GPU',records=ps,validation_report='m5h_pixel_val.json',test_report='m5h_pixel_test.json',teacher_report='m5h_teacher_test.json',limits='Teacherprimarysupportedvehicle/person, Studentprimaryall3classes; use supported_metrics for teachercomparison. Sixindependentpositions; pixels/views notindependentsamples.'))
    da=depth['aggregate'];near={(r['name'],r['seed']):r['metrics']['expected_mae']['mean']for r in da if r['scope']=='all'and r['distance']=='1.5_10'};orig=near[('original',0)];nearchecks=[dict(name=k[0],seed=k[1],near_mae=v,original_near_mae=orig,ratio=v/orig,passed=v<=orig*1.05)for k,v in near.items()if k[0]!='original'];scenechecks=[]
    for group in sorted({r['group']for r in depth['records']}):
        per=[]
        for name,seed in near:
            rr=[r for r in depth['records']if r['group']==group and r['name']==name and r['seed']==seed and r['scope']=='all'and r['distance']=='1.5_10'];n=sum(r['n']for r in rr);per.append(dict(name=name,seed=seed,near_mae=sum((r['expected_mae']or 0)*r['n']for r in rr)/n if n else None,n=n))
        base=next(r['near_mae']for r in per if r['name']=='original');scenechecks.extend(dict(group=group,**r,ratio=r['near_mae']/base,passed=r['near_mae']<=1.05*base)for r in per if r['name']!='original')
    db=[]
    for name,seed in near:
        rr=[r for r in depth['bev']if r['name']==name and r['seed']==seed];n=sum(r['matching']['tolerances']['1.0']['gt_count']for r in rr);hit=sum(r['matching']['tolerances']['1.0']['matched_gt']for r in rr);db.append(dict(name=name,seed=seed,gt_cells=n,matched_gt=hit,recall_1m=ratio(hit,n)))
    atomic_json('reports/m5h_depth_spatial_quality.json',dict(kind='offline_summary_actual_GPU_frozen_Depth_test',aggregate=da,near_checks=nearchecks,scene_near_checks=scenechecks,oracle2d_learned_depth=db,source_report='m5h_depth_test.json',limits='Neargate pooled plus per-scene disclosed. MAEexpecteddepth notargmax. GTinsideBEV denominators separate. Strict20-35targetcount small, notfar generalizationpass. OriginalS16Valchecks inpreservationreport; originaltestnotreselected.'))
    ss=[]
    for seed in [101,102]:
        for mode in sorted({r['mode']for r in shadow['records']}):
            rr=[r for r in shadow['records']if r['seed']==seed and r['mode']==mode];n=sum(r['matching']['tolerances']['1.0']['gt_count']for r in rr);hits=sum(r['matching']['tolerances']['1.0']['matched_gt']for r in rr);tp=sum(r['metrics']['tp']for r in rr);fp=sum(r['metrics']['fp']for r in rr);neg=sum(r['metrics']['negative']for r in rr);sn=sum(r['supported_matching']['tolerances']['1.0']['gt_count']for r in rr);sh=sum(r['supported_matching']['tolerances']['1.0']['matched_gt']for r in rr);ss.append(dict(seed=seed,mode=mode,gt_cells=n,recall_1m=ratio(hits,n),visible_surface_precision=ratio(tp,tp+fp),visible_background_fpr=ratio(fp,neg),supported_recall_1m=ratio(sh,sn),unknown_high_fraction_mean=float(np.mean([r['unknown_high_fraction']for r in rr]))))
    atomic_json('reports/m5h_shadow_digest.json',dict(kind='offline_aggregation_actual_GPU_shadow',records=ss,source_report='m5h_shadow_comparison.json'))
    ro=[]
    for seed in [101,102]:
        for mode in sorted({r['mode']for r in roi['records']}):
            rr=[r for r in roi['records']if r['seed']==seed and r['mode']==mode];n=sum(r['matching']['tolerances']['1.0']['gt_count']for r in rr);ro.append(dict(seed=seed,mode=mode,mean_rays=float(np.mean([r['ray_count']for r in rr])),max_rays=max(r['ray_count']for r in rr),mean_cpu_geometry_ms=float(np.mean([r['cpu_geometry_ms']for r in rr])),recall_1m=ratio(sum(r['matching']['tolerances']['1.0']['matched_gt']for r in rr),n),roi_target_miss_fraction=1-ratio(sum(r['roi_target_rays']for r in rr),sum(r['s4_target_rays']for r in rr))))
    atomic_json('reports/m5h_roi_digest.json',dict(kind='offline_aggregation',records=ro,limits='OracleDepth no realS4Depthmodel; CPUcost includesdiagnosticmatching/setup.'))
    gates={
      'engineering_interfaces':dict(status='Passed',evidence='268passed5skipped + actualGPUisolation + 23CARLAenvironmentcontracts'),
      'teacher_data_mapping':dict(status='PassedWithLimitations',evidence='RGBonly instancecache, supportedclassagreement, conesGTonly; boundariesUnknown; resizedmetadata corrected'),
      'student_visual_quality':dict(status='Partial',evidence='No jointper-scene P80/R80/EmptyFPR1 pass; KDValAPgain doesnotimprovejointcriterion'),
      'background_false_positives':dict(status='Partial',evidence='LockedlowFPRpoints and RemovedFPR reported; thresholding cannotreplace Recall'),
      'actor_small_target_generalization':dict(status='Partial',evidence='6positions only; rear_righttrain2positiveviews; Actor-viewmisses remain'),
      'near_depth_preservation':dict(status='PassedLimited'if all(r['passed']for r in nearchecks) and all(r['passed']for r in scenechecks)else'Partial',evidence=nearchecks,scope='Poolednewtestcurrent-surfaces; originalS16Val also checked; scene regressions separatelyreported'),
      'midfar_depth_space':dict(status='Partial',evidence='Strict20-35targetsupport disclosed, same-layerprobability alone notspatiallocalization'),
      'joint_BEV_space':dict(status='Partial',evidence='Visiblefirstsurfaceevent metrics; no calibratedoccupancy/safety pass'),
      'six_camera_inference':dict(status='Measured',evidence='m5h_inference_latency.json; isolatedGPUonly notcontrolperiod'),
      'runtime':dict(status='PartialInherited',evidence='Unchanged; no newstressmatrix'),
      'oracle_driving_safety':dict(status='NotRevalidated',evidence='Captureonly noOracleclosedloopretest'),
      'pure_visual_safety':dict(status='NotValidated',evidence='No visualcontrolcommands'),
      'formal_M5_scaleup':dict(status='NotApproved',evidence='2Dquality/spatialreliability unresolved')}
    acceptance=dict(stage='M5-H',status='Partial',base='bf0c38d',implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),gates=gates,actual_execution=dict(gpu='RTX4090',pixel_runs=4,pixel_steps_each=2400,depth_runs=4,depth_steps_each=1200,seeds=[101,102],new_training_positions=2,new_training_frames=16,newtest_positions=data['independent_location_groups'],newtest_frames=data['frames'],egoremoved_reference_frames=88,teacher_carla_finetuning=False,visual_control=False),tests=dict(full_passed=268,full_skipped=5,carla_environment_passed=23,counts_overlap=True),choices=frozen['choices'],source_sha256=file_hash(__file__))
    atomic_json(dest,acceptance)
    lines=['# M5-H — Efficient Visual Distillation & Near-Preserving Depth','', '**总体 Partial。** 完成实际RTX4090匹配蒸馏/深度研究、CARLA六位置封存测试、空间Shadow和独立六摄计时。未开启大型Occupancy、Temporal、SparseQuery或视觉闭环。', '', '基线`bf0c38d`；源码冻结提交`f0c636c`；独立分支`feat/m5h-distill-near-depth`，不合并main。复现和数学定义见[实现文档](../docs/M5H_IMPLEMENTATION.md)。原始模型/报告/数据不覆盖；完整SHA在`m5h_artifact_manifest.json`。', '', '## 实际执行与实验控制','', 'Train从192帧/12位置增加至208帧/14位置，Val96帧/6位置不变；新增两个训练位置补相机盲区。测试72六摄帧来自Town01/02/03/04/05/10HD六位置，每处12配置，另有对应ego-removed参考。不是72个独立场景，资产/布局模板仍重复。新测试采集前锁定全部模型/阈值，之后无重新训练或模型选择。', '', '两个seed×GT/Distill各2400步；Depth Replay/Replay+KL各1200步。Student主干/48LID/HeightSplat/默认控制器不变。Teacher缓存304训练验证帧和72测试帧的RGB-only预测，非每step推理。Teacher支持车辆/行人，cone仅GT监督。官方权重保留本地，不分发。', '', '实际Student每组约39.6–41.5秒，峰值allocated约6.04GiB；Depth每组约28.5–39.6秒，峰值约2.04GiB。含缓存驻留与周期评测，期间部分独立作业并发，不是从原图完整六摄训练吞吐。参数与逐任务Loss、梯度、实际选择step在原始训练JSON。', '', '完整训练覆盖1248/1248视角；右后仅2个正视角，不能声称外观充分。seed101最终选1200步，尚48视角未见；seed102选1600步。最终曝光与所选Checkpoint分开。相较M5G同时改变了样本和采样，不能把差异单独归因于覆盖；教师损失A/B是严格配对因素。', '', '## 蒸馏与低误报','', '|模型|seed|Test P|Test R|AP|整体背景FPR|每百万有效像素FP|Removed局部FPR|Actor-view漏检|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def pct(v):return 'n/a'if v is None else f'{100*v:.2f}%'
    for r in ps:
        m=r['metrics'];lines.append(f"|{r['name']}|{r['seed']}|{pct(m['precision'])}|{pct(m['recall'])}|{pct(m['histogram_ap'])}|{pct(m['fpr'])}|{r['fp_per_million']:.1f}|{pct(r['removed']['removed_fpr'])}|{pct(r['actor_miss_rate'])}|")
    lines+=['', 'Teacher主行仅vehicle/person，Student主行含cone，严格同域对比请读`m5h_pixel_quality.json`的`supported_metrics`。Actor指标为可见>=3像素Actor-view，误检组件不等于物理Actor级Precision。所有主阈值由Val冻结；测试PR曲线只描述，不能重新挑阈值。整体背景FPR分母是全部可靠非目标像素，不是Empty专用分母。每百万有效像素FP分母还包含正像素。', '', '蒸馏Val macro AP分别增加2.34/5.51个百分点，但联合门禁violation两seed均变差；故不启动C困难负样本或Backbone升级。保守正区soft BCE已与硬GT高度重叠，缺少背景/边界/特征知识迁移；这限制了本次方法，不能证明所有蒸馏无效。教师在Train/Val表现明显有域差，具体漏检、小目标、天气与camera审计在`m5h_teacher_quality.json`。低置信/冲突不作为强负样本。','', '## 近场保护与空间概率','',f'新Test原Head近场Expected MAE **{orig:.4f}m**。以下均为Val选中非零step，未用Test选择。','', '|模型|seed|Test近场MAE m|相对原Head|1.05门禁|','|---|---:|---:|---:|---|']
    for r in nearchecks:lines.append(f"|{r['name']}|{r['seed']}|{r['near_mae']:.4f}|{r['ratio']:.3f}|{'Passed'if r['passed']else'Partial'}|")
    lines+=['', '|模型|seed|全局MAE/Bias m|10–20m目标MAE / n|20–35m目标MAE / n|20–35m同层1m质量 / BEV n|','|---|---:|---|---|---|---|']
    for name,seed in near:
        get=lambda scope,dist:next(r for r in da if r['name']==name and r['seed']==seed and r['scope']==scope and r['distance']==dist)
        glob=get('all','1.5_80');a=get('actor','10_20');b=get('actor','20_35');g=glob['metrics'];bm=b['metrics'];lines.append(f"|{name}|{seed}|{g['expected_mae']['mean']:.3f} / {g['signed_bias']['mean']:.3f}|{a['metrics']['expected_mae']['mean']} / {a['n']}|{bm['expected_mae']['mean']} / {b['n']}|{pct(bm['same_layer_1m_mass']['mean'])} / {b['bev_inside_n']}|")
    lines+=['', 'MAE为Softmax期望光轴深度，Mode另报。分层0.5/1/2m概率、GTBin、范围外质量、期望三维误差与严格Mask Oracle2D投影见`m5h_depth_spatial_quality.json`及逐帧原报告。原M4 S16 Val近场也通过，跨场景单独检查，不能只看池化均值。', '', '与M5G相比同时降低LR、取消距离/Actor等权并加入Replay，因此不能单独证明Replay贡献；B/C只隔离KL。近/远GT梯度余弦及KL梯度冲突已实际记录，初始化KL梯度为零。固定KL10保留原分布也约束可改善的近场误差，不是强制旧分布等于GT。条件D有资格但本轮明确延期，先验证B/C泛化，无额外空间损失训练。','', '## 联合BEV空间结果','', '|seed|组合|同层1m Recall|可见表面Precision|确认背景FPR|','|---:|---|---:|---:|---:|']
    for r in ss:lines.append(f"|{r['seed']}|{r['mode']}|{pct(r['recall_1m'])}|{pct(r['visible_surface_precision'])}|{pct(r['visible_background_fpr'])}|")
    lines+=['', '新Pixel是Val选出的GT组，Distill另列；旧Pixel是M5G FPN1200最佳Checkpoint。Oracle2D/Depth是同渲染器诊断，不是独立Mesh真值。正式学生输出无GTMask；Unknown不算安全负例。Teacher+OracleDepth对cone无支持，同域空间结果单列supported。粗高度层中心3D误差仅空间格定位代理，不是精确Actor三维误差。','', '## 推理、ROI与安全','', '|模型|六摄P50 ms|P95 ms|峰值allocated MiB|','|---|---:|---:|---:|']
    for r in lat['records']:lines.append(f"|{r['name']}|{r['p50']:.2f}|{r['p95']:.2f}|{r['peak_allocated_mib']:.1f}|")
    lines+=['', '独立GPU计时关闭CARLA和其他GPU作业；不含磁盘/渲染/tick/控制。Student结构不变，蒸馏无额外部署教师。不能由此宣称20Hz控制严格实时。ROI选取仅使用Student输出，OracleDepth离线测射线覆盖，未实现S4Depth网络或替换Splat。','', '|seed|采样|平均射线数|同层1m Recall|目标射线遗漏|','|---:|---|---:|---:|---:|']
    for r in ro:lines.append(f"|{r['seed']}|{r['mode']}|{r['mean_rays']:.0f}|{pct(r['recall_1m'])}|{pct(r['roi_target_miss_fraction'])}|")
    lines+=['', '## 验收、失败记录和下一步','', '全套268 passed、5 skipped；CARLA环境相关23 passed（与全套有重叠）；实际CUDA隔离验证通过。合成单测只验证Mask/梯度/计数合同，不能替代CARLA质量实测。采集/训练/冻结/评测完整日志在`reports/m5h_logs`。', '', '保留初版地图名/世界坐标混用造成的错误审计，最终隔离依据`m5h_split_audit_final.json`的Pixel距离和`m5h_replay_coordinate_audit.json`更正的M4距离；最小距当前Train/Val/M4 Replay为49.57m。教师名义resize记录独立更正；首轮Train自选阈值指标不用于模型选择，采用`teacher_train_locked`替代一致工作点。某些报告在结束时对源文件取SHA，期间仅增加分组输出字段；实际执行版本归档`runs/m5h_source_archive`，冻结后科学代码不变。', '', '**七项结论：**', '', '1. 本次正区软分数蒸馏尚未提供稳定联合质量增益：seed101 P29.12→29.26%、R28.96→26.04%，seed102 P40.83→46.00%、R37.71→48.65%。结构和推理成本保持不变。Teacher更强不等于这种KD目标足够。', '2. 仅seed102同时改善P/R/Removed（局部FPR8.56→2.31%）；seed101局部FPR10.63→8.86%但Recall下降。两个seed不稳定，不能宣称新位置可靠。', '3. 完整预算曝光确实提升到100%，但最佳早期Checkpoint仍可能未全见；数据增补与采样未单独消融。Student未更换视觉预训练权重，教师loss才是A/B单变量。', '4. 本轮降低LR、均匀射线CE+原分布Replay建立了通过Val的近场保护候选；新Test严格结果见近场表，默认Head仍不覆盖。', '5. BEV收益需同时看空间Recall/Precision/背景；近场MAE改善不能自动视为Actor空间定位通过，联合门禁继续Partial。', '6. 高分辨率采样仍是独立瓶颈，但二维识别也未可靠，不能宣称瓶颈已经完全转移到投影。预测ROI遗漏会限制更密采样收益。', '7. 继续受控视觉特征/可靠负例覆盖与Depth空间概率研究，暂不扩大正式BEV/Occupancy训练，也不自动启动Temporal或Sparse Query。', '', '工程接口、研究执行、泛化质量、运行时与纯视觉安全分别验收；详见`m5h_acceptance.json`。']
    Path('reports/m5h_summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(pixel=ps,near=nearchecks,shadow=ss,roi=ro),ensure_ascii=False))
if __name__=='__main__':main()
