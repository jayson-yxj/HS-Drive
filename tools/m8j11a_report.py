"""Aggregate-only closeout. Private predictions, labels, images and memory remain private."""
from pathlib import Path
import json
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.m8j11a_common import read,write,sha,PROTOCOL
from training.m8j7_metrics import distribution


def main():
    cfg=read(PROTOCOL);r=read('reports/m8j11a_comparison.json');v=read('reports/m8j11a_validation.json');root=Path(cfg['private_root'])
    tests=read('reports/m8j11a_sources/final_contract_tests.json');assert tests['exit_code']==0
    assert '24 passed, 1 skipped' in Path(tests['log_path']).read_text()
    intervals={};ages={};resources=v['resources'];runs={t:read(root/t/'result.json') for t in ('T1','T2','T3')}
    for t,rows in cfg['schedule'].items():
        intervals[t]=distribution([x['input_dt_s'] for x in rows if x['input_dt_s'] is not None])
        mem=[json.loads(l) for l in (root/t/'memory.jsonl').read_text().splitlines()]
        ages[t]=distribution([m['age_seconds_max'] for m in mem if m['age_seconds_max'] is not None])
    support=read('reports/m8j11_sources/official_filters.json')
    figdir=Path('reports/m8j11a_figures');figdir.mkdir(exist_ok=False)
    fig,axes=plt.subplots(2,3,figsize=(15,7))
    for row,(scene,data) in enumerate(r['recovery'].items()):
        seq=data['remaining_frames'];x=[f['elapsed_s'] for f in seq]
        ys=[[f['T2']['FP']-f['T0']['FP'] for f in seq],
            [(f['T2']['recall']-f['T0']['recall'])*100 for f in seq],
            [f['paired']['paired_errors']['xy_m_B_minus_A']['mean'] for f in seq]]
        for col,(y,label) in enumerate(zip(ys,['FP delta','Recall delta (pp)','Common GT XY error delta (m)'])):
            ax=axes[row,col];ax.plot(x,y,'o-');ax.axhline(0,color='gray',linestyle='--');ax.set_title(scene+' '+label);ax.set_xlabel('Seconds after reset')
    fig.suptitle('T2 minus T0: two correlated resets, no recovery deadline certification');fig.tight_layout();fig.savefig(figdir/'reset_trends.png',dpi=140);plt.close(fig)
    acceptance=dict(stage='M8-J11A',client_task_date='2026-11-01',
        run_ids={t:runs[t]['run_id'] for t in runs},
        host_execution_clock={t:read('reports/m8j11a_sources/'+t+'_run.json')['host_utc'] for t in runs},
        clock_note='Client task date and recorded execution host UTC date are both 2026-11-01; exact unmodified timestamps are retained in phase logs.',
        InterventionImplementation='Passed_ObservedHeadStateAndExactCrossChecks',
        ExperimentCoverage='Completed_T1_81_T2_81_T3_41',
        TemporalSensitivity='Measured_HistoryRemovalAndSparseHistoryDegradation_ResetEffectsMixed_NoRobustnessCertification',
        actual_forwards=203,extra_forwards=0,warmup=0,total_budget=243,
        training_steps=0,backward_calls=0,CARLA=False,dependency_changes=False,official_source_changes=False,
        reference='PETR-A retained; T0 comparison in this experiment is StreamPETR only',
        model_default_changed=False,main_updated=False,branch_deleted=False,
        merge_eligibility='CodeReady_DefaultUnchanged_RelevantTestsPassed_UserConfirmationPending',
        tests=dict(passed=24,skipped=1,skip='Existing CUDA Attention test not rerun; actual203FP32 forwards separately executed',full_model_mock_forwards_counted=False),
        paired_consistency_checks=v['all_pairwise_checks'],paired_max_abs=v['max_pairwise_abs'],atol=cfg['atol'],rtol=cfg['rtol'],
        parameter_buffer_integrity=True,GT_input=False,pose_source=cfg['pose_source'],T3_official_eval='NotRun_Intentional41FrameProjectOnly',
        intervals_s=intervals,oldest_observed_age_s=ages,resources=resources,
        private_root=str(root),visual_groups=6,actual_visual_inspection=dict(reviewer='Assistant visual tool; not human signoff',
            files=['T2_20_six.jpg','T3_70_bev.jpg'],observations=['Reset view shows extra predictions around visible traffic; no geometry correction applied',
                'Sparse-history BEV shows denser overlapping prediction clusters in selected scene'],other_four_groups='Generated_NotClaimedReviewed'),
        execution_source_commit=runs['T2']['code_commit'],configuration_sha256=sha(PROTOCOL),
        limits=['Only two reused development scenes; no independent safety/generalization inference',
            'T1 uses temporally trained weights; not fairly retrained single-frame model',
            'T3 jointly changes observation interval and memory time span',
            'No evidence of universal four-frame recovery; scene and class effects differ',
            'Instrumented cache timings are not comparable with J11 raw-image timing'],
        closeout='Finish this experiment; no training repairs or extension. Next separately scoped direction: SparseDriveV2 Bench2Drive branch integration review; not executed or approved here')
    write('reports/m8j11a_acceptance.json',acceptance)
    official_rows=[]
    for t,m in r['official'].items():
        vals=[m['mean_ap'],m['nd_score']]+[m['tp_errors'][k] for k in ('trans_err','scale_err','orient_err','vel_err','attr_err')]
        official_rows.append('|'+t+'|'+'|'.join('{:.6f}'.format(x) for x in vals)+'|')
    project_rows=[]
    for name,d in [('T0全81',r['project']['T1']['T0']),('T1全81',r['project']['T1']['intervention']),('T2全81',r['project']['T2']['intervention']),
                   ('T0共同41',r['project']['T3']['T0']),('T3共同41',r['project']['T3']['intervention'])]:
        p=d['primary'];z=d['vertical']['all']
        project_rows.append('|{}|{}|{}|{}|{}|{}|{:.2%}|{:.2%}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|'.format(name,p['GT'],p['TP'],p['FP'],p['FN'],p['duplicates'],p['precision'],p['recall'],p['matched_errors']['xy_m']['mean'],p['matched_errors']['xy_m']['p95'],z['abs_dz_m']['mean'],z['abs_dh_m']['mean']))
    pair_rows=[]
    for t,p in r['project'].items():
        q=p['paired'];e=q['paired_errors']['xy_m_B_minus_A']
        pair_rows.append('|{}|{}|{}|{}|{}|{:+.4f}|'.format(t,q['common'],q['A_only'],q['B_only'],q['neither'],e['mean']))
    reset_rows=[]
    for scene,values in r['recovery'].items():
        for x in values['selected_offsets']:
            reset_rows.append('|{}|{}|{:.6f}|{}→{}|{}→{}|{:+.4f}|'.format(scene,x['offset'],x['elapsed_s'],x['T0']['TP'],x['T2']['TP'],x['T0']['FP'],x['T2']['FP'],x['paired']['paired_errors']['xy_m_B_minus_A']['mean']))
    class_rows=[]
    for c in read(cfg['source_protocol'])['classes']:
        ap=[r['official'][t]['mean_dist_aps'][c] for t in ('T0','T1','T2')]
        class_rows.append('|{}|{} / {}|{:.4f}|{:.4f}|{:.4f}|'.format(c,support['official_GT_filtered']['classes'][c],support['GT_instances_by_class'][c],*ap))
    resource_rows=[]
    for t,x in resources.items():
        d=x['summary'];resource_rows.append('|{}|{}|{:.2f}|{:.2f}|{:.3f}|{:.3f}|'.format(t,x['actual_forwards'],d['gpu_instrumented_ms']['median'],d['gpu_instrumented_ms']['p95'],d['peak_allocated_bytes']['max']/2**30,d['peak_reserved_bytes']['max']/2**30))
    text='''# M8-J11A：冻结StreamPETR时序敏感性实验

**干预实现通过，三个实验全部完成；观测到历史依赖和稀疏历史退化，不认证时序鲁棒性或驾驶安全。** 单一分支 `feat/streampetr-temporal-sensitivity`；基础main `1d7472dea3941132a0e5946dc7f80feddc26b4f6` 与执行前远程一致。PETR-A继续参考，默认运行模式不变。

任务客户端日期与实际执行日志的UTC日期均为2026-11-01；精确时间保留在原始阶段记录中。GPU执行源码 `6e36963b96b76e1cb68b84ff445eabed3a3ee6a6`；评测源码与命令逐阶段记录于 `m8j11a_sources`。

## 固定输入与预算

唯一权重SHA `e6323ae5c31adf1eedd46d6dd4fd3c73d95aa26f18cc8aa23c196494b7de3451`，官方源码 `95f64702306ccdb7a78889578b2a55b5deb35b2a`，已验证非Flash FP32后端，bs1，256×704×6，TF32关闭。与T0相同的resolved config SHA `7d28a79387a745dda53fd713c769d229b08869102f29be3fe60bce90fc8ba3c3`。全部77个固定文件、81个T0缓存和输入manifest按哈希验证；没有PETR-A/B/C缓存混入T0。

清单先于任何前向冻结：scene-0103为40帧、scene-0916为41帧，按真实时间排序。T2各scene零基20重置，即全局20/60；T3保留0,2,4,…，20+21=41帧。配置记录完整sample/scene/索引/时间间隔及run_id，不按GT或预测选样。新前向按T2→T1→T3执行81+81+41=203，预热0、额外诊断0，未使用剩余40次预算。每次simple_test前持久化预约，失败尝试也计数；无optimizer/backward。

自车位姿来自nuScenes标定及ego_pose，沿用LIDAR_TOP参考坐标到global变换，不是视觉里程计。纯视觉分支只读取既有RGB/标定/真实时间/位姿缓存；GT在独立CPU评测进程加载，点云/Radar载荷未进入推理。真实相机时间戳及位姿未经修改。

## 重置与状态证据

官方 `simple_test_pts` 会覆盖外部prev_exists，因此干预使用 `prev_scene_token=None` 加官方 `reset_memory()`，随后由官方入口处理真实scene token。只设置输入prev_exists=0不能替代该过程。适配器默认normal，不修改官方模型和历史runner。

实际状态包括检测器prev_scene_token/test_flag/single_test，以及memory_embedding/reference_point/timestamp/egopose/velo。五类记忆一起清空；cold路径合法添加256个pseudo-reference及单位位姿，不把非零初始参考点误判为旧历史。无独立逐槽可见性标记，实际有效性依赖prev_exists及已写入历史；padding不计观测。每帧读1024槽、写入256个提议，写完保存1280槽。

memory_timestamp在post-update后保存负的来源秒数；下一帧加当前时间变为相对年龄。检查器按实际写入时间逐槽核验，没有将相对年龄当绝对时间比较。历史scene和时间、跳帧未参与、头入口prev_exists、参数/注册buffer均核验。官方实现只按scene变化自动清空，没有时间间隔触发清空的规则，本轮T3也未观测到该类重置。

__PAIRED_COUNT__个缓存配对检查全部逐数组相同（最大差0，仍保留原atol=rtol=1e-5）：T2重置前40帧、T1/T3各两场景首帧、T2两次重置与T1同帧。40+4+2=46个检查（T2首帧已含在40帧中）。参数/固定buffer初末指纹均为 `b4a9e08d2db06e0dbafd8ae45633ef8e249e8b2545425b916e31b2a01edaedef`。没有跨场景、跨实验或未来观测进入记忆。

## 官方完整81帧评价

nuScenes-devkit1.1.9 / detection_cvpr_2019，官方GT3393；固定Top300、0.2阈值、范围和属性/速度导出，没有NMS或阈值搜索。T0直接读取精确旧指标；T1/T2为人为干预，不是自然运行或排行榜成绩。

|条件|mAP|NDS|mATE m|mASE|mAOE rad|mAVE m/s|mAAE|
|---|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(official_rows)+'''

|类别|官方GT记录 / 独立instance|T0 AP|T1 AP|T2 AP|
|---|---:|---:|---:|---:|
'''+ '\n'.join(class_rows)+'''

零支持类保留官方平均行为，N/A误差在官方原文件保留NaN，聚合JSON用null；公交仅一个独立实例。T3未更改官方split、未补空预测，只做41帧项目共同子集诊断。

## 固定工作点与匹配集合变化

同类一对一XY<2m；完整项目GT3559，稀疏共同子集GT1795，与官方3393不混用。误差统计只针对各自匹配目标，配对统计另列。GT记录含同一实例连续帧，不能当作独立物体。

|条件|GT|TP|FP|FN|重复|Precision|Recall|XY均值m|XY P95m|absΔz均值m|absΔh均值m|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(project_rows)+'''

|干预相对同帧T0|共同匹配|仅T0|仅干预|两者未匹配|共同GT平均XY误差变化m|
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(pair_rows)+'''

T1：两个场景召回均下降，汽车/行人贡献大部分新增FP；共同GT平均XY误差+0.2004m。支持“该时序权重正常运行依赖历史”的有限结论，不能当作时序模型与公平重训单帧模型的优劣结论。

T2：整体TP相同，但70条T0-only与70条T2-only互相抵消，新增573条FP、106个重复。scene-0103召回86.80%→88.16%，scene-0916召回89.16%→88.03%；公交召回提高、摩托车召回下降。整体召回相同不表示每个对象或重置后行为相同。

T3：相同41帧上FP1668→4460、重复407→2272，TP净减少26。scene-0103召回下降，scene-0916召回提高但误检显著增加；增加FP主要来自汽车和行人。共同目标XY误差+0.1202m。直接证据是稀疏输入导致预测集合、重复和位置变化；“过旧特征、传播Query竞争”等具体内部原因尚未单独隔离，不能断言。实际输入间隔中位数0.499881→0.999768秒；T2正常历史读取的最老年龄中位数约1.999503秒，T3约3.999175秒。固定容量下观测间隔和历史时间跨度同时扩大，不是固定时间窗消融。

## 两次重置的局部影响

|场景|重置后偏移帧|实际经过秒|TP T0→T2|FP T0→T2|共同GT平均XY误差变化m|
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(reset_rows)+'''

重置后第4帧约2秒时，两个场景FP仍分别30→77、25→50；不能宣称四帧内稳定恢复。后段scene-0103在约7.65–9.55秒的最后5帧FP差为0、+1、+2、0、−3；scene-0916在约8.00–10.00秒为−1、+3、−2、−2、−1，差异缩小但不单调。全部剩余20/21帧趋势保留在JSON和下图；单帧超过T0不被当成恢复认证。

![T2 local change](m8j11a_figures/reset_trends.png)

## 资源与验证

|条件|新前向|仪器化GPU区间P50 ms|P95 ms|峰值allocated GiB|峰值reserved GiB|
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(resource_rows)+'''

CUDA事件区间包含状态审计中的同步和CPU检查空隙；每帧wall还包含缓存读取、H2D、输出、完整冻结检查和落盘。实际本轮GPU P95约29–34ms，与历史J11约160ms并不接近；除边界不同外，跨运行环境负载/时钟等差异尚未归因，不宣称这里获得了加速。T2首帧仪器化区间约2784ms的长尾也保留。本轮未为计时增跑，不能与J11原始图像完整路径或无仪器化GPU P95直接比较，也不能据此升级实时门禁。

24项有限测试通过（含官方pre_update_memory小张量冷初始化与历史合同回归），1项已有CUDA Attention测试跳过；203次真实RTX4090 FP32前向单独记录。没有训练、CARLA、依赖升级、架构/解码/默认后端修改，旧来源及报告哈希保持不变。

6组私有对照按运行前固定索引生成：T1/10、T2/19/20/59/60、T3/70。私有入口 `'''+str(root/ 'visuals/index.html')+'''`。助手实际查看T2_20六摄与T3_70 BEV，仅作解释；其他生成内容不称人工复核，未要求或伪造签核。

## 结案与分支处理

A InterventionImplementation：通过。B ExperimentCoverage：203/203完成。C TemporalSensitivity：历史移除和稀疏历史出现明确损失，中途重置影响局部明显且场景/类别混合；不作鲁棒性全面通过或架构无效结论。

代码具备合并条件：默认关闭、历史源码不变、相关回归和真实状态检查通过。建议用户一次确认后，将本功能分支最终提交快进main（若main前进则先检查兼容性），核验远程main包含后，以准确预期tip保护删除远程和本地 `feat/streampetr-temporal-sensitivity`。不新增实验分支或tag，不因性能下降长期保留分支。本轮不自动合并或删除。

本实验线到此结束，不用剩余预算追加修复训练。下一步转向另行限定范围的SparseDriveV2 Bench2Drive分支接入评估；本轮未执行该接入，不以消除全部时序敏感性为前置条件。PETR-A和默认运行模式继续不变。

[验收](m8j11a_acceptance.json) · [完整聚合统计](m8j11a_comparison.json) · [交叉验证及资源](m8j11a_validation.json) · [实现与复现](../docs/M8J11A_IMPLEMENTATION.md)
'''
    text=text.replace('__PAIRED_COUNT__',str(v['all_pairwise_checks']))
    Path('reports/m8j11a_summary.md').write_text(text)
    print('Report complete; merge pending user confirmation')

if __name__=='__main__':main()
